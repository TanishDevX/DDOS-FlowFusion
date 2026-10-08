"""
scripts/train_flow_fusion_v2.py
--------------------------------
FlowFusion v2 training script — Optimized Multimodal DDoS Detection.

Runs the all_gated configuration (all 3 branches, contextual scalar gating)
with all v2 enhancements enabled:
  - ClassWeightedFocalLoss (gamma=2.0, per-class alpha from inverse frequency)
  - 3-stage CNN Branch [64,128,256] with BatchNorm
  - d_token=128, d_group=64, n_layers=4, n_group_layers=3
  - Cross-branch multi-head attention fusion
  - 120 epochs, early_stopping patience=25, ReduceLROnPlateau fallback

Results saved to:
  experiments/results/flow_fusion_v2/all_gated/results.json

After completion, prints a direct comparison against v1 FlowFusion and
all tree-based baselines.

Usage:
    python scripts/train_flow_fusion_v2.py               # full 5-seed run
    python scripts/train_flow_fusion_v2.py --smoke-test  # 1 seed, 3 epochs
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.augmentation import augment_benign
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.gpu import setup_gpu, log_hardware_summary
from src.models.flow_fusion import (
    build_flow_fusion,
    get_group_indices,
    compute_correlation_reorder,
)
from src.evaluation.metrics import evaluate, aggregate_seeds

logger  = get_logger("train_flow_fusion_v2")
RESULTS = ROOT / "experiments" / "results" / "flow_fusion_v2"
MODELS  = ROOT / "models"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_processed(split: str, mmap_mode=None) -> tuple:
    base          = ROOT / "data" / "processed" / split
    X             = np.load(base / "X.npy", mmap_mode=mmap_mode)
    y             = np.load(base / "y.npy", mmap_mode=mmap_mode)
    label_classes = np.load(base / "label_classes.npy", allow_pickle=True)
    feature_names = np.load(base / "feature_names.npy",  allow_pickle=True)
    logger.info(f"Loaded {split}: X{X.shape}  y{y.shape}  classes={list(label_classes)}")
    return X, y, label_classes, list(feature_names)


# ═══════════════════════════════════════════════════════════════════════════
# Training one seed
# ═══════════════════════════════════════════════════════════════════════════

def train_one_seed(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list,
    label_classes: np.ndarray,
    cfg_ff: dict, cfg_train: dict,
    seed: int,
    seed_idx: int = 1, total_seeds: int = 1,
    smoke_test: bool = False,
) -> tuple:
    set_seed(seed)
    tf.keras.backend.clear_session()

    input_dim   = X_tr.shape[1]
    num_classes = len(label_classes)

    # ── Scaling ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # ── Augmentation ──────────────────────────────────────────────────────
    if cfg_train.get("use_augmentation", False):
        benign_idx = list(label_classes).index("BENIGN")
        X_tr_scaled, y_tr = augment_benign(
            X_tr_scaled, y_tr,
            benign_class_idx=benign_idx,
            gaussian_sigma=cfg_train.get("aug_gaussian_sigma", 0.01),
            mixup_alpha=cfg_train.get("aug_mixup_alpha", 0.2),
            n_gaussian=cfg_train.get("aug_n_gaussian", 5000),
            n_mixup=cfg_train.get("aug_n_mixup", 5000),
            seed=seed,
        )

    # FIN (Branch 2) uses group_indices directly -- no correlation reorder needed
    group_indices = get_group_indices(feature_names)


    # ── v2: Per-class inverse-frequency focal loss alpha ──────────────────
    class_counts = np.bincount(y_tr.astype(int), minlength=num_classes).astype(np.float32)
    class_alpha  = 1.0 / (class_counts + 1e-8)
    class_alpha /= class_alpha.sum()
    logger.info(
        f"  [Seed {seed}] Focal alpha: "
        + "  ".join(f"{label_classes[i]}={class_alpha[i]:.4f}" for i in range(num_classes))
    )

    base_lr      = cfg_ff.get("learning_rate", 1.5e-4)
    total_epochs = 3 if smoke_test else cfg_ff.get("epochs", 120)
    warmup_epochs= cfg_ff.get("warmup_epochs", 8)

    # ── Build model ───────────────────────────────────────────────────────
    model = build_flow_fusion(
        input_dim=input_dim,
        num_classes=num_classes,
        group_indices=group_indices,
        cfg=cfg_ff,
        branch_flags=(True, True, True),
        fusion_mode="gated",
        class_alpha=class_alpha,   # v2 focal loss weights
    )


    # ── Re-compile with updated AdamW (clipnorm) ──────────────────────────
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=base_lr,
        weight_decay=cfg_ff.get("weight_decay", 1e-5),
        clipnorm=cfg_ff.get("clipnorm", 1.0),
    )

    output_names  = model.output_names
    existing_loss = model.loss
    main_acc      = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    if len(output_names) == 1:
        model.compile(optimizer=optimizer, loss=existing_loss, metrics=[main_acc])
        y_tr_fit  = y_tr.astype(np.int32)
        y_val_fit = y_val.astype(np.int32)
    else:
        # Gate outputs have zero-weight loss (they're only for interpretability)
        loss_dict    = {}
        loss_weights = {}
        for name in output_names:
            if name == "output":
                loss_dict[name]    = existing_loss if not isinstance(existing_loss, dict) else existing_loss.get("output", existing_loss)
                loss_weights[name] = 1.0
            else:
                loss_dict[name]    = tf.keras.losses.MeanSquaredError()
                loss_weights[name] = 0.0
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics={"output": [main_acc]},
        )
        y_tr_fit  = {"output": y_tr.astype(np.int32)}
        y_val_fit = {"output": y_val.astype(np.int32)}
        for name in output_names:
            if name != "output":
                y_tr_fit[name]  = np.zeros((len(y_tr),  1), dtype=np.float32)
                y_val_fit[name] = np.zeros((len(y_val), 1), dtype=np.float32)

    # ── Callbacks ─────────────────────────────────────────────────────────
    from src.utils.progress import (
        CosineDecayCallback,
        NaNDetectionCallback,
        CosineGateTemperatureCallback,
        GateMonitorCallback,
        MacroF1ValidationCallback,
        RichTrainingCallback,
    )

    callbacks = [
        CosineDecayCallback(
            base_lr=base_lr,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=cfg_train.get("min_lr", 1e-6),
        ),
        CosineGateTemperatureCallback(
            total_epochs=total_epochs,
            start_temp=cfg_ff.get("gate_temperature_start", 1.0),
            end_temp=cfg_ff.get("gate_temperature_end", 0.3),
        ),
        NaNDetectionCallback(threshold=1e6),
        GateMonitorCallback(),
        MacroF1ValidationCallback(
            val_x=X_val_scaled,
            val_y=y_val,
            num_classes=num_classes,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=25,   # v2: 25 epochs — allow branches time to co-converge
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(  # v2: fallback LR reducer
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=cfg_train.get("min_lr", 1e-6),
            verbose=0,
        ),
        RichTrainingCallback(
            model_name="FlowFusion-v2[all_gated]",
            seed=seed,
            seed_idx=seed_idx,
            total_seeds=total_seeds,
            total_epochs=total_epochs,
        ),
    ]

    # ── Class sample weights (balanced) ───────────────────────────────────
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}
    sample_weight_tr  = np.array([cw_dict[int(c)] for c in y_tr],  dtype=np.float32)
    sample_weight_val = np.array([cw_dict[int(c)] for c in y_val], dtype=np.float32)

    # ── Fit ───────────────────────────────────────────────────────────────
    t0 = time.time()
    model.fit(
        X_tr_scaled.astype(np.float32), y_tr_fit,
        validation_data=(X_val_scaled.astype(np.float32), y_val_fit, sample_weight_val),
        epochs=total_epochs,
        batch_size=cfg_train["batch_size"],
        callbacks=callbacks,
        sample_weight=sample_weight_tr,
        verbose=0,
    )
    elapsed = time.time() - t0
    logger.info(f"  Training done in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── Evaluation ────────────────────────────────────────────────────────
    def _predict(X_input):
        chunk_size = 1_000_000
        all_logits = []
        all_gates  = []
        for i in range(0, len(X_input), chunk_size):
            X_chunk = X_input[i : i + chunk_size]
            X_chunk = scaler.transform(X_chunk).astype(np.float32)
            out = model.predict(X_chunk, batch_size=8192, verbose=0)
            if isinstance(out, dict):
                all_logits.append(out["output"])
                gate_keys = [k for k in out.keys() if k.startswith("gate_")]
                if not all_gates:
                    all_gates = [[] for _ in gate_keys]
                for g_idx, k in enumerate(gate_keys):
                    all_gates[g_idx].append(out[k])
            elif isinstance(out, (list, tuple)):
                all_logits.append(out[0])
                if not all_gates:
                    all_gates = [[] for _ in out[1:]]
                for g_idx, g_val in enumerate(out[1:]):
                    all_gates[g_idx].append(g_val)
            else:
                all_logits.append(out)
        logits_concat = np.concatenate(all_logits, axis=0)
        gates_concat  = [np.concatenate(g, axis=0) for g in all_gates] if all_gates else []
        return logits_concat, gates_concat

    val_out, _      = _predict(X_val)
    val_pred        = np.argmax(val_out, axis=1)
    val_metrics     = evaluate(y_val, val_pred, val_out, label_classes,
                               split_name=f"val(v2,seed={seed})")

    test_out, gates = _predict(X_test)
    test_pred       = np.argmax(test_out, axis=1)
    test_metrics    = evaluate(y_test, test_pred, test_out, label_classes,
                               split_name=f"test(v2,seed={seed})")

    gate_log = {}
    if gates:
        for i, g_arr in enumerate(gates):
            gate_log[f"gate_mean_{i}"] = float(np.mean(g_arr))
        logger.info(
            f"  Gate means (v2): { {k: f'{v:.3f}' for k, v in gate_log.items()} }\n"
            f"  (B1=tabular, B2=spatial/CNN, B3=structural/group-attn)"
        )

    return val_metrics, test_metrics, model, elapsed, gate_log


# ═══════════════════════════════════════════════════════════════════════════
# Comparison table against v1 + baselines
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(test_agg_v2: dict):
    """Print v2 vs v1 FlowFusion and all tree baselines."""
    comparisons = [
        ("FlowFusion v2 [ALL_GATED]", test_agg_v2, "⭐"),
    ]

    # Load existing results if available
    for label, path in [
        ("FlowFusion v1 [all_gated]", ROOT / "experiments/results/flow_fusion/all_gated/results.json"),
        ("FT-Transformer",             ROOT / "experiments/results/ft_transformer/results.json"),
        ("LightGBM",                   ROOT / "experiments/results/lightgbm/results.json"),
        ("XGBoost",                    ROOT / "experiments/results/xgboost/results.json"),
        ("FlowMLP",                    ROOT / "experiments/results/mlp/results.json"),
    ]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                comparisons.append((label, data.get("test_aggregated", {}), "  "))
            except Exception:
                pass

    print("\n" + "=" * 85)
    print("  FLOWFUSION v2 vs BASELINES — FINAL COMPARISON")
    print("=" * 85)
    header = f"  {'Model':<35} {'Accuracy':>10} {'Macro-F1':>10} {'ROC-AUC':>10}"
    print(header)
    print("  " + "-" * 80)

    for label, agg, star in comparisons:
        acc  = agg.get("accuracy",  {}).get("mean", 0.0)
        f1   = agg.get("macro_f1", {}).get("mean", 0.0)
        auc  = agg.get("roc_auc",  {}).get("mean", 0.0)
        acc_s = agg.get("accuracy",  {}).get("std",  0.0)
        f1_s  = agg.get("macro_f1", {}).get("std",   0.0)
        print(f"  {star} {label:<34} {acc:.4f}±{acc_s:.4f}  {f1:.4f}±{f1_s:.4f}  {auc:.4f}")

    print("=" * 85)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train FlowFusion v2 (optimized multimodal DDoS detection).")
    parser.add_argument("--smoke-test", action="store_true", help="1 seed, 3 epochs — fast sanity check")
    args = parser.parse_args()

    setup_gpu()
    log_hardware_summary()

    t_total = time.time()

    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    cfg_ff    = dict(cfg["models"]["flow_fusion"])
    cfg_train = dict(cfg["training"])

    seeds     = [42] if args.smoke_test else cfg_train["seeds"]
    val_split = cfg["data"]["val_split"]

    logger.info(f"\nFlowFusion v2 Configuration:")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  d_token={cfg_ff['d_token']}  n_layers={cfg_ff['n_layers']}")
    logger.info(f"  B2=FIN(d_inter={cfg_ff.get('d_interaction', 64)}, heads={cfg_ff.get('n_interaction_heads', 4)}, layers={cfg_ff.get('n_interaction_layers', 2)})  d_group={cfg_ff['d_group']}")

    logger.info(f"  use_focal_loss={cfg_ff.get('use_focal_loss', True)}")
    logger.info(f"  use_cross_branch_attention={cfg_ff.get('use_cross_branch_attention', True)}")
    logger.info(f"  epochs={cfg_ff['epochs']}")

    # ── Load data ─────────────────────────────────────────────────────────
    X_full, y_full, label_classes, feature_names = load_processed("train")
    X_test, y_test, _, _                         = load_processed("test", mmap_mode="r")

    # Stratified test sampling (2M rows = same metrics, faster eval)
    MAX_TEST = 2_000_000
    if len(X_test) > MAX_TEST:
        rng     = np.random.default_rng(42)
        classes = np.unique(y_test)
        n_per   = MAX_TEST // len(classes)
        idx = np.concatenate([
            rng.choice(np.where(y_test == c)[0],
                       size=min(n_per, int((y_test == c).sum())),
                       replace=False)
            for c in classes
        ])
        rng.shuffle(idx)
        X_test = np.array(X_test[idx])
        y_test = y_test[idx]
        logger.info(f"Test stratified-sampled: {len(X_test):,} rows")

    print("\n" + "=" * 70)
    print("      FLOWFUSION v2 — MULTIMODAL DDOS DETECTION BENCHMARK       ")
    print("      Focal Loss | 128-dim | 3-stage CNN | Cross-Branch Attn    ")
    print("=" * 70 + "\n")

    # ── Train all seeds ───────────────────────────────────────────────────
    val_results_all  = []
    test_results_all = []
    train_times      = []
    all_gate_logs    = []
    best_val_f1      = -1.0
    best_model       = None

    for seed_idx, seed in enumerate(seeds, 1):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_full,
            test_size=val_split,
            stratify=y_full,
            random_state=seed,
        )

        val_m, test_m, model, elapsed, gate_log = train_one_seed(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            feature_names, label_classes,
            cfg_ff, cfg_train,
            seed=seed, seed_idx=seed_idx, total_seeds=len(seeds),
            smoke_test=args.smoke_test,
        )

        val_results_all.append(val_m)
        test_results_all.append(test_m)
        train_times.append(elapsed)
        all_gate_logs.append(gate_log)

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_model  = model

        logger.info(
            f"  [Seed {seed}] val_f1={val_m['macro_f1']:.4f}  "
            f"test_f1={test_m['macro_f1']:.4f}  "
            f"DrDoS_MSSQL_f1={test_m.get('f1_DrDoS_MSSQL', 0.0):.4f}"
        )
        tf.keras.backend.clear_session()

    # ── Aggregate ─────────────────────────────────────────────────────────
    val_agg  = aggregate_seeds(val_results_all)
    test_agg = aggregate_seeds(test_results_all)

    # ── Save best model ───────────────────────────────────────────────────
    model_path = MODELS / "flow_fusion_v2_all_gated_best.keras"
    if best_model is not None:
        best_model.save(str(model_path))
        logger.info(f"  Best model saved → {model_path}")

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = RESULTS / "all_gated"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":             "FlowFusion_v2",
        "ablation":          "all_gated",
        "branch_flags":      [True, True, True],
        "fusion_mode":       "gated",
        "seeds":             seeds,
        "v2_enhancements": {
            "focal_loss":              cfg_ff.get("use_focal_loss", True),
            "focal_gamma":             cfg_ff.get("focal_gamma", 2.0),
            "cross_branch_attention":  cfg_ff.get("use_cross_branch_attention", True),
            "d_token":                 cfg_ff["d_token"],
            "b2_branch":               "FIN",
            "d_interaction":           cfg_ff.get("d_interaction", 64),
            "n_interaction_layers":    cfg_ff.get("n_interaction_layers", 2),
            "n_group_layers":          cfg_ff["n_group_layers"],
        },

        "val_aggregated":    val_agg,
        "test_aggregated":   test_agg,
        "val_per_seed":      val_results_all,
        "test_per_seed":     test_results_all,
        "train_times_s":     train_times,
        "gate_logs":         all_gate_logs,
        "best_val_macro_f1": best_val_f1,
    }

    results_file = out_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"  Results saved → {results_file}")

    total_time = (time.time() - t_total) / 60.0

    # ── Final comparison table ────────────────────────────────────────────
    print_comparison_table(test_agg)

    # ── Per-class F1 summary ──────────────────────────────────────────────
    print("\n  Per-class F1 (test, 5-seed average):")
    for cls in label_classes:
        key = f"f1_{cls}"
        if key in test_agg:
            mean_f1 = test_agg[key]["mean"]
            std_f1  = test_agg[key]["std"]
            delta   = ""
            # Compare against v1 all_gated if available
            v1_path = ROOT / "experiments/results/flow_fusion/all_gated/results.json"
            if v1_path.exists():
                try:
                    with open(v1_path) as fv1:
                        v1_data = json.load(fv1)
                    v1_f1 = v1_data.get("test_aggregated", {}).get(key, {}).get("mean", 0.0)
                    diff  = (mean_f1 - v1_f1) * 100
                    delta = f"  (Δ {'+' if diff>=0 else ''}{diff:.2f}% vs v1)"
                except Exception:
                    pass
            print(f"    {cls:<20}: {mean_f1:.4f} ± {std_f1:.4f}{delta}")

    # ── Gate analysis ─────────────────────────────────────────────────────
    if all_gate_logs and all(all_gate_logs):
        print("\n  Gate weight averages across seeds:")
        branch_labels = ["B1-Tabular", "B2-Spatial/CNN", "B3-Structural"]
        for i, blabel in enumerate(branch_labels):
            key    = f"gate_mean_{i}"
            values = [g.get(key, 0.0) for g in all_gate_logs if key in g]
            if values:
                print(f"    {blabel:<20}: {np.mean(values):.3f}  (target >0.50 for CNN)")

    print(f"\n  Total training time: {total_time:.1f} min")
    print(f"  Model saved:   {model_path}")
    print(f"  Results saved: {results_file}\n")


if __name__ == "__main__":
    main()
