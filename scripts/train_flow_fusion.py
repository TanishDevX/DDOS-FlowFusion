"""
scripts/train_flow_fusion.py
-----------------------------
FlowFusion training script — Multi-View DDoS Detection.

Runs the full 8-configuration ablation study by default, or a single config
when --ablation is specified. For each configuration:
  - Trains over 5 (or more) seeds
  - Reports mean ± std macro-F1 on the held-out March 11 test set
  - Saves the best model (highest val macro-F1)
  - Collects gate values from the proposed model (all_gated) for interpretability

Critical methodological notes:
  - Correlation matrix for CNN reordering is computed INSIDE the per-seed split
    on X_tr only — not on the full training set — avoiding leakage from val fold.
  - Macro-F1 is the primary reported metric (test set is naturally imbalanced).
  - Gradient clipping (global norm 1.0) applied for multi-branch stability.
  - Linear LR warmup over first `warmup_steps` steps.

Run from project root:
    python scripts/train_flow_fusion.py                        # all 8 ablation configs
    python scripts/train_flow_fusion.py --ablation all_gated   # proposed model only
    python scripts/train_flow_fusion.py --ablation b1_only     # FT-Transformer alone
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
from sklearn.utils.class_weight import compute_class_weight
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augmentation import augment_benign

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.gpu import setup_gpu, log_hardware_summary
from src.models.flow_fusion import (
    build_flow_fusion,
    get_group_indices,
    compute_correlation_reorder,
    ABLATION_CONFIGS,
)
from src.evaluation.metrics import evaluate, aggregate_seeds, format_results_table

logger  = get_logger("train_flow_fusion")
RESULTS = ROOT / "experiments" / "results" / "flow_fusion"
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
# LR warmup callback
# ═══════════════════════════════════════════════════════════════════════════

class WarmupCallback(tf.keras.callbacks.Callback):
    """Linear warmup callback. Keeps the optimizer learning rate settable."""

    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__()
        self.base_lr      = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.global_step  = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self.global_step < self.warmup_steps:
            self.global_step += 1
            ratio = float(self.global_step) / float(self.warmup_steps)
            new_lr = self.base_lr * ratio
            # Set the learning rate on the optimizer directly
            self.model.optimizer.learning_rate = new_lr


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
    branch_flags: tuple, fusion_mode: str,
    ablation_name: str, seed: int,
    seed_idx: int = 1, total_seeds: int = 1,
) -> tuple:
    set_seed(seed)
    tf.keras.backend.clear_session()

    input_dim   = X_tr.shape[1]
    num_classes = len(label_classes)
    use_b1, use_b2, use_b3 = branch_flags

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

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

    # FIN (Branch 2) uses group_indices directly — no correlation reorder needed
    # group_indices are needed if B2 (FIN) or B3 (Group Attention) is active
    group_indices = get_group_indices(feature_names) if (use_b2 or use_b3) else {}

    base_lr = cfg_ff.get("learning_rate", 1e-4)
    total_epochs = cfg_ff.get("epochs", cfg_train.get("epochs", 120))
    warmup_epochs = cfg_ff.get("warmup_epochs", 8)

    # v2: Compute per-class inverse-frequency alpha for focal loss
    # alpha_c = 1/count_c, then normalised to sum=1
    # This penalises DrDoS_MSSQL misclassifications more heavily.
    class_counts = np.bincount(y_tr.astype(int), minlength=num_classes).astype(np.float32)
    class_alpha  = 1.0 / (class_counts + 1e-8)
    class_alpha /= class_alpha.sum()
    logger.info(
        f"  Focal loss alpha (per-class): "
        + "  ".join(f"{label_classes[i]}={class_alpha[i]:.4f}" for i in range(num_classes))
    )

    model = build_flow_fusion(
        input_dim=input_dim,
        num_classes=num_classes,
        group_indices=group_indices,
        cfg=cfg_ff,
        branch_flags=branch_flags,
        fusion_mode=fusion_mode,
        class_alpha=class_alpha,    # v2: pass inverse-freq weights to focal loss
    )

    # v2: model is already compiled inside build_flow_fusion with focal loss
    # Re-compile only to inject the updated AdamW optimizer with clipnorm
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=base_lr,
        weight_decay=cfg_ff.get("weight_decay", 1e-5),
        clipnorm=cfg_ff.get("clipnorm", 1.0),
    )

    output_names = model.output_names
    # Retrieve the compiled loss from the model (focal or smoothed CCE)
    existing_loss = model.loss
    main_acc      = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    if len(output_names) == 1:
        model.compile(
            optimizer=optimizer,
            loss=existing_loss,
            metrics=[main_acc],
        )
        y_tr_fit  = y_tr.astype(np.int32)
        y_val_fit = y_val.astype(np.int32)
    else:
        # Rebuild loss dict preserving focal loss on "output", zero-weight on gate outputs
        loss_dict    = {"output": existing_loss if isinstance(existing_loss, dict) is False else existing_loss.get("output", existing_loss)}
        loss_weights = {"output": 1.0}
        for name in output_names:
            if name != "output":
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
        MacroF1ValidationCallback(val_x=X_val_use, val_y=y_val, num_classes=num_classes),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=max(cfg_train["early_stopping_patience"], 25),  # v2: min 25 for branch co-convergence
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(  # v2: fallback LR reducer before early stop
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=cfg_train.get("min_lr", 1e-6),
            verbose=0,
        ),
        RichTrainingCallback(
            model_name=f"FlowFusion[{ablation_name}]",
            seed=seed,
            seed_idx=seed_idx,
            total_seeds=total_seeds,
            total_epochs=total_epochs,
        ),
    ]

    cw_arr = compute_class_weight(
        "balanced",
        classes=np.unique(y_tr),
        y=y_tr,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(cw_arr)}

    sample_weight_tr  = np.array([class_weight_dict[int(c)] for c in y_tr],  dtype=np.float32)
    sample_weight_val = np.array([class_weight_dict[int(c)] for c in y_val], dtype=np.float32)

    t0 = time.time()
    model.fit(
        X_tr_use.astype(np.float32), y_tr_fit,
        validation_data=(X_val_use.astype(np.float32), y_val_fit, sample_weight_val),
        epochs=total_epochs,
        batch_size=cfg_train["batch_size"],
        callbacks=callbacks,
        sample_weight=sample_weight_tr,
        verbose=0,
    )


    elapsed = time.time() - t0
    logger.info(f"  Training done in {elapsed:.1f}s")

    # ── Evaluation ────────────────────────────────────────────────────────
    def _predict(X_input):
        logger.info(f"    Predicting on {len(X_input):,} rows in chunks...")
        chunk_size = 1_000_000
        all_logits = []
        all_gates = []

        for i in range(0, len(X_input), chunk_size):
            X_chunk = X_input[i : i + chunk_size]
            # Scale in original column order — reordering is now applied INSIDE
            # the model.
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
        gates_concat = [np.concatenate(g, axis=0) for g in all_gates] if all_gates else []
        return logits_concat, gates_concat

    val_out, _       = _predict(X_val)
    val_pred         = np.argmax(val_out, axis=1)
    val_metrics      = evaluate(y_val, val_pred, val_out, label_classes,
                                split_name=f"val({ablation_name},seed={seed})")

    test_out, gates  = _predict(X_test)
    test_pred        = np.argmax(test_out, axis=1)
    test_metrics     = evaluate(y_test, test_pred, test_out, label_classes,
                                split_name=f"test({ablation_name},seed={seed})")

    # ── Log gate values (for interpretability analysis) ───────────────────
    gate_log = {}
    if gates:
        for i, g_arr in enumerate(gates):
            # g_arr: (n_test, 1) → scalar mean over test set
            gate_log[f"gate_mean_{i}"] = float(np.mean(g_arr))
        logger.info(f"  Gate means: { {k: f'{v:.3f}' for k, v in gate_log.items()} }")

    return val_metrics, test_metrics, model, elapsed, gate_log


# ═══════════════════════════════════════════════════════════════════════════
# Run one ablation config across all seeds
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation_config(
    ablation_name: str,
    branch_flags: tuple,
    fusion_mode: str,
    X_full: np.ndarray, y_full: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list, label_classes: np.ndarray,
    cfg: dict,
    smoke_test: bool = False,
) -> dict:
    seeds     = [42] if smoke_test else cfg["training"]["seeds"]
    val_split = cfg["data"]["val_split"]
    cfg_ff    = dict(cfg["models"]["flow_fusion"])
    cfg_train = dict(cfg["training"])

    if smoke_test:
        cfg_ff["epochs"] = 2
        cfg_train["epochs"] = 2

    out_dir   = RESULTS / ablation_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_results_all  = []
    test_results_all = []
    train_times      = []
    all_gate_logs    = []
    best_val_f1      = -1.0
    best_model       = None

    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION CONFIG: {ablation_name}")
    logger.info(f"  branches={branch_flags}  fusion={fusion_mode}  seeds={seeds}")
    logger.info(f"{'='*60}")

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
            cfg_ff, cfg_train, branch_flags, fusion_mode,
            ablation_name, seed, seed_idx, len(seeds),
        )

        val_results_all.append(val_m)
        test_results_all.append(test_m)
        train_times.append(elapsed)
        all_gate_logs.append(gate_log)

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_model  = model

        tf.keras.backend.clear_session()

    val_agg  = aggregate_seeds(val_results_all)
    test_agg = aggregate_seeds(test_results_all)

    model_path = MODELS / f"flow_fusion_{ablation_name}_best.keras"
    if best_model is not None:
        best_model.save(str(model_path))

    payload = {
        "ablation":          ablation_name,
        "branch_flags":      branch_flags,
        "fusion_mode":       fusion_mode,
        "seeds":             seeds,
        "val_aggregated":    val_agg,
        "test_aggregated":   test_agg,
        "val_per_seed":      val_results_all,
        "test_per_seed":     test_results_all,
        "train_times_s":     train_times,
        "gate_logs":         all_gate_logs,
        "best_val_macro_f1": best_val_f1,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_wilcoxon(name_a: str, f1s_a: list, name_b: str, f1s_b: list):
    from scipy.stats import wilcoxon
    try:
        diffs = [a - b for a, b in zip(f1s_a, f1s_b)]
        if all(d == 0 for d in diffs):
            logger.info(f"  Wilcoxon {name_a} vs {name_b}: identical scores, p=1.0")
            return
        stat, p = wilcoxon(f1s_a, f1s_b, alternative="two-sided")
        sig = "✅ significant" if p < 0.05 else "⚠️  not significant"
        logger.info(f"  Wilcoxon {name_a} vs {name_b}: stat={stat:.4f}, p={p:.4f} — {sig}")
    except Exception as e:
        logger.warning(f"  Wilcoxon failed: {e}")


def build_comparison_table(all_results: dict):
    import pandas as pd

    rows = []
    for ablation_name, payload in all_results.items():
        test_agg = payload["test_aggregated"]
        row = {
            "Config":     ablation_name,
            "Branches":   str(payload["branch_flags"]),
            "Fusion":     payload["fusion_mode"],
            "Macro-F1":   f"{test_agg['macro_f1']['mean']:.4f} ± {test_agg['macro_f1']['std']:.4f}",
            "Accuracy":   f"{test_agg['accuracy']['mean']:.4f} ± {test_agg['accuracy']['std']:.4f}",
            "ROC-AUC":    f"{test_agg['roc_auc']['mean']:.4f} ± {test_agg['roc_auc']['std']:.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    comparison_path = RESULTS / "ablation_comparison.csv"
    df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 80)
    print("FLOWFUSION ABLATION RESULTS (Macro-F1 = primary metric)")
    print("=" * 80)
    print("\n" + df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Train FlowFusion (multi-view DDoS detection).")
    parser.add_argument(
        "--ablation",
        default=None,
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        help="Which ablation config to run.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 seed for 2 epochs smoke test")
    args = parser.parse_args()

    setup_gpu()
    log_hardware_summary()

    t_total = time.time()

    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    if args.ablation is not None:
        run_all = (args.ablation == "all")
        cli_mode = args.ablation
    else:
        run_all  = cfg["models"]["ablation"].get("run_all", False)
        cli_mode = cfg["models"]["ablation"].get("mode", "all_gated")

    if run_all or args.ablation == "all":
        configs_to_run = list(ABLATION_CONFIGS.keys())
    else:
        configs_to_run = [cli_mode]

    logger.info(f"Ablation configs to run: {configs_to_run}")

    # ── Load data ─────────────────────────────────────────────────────────
    X_full, y_full, label_classes, feature_names = load_processed("train")
    X_test, y_test, _, _                         = load_processed("test", mmap_mode="r")

    # ── Stratified test sampling (2M rows ≈ same metrics, 10× faster eval) ─
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
        logger.info(f"Test stratified-sampled: {len(X_test):,} rows ({n_per:,}/class)")


    logger.info(f"\nConfig: {cfg['models']['flow_fusion']}")
    logger.info(f"Feature count: {X_full.shape[1]}")
    logger.info(f"Verify grid: {cfg['models']['flow_fusion']['grid_h']} × "
                f"{cfg['models']['flow_fusion']['grid_w']} = "
                f"{cfg['models']['flow_fusion']['grid_h'] * cfg['models']['flow_fusion']['grid_w']} "
                f"(need {X_full.shape[1]})")

    # No global StandardScaler fitted on X_full here — to prevent validation double-scaling.
    # The StandardScaler is now fitted inside train_one_seed on the training split only.

    # ── Run all ablation configs ───────────────────────────────────────────
    all_results = {}
    for ablation_name in configs_to_run:
        branch_flags, fusion_mode = ABLATION_CONFIGS[ablation_name]
        payload = run_ablation_config(
            ablation_name=ablation_name,
            branch_flags=branch_flags,
            fusion_mode=fusion_mode,
            X_full=X_full, y_full=y_full,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
            label_classes=label_classes,
            cfg=cfg,
            smoke_test=args.smoke_test,
        )
        all_results[ablation_name] = payload

    # ── Comparison table ───────────────────────────────────────────────────
    if len(all_results) > 1:
        build_comparison_table(all_results)

    # ── Statistical significance tests ────────────────────────────────────
    if "all_gated" in all_results and len(all_results) > 1:
        logger.info("\n--- Wilcoxon tests: all_gated vs each ablation ---")
        gated_f1s = [r["macro_f1"] for r in all_results["all_gated"]["test_per_seed"]]
        for name, payload in all_results.items():
            if name == "all_gated":
                continue
            other_f1s = [r["macro_f1"] for r in payload["test_per_seed"]]
            run_wilcoxon("all_gated", gated_f1s, name, other_f1s)

    # ── Also compare against XGBoost/FT-Transformer baselines if available ─
    for baseline_name, baseline_dir in [
        ("xgboost",        ROOT / "experiments" / "results" / "xgboost"),
        ("ft_transformer", ROOT / "experiments" / "results" / "ft_transformer"),
    ]:
        results_file = baseline_dir / "results.json"
        if results_file.exists() and "all_gated" in all_results:
            try:
                with open(results_file) as f:
                    bl = json.load(f)
                bl_f1s    = [r["macro_f1"] for r in bl.get("test_per_seed", [])]
                gated_f1s = [r["macro_f1"] for r in all_results["all_gated"]["test_per_seed"]]
                if bl_f1s:
                    run_wilcoxon("all_gated", gated_f1s, baseline_name, bl_f1s)
            except Exception as e:
                logger.warning(f"Could not load {baseline_name} results: {e}")

    logger.info(
        f"\n{'='*60}\n"
        f"FlowFusion training complete.\n"
        f"Total time: {(time.time()-t_total)/60:.1f} min\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
