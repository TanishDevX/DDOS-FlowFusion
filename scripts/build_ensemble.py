"""
scripts/build_ensemble.py
--------------------------
Phase 1: Test-Time Ensemble — FlowFusion v2 + LightGBM.

Combines the probability outputs of both models at inference time:

    p_final = alpha * p_flowfusion + (1 - alpha) * p_lightgbm

`alpha` is optimized on the validation set via grid search.
No retraining required — loads existing model checkpoints.

Why this works:
    FlowFusion v2 excels at: DrDoS_UDP, BENIGN, Syn, NetBIOS
    LightGBM excels at:      DrDoS_MSSQL, DrDoS_LDAP (sharp tree thresholds)

Usage:
    python scripts/build_ensemble.py               # full test set
    python scripts/build_ensemble.py --smoke-test  # fast 5k sample check
    python scripts/build_ensemble.py --alpha 0.65  # fixed alpha, skip search
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import yaml
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.gpu import setup_gpu
from src.models.flow_fusion import (
    build_flow_fusion,
    get_group_indices,
    ClassWeightedFocalLoss,
)
from src.evaluation.metrics import evaluate

logger  = get_logger("build_ensemble")
OUT_DIR = ROOT / "experiments" / "results" / "ensemble"
MODELS  = ROOT / "models"


# ── Data loading ──────────────────────────────────────────────────────────

def load_processed(split: str, mmap_mode=None) -> tuple:
    base          = ROOT / "data" / "processed" / split
    X             = np.load(base / "X.npy", mmap_mode=mmap_mode)
    y             = np.load(base / "y.npy", mmap_mode=mmap_mode)
    label_classes = np.load(base / "label_classes.npy", allow_pickle=True)
    feature_names = np.load(base / "feature_names.npy",  allow_pickle=True)
    logger.info(f"Loaded {split}: X{X.shape}  classes={list(label_classes)}")
    return X, y, label_classes, feature_names


# ── Probability getters ───────────────────────────────────────────────────

def get_flowfusion_probs(
    X_scaled: np.ndarray,
    label_classes: np.ndarray,
    feature_names: np.ndarray,
    cfg_ff: dict,
    seed: int = 42,
) -> np.ndarray:
    """Load FlowFusion v2 checkpoint and return predicted probabilities."""
    set_seed(seed)
    num_classes   = len(label_classes)
    input_dim     = X_scaled.shape[1]
    group_indices = get_group_indices(feature_names)

    dummy_alpha = np.ones(num_classes, dtype=np.float32) / num_classes
    model = build_flow_fusion(
        input_dim=input_dim,
        num_classes=num_classes,
        group_indices=group_indices,
        cfg=cfg_ff,
        branch_flags=(True, True, True),
        fusion_mode="gated",
        class_alpha=dummy_alpha,
    )

    ckpt_path = MODELS / "flow_fusion_v2_all_gated_best.keras"
    model.load_weights(str(ckpt_path))
    logger.info(f"FlowFusion v2 weights loaded <- {ckpt_path}")

    raw_preds = model.predict(X_scaled, batch_size=2048, verbose=0)

    # Keras returns a dict when model has multiple named outputs
    # (e.g. 'output', 'gate_interaction', 'gate_structural', 'gate_tabular')
    if isinstance(raw_preds, dict):
        return raw_preds["output"].astype(np.float32)
    elif isinstance(raw_preds, list):
        for p in raw_preds:
            if hasattr(p, "ndim") and p.ndim == 2 and p.shape[1] == num_classes:
                return p.astype(np.float32)
        raise ValueError("Could not find main output among model outputs.")
    return raw_preds.astype(np.float32)




def get_lightgbm_probs(X_raw: np.ndarray) -> np.ndarray:
    """Load LightGBM best model and return predicted probabilities."""
    lgbm_path = MODELS / "lgbm_best.joblib"
    if not lgbm_path.exists():
        raise FileNotFoundError(
            f"LightGBM model not found at {lgbm_path}.\n"
            f"Run: python scripts/train_lightgbm.py  first."
        )
    model = joblib.load(lgbm_path)
    logger.info(f"LightGBM loaded <- {lgbm_path}")
    return model.predict_proba(X_raw).astype(np.float32)


# ── Ensemble helpers ──────────────────────────────────────────────────────

def blend(p_ff: np.ndarray, p_lgbm: np.ndarray, alpha: float) -> np.ndarray:
    blended = alpha * p_ff + (1.0 - alpha) * p_lgbm
    return blended / blended.sum(axis=1, keepdims=True)


def optimize_alpha(
    p_ff_val:   np.ndarray,
    p_lgbm_val: np.ndarray,
    y_val:      np.ndarray,
) -> tuple:
    alpha_grid = np.linspace(0.40, 0.90, 51)
    best_alpha, best_f1 = 0.5, 0.0
    logger.info(f"Grid-searching alpha in [{alpha_grid[0]:.2f}, {alpha_grid[-1]:.2f}] ...")
    for alpha in alpha_grid:
        y_pred   = np.argmax(blend(p_ff_val, p_lgbm_val, alpha), axis=1)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        if macro_f1 > best_f1:
            best_f1, best_alpha = macro_f1, alpha
    logger.info(f"  Best alpha={best_alpha:.3f}  val_macro_f1={best_f1:.5f}")
    return float(best_alpha), float(best_f1)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlowFusion v2 + LightGBM ensemble")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--alpha",      type=float, default=None, help="Fixed blend weight (skip grid search)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_gpu()

    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_ff    = cfg["models"]["flow_fusion"]
    val_split = cfg["data"]["val_split"]

    # Load data
    X_full, y_full, label_classes, feature_names = load_processed("train")
    X_test, y_test, _, _                         = load_processed("test", mmap_mode="r")

    # ── Stratified test subsample (same as training scripts) ─────────────
    # Training scripts cap test at 2M samples using n_per = 2M // n_classes
    # per class to keep evaluation consistent across all models.
    MAX_TEST   = 2_000_000
    n_classes  = len(label_classes)
    if len(X_test) > MAX_TEST:
        rng   = np.random.default_rng(42)
        n_per = MAX_TEST // n_classes
        idx   = np.concatenate([
            rng.choice(
                np.where(y_test == c)[0],
                size=min(n_per, int((y_test == c).sum())),
                replace=False,
            )
            for c in range(n_classes)
        ])
        rng.shuffle(idx)
        X_test = X_test[idx]
        y_test = y_test[idx]
        logger.info(
            f"Test set stratified-subsampled: {len(y_test):,} samples "
            f"(max {n_per:,}/class)"
        )


    # Same val split as training seed=42
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full,
        test_size=val_split,
        stratify=y_full,
        random_state=args.seed,
    )

    # Scale (FlowFusion trained on StandardScaler output)
    scaler    = StandardScaler()
    scaler.fit(X_tr)
    X_val_sc  = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Smoke test: subsample test to 5k
    if args.smoke_test:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
        X_test_sc  = X_test_sc[idx]
        X_test_raw = X_test[idx]
        y_test     = y_test[idx]
        logger.info(f"[Smoke-test] Test subsampled to {len(y_test)} samples")
    else:
        X_test_raw = X_test

    t0 = time.time()

    # FlowFusion probabilities
    logger.info("\n=== FlowFusion v2 — validation probs ===")
    p_ff_val  = get_flowfusion_probs(X_val_sc, label_classes, feature_names, cfg_ff, seed=args.seed)
    logger.info("\n=== FlowFusion v2 — test probs ===")
    p_ff_test = get_flowfusion_probs(X_test_sc, label_classes, feature_names, cfg_ff, seed=args.seed)

    # LightGBM probabilities (runs on raw unscaled features)
    logger.info("\n=== LightGBM — validation probs ===")
    p_lgbm_val  = get_lightgbm_probs(X_val)
    logger.info("\n=== LightGBM — test probs ===")
    p_lgbm_test = get_lightgbm_probs(X_test_raw)

    logger.info(f"Inference completed in {time.time()-t0:.1f}s")

    # Optimize alpha on validation
    if args.alpha is not None:
        best_alpha = args.alpha
        y_pred_val = np.argmax(blend(p_ff_val, p_lgbm_val, best_alpha), axis=1)
        best_val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        logger.info(f"Fixed alpha={best_alpha:.3f}  val_macro_f1={best_val_f1:.5f}")
    else:
        best_alpha, best_val_f1 = optimize_alpha(p_ff_val, p_lgbm_val, y_val)

    # Standalone val F1s (for baseline comparison)
    ff_val_f1   = f1_score(y_val, np.argmax(p_ff_val,   axis=1), average="macro", zero_division=0)
    lgbm_val_f1 = f1_score(y_val, np.argmax(p_lgbm_val, axis=1), average="macro", zero_division=0)

    # Clean probabilities to prevent float precision NaN in ROC-AUC calculation
    p_blend_test = np.nan_to_num(blend(p_ff_test, p_lgbm_test, best_alpha), nan=1.0/len(label_classes))
    p_ff_clean   = np.nan_to_num(p_ff_test, nan=1.0/len(label_classes))
    p_lgbm_clean = np.nan_to_num(p_lgbm_test, nan=1.0/len(label_classes))

    y_pred_test  = np.argmax(p_blend_test, axis=1)
    ens_metrics  = evaluate(y_test, y_pred_test, p_blend_test, label_classes,
                            split_name=f"ensemble(alpha={best_alpha:.3f})")
    ff_metrics   = evaluate(y_test, np.argmax(p_ff_clean,   axis=1), p_ff_clean,   label_classes, split_name="ff_test")
    lgbm_metrics = evaluate(y_test, np.argmax(p_lgbm_clean, axis=1), p_lgbm_clean, label_classes, split_name="lgbm_test")


    # Print comparison table
    print("\n" + "=" * 80)
    print("  ENSEMBLE RESULTS  —  FlowFusion v2  +  LightGBM")
    print("=" * 80)
    print(f"  Blend weight: alpha={best_alpha:.3f} (FlowFusion) + {1-best_alpha:.3f} (LightGBM)")
    print(f"\n  {'Model':<38} {'Val Macro-F1':>14}  {'Test Macro-F1':>14}")
    print("  " + "-" * 70)
    print(f"  {'FlowFusion v2 (standalone)':<38} {ff_val_f1:>14.5f}  {ff_metrics['macro_f1']:>14.5f}")
    print(f"  {'LightGBM (standalone)':<38} {lgbm_val_f1:>14.5f}  {lgbm_metrics['macro_f1']:>14.5f}")
    print(f"  {'⭐ Ensemble (alpha={:.3f})'.format(best_alpha):<38} {best_val_f1:>14.5f}  {ens_metrics['macro_f1']:>14.5f}")
    print("=" * 80)

    print("\n  Per-class F1 (test):  FF v2  vs  LGBM  vs  Ensemble  vs  Δ_FF")
    for c in label_classes:
        ens_f1  = ens_metrics.get(f"f1_{c}", 0)
        ff_f1   = ff_metrics.get(f"f1_{c}", 0)
        lgbm_f1 = lgbm_metrics.get(f"f1_{c}", 0)
        delta   = (ens_f1 - ff_f1) * 100
        print(f"    {c:<18}  {ff_f1:.4f}    {lgbm_f1:.4f}    {ens_f1:.4f}    {delta:+.3f}%")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":                         "FlowFusion_v2_LightGBM_Ensemble",
        "best_alpha":                    best_alpha,
        "best_val_macro_f1":             best_val_f1,
        "ff_standalone_test_macro_f1":   ff_metrics["macro_f1"],
        "lgbm_standalone_test_macro_f1": lgbm_metrics["macro_f1"],
        "ensemble_test_macro_f1":        ens_metrics["macro_f1"],
        "test_metrics":                  ens_metrics,
        "ff_standalone_metrics":         ff_metrics,
        "lgbm_standalone_metrics":       lgbm_metrics,
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved -> {out_path}")
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
