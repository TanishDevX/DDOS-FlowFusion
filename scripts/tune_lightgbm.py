"""
scripts/tune_lightgbm.py
-------------------------
LightGBM hyperparameter grid search + tuned 5-seed training.

Stage 1: Grid search over key hyperparameters using seed=42.
Stage 2: Retrain with best config over all 5 seeds and save results.

Key improvements over baseline:
  - DART boosting mode (dropout regularized trees)
  - Wider leaf search (num_leaves up to 511)
  - min_data_in_leaf tuning for better BENIGN generalization
  - Node-level feature sampling (feature_fraction_bynode)
  - Balanced class_weight to upweight BENIGN samples

Run from project root:
    python scripts/tune_lightgbm.py
    python scripts/tune_lightgbm.py --skip-search
"""

import sys
import json
import time
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.evaluation.metrics import evaluate, aggregate_seeds, format_results_table

logger = get_logger("tune_lightgbm")

RESULTS_DIR = ROOT / "experiments" / "results" / "lightgbm"
MODELS_DIR  = ROOT / "models"

# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter search space
# ═══════════════════════════════════════════════════════════════════════════

SEARCH_SPACE = {
    "boosting_type":          ["gbdt", "dart"],
    "num_leaves":             [127, 255, 511],
    "min_data_in_leaf":       [20, 50, 100],
    "feature_fraction_bynode": [0.5, 0.8],
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_processed(split: str) -> tuple:
    base          = ROOT / "data" / "processed" / split
    X             = np.load(base / "X.npy")
    y             = np.load(base / "y.npy")
    label_classes = np.load(base / "label_classes.npy", allow_pickle=True)
    feature_names = np.load(base / "feature_names.npy",  allow_pickle=True)
    logger.info(f"Loaded {split}: X{X.shape}  y{y.shape}  classes={list(label_classes)}")
    return X, y, label_classes, feature_names


def sample_test(X_test, y_test, max_rows: int = 2_000_000):
    """Stratified subsample of test set for faster evaluation."""
    if len(X_test) <= max_rows:
        return X_test, y_test
    rng     = np.random.default_rng(42)
    classes = np.unique(y_test)
    n_per   = max_rows // len(classes)
    idx = np.concatenate([
        rng.choice(np.where(y_test == c)[0],
                   size=min(n_per, int((y_test == c).sum())),
                   replace=False)
        for c in classes
    ])
    rng.shuffle(idx)
    logger.info(f"Test stratified-sampled: {len(idx):,} rows")
    return X_test[idx], y_test[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Training one config / seed
# ═══════════════════════════════════════════════════════════════════════════

def train_one(
    X_tr, y_tr, X_val, y_val, X_test, y_test,
    params: dict, label_classes, feature_names, seed: int,
) -> tuple:
    """Train one LightGBM model, return (val_metrics, test_metrics, model, elapsed)."""
    set_seed(seed)

    # DART does not support early stopping in LightGBM; fall back to fixed n_estimators
    is_dart = params.get("boosting_type", "gbdt") == "dart"
    callbacks = [lgb.log_evaluation(period=-1)]
    if not is_dart:
        callbacks.append(
            lgb.early_stopping(params.get("early_stopping_rounds", 30), verbose=False)
        )

    model = lgb.LGBMClassifier(**{k: v for k, v in params.items()
                                   if k != "early_stopping_rounds"})
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
        feature_name=list(feature_names),
    )
    elapsed = time.time() - t0
    best_iter = getattr(model, "best_iteration_", params.get("n_estimators", 500))
    logger.info(f"  Seed={seed}  best_iter={best_iter}  elapsed={elapsed:.1f}s")

    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)
    val_m = evaluate(y_val, y_val_pred, y_val_prob, label_classes,
                     split_name=f"val(seed={seed})")

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    test_m = evaluate(y_test, y_test_pred, y_test_prob, label_classes,
                      split_name=f"test(seed={seed})")

    return val_m, test_m, model, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(
    X_tr, y_tr, X_val, y_val, X_test, y_test,
    base_params: dict, label_classes, feature_names,
) -> dict:
    """Exhaustive grid search over SEARCH_SPACE using seed=42. Returns best overrides."""
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(product(*values))
    logger.info(f"Grid search: {len(combos)} combinations")

    best_f1 = -1.0
    best_overrides = {}

    for i, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        params    = {**base_params, **overrides, "random_state": 42}
        logger.info(f"  [{i+1}/{len(combos)}] {overrides}")
        try:
            val_m, test_m, _, _ = train_one(
                X_tr, y_tr, X_val, y_val, X_test, y_test,
                params, label_classes, feature_names, seed=42,
            )
            val_f1 = val_m["macro_f1"]
            logger.info(f"    val macro-F1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_overrides = overrides
        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.info(f"\nBest config (val macro-F1={best_f1:.4f}): {best_overrides}")
    return best_overrides


# ═══════════════════════════════════════════════════════════════════════════
# Full 5-seed run
# ═══════════════════════════════════════════════════════════════════════════

def full_run(
    X_full, y_full, X_test, y_test, params: dict,
    seeds: list, val_split: float, label_classes, feature_names,
) -> None:
    val_results_all  = []
    test_results_all = []
    train_times      = []
    best_val_f1      = -1.0
    best_model       = None

    for seed in seeds:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_full,
            test_size=val_split,
            stratify=y_full,
            random_state=seed,
        )
        p = {**params, "random_state": seed}
        val_m, test_m, model, elapsed = train_one(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            p, label_classes, feature_names, seed=seed,
        )
        val_results_all.append(val_m)
        test_results_all.append(test_m)
        train_times.append(elapsed)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_model  = model
            logger.info(f"  → New best val macro-F1: {best_val_f1:.4f}")

    val_agg  = aggregate_seeds(val_results_all)
    test_agg = aggregate_seeds(test_results_all)

    logger.info("\n" + "=" * 60)
    logger.info("TUNED LIGHTGBM RESULTS  (mean ± std across 5 seeds)")
    logger.info("=" * 60)
    for m in ["accuracy", "macro_f1", "weighted_f1", "roc_auc", "f1_BENIGN"]:
        if m in test_agg:
            logger.info(f"  {m:<20} {test_agg[m]['mean']:.4f} ± {test_agg[m]['std']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":           "LightGBM_tuned",
        "seeds":           seeds,
        "tuned_params":    params,
        "val_aggregated":  val_agg,
        "test_aggregated": test_agg,
        "val_per_seed":    val_results_all,
        "test_per_seed":   test_results_all,
        "train_times_s":   train_times,
    }
    with open(RESULTS_DIR / "results_tuned.json", "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Tuned results → {RESULTS_DIR}/results_tuned.json")

    joblib.dump(best_model, MODELS_DIR / "lgbm_tuned_best.joblib")
    logger.info(f"Best model → models/lgbm_tuned_best.joblib  (val F1={best_val_f1:.4f})")
    table = format_results_table(test_agg, "LightGBM_tuned")
    logger.info(f"\n{table.to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tune LightGBM for DDoS detection.")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()

    t_total  = time.time()
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seeds     = cfg["training"]["seeds"]
    val_split = cfg["data"]["val_split"]
    cfg_lgbm  = cfg["models"]["lightgbm"]

    base_params = {
        "objective":          "multiclass",
        "num_class":          6,
        "learning_rate":      cfg_lgbm["learning_rate"],
        "n_estimators":       1000,
        "subsample":          cfg_lgbm["subsample"],
        "colsample_bytree":   cfg_lgbm["colsample_bytree"],
        "reg_alpha":          cfg_lgbm["reg_alpha"],
        "reg_lambda":         cfg_lgbm["reg_lambda"],
        "class_weight":       "balanced",    # upweights BENIGN
        "device":             "gpu",
        "n_jobs":             -1,
        "verbosity":          -1,
        "early_stopping_rounds": cfg_lgbm.get("early_stopping_rounds", 30),
        # tunable defaults
        "boosting_type":      "gbdt",
        "num_leaves":         cfg_lgbm["num_leaves"],
        "min_data_in_leaf":   cfg_lgbm["min_child_samples"],
        "feature_fraction_bynode": 0.8,
    }

    X_full, y_full, label_classes, feature_names = load_processed("train")
    X_test, y_test, _, _                         = load_processed("test")
    X_test, y_test = sample_test(X_test, y_test)
    base_params["num_class"] = len(label_classes)

    if args.skip_search:
        best_overrides = {"boosting_type": "dart", "num_leaves": 255,
                          "min_data_in_leaf": 50, "feature_fraction_bynode": 0.8}
    else:
        set_seed(42)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_full, test_size=val_split, stratify=y_full, random_state=42
        )
        best_overrides = grid_search(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            base_params, label_classes, feature_names,
        )

    best_params = {**base_params, **best_overrides}
    logger.info(f"\nFinal params: {best_params}")
    full_run(X_full, y_full, X_test, y_test, best_params, seeds, val_split, label_classes, feature_names)
    logger.info(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
