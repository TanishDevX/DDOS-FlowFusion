"""
scripts/tune_xgboost.py
------------------------
XGBoost hyperparameter grid search + tuned 5-seed training.

Stage 1: Grid search over key hyperparameters using seed=42, val macro-F1 as criterion.
Stage 2: Retrain with best config over all 5 seeds and save results.

Key improvements over baseline:
  - Deeper trees (max_depth up to 12)
  - colsample_bylevel (node-level feature sampling)
  - gamma (min split loss) regularization
  - Class-weighted sample_weight (upweights BENIGN)
  - More estimators with tighter early stopping

Run from project root:
    python scripts/tune_xgboost.py
    python scripts/tune_xgboost.py --skip-search   # use defaults from config
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
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.evaluation.metrics import evaluate, aggregate_seeds, format_results_table

logger = get_logger("tune_xgboost")

RESULTS_DIR = ROOT / "experiments" / "results" / "xgboost"
MODELS_DIR  = ROOT / "models"

# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter search space
# ═══════════════════════════════════════════════════════════════════════════

SEARCH_SPACE = {
    "max_depth":         [8, 10, 12],
    "colsample_bylevel": [0.6, 0.8, 1.0],
    "min_child_weight":  [3, 5, 10],
    "gamma":             [0.0, 0.1, 0.3],
}


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
    return np.array(X_test[idx]), y_test[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Training one config / seed
# ═══════════════════════════════════════════════════════════════════════════

def train_one(
    X_tr, y_tr, X_val, y_val, X_test, y_test,
    params: dict, label_classes, seed: int,
    use_class_weights: bool = True,
) -> tuple:
    """Train one XGBoost model, return (val_metrics, test_metrics, model, elapsed)."""
    set_seed(seed)

    sample_weight = compute_sample_weight("balanced", y_tr) if use_class_weights else None
    val_weight    = compute_sample_weight("balanced", y_val) if use_class_weights else None

    model = xgb.XGBClassifier(**params)
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[val_weight],
        verbose=False,
    )
    elapsed = time.time() - t0
    logger.info(f"  Seed={seed}  best_iter={model.best_iteration}  elapsed={elapsed:.1f}s")

    # Validation
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)
    val_m = evaluate(y_val, y_val_pred, y_val_prob, label_classes,
                     split_name=f"val(seed={seed})")

    # Test (chunked)
    chunk = 1_000_000
    preds, probs = [], []
    for i in range(0, len(X_test), chunk):
        preds.append(model.predict(X_test[i:i+chunk]))
        probs.append(model.predict_proba(X_test[i:i+chunk]))
    test_m = evaluate(y_test, np.concatenate(preds), np.concatenate(probs, axis=0),
                      label_classes, split_name=f"test(seed={seed})")

    return val_m, test_m, model, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════

def grid_search(
    X_tr, y_tr, X_val, y_val, X_test, y_test,
    base_params: dict, label_classes,
) -> dict:
    """Exhaustive grid search over SEARCH_SPACE using seed=42. Returns best overrides."""
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(product(*values))
    logger.info(f"Grid search: {len(combos)} combinations × 1 seed = {len(combos)} fits")

    best_f1      = -1.0
    best_overrides = {}
    results = []

    for i, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        params    = {**base_params, **overrides, "random_state": 42}
        logger.info(f"  [{i+1}/{len(combos)}] {overrides}")
        try:
            val_m, test_m, _, elapsed = train_one(
                X_tr, y_tr, X_val, y_val, X_test, y_test,
                params, label_classes, seed=42,
            )
            val_f1 = val_m["macro_f1"]
            logger.info(f"    val macro-F1={val_f1:.4f}  (test macro-F1={test_m['macro_f1']:.4f})")
            results.append({**overrides, "val_f1": val_f1, "test_f1": test_m["macro_f1"]})
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_overrides = overrides
        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.info(f"\nBest config (val macro-F1={best_f1:.4f}): {best_overrides}")
    return best_overrides


# ═══════════════════════════════════════════════════════════════════════════
# Full 5-seed run with best params
# ═══════════════════════════════════════════════════════════════════════════

def full_run(
    X_full, y_full, X_test, y_test, params: dict,
    seeds: list, val_split: float, label_classes,
) -> None:
    """Run full 5-seed training with tuned params and save results."""
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
        logger.info(f"\nTrain: {X_tr.shape}  Val: {X_val.shape}")

        p = {**params, "random_state": seed}
        val_m, test_m, model, elapsed = train_one(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            p, label_classes, seed=seed,
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
    logger.info("TUNED XGBOOST RESULTS  (mean ± std across 5 seeds)")
    logger.info("=" * 60)
    for m in ["accuracy", "macro_f1", "weighted_f1", "roc_auc", "f1_BENIGN"]:
        if m in test_agg:
            logger.info(f"  {m:<20} {test_agg[m]['mean']:.4f} ± {test_agg[m]['std']:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":           "XGBoost_tuned",
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
    logger.info(f"Tuned results saved → {RESULTS_DIR}/results_tuned.json")

    joblib.dump(best_model, MODELS_DIR / "xgb_tuned_best.joblib")
    logger.info(f"Best tuned model saved → models/xgb_tuned_best.joblib  (val F1={best_val_f1:.4f})")

    table = format_results_table(test_agg, "XGBoost_tuned")
    logger.info(f"\n{table.to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tune XGBoost for DDoS detection.")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip grid search and use best known defaults.")
    args = parser.parse_args()

    t_total = time.time()
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seeds     = cfg["training"]["seeds"]
    val_split = cfg["data"]["val_split"]
    cfg_xgb   = cfg["models"]["xgboost"]

    # Base params (from config, with improvements)
    base_params = {
        "objective":             "multi:softprob",
        "num_class":             6,  # updated below after loading data
        "learning_rate":         cfg_xgb["learning_rate"],
        "n_estimators":          1000,          # more estimators, rely on early stopping
        "subsample":             cfg_xgb["subsample"],
        "colsample_bytree":      cfg_xgb["colsample_bytree"],
        "reg_alpha":             cfg_xgb["reg_alpha"],
        "reg_lambda":            cfg_xgb["reg_lambda"],
        "tree_method":           "hist",
        "device":                "cuda",
        "n_jobs":                -1,
        "verbosity":             0,
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": cfg_xgb.get("early_stopping_rounds", 30),
        # tunable defaults (overridden by grid search)
        "max_depth":             cfg_xgb["max_depth"],
        "min_child_weight":      cfg_xgb["min_child_weight"],
        "gamma":                 0.0,
        "colsample_bylevel":     0.8,
    }

    # Load data
    X_full, y_full, label_classes, _ = load_processed("train")
    X_test, y_test, _, _             = load_processed("test", mmap_mode="r")
    X_test, y_test                   = sample_test(X_test, y_test)
    base_params["num_class"]          = len(label_classes)

    # Grid search
    if args.skip_search:
        logger.info("Skipping grid search — using baseline tuned defaults.")
        best_overrides = {"max_depth": 10, "colsample_bylevel": 0.8,
                          "min_child_weight": 5, "gamma": 0.1}
    else:
        set_seed(42)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_full, test_size=val_split, stratify=y_full, random_state=42
        )
        best_overrides = grid_search(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            base_params, label_classes,
        )

    best_params = {**base_params, **best_overrides}
    logger.info(f"\nFinal tuned params: {best_params}")

    # Full 5-seed run
    full_run(X_full, y_full, X_test, y_test, best_params, seeds, val_split, label_classes)

    logger.info(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
