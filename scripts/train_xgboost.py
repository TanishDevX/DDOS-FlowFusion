import sys, json, time, argparse
from pathlib import Path
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.gpu import get_xgb_device, log_hardware_summary
from src.utils.progress import print_feature_importance
from src.evaluation.metrics import evaluate, aggregate_seeds, format_results_table

logger = get_logger("train_xgboost")

RESULTS_DIR = ROOT / "experiments" / "results" / "xgboost"
MODELS_DIR  = ROOT / "models"


def load_processed(split: str, mmap_mode=None) -> tuple:
    base = ROOT / "data" / "processed" / split
    X            = np.load(base / "X.npy", mmap_mode=mmap_mode)
    y            = np.load(base / "y.npy", mmap_mode=mmap_mode)
    label_classes = np.load(base / "label_classes.npy", allow_pickle=True)
    feature_names = np.load(base / "feature_names.npy",  allow_pickle=True)
    logger.info(f"Loaded {split}: X{X.shape}  y{y.shape}  classes={list(label_classes)}")
    return X, y, label_classes, feature_names


def train_one_seed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cfg_xgb: dict,
    label_classes: np.ndarray,
    feature_names: np.ndarray,
    seed: int,
    seed_idx: int,
    total_seeds: int,
    eval_period: int = 50,
) -> tuple:
    banner = (
        f"\n==========================================================\n"
        f"  XGBoost (Hist GPU)| Seed {seed_idx}/{total_seeds} (seed={seed:<4})\n"
        f"=========================================================="
    )
    print(banner, flush=True)

    xgb_device = get_xgb_device()

    params = {
        "objective":           "multi:softprob",
        "num_class":           len(label_classes),
        "max_depth":           cfg_xgb["max_depth"],
        "learning_rate":       cfg_xgb["learning_rate"],
        "n_estimators":        cfg_xgb["n_estimators"],
        "subsample":           cfg_xgb["subsample"],
        "colsample_bytree":    cfg_xgb["colsample_bytree"],
        "colsample_bylevel":   cfg_xgb.get("colsample_bylevel", 1.0),
        "min_child_weight":    cfg_xgb["min_child_weight"],
        "gamma":               cfg_xgb.get("gamma", 0.0),
        "reg_alpha":           cfg_xgb["reg_alpha"],
        "reg_lambda":          cfg_xgb["reg_lambda"],
        "tree_method":         "hist",
        "device":              xgb_device,
        "random_state":        seed,
        "n_jobs":              1 if xgb_device == "cuda" else -1,
        "verbosity":           0,
        "eval_metric":         "mlogloss",
        "early_stopping_rounds": cfg_xgb["early_stopping_rounds"],
    }

    model = xgb.XGBClassifier(**params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=eval_period,
    )
    elapsed = time.time() - t0

    best_iter = getattr(model, "best_iteration", cfg_xgb["n_estimators"])
    logger.info(f"Seed {seed} completed in {elapsed:.1f}s | Best iteration: {best_iter}")

    print_feature_importance(model, list(feature_names), top_n=15)

    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)
    val_metrics = evaluate(y_val, y_val_pred, y_val_prob, label_classes, split_name=f"val(seed={seed})")

    chunk_size = 1_000_000
    y_test_pred_list = []
    y_test_prob_list = []
    
    for i in range(0, len(X_test), chunk_size):
        X_chunk = X_test[i : i + chunk_size]
        y_test_pred_list.append(model.predict(X_chunk))
        y_test_prob_list.append(model.predict_proba(X_chunk))

    y_test_pred = np.concatenate(y_test_pred_list)
    y_test_prob = np.concatenate(y_test_prob_list)
    
    test_metrics = evaluate(y_test, y_test_pred, y_test_prob, label_classes, split_name=f"test(seed={seed})")

    return val_metrics, test_metrics, model, elapsed


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 seed with 50 trees smoke test")
    args = parser.parse_args()

    log_hardware_summary()
    t_total = time.time()

    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seeds      = [42] if args.smoke_test else cfg["training"]["seeds"]
    val_split  = cfg["data"]["val_split"]
    cfg_xgb    = dict(cfg["models"]["xgboost"])

    if args.smoke_test:
        cfg_xgb["n_estimators"] = 50

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_full, y_train_full, label_classes, feature_names = load_processed("train")
    X_test,       y_test,       _,             _             = load_processed("test", mmap_mode="r")

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

    val_results_all  = []
    test_results_all = []
    best_val_f1      = -1.0
    best_model       = None
    train_times      = []

    eval_period = 10 if args.smoke_test else 50

    for seed_idx, seed in enumerate(seeds, 1):
        set_seed(seed)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_split,
            stratify=y_train_full,
            random_state=seed,
        )

        val_m, test_m, model, elapsed = train_one_seed(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            cfg_xgb, label_classes, feature_names, seed, seed_idx, len(seeds), eval_period=eval_period
        )

        val_results_all.append(val_m)
        test_results_all.append(test_m)
        train_times.append(elapsed)

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_model  = model

    val_agg  = aggregate_seeds(val_results_all)
    test_agg = aggregate_seeds(test_results_all)

    results_payload = {
        "model":            "XGBoost",
        "seeds":            seeds,
        "val_aggregated":   val_agg,
        "test_aggregated":  test_agg,
        "val_per_seed":     val_results_all,
        "test_per_seed":    test_results_all,
        "train_times_s":    train_times,
        "config":           cfg_xgb,
    }
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    if best_model is not None:
        best_model_path = MODELS_DIR / "xgb_best.joblib"
        joblib.dump(best_model, best_model_path)

    print("\n=== Validation results (aggregated) ===")
    print(format_results_table(val_agg, "XGBoost"))
    print("\n=== Test results (aggregated) ===")
    print(format_results_table(test_agg, "XGBoost"))


if __name__ == "__main__":
    main()
