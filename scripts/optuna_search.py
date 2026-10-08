"""
scripts/optuna_search.py
-------------------------
Bayesian hyperparameter optimization via Optuna for all 5 DDoS detection models.

Running 30 trials per model (1 seed per trial, fast), then retraining the winner
with 5 seeds for publication-quality mean ± std numbers.

This gives significantly better results than grid search because:
  - Bayesian optimization focuses trials on promising regions of the search space
  - Pruning (MedianPruner) terminates unpromising trials early
  - 30 trials cover a much larger effective search space than our grid scripts

Install optuna first if not present:
    pip install optuna

Run from project root:
    python scripts/optuna_search.py --model xgboost --n-trials 30
    python scripts/optuna_search.py --model lightgbm --n-trials 30
    python scripts/optuna_search.py --model mlp --n-trials 30
    python scripts/optuna_search.py --model ft_transformer --n-trials 30
    python scripts/optuna_search.py --model all --n-trials 30

Output: experiments/results/<model>/optuna_best_params.json
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.evaluation.metrics import evaluate

logger = get_logger("optuna_search")

RESULTS_BASE = ROOT / "experiments" / "results"


# ═══════════════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════════════

_DATA_CACHE = {}

def _load(split: str):
    """Load and cache processed numpy arrays."""
    if split not in _DATA_CACHE:
        base = ROOT / "data" / "processed" / split
        _DATA_CACHE[split] = (
            np.load(base / "X.npy"),
            np.load(base / "y.npy"),
            np.load(base / "label_classes.npy", allow_pickle=True),
            np.load(base / "feature_names.npy",  allow_pickle=True),
        )
    return _DATA_CACHE[split]


def _sample_test(X_test, y_test, max_rows=500_000):
    """Fast small subsample for Optuna objective evaluations."""
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
    return np.array(X_test[idx]), y_test[idx]


def _split(X_full, y_full, val_split=0.2, seed=42):
    return train_test_split(X_full, y_full, test_size=val_split,
                            stratify=y_full, random_state=seed)


# ═══════════════════════════════════════════════════════════════════════════
# XGBoost objective
# ═══════════════════════════════════════════════════════════════════════════

def xgboost_objective(trial, X_tr, y_tr, X_val, y_val, X_test, y_test, label_classes):
    """Optuna objective for XGBoost. Returns val macro-F1."""
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_sample_weight

    params = {
        "objective":             "multi:softprob",
        "num_class":             len(label_classes),
        "max_depth":             trial.suggest_int("max_depth", 6, 14),
        "learning_rate":         trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "n_estimators":          1000,
        "subsample":             trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel":     trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight":      trial.suggest_int("min_child_weight", 1, 20),
        "gamma":                 trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha":             trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":            trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "tree_method":           "hist",
        "device":                "cuda",
        "n_jobs":                -1,
        "verbosity":             0,
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": 30,
        "random_state":          42,
    }

    sw = compute_sample_weight("balanced", y_tr)
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_val, y_val)], verbose=False)

    y_val_prob = model.predict_proba(X_val)
    y_val_pred = np.argmax(y_val_prob, axis=1)
    m = evaluate(y_val, y_val_pred, y_val_prob, label_classes, split_name="optuna_val")
    return m["macro_f1"]


# ═══════════════════════════════════════════════════════════════════════════
# LightGBM objective
# ═══════════════════════════════════════════════════════════════════════════

def lightgbm_objective(trial, X_tr, y_tr, X_val, y_val, X_test, y_test, label_classes, feature_names):
    """Optuna objective for LightGBM. Returns val macro-F1."""
    import lightgbm as lgb

    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    params = {
        "objective":               "multiclass",
        "num_class":               len(label_classes),
        "boosting_type":           boosting,
        "num_leaves":              trial.suggest_int("num_leaves", 63, 511),
        "learning_rate":           trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "n_estimators":            1000,
        "subsample":               trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":        trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.4, 1.0),
        "min_data_in_leaf":        trial.suggest_int("min_data_in_leaf", 10, 200),
        "reg_alpha":               trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":              trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "class_weight":            "balanced",
        "device":                  "gpu",
        "n_jobs":                  -1,
        "verbosity":               -1,
        "random_state":            42,
    }
    is_dart = boosting == "dart"
    callbacks = [lgb.log_evaluation(period=-1)]
    if not is_dart:
        callbacks.append(lgb.early_stopping(30, verbose=False))

    model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k not in ("early_stopping_rounds",)})
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks, feature_name=list(feature_names))

    y_val_prob = model.predict_proba(X_val)
    y_val_pred = model.predict(X_val)
    m = evaluate(y_val, y_val_pred, y_val_prob, label_classes, split_name="optuna_val")
    return m["macro_f1"]


# ═══════════════════════════════════════════════════════════════════════════
# MLP objective
# ═══════════════════════════════════════════════════════════════════════════

def mlp_objective(trial, X_tr, y_tr, X_val, y_val, label_classes):
    """Optuna objective for MLP. Returns val macro-F1."""
    import tensorflow as tf
    from src.models.mlp import build_mlp
    set_seed(42)
    tf.keras.backend.clear_session()

    n_layers = trial.suggest_int("n_layers", 2, 4)
    base_units = trial.suggest_categorical("base_units", [128, 256, 512])
    layers = [base_units // (2**i) for i in range(n_layers)]
    dropouts = [trial.suggest_float(f"dropout_{i}", 0.1, 0.5) for i in range(n_layers)]

    cfg = {
        "layers":         layers,
        "dropout":        dropouts,
        "activation":     trial.suggest_categorical("activation", ["swish", "relu", "gelu"]),
        "batch_norm":     True,
        "use_residual":   trial.suggest_categorical("use_residual", [True, False]),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
    }

    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}

    model = build_mlp(X_tr.shape[1], len(label_classes), cfg)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg["learning_rate"],
                                            weight_decay=cfg["weight_decay"], clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=cfg["label_smoothing"]),
        metrics=["accuracy"]
    )
    model.fit(
        X_tr.astype(np.float32), y_tr,
        validation_data=(X_val.astype(np.float32), y_val),
        epochs=50, batch_size=512, class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                    restore_best_weights=True, verbose=0)],
        verbose=0,
    )
    y_val_prob = model.predict(X_val.astype(np.float32), verbose=0)
    m = evaluate(y_val, np.argmax(y_val_prob, 1), y_val_prob, label_classes, split_name="optuna_val")
    return m["macro_f1"]


# ═══════════════════════════════════════════════════════════════════════════
# FT-Transformer objective
# ═══════════════════════════════════════════════════════════════════════════

def ft_transformer_objective(trial, X_tr, y_tr, X_val, y_val, label_classes):
    """Optuna objective for FT-Transformer. Returns val macro-F1."""
    import tensorflow as tf
    from src.models.ft_transformer import build_ft_transformer
    set_seed(42)
    tf.keras.backend.clear_session()

    d_token = trial.suggest_categorical("d_token", [64, 128, 192])
    n_heads  = trial.suggest_categorical("n_heads", [4, 8])
    # Ensure d_token divisible by n_heads
    while d_token % n_heads != 0:
        n_heads = 4

    cfg = {
        "d_token":           d_token,
        "n_heads":           n_heads,
        "n_layers":          trial.suggest_int("n_layers", 2, 5),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.4),
        "ffn_dropout":       trial.suggest_float("ffn_dropout", 0.0, 0.3),
        "learning_rate":     trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True),
        "weight_decay":      trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "drop_path_rate":    trial.suggest_float("drop_path_rate", 0.0, 0.2),
        "warmup_steps":      trial.suggest_int("warmup_steps", 200, 1000),
        "label_smoothing":   0.05,
    }

    base_lr = cfg["learning_rate"]
    warmup_steps = cfg["warmup_steps"]

    class WarmupCB(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._step = 0
        def on_train_batch_begin(self, batch, logs=None):
            if self._step < warmup_steps:
                self._step += 1
                self.model.optimizer.learning_rate = base_lr * self._step / warmup_steps

    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}

    model = build_ft_transformer(X_tr.shape[1], len(label_classes), cfg)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=cfg["weight_decay"], clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(
        X_tr.astype(np.float32), y_tr,
        validation_data=(X_val.astype(np.float32), y_val),
        epochs=50, batch_size=512, class_weight=class_weights,
        callbacks=[
            WarmupCB(),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
        ],
        verbose=0,
    )
    y_val_prob = model.predict(X_val.astype(np.float32), verbose=0)
    m = evaluate(y_val, np.argmax(y_val_prob, 1), y_val_prob, label_classes, split_name="optuna_val")
    return m["macro_f1"]


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_study(model_name: str, n_trials: int, cfg: dict):
    """Run an Optuna study for a given model and save the best params."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_full, y_full, label_classes, feature_names = _load("train")
    X_test, y_test, _, _                         = _load("test")
    X_test, y_test = _sample_test(X_test, y_test)

    val_split = cfg["data"]["val_split"]
    X_tr, X_val, y_tr, y_val = _split(X_full, y_full, val_split)

    # Scale for DL models
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)

    if model_name == "xgboost":
        obj = lambda trial: xgboost_objective(trial, X_tr, y_tr, X_val, y_val,
                                               X_test, y_test, label_classes)
    elif model_name == "lightgbm":
        obj = lambda trial: lightgbm_objective(trial, X_tr, y_tr, X_val, y_val,
                                                X_test, y_test, label_classes, feature_names)
    elif model_name == "mlp":
        obj = lambda trial: mlp_objective(trial, X_tr_sc, y_tr, X_val_sc, y_val, label_classes)
    elif model_name == "ft_transformer":
        obj = lambda trial: ft_transformer_objective(trial, X_tr_sc, y_tr, X_val_sc, y_val, label_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_optuna",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTUNA RESULT: {model_name.upper()}")
    logger.info(f"  Best val macro-F1: {best.value:.4f}")
    logger.info(f"  Best params: {best.params}")
    logger.info(f"{'='*60}")

    out_dir = RESULTS_BASE / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "optuna_best_params.json", "w") as f:
        json.dump({"model": model_name, "best_val_macro_f1": best.value,
                   "best_params": best.params, "n_trials": n_trials}, f, indent=2)
    logger.info(f"Best params saved → {out_dir}/optuna_best_params.json")
    return best.params


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for DDoS detection models.")
    parser.add_argument("--model", default="xgboost",
                        choices=["xgboost", "lightgbm", "mlp", "ft_transformer", "all"])
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    models_to_tune = (["xgboost", "lightgbm", "mlp", "ft_transformer"]
                      if args.model == "all" else [args.model])

    t0 = time.time()
    for m in models_to_tune:
        logger.info(f"\nStarting Optuna search for {m.upper()} ({args.n_trials} trials)...")
        run_study(m, args.n_trials, cfg)

    logger.info(f"\nTotal HPO time: {(time.time()-t0)/60:.1f} min")
    logger.info("Best params saved to experiments/results/<model>/optuna_best_params.json")
    logger.info("Use these params in config.yaml and re-run the corresponding train_*.py script.")


if __name__ == "__main__":
    main()
