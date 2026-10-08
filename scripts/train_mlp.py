import sys, json, time, argparse
from pathlib import Path
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.gpu import setup_gpu
from src.utils.progress import (
    CosineDecayCallback,
    NaNDetectionCallback,
    MacroF1ValidationCallback,
    RichTrainingCallback,
)
from src.models.mlp import build_mlp
from src.evaluation.metrics import evaluate, aggregate_seeds, format_results_table
from src.data.augmentation import augment_benign

logger = get_logger('train_mlp')
RESULTS_DIR = ROOT / 'experiments' / 'results' / 'mlp'
MODELS_DIR = ROOT / 'models'


def load_processed(split, mmap_mode=None):
    base = ROOT / 'data' / 'processed' / split
    X = np.load(base / 'X.npy', mmap_mode=mmap_mode)
    y = np.load(base / 'y.npy', mmap_mode=mmap_mode)
    label_classes = np.load(base / 'label_classes.npy', allow_pickle=True)
    feature_names = np.load(base / 'feature_names.npy', allow_pickle=True)
    logger.info(f'Loaded {split}: X{X.shape}  y{y.shape}  classes={list(label_classes)}')
    return X, y, label_classes, feature_names


def train_one_seed(
    X_train, y_train, X_val, y_val, X_test, y_test,
    cfg_mlp, cfg_train, label_classes, seed, seed_idx, total_seeds, scaler
) -> tuple:
    set_seed(seed)
    input_dim = X_train.shape[1]
    num_classes = len(label_classes)

    model = build_mlp(input_dim, num_classes, cfg_mlp)

    classes = np.arange(num_classes)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {i: float(w) for i, w in enumerate(cw)}

    base_lr = cfg_mlp.get("learning_rate", cfg_train["learning_rate"])
    total_epochs = cfg_mlp.get("epochs", cfg_train.get("epochs", 80))
    warmup_epochs = cfg_mlp.get("warmup_epochs", 5)

    callbacks = [
        CosineDecayCallback(
            base_lr=base_lr,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=cfg_train.get("min_lr", 1e-6),
        ),
        NaNDetectionCallback(threshold=1e6),
        MacroF1ValidationCallback(val_x=X_val, val_y=y_val, num_classes=num_classes),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_macro_f1',
            mode='max',
            patience=cfg_train['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0,
        ),
        RichTrainingCallback(
            model_name="FlowMLP",
            seed=seed,
            seed_idx=seed_idx,
            total_seeds=total_seeds,
            total_epochs=total_epochs,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        batch_size=cfg_train['batch_size'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0,
    )
    elapsed = time.time() - t0

    y_val_prob = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_prob, axis=1)
    val_metrics = evaluate(y_val, y_val_pred, y_val_prob, label_classes, split_name=f'val(seed={seed})')

    chunk_size = 1_000_000
    y_test_pred_list, y_test_prob_list = [], []
    for i in range(0, len(X_test), chunk_size):
        X_chunk = scaler.transform(X_test[i:i + chunk_size])
        y_prob_chunk = model.predict(X_chunk, batch_size=8192, verbose=0)
        y_test_prob_list.append(y_prob_chunk)
        y_test_pred_list.append(np.argmax(y_prob_chunk, axis=1))
    y_test_prob = np.concatenate(y_test_prob_list, axis=0)
    y_test_pred = np.concatenate(y_test_pred_list, axis=0)
    test_metrics = evaluate(y_test, y_test_pred, y_test_prob, label_classes, split_name=f'test(seed={seed})')

    return val_metrics, test_metrics, model, elapsed, history.history


def main():
    parser = argparse.ArgumentParser(description="Train FlowMLP model")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 seed for 2 epochs smoke test")
    args = parser.parse_args()

    setup_gpu()
    t_total = time.time()

    cfg_path = ROOT / 'config' / 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seeds = [42] if args.smoke_test else cfg['training']['seeds']
    val_split = cfg['data']['val_split']
    cfg_mlp = dict(cfg['models']['mlp'])
    cfg_train = dict(cfg['training'])

    if args.smoke_test:
        cfg_mlp['epochs'] = 2
        cfg_train['epochs'] = 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_full, y_train_full, label_classes, feature_names = load_processed('train')
    X_test, y_test, _, _ = load_processed('test', mmap_mode='r')

    MAX_TEST = 2_000_000
    if len(X_test) > MAX_TEST:
        rng = np.random.default_rng(42)
        classes = np.unique(y_test)
        n_per = MAX_TEST // len(classes)
        idx = np.concatenate([
            rng.choice(
                np.where(y_test == c)[0],
                size=min(n_per, int((y_test == c).sum())),
                replace=False,
            )
            for c in classes
        ])
        rng.shuffle(idx)
        X_test = np.array(X_test[idx])
        y_test = y_test[idx]

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)

    val_results_all, test_results_all = [], []
    best_val_f1, best_model = -1.0, None
    train_times = []

    for seed_idx, seed in enumerate(seeds, 1):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_split,
            stratify=y_train_full,
            random_state=seed,
        )

        if cfg_train.get("use_augmentation", False):
            benign_idx = list(label_classes).index("BENIGN")
            X_tr, y_tr = augment_benign(
                X_tr, y_tr,
                benign_class_idx=benign_idx,
                gaussian_sigma=cfg_train.get("aug_gaussian_sigma", 0.01),
                mixup_alpha=cfg_train.get("aug_mixup_alpha", 0.2),
                n_gaussian=cfg_train.get("aug_n_gaussian", 5000),
                n_mixup=cfg_train.get("aug_n_mixup", 5000),
                seed=seed,
            )

        val_m, test_m, model, elapsed, history = train_one_seed(
            X_tr, y_tr, X_val, y_val, X_test, y_test,
            cfg_mlp, cfg_train, label_classes, seed, seed_idx, len(seeds), scaler,
        )
        val_results_all.append(val_m)
        test_results_all.append(test_m)
        train_times.append(elapsed)

        if val_m['macro_f1'] > best_val_f1:
            best_val_f1 = val_m['macro_f1']
            best_model = model

        tf.keras.backend.clear_session()

    val_agg = aggregate_seeds(val_results_all)
    test_agg = aggregate_seeds(test_results_all)

    results_payload = {
        'model': 'FlowMLP',
        'seeds': seeds,
        'val_aggregated': val_agg,
        'test_aggregated': test_agg,
        'val_per_seed': val_results_all,
        'test_per_seed': test_results_all,
        'train_times_s': train_times,
        'config': cfg_mlp,
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results_payload, f, indent=2)

    if best_model is not None:
        best_model.save(str(MODELS_DIR / 'mlp_best.keras'))

    total_elapsed = time.time() - t_total
    logger.info(f'Total wall time: {total_elapsed:.1f}s')
    logger.info(f'Best val macro-F1 across seeds: {best_val_f1:.4f}')

    print('\n=== Validation results (aggregated) ===')
    print(format_results_table(val_agg, "FlowMLP"))
    print('\n=== Test results (aggregated) ===')
    print(format_results_table(test_agg, "FlowMLP"))


if __name__ == '__main__':
    main()
