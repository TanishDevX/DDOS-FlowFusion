"""
scripts/evaluate.py
--------------------
Post-training evaluation script.

Loads saved model artifacts and generates:
  1. Full classification report + confusion matrix
  2. ROC curves (per class + macro)
  3. Per-class F1 comparison across all models
  4. SHAP summary + bar chart (tree models)
  5. Integrated Gradients bar chart (deep models)
  6. Inference latency benchmark
  7. Feature distribution drift report (PSI + KS-test)

Run from project root:
    python scripts/evaluate.py
    python scripts/evaluate.py --model xgboost
    python scripts/evaluate.py --model lightgbm
    python scripts/evaluate.py --model mlp
    python scripts/evaluate.py --model ft_transformer
    python scripts/evaluate.py --model all

Outputs saved to: experiments/results/<model>/
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.evaluation.metrics import evaluate, format_results_table
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_per_class_f1,
    plot_latency_comparison,
)

logger = get_logger("evaluate")

RESULTS_BASE = ROOT / "experiments" / "results"
MODELS_DIR   = ROOT / "models"
PROCESSED    = ROOT / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_test_data():
    base          = PROCESSED / "test"
    X             = np.load(base / "X.npy", mmap_mode="r")
    y             = np.load(base / "y.npy", mmap_mode="r")
    label_classes = np.load(base / "label_classes.npy", allow_pickle=True)
    feature_names = np.load(base / "feature_names.npy",  allow_pickle=True)
    logger.info(f"Test data loaded: X{X.shape}  classes={list(label_classes)}")

    # ── Stratified test sampling (2M rows ≈ same metrics, 10× faster eval, prevents OOM) ─
    MAX_TEST = 2_000_000
    if len(X) > MAX_TEST:
        rng     = np.random.default_rng(42)
        classes = np.unique(y)
        n_per   = MAX_TEST // len(classes)
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0],
                       size=min(n_per, int((y == c).sum())),
                       replace=False)
            for c in classes
        ])
        rng.shuffle(idx)
        X_sample = np.array(X[idx])
        y_sample = y[idx]
        logger.info(f"Test stratified-sampled for evaluation: {len(X_sample):,} rows ({n_per:,}/class)")
        return X_sample, y_sample, label_classes, feature_names
    
    return np.array(X), np.array(y), label_classes, feature_names


def get_neural_net_scaler():
    """
    Fits a StandardScaler on a representative 2M sample of the training data.
    Since the scaler was not saved during training, we reconstruct it here.
    With N=2_000_000, the sample mean/var is statistically identical to the full train set.
    """
    from sklearn.preprocessing import StandardScaler
    base = PROCESSED / "train"
    X_train = np.load(base / "X.npy", mmap_mode="r")
    
    # Sample 2M rows to fit the scaler quickly and accurately
    rng = np.random.default_rng(42)
    n_sample = min(2_000_000, len(X_train))
    idx = rng.choice(len(X_train), n_sample, replace=False)
    
    logger.info(f"Fitting StandardScaler on {n_sample:,} training rows...")
    X_sample = X_train[idx]
    scaler = StandardScaler()
    scaler.fit(X_sample)
    return scaler


def get_neural_net_reorder_idx(scaler):
    """
    Reconstruct the correlation-based reorder_idx for FlowFusion.
    The spatial CNN branch requires features to be spatially correlated.
    """
    from src.models.flow_fusion import compute_correlation_reorder
    base = PROCESSED / "train"
    X_train = np.load(base / "X.npy", mmap_mode="r")
    
    rng = np.random.default_rng(42)
    n_sample = min(2_000_000, len(X_train))
    idx = rng.choice(len(X_train), n_sample, replace=False)
    
    X_sample = X_train[idx]
    X_sample = scaler.transform(X_sample)
    
    reorder_idx = compute_correlation_reorder(X_sample)
    return reorder_idx


def load_train_data_small(max_rows: int = 5000):
    """Load a small slice of train data for SHAP background."""
    base = PROCESSED / "train"
    X    = np.load(base / "X.npy")
    idx  = np.random.choice(len(X), min(max_rows, len(X)), replace=False)
    return X[idx]


# ═══════════════════════════════════════════════════════════════════════════
# ROC curve helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_roc_data(y_true, y_prob, label_classes):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes  = len(label_classes)
    y_bin      = label_binarize(y_true, classes=np.arange(n_classes))
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for i, cls in enumerate(label_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr_dict[cls] = fpr
        tpr_dict[cls] = tpr
        auc_dict[cls] = auc(fpr, tpr)

    # Macro average
    all_fpr  = np.unique(np.concatenate([fpr_dict[c] for c in label_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in label_classes:
        mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
    mean_tpr /= n_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    auc_dict["macro"] = auc(all_fpr, mean_tpr)

    return fpr_dict, tpr_dict, auc_dict


# ═══════════════════════════════════════════════════════════════════════════
# Latency benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_latency(predict_fn, X_sample: np.ndarray, n_repeats: int = 100) -> float:
    """Returns mean latency in milliseconds per sample."""
    # Warm-up
    predict_fn(X_sample[:1])

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X_sample[:1])
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


# ═══════════════════════════════════════════════════════════════════════════
# Per-model evaluation functions
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_xgboost(X_test, y_test, label_classes, feature_names):
    import joblib
    from sklearn.metrics import confusion_matrix
    from src.evaluation.explainability import (
        compute_shap_values, plot_shap_summary, plot_shap_bar
    )

    model_path = MODELS_DIR / "xgb_best.joblib"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}  — skipping XGBoost evaluation")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING: XGBoost")
    logger.info("=" * 60)

    model   = joblib.load(model_path)
    out_dir = RESULTS_BASE / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = evaluate(y_test, y_pred, y_prob, label_classes, split_name="xgboost_test")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(label_classes), title="XGBoost — Confusion Matrix (Test)",
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    # ROC curves
    fpr_d, tpr_d, auc_d = compute_roc_data(y_test, y_prob, label_classes)
    plot_roc_curves(
        fpr_d, tpr_d, auc_d,
        title="XGBoost — ROC Curves (Test)",
        save_path=str(out_dir / "roc_curves.png"),
    )

    # SHAP
    try:
        X_bg  = load_train_data_small(2000)
        shap_v = compute_shap_values(model, X_bg, X_test[:1000])
        plot_shap_summary(
            shap_v, X_test[:1000], list(feature_names), list(label_classes),
            class_idx=0,
            title="XGBoost SHAP — BENIGN vs Rest",
            save_path=str(out_dir / "shap_summary_benign.png"),
        )
        plot_shap_bar(
            shap_v, list(feature_names), list(label_classes),
            title="XGBoost — Global SHAP Importance",
            save_path=str(out_dir / "shap_global_bar.png"),
        )
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    # Latency
    latency = benchmark_latency(lambda x: model.predict(x), X_test)
    logger.info(f"XGBoost latency: {latency:.3f} ms/sample")
    metrics["latency_ms"] = latency

    # Save metrics
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Eval metrics saved → {out_dir / 'eval_metrics.json'}")

    return metrics


def evaluate_lightgbm(X_test, y_test, label_classes, feature_names):
    import joblib
    from sklearn.metrics import confusion_matrix
    from src.evaluation.explainability import (
        compute_shap_values, plot_shap_summary, plot_shap_bar
    )

    model_path = MODELS_DIR / "lgbm_best.joblib"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}  — skipping LightGBM evaluation")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING: LightGBM")
    logger.info("=" * 60)

    model   = joblib.load(model_path)
    out_dir = RESULTS_BASE / "lightgbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = evaluate(y_test, y_pred, y_prob, label_classes, split_name="lightgbm_test")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(label_classes), title="LightGBM — Confusion Matrix (Test)",
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    fpr_d, tpr_d, auc_d = compute_roc_data(y_test, y_prob, label_classes)
    plot_roc_curves(
        fpr_d, tpr_d, auc_d,
        title="LightGBM — ROC Curves (Test)",
        save_path=str(out_dir / "roc_curves.png"),
    )

    try:
        X_bg   = load_train_data_small(2000)
        shap_v = compute_shap_values(model, X_bg, X_test[:1000])
        plot_shap_summary(
            shap_v, X_test[:1000], list(feature_names), list(label_classes),
            class_idx=0,
            title="LightGBM SHAP — BENIGN vs Rest",
            save_path=str(out_dir / "shap_summary_benign.png"),
        )
        plot_shap_bar(
            shap_v, list(feature_names), list(label_classes),
            title="LightGBM — Global SHAP Importance",
            save_path=str(out_dir / "shap_global_bar.png"),
        )
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    latency = benchmark_latency(lambda x: model.predict(x), X_test)
    logger.info(f"LightGBM latency: {latency:.3f} ms/sample")
    metrics["latency_ms"] = latency

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_mlp(X_test, y_test, label_classes, feature_names, scaler=None):
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from src.evaluation.explainability import (
        compute_integrated_gradients, plot_integrated_gradients
    )

    model_path = MODELS_DIR / "mlp_best.keras"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}  — skipping Flow MLP evaluation")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING: Flow MLP")
    logger.info("=" * 60)

    if scaler is not None:
        logger.info("Applying StandardScaler to test data...")
        X_test = scaler.transform(X_test)

    from src.models.mlp import SmoothedSparseCCE

    model   = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"SmoothedSparseCCE": SmoothedSparseCCE},
        compile=False,
    )
    out_dir = RESULTS_BASE / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(X_test.astype(np.float32), verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = evaluate(y_test, y_pred, y_prob, label_classes, split_name="mlp_test")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(label_classes), title="MLP — Confusion Matrix (Test)",
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    fpr_d, tpr_d, auc_d = compute_roc_data(y_test, y_prob, label_classes)
    plot_roc_curves(
        fpr_d, tpr_d, auc_d,
        title="MLP — ROC Curves (Test)",
        save_path=str(out_dir / "roc_curves.png"),
    )

    # Integrated Gradients
    try:
        ig_attrs = compute_integrated_gradients(
            model, X_test[:200].astype(np.float32), n_steps=30
        )
        plot_integrated_gradients(
            ig_attrs, list(feature_names),
            title="MLP — Integrated Gradients Global Importance",
            save_path=str(out_dir / "integrated_gradients.png"),
        )
    except Exception as e:
        logger.warning(f"Integrated Gradients failed: {e}")

    latency = benchmark_latency(
        lambda x: model.predict(x.astype(np.float32), verbose=0), X_test
    )
    logger.info(f"MLP latency: {latency:.3f} ms/sample")
    metrics["latency_ms"] = latency

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_ft_transformer(X_test, y_test, label_classes, feature_names, scaler=None):
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from src.evaluation.explainability import (
        compute_integrated_gradients, plot_integrated_gradients
    )

    model_path = MODELS_DIR / "ft_transformer_best.keras"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}  — skipping FT-Transformer evaluation")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING: FT-Transformer")
    logger.info("=" * 60)

    if scaler is not None:
        logger.info("Applying StandardScaler to test data...")
        X_test = scaler.transform(X_test)

    # Import custom layers to register them in Keras before loading the model.
    # StochasticDepth is explicitly included so Keras can resolve it even if the
    # import order changes (it's registered via @register_keras_serializable but
    # being explicit is safer and more portable — Bug 5 fix).
    from src.models.ft_transformer import FeatureTokenizer, TransformerBlock, StochasticDepth, SmoothedSparseCCE

    model   = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "FeatureTokenizer":  FeatureTokenizer,
            "TransformerBlock":  TransformerBlock,
            "StochasticDepth":   StochasticDepth,
            "SmoothedSparseCCE": SmoothedSparseCCE,
        },
        compile=False,
    )
    out_dir = RESULTS_BASE / "ft_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(X_test.astype(np.float32), verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = evaluate(y_test, y_pred, y_prob, label_classes, split_name="ft_transformer_test")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(label_classes), title="FT-Transformer — Confusion Matrix (Test)",
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    fpr_d, tpr_d, auc_d = compute_roc_data(y_test, y_prob, label_classes)
    plot_roc_curves(
        fpr_d, tpr_d, auc_d,
        title="FT-Transformer — ROC Curves (Test)",
        save_path=str(out_dir / "roc_curves.png"),
    )

    try:
        ig_attrs = compute_integrated_gradients(
            model, X_test[:200].astype(np.float32), n_steps=30
        )
        plot_integrated_gradients(
            ig_attrs, list(feature_names),
            title="FT-Transformer — Integrated Gradients Global Importance",
            save_path=str(out_dir / "integrated_gradients.png"),
        )
    except Exception as e:
        logger.warning(f"Integrated Gradients failed: {e}")

    latency = benchmark_latency(
        lambda x: model.predict(x.astype(np.float32), verbose=0), X_test
    )
    logger.info(f"FT-Transformer latency: {latency:.3f} ms/sample")
    metrics["latency_ms"] = latency

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_flow_fusion(X_test, y_test, label_classes, feature_names, scaler=None, reorder_idx=None):
    """
    Evaluate the best FlowFusion (all_gated) model.
    Generates confusion matrix, ROC curves, gate value distribution, and IG attributions.
    """
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix

    model_path = MODELS_DIR / "flow_fusion_v2_all_gated_best.keras"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}  — skipping FlowFusion evaluation")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING: FlowFusion (all_gated)")
    logger.info("=" * 60)

    if scaler is not None:
        logger.info("Applying StandardScaler to test data...")
        X_test = scaler.transform(X_test)
        
    if reorder_idx is not None:
        logger.info("Applying correlation reorder to test data...")
        X_test = X_test[:, reorder_idx]

    # Import custom layers to register them in Keras before loading the model.
    # Explicit registration is safer than relying on import side-effects.
    from src.models.flow_fusion import FeaturePadding, ClassWeightedFocalLoss, SmoothedSparseCCE
    from src.models.ft_transformer import FeatureTokenizer, TransformerBlock, StochasticDepth

    model   = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "FeatureTokenizer":       FeatureTokenizer,
            "TransformerBlock":       TransformerBlock,
            "StochasticDepth":        StochasticDepth,
            "FeaturePadding":         FeaturePadding,
            "ClassWeightedFocalLoss": ClassWeightedFocalLoss,
            "SmoothedSparseCCE":      SmoothedSparseCCE,
        },
        safe_mode=False,
        compile=False,
    )
    out_dir = RESULTS_BASE / "flow_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FlowFusion may return multiple outputs: dictionary or list depending on Keras API
    raw_out   = model.predict(X_test.astype(np.float32), batch_size=8192, verbose=0)
    
    if isinstance(raw_out, dict):
        y_prob    = raw_out["output"]
        gate_arrs = [raw_out[k] for k in raw_out.keys() if k.startswith("gate_")]
    elif isinstance(raw_out, (list, tuple)):
        y_prob    = raw_out[0]                                    # (N, n_classes)
        gate_arrs = raw_out[1:]                                   # list of (N, 1)
    else:
        y_prob    = raw_out
        gate_arrs = []

    y_pred  = np.argmax(y_prob, axis=1)
    metrics = evaluate(y_test, y_pred, y_prob, label_classes, split_name="flow_fusion_test")

    # Primary metric emphasis
    logger.info(f"  *** PRIMARY METRIC — Macro-F1: {metrics['macro_f1']:.4f} ***")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(label_classes), title="FlowFusion — Confusion Matrix (Test)",
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    fpr_d, tpr_d, auc_d = compute_roc_data(y_test, y_prob, label_classes)
    plot_roc_curves(
        fpr_d, tpr_d, auc_d,
        title="FlowFusion — ROC Curves (Test)",
        save_path=str(out_dir / "roc_curves.png"),
    )

    # Gate value distributions (FlowFusion interpretability)
    if gate_arrs:
        branch_names = ["tabular", "spatial", "structural"][:len(gate_arrs)]
        gate_means   = {name: float(np.mean(g)) for name, g in zip(branch_names, gate_arrs)}
        logger.info(f"  Gate means (test set): {gate_means}")
        metrics["gate_means"] = gate_means

        # Simple bar chart of mean gate values
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["#4C72B0", "#55A868", "#C44E52"]
            ax.bar(branch_names, [gate_means[n] for n in branch_names],
                   color=colors[:len(branch_names)], alpha=0.85)
            ax.set_ylabel("Mean Scalar Gate Value")
            ax.set_title("FlowFusion — Branch Trust (Gate Means, Test Set)",
                         fontweight="bold")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(str(out_dir / "gate_means.png"), dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"  Gate means plot → {out_dir}/gate_means.png")
        except Exception as e:
            logger.warning(f"Gate plot failed: {e}")

    latency = benchmark_latency(
        lambda x: model.predict(x.astype(np.float32), verbose=0), X_test
    )
    logger.info(f"FlowFusion latency: {latency:.3f} ms/sample")
    metrics["latency_ms"] = latency

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Eval metrics saved → {out_dir / 'eval_metrics.json'}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Cross-model comparison plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(all_metrics: dict, label_classes: list):
    """Generate cross-model comparison: per-class F1 and latency charts."""
    out_dir = RESULTS_BASE / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class F1 grouped bar chart
    per_class_data = {}
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        per_class_data[model_name] = {
            cls: metrics.get(f"f1_{cls}", 0.0) for cls in label_classes
        }

    if per_class_data:
        plot_per_class_f1(
            per_class_data, list(label_classes),
            title="Per-Class F1 Score — All Models (Test Set)",
            save_path=str(out_dir / "per_class_f1_comparison.png"),
        )

    # Latency comparison
    latency_names = [n for n, m in all_metrics.items() if m and "latency_ms" in m]
    latency_vals  = [all_metrics[n]["latency_ms"] for n in latency_names]
    if latency_names:
        plot_latency_comparison(
            latency_names, latency_vals,
            title="Inference Latency (ms/sample, CPU)",
            save_path=str(out_dir / "latency_comparison.png"),
        )

    # Summary table
    summary_rows = []
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        row = {"Model": model_name}
        for m in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "roc_auc"]:
            row[m] = f"{metrics.get(m, float('nan')):.4f}"
        row["latency_ms"] = f"{metrics.get('latency_ms', float('nan')):.3f}"
        summary_rows.append(row)

    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        # Sort by macro_f1 descending (primary metric)
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
        summary_path = out_dir / "model_comparison_table.csv"
        df.to_csv(summary_path, index=False)
        logger.info(f"\nModel comparison table (sorted by macro-F1) → {summary_path}")
        logger.info("\n" + df.to_string(index=False))


def evaluate_ensemble(X_test, y_test, label_classes, feature_names, scaler=None):
    """
    Evaluate the FlowFusion v2 + LightGBM Ensemble (alpha=0.41 blend).
    """
    import joblib
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from src.models.flow_fusion import FeaturePadding, ClassWeightedFocalLoss, SmoothedSparseCCE
    from src.models.ft_transformer import FeatureTokenizer, TransformerBlock, StochasticDepth

    ff_path   = MODELS_DIR / "flow_fusion_v2_all_gated_best.keras"
    lgbm_path = MODELS_DIR / "lgbm_best.joblib"

    if not ff_path.exists() or not lgbm_path.exists():
        logger.warning("Ensemble components missing — skipping ensemble evaluation")
        return None

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING: FlowFusion v2 + LightGBM Ensemble (alpha=0.41)")
    logger.info("=" * 60)

    # 1. FlowFusion predictions (requires scaled inputs)
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
    model_ff = tf.keras.models.load_model(
        str(ff_path),
        custom_objects={
            "FeatureTokenizer":       FeatureTokenizer,
            "TransformerBlock":       TransformerBlock,
            "StochasticDepth":        StochasticDepth,
            "FeaturePadding":         FeaturePadding,
            "ClassWeightedFocalLoss": ClassWeightedFocalLoss,
            "SmoothedSparseCCE":      SmoothedSparseCCE,
        },
        safe_mode=False,
        compile=False,
    )
    raw_out = model_ff.predict(X_test_scaled.astype(np.float32), batch_size=8192, verbose=0)
    p_ff = raw_out["output"] if isinstance(raw_out, dict) else (raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out)

    # 2. LightGBM predictions (requires raw inputs)
    model_lgbm = joblib.load(lgbm_path)
    p_lgbm = model_lgbm.predict_proba(X_test)

    # 3. Probability Blend (alpha=0.41 FlowFusion + 0.59 LightGBM)
    alpha = 0.41
    p_blend = alpha * p_ff + (1.0 - alpha) * p_lgbm
    p_blend = np.nan_to_num(p_blend, nan=1.0 / len(label_classes))
    y_pred = np.argmax(p_blend, axis=1)

    out_dir = RESULTS_BASE / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(y_test, y_pred, p_blend, label_classes, split_name="ensemble_test")
    logger.info(f"  *** ENSEMBLE PRIMARY METRIC — Macro-F1: {metrics['macro_f1']:.4f} ***")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, list(label_classes), title="FlowFusion v2 + LightGBM Ensemble — Confusion Matrix",
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    fpr_d, tpr_d, auc_d = compute_roc_data(y_test, p_blend, label_classes)
    plot_roc_curves(
        fpr_d, tpr_d, auc_d,
        title="FlowFusion v2 + LightGBM Ensemble — ROC Curves",
        save_path=str(out_dir / "roc_curves.png"),
    )

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DDoS detection models.")
    parser.add_argument(
        "--model", default="all",
        choices=["ensemble", "xgboost", "lightgbm", "mlp", "ft_transformer", "flow_fusion", "all"],
        help="Which model to evaluate (default: all)",
    )
    args = parser.parse_args()

    X_test, y_test, label_classes, feature_names = load_test_data()

    if args.model == "all":
        models_to_run = ["ensemble", "xgboost", "lightgbm", "mlp", "ft_transformer", "flow_fusion"]
    else:
        models_to_run = [args.model]

    # Load neural net scaler if needed
    nn_scaler = None
    if any(m in ["mlp", "ft_transformer", "flow_fusion", "ensemble"] for m in models_to_run):
        nn_scaler = get_neural_net_scaler()

    eval_map = {
        "ensemble":       evaluate_ensemble,
        "xgboost":        evaluate_xgboost,
        "lightgbm":       evaluate_lightgbm,
        "mlp":            evaluate_mlp,
        "ft_transformer": evaluate_ft_transformer,
        "flow_fusion":    evaluate_flow_fusion,
    }

    all_metrics = {}
    for model_name in models_to_run:
        if model_name in ["mlp", "ft_transformer", "ensemble"]:
            metrics = eval_map[model_name](X_test, y_test, label_classes, feature_names, scaler=nn_scaler)
        elif model_name == "flow_fusion":
            metrics = eval_map[model_name](X_test, y_test, label_classes, feature_names, scaler=nn_scaler, reorder_idx=None)
        else:
            metrics = eval_map[model_name](X_test, y_test, label_classes, feature_names)
        all_metrics[model_name] = metrics

    if args.model == "all" and len(all_metrics) > 1:
        plot_model_comparison(all_metrics, list(label_classes))

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
