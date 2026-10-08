"""
src/evaluation/explainability.py
---------------------------------
Model explainability utilities.

  - TreeSHAP : XGBoost / LightGBM  (uses SHAP TreeExplainer — exact, fast)
  - IntGrad   : Keras MLP / FT-Transformer  (Integrated Gradients, TF-native)

Usage:
    from src.evaluation.explainability import (
        compute_shap_values,
        plot_shap_summary,
        compute_integrated_gradients,
        plot_integrated_gradients,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for script runs

from pathlib import Path
from typing import Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Tree-based SHAP  (XGBoost / LightGBM)
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    max_explain: int = 2000,
) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer.

    Args:
        model:         Trained XGBoost or LightGBM model object.
        X_background:  Background dataset (typically train split, used for E[f(X)]).
        X_explain:     Data to explain (typically test split).
        max_explain:   Cap number of explained rows to keep runtime manageable.

    Returns:
        shap_values array of shape (n_samples, n_features, n_classes) or
        (n_samples, n_features) depending on model type.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install SHAP: pip install shap")

    logger.info(f"Computing TreeSHAP for {min(len(X_explain), max_explain)} samples …")

    # Subsample for speed
    if len(X_explain) > max_explain:
        idx = np.random.choice(len(X_explain), max_explain, replace=False)
        X_explain = X_explain[idx]

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_explain)

    logger.info(
        f"SHAP values computed. "
        f"Type: {type(shap_vals)}  "
        f"Shape: {np.array(shap_vals).shape}"
    )
    return shap_vals


def plot_shap_summary(
    shap_values,
    X_explain: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    class_idx: int = 0,
    max_features: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """
    Beeswarm summary plot for one class.

    Args:
        shap_values:   Output of compute_shap_values (list of arrays, one per class).
        X_explain:     Data that was explained.
        feature_names: List of feature name strings.
        class_names:   List of class name strings.
        class_idx:     Which class to plot (index into shap_values list).
        max_features:  Number of top features to display.
        title:         Plot title.
        save_path:     If provided, save figure here.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install SHAP: pip install shap")

    # shap_values is a list of (n_samples, n_features) arrays or a 3D numpy array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
        cls_label = class_names[class_idx] if class_names else str(class_idx)
        full_title = f"{title} — Class: {cls_label}"
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, class_idx]
        cls_label = class_names[class_idx] if class_names else str(class_idx)
        full_title = f"{title} — Class: {cls_label}"
    else:
        sv = shap_values
        full_title = title

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_explain,
        feature_names=feature_names,
        max_display=max_features,
        show=False,
        plot_type="dot",
    )
    plt.title(full_title, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"SHAP summary plot saved → {save_path}")
    plt.close()


def plot_shap_bar(
    shap_values,
    feature_names: List[str],
    class_names: List[str],
    max_features: int = 20,
    title: str = "Mean |SHAP| — Global Feature Importance",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of mean absolute SHAP values across all classes (global importance).
    """
    # Average |SHAP| across all classes and all samples
    if isinstance(shap_values, list):
        # List of (n_samples, n_features) — one per class
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape (n_samples, n_features, n_classes) -> average over axis 0 (samples) and 2 (classes)
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    top_idx  = np.argsort(mean_abs)[::-1][:max_features]
    top_vals = mean_abs[top_idx]
    top_feat = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, max_features * 0.35 + 1))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top_feat)))
    ax.barh(top_feat[::-1], top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"SHAP bar chart saved → {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Integrated Gradients  (Keras MLP / FT-Transformer)
# ═══════════════════════════════════════════════════════════════════════════

def _interpolate_inputs(
    baseline: np.ndarray,
    inputs: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    Interpolate between baseline and inputs along the IG path.

    Returns array of shape (n_steps, n_features).
    """
    alphas = alphas[:, np.newaxis]                            # (steps, 1)
    return baseline + alphas * (inputs - baseline)            # (steps, n_features)


def compute_integrated_gradients(
    model,
    X_explain: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    batch_size: int = 256,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients attributions for a Keras model.

    Args:
        model:        Trained Keras model (MLP or FT-Transformer).
        X_explain:    Samples to explain, shape (n_samples, n_features).
        baseline:     Reference input (default: zeros — "no information").
        n_steps:      Number of Riemann steps along the IG path.
        batch_size:   Batch size for forward passes (avoids OOM).
        target_class: Class index to explain. If None, uses the predicted class per sample.

    Returns:
        attributions array, shape (n_samples, n_features).
    """
    import tensorflow as tf

    if baseline is None:
        baseline = np.zeros((1, X_explain.shape[1]), dtype=np.float32)
    else:
        baseline = baseline.astype(np.float32)

    alphas    = np.linspace(0.0, 1.0, n_steps + 1)           # (n_steps+1,)
    n_samples = X_explain.shape[0]
    n_features= X_explain.shape[1]
    all_attrs = np.zeros((n_samples, n_features), dtype=np.float32)

    logger.info(
        f"Computing Integrated Gradients: {n_samples} samples, "
        f"{n_steps} steps, batch_size={batch_size}"
    )

    for sample_idx in range(n_samples):
        x_single   = X_explain[sample_idx : sample_idx + 1].astype(np.float32)  # (1, F)
        interp_pts = _interpolate_inputs(baseline, x_single, alphas)             # (steps, F)

        # Get target class
        if target_class is not None:
            tc = target_class
        else:
            tc = int(np.argmax(
                model.predict(x_single, verbose=0), axis=1
            )[0])

        # Compute gradients in batches over the interpolation path
        grads_all = []
        for start in range(0, len(interp_pts), batch_size):
            batch = tf.constant(interp_pts[start : start + batch_size])
            with tf.GradientTape() as tape:
                tape.watch(batch)
                preds = model(batch, training=False)
                target_preds = preds[:, tc]
            grads = tape.gradient(target_preds, batch)  # (batch, F)
            grads_all.append(grads.numpy())

        grads_array = np.concatenate(grads_all, axis=0)  # (n_steps+1, F)

        # Trapezoidal integration
        avg_grads   = (grads_array[:-1] + grads_array[1:]) / 2.0   # (n_steps, F)
        mean_grads  = avg_grads.mean(axis=0)                         # (F,)
        delta       = x_single[0] - baseline[0]                      # (F,)
        all_attrs[sample_idx] = mean_grads * delta

    logger.info("Integrated Gradients computation complete.")
    return all_attrs


def plot_integrated_gradients(
    attributions: np.ndarray,
    feature_names: List[str],
    max_features: int = 20,
    title: str = "Integrated Gradients — Mean |Attribution|",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of mean absolute IG attributions (global feature importance).

    Args:
        attributions:  Array (n_samples, n_features) from compute_integrated_gradients.
        feature_names: Feature name strings.
        max_features:  Number of top features to show.
        title:         Plot title.
        save_path:     If provided, save figure here.
    """
    mean_abs = np.abs(attributions).mean(axis=0)               # (n_features,)
    top_idx  = np.argsort(mean_abs)[::-1][:max_features]
    top_vals = mean_abs[top_idx]
    top_feat = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, max_features * 0.35 + 1))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(top_feat)))
    ax.barh(top_feat[::-1], top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mean |Integrated Gradient Attribution|", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"IG attribution plot saved → {save_path}")
    plt.close()
