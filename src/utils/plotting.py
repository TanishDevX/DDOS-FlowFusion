"""
src/utils/plotting.py
---------------------
All visualization functions for the project.
Import and call these from notebooks and evaluation scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import List, Optional

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

CLASS_NAMES = ["BENIGN", "DrDoS_LDAP", "DrDoS_MSSQL", "DrDoS_NetBIOS", "DrDoS_UDP", "Syn"]
PALETTE     = sns.color_palette("tab10", n_colors=len(CLASS_NAMES))


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm:          Confusion matrix array (N x N).
        class_names: List of class label strings.
        title:       Plot title.
        save_path:   If provided, saves figure to this path.
        normalize:   If True, normalize by row (recall per class).
    """
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── 2. ROC Curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(
    fpr_dict: dict,
    tpr_dict: dict,
    auc_dict: dict,
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-class and macro-average ROC curves.

    Args:
        fpr_dict:  {class_name: fpr_array} — include "macro" key for macro avg.
        tpr_dict:  {class_name: tpr_array}
        auc_dict:  {class_name: auc_score}
        title:     Plot title.
        save_path: If provided, saves figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("tab10", n_colors=len(fpr_dict))
    for (name, fpr), color in zip(fpr_dict.items(), colors):
        lw = 2.5 if name == "macro" else 1.5
        ls = "--" if name == "macro" else "-"
        ax.plot(
            fpr, tpr_dict[name],
            color=color, lw=lw, ls=ls,
            label=f"{name} (AUC = {auc_dict[name]:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── 3. Per-Class F1 Bar Chart ─────────────────────────────────────────────────
def plot_per_class_f1(
    results: dict,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Per-Class F1 Score Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing per-class F1 across multiple models.

    Args:
        results:     {model_name: {class_name: f1_score}}
        class_names: List of class names.
        title:       Plot title.
        save_path:   If provided, saves figure to this path.
    """
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_classes   = len(class_names)
    x           = np.arange(n_classes)
    width       = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = sns.color_palette("tab10", n_colors=n_models)

    for i, (model, color) in enumerate(zip(model_names, colors)):
        f1_scores = [results[model].get(c, 0) for c in class_names]
        offset    = (i - n_models / 2 + 0.5) * width
        bars      = ax.bar(x + offset, f1_scores, width, label=model, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7
                )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── 4. Training Curves ────────────────────────────────────────────────────────
def plot_training_curves(
    history: dict,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot loss and accuracy training curves from Keras history.

    Args:
        history:   Keras History.history dict (or equivalent).
        title:     Plot title.
        save_path: If provided, saves figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["loss"],     label="Train Loss", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss",   lw=2, ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss", fontweight="bold")
    axes[0].legend()

    # Accuracy
    acc_key = "accuracy" if "accuracy" in history else "sparse_categorical_accuracy"
    val_acc_key = f"val_{acc_key}"
    if acc_key in history:
        axes[1].plot(history[acc_key],     label="Train Acc", lw=2)
        axes[1].plot(history[val_acc_key], label="Val Acc",   lw=2, ls="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} — Accuracy", fontweight="bold")
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── 5. Inference Latency Bar Chart ────────────────────────────────────────────
def plot_latency_comparison(
    model_names: List[str],
    latencies_ms: List[float],
    title: str = "Inference Latency Comparison (per sample, CPU)",
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart comparing inference latency across models.

    Args:
        model_names:  List of model name strings.
        latencies_ms: Corresponding latency in milliseconds per sample.
        title:        Plot title.
        save_path:    If provided, saves figure to this path.
    """
    colors = sns.color_palette("viridis", n_colors=len(model_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(model_names, latencies_ms, color=colors, alpha=0.85)

    for bar, val in zip(bars, latencies_ms):
        ax.text(
            val + max(latencies_ms) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} ms",
            va="center", fontsize=9
        )

    ax.set_xlabel("Latency (ms per sample)")
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── 6. KDE Drift Plots ────────────────────────────────────────────────────────
def plot_feature_drift(
    train_series,
    test_series,
    feature_name: str,
    psi_score: float,
    save_path: Optional[str] = None,
) -> None:
    """
    KDE overlay comparing train vs. test distribution for one feature.

    Args:
        train_series:  pandas Series — training data values.
        test_series:   pandas Series — test data values.
        feature_name:  Name of the feature being plotted.
        psi_score:     PSI score (shown in title).
        save_path:     If provided, saves figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    train_series.plot.kde(ax=ax, label="Train (Jan 12)", color="#1f77b4", lw=2)
    test_series.plot.kde(ax=ax,  label="Test  (Mar 11)", color="#ff7f0e", lw=2, ls="--")

    ax.set_title(
        f"{feature_name}  |  PSI = {psi_score:.4f}",
        fontweight="bold"
    )
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
