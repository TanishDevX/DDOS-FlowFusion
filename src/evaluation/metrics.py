"""
src/evaluation/metrics.py
--------------------------
Centralized evaluation metrics for all experiments.

Usage:
    from src.evaluation.metrics import evaluate, format_results_table
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    label_classes: np.ndarray = None,
    split_name: str = "eval",
) -> dict:
    """
    Compute all primary metrics for a single run.

    Args:
        y_true:        Ground truth integer labels.
        y_pred:        Predicted integer labels.
        y_prob:        Predicted probabilities (N x C). Required for ROC-AUC / PR-AUC.
        label_classes: Array of class name strings (from LabelEncoder.classes_).
        split_name:    Name of the split for logging (e.g. 'val', 'test').

    Returns:
        Dict with all metric values.
    """
    results = {}
    labels = np.arange(len(label_classes)) if label_classes is not None else None

    results["accuracy"]          = accuracy_score(y_true, y_pred)
    results["macro_f1"]          = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    results["weighted_f1"]       = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    results["macro_precision"]   = precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    results["macro_recall"]      = recall_score(y_true, y_pred, labels=labels, average="macro",       zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    if label_classes is not None:
        for cls, f1 in zip(label_classes, per_class_f1):
            results[f"f1_{cls}"] = f1
    else:
        for i, f1 in enumerate(per_class_f1):
            results[f"f1_class_{i}"] = f1

    # ROC-AUC (requires probabilities)
    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(
                y_true, y_prob, labels=labels, multi_class="ovr", average="macro"
            )
        except Exception as e:
            logger.warning(f"ROC-AUC failed: {e}")
            results["roc_auc"] = float("nan")
    else:
        results["roc_auc"] = float("nan")

    # Log summary
    logger.info(f"\n[{split_name}] Accuracy={results['accuracy']:.4f}  "
                f"Macro-F1={results['macro_f1']:.4f}  "
                f"ROC-AUC={results['roc_auc']:.4f}")

    if label_classes is not None:
        logger.info(classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=label_classes,
            zero_division=0,
            digits=4,
        ))

    return results


def aggregate_seeds(results_list: list) -> dict:
    """
    Aggregate metric dicts from multiple seeds into mean ± std.

    Args:
        results_list: List of metric dicts (one per seed).

    Returns:
        Dict: {metric_name: {"mean": float, "std": float}}
    """
    all_keys = results_list[0].keys()
    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in results_list if not np.isnan(r[key])]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)),
        }
    return aggregated


def format_results_table(aggregated: dict, model_name: str) -> pd.DataFrame:
    """
    Format aggregated results as a readable DataFrame row.

    Args:
        aggregated:  Output of aggregate_seeds().
        model_name:  Name of the model (e.g. 'XGBoost').

    Returns:
        Single-row DataFrame with mean ± std columns.
    """
    primary_metrics = [
        "accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall", "roc_auc",
    ]
    row = {"model": model_name}
    for m in primary_metrics:
        if m in aggregated:
            mean = aggregated[m]["mean"]
            std  = aggregated[m]["std"]
            row[m] = f"{mean:.4f} ± {std:.4f}"
    return pd.DataFrame([row])
