"""
src/evaluation/__init__.py
"""
from src.evaluation.metrics import evaluate, aggregate_seeds, format_results_table
from src.evaluation.explainability import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    compute_integrated_gradients,
    plot_integrated_gradients,
)

__all__ = [
    "evaluate", "aggregate_seeds", "format_results_table",
    "compute_shap_values", "plot_shap_summary", "plot_shap_bar",
    "compute_integrated_gradients", "plot_integrated_gradients",
]
