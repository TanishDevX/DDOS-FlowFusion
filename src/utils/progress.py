"""
src/utils/progress.py
---------------------
Central training progress, callbacks, and visualization utilities.
Provides clean terminal displays, custom Keras callbacks for training schedules,
and feature importance printing for tree-based models.
"""

import sys
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Force UTF-8 stdout if possible on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Keras Callbacks
# ═══════════════════════════════════════════════════════════════════════════

class CosineDecayCallback(keras.callbacks.Callback):
    """Cosine annealing learning-rate schedule with a linear warm-up phase."""

    def __init__(
        self,
        base_lr: float,
        total_epochs: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.min_lr = float(min_lr)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        self.model.optimizer.learning_rate = lr


class NaNDetectionCallback(keras.callbacks.Callback):
    """Stop training if loss goes NaN/Inf or explodes."""

    def __init__(self, threshold: float = 1e6):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)
        if np.isnan(loss) or np.isinf(loss) or loss > self.threshold:
            logger.warning(
                f"  [!] Epoch {epoch+1}: loss={loss:.4f} - divergence detected. Stopping training."
            )
            self.model.stop_training = True


class CosineGateTemperatureCallback(keras.callbacks.Callback):
    """Anneals gate temperature t from start_temp -> end_temp over total_epochs."""

    def __init__(
        self,
        total_epochs: int,
        start_temp: float = 1.0,
        end_temp: float = 0.3,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.current_temp = start_temp

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        progress = epoch / max(1, self.total_epochs - 1)
        self.current_temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (
            1 + math.cos(math.pi * progress)
        )


class GateMonitorCallback(keras.callbacks.Callback):
    """Logs FlowFusion gate values per epoch for interpretability tracking."""

    def __init__(self, val_data=None):
        super().__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        gate_info = []
        for k, v in logs.items():
            if k.startswith("gate_") or "gate" in k:
                gate_info.append(f"{k}={v:.3f}")
        if gate_info:
            logger.info(f"    [Gate State Epoch {epoch+1:02d}] " + "  ".join(gate_info))


class MacroF1ValidationCallback(keras.callbacks.Callback):
    """Calculates macro-F1 score on validation set at end of each epoch."""

    def __init__(self, val_x: np.ndarray, val_y: np.ndarray, num_classes: int):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y
        self.num_classes = num_classes

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        val_pred = self.model.predict(self.val_x, batch_size=8192, verbose=0)
        if isinstance(val_pred, dict):
            val_pred = val_pred["output"]
        elif isinstance(val_pred, (list, tuple)):
            val_pred = val_pred[0]
        y_pred = np.argmax(val_pred, axis=1)
        macro_f1 = f1_score(self.val_y, y_pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = macro_f1


class RichTrainingCallback(keras.callbacks.Callback):
    """Terminal progress logger displaying epoch progress, metrics, and seed banners using ASCII."""

    def __init__(self, model_name: str, seed: int, seed_idx: int, total_seeds: int, total_epochs: int):
        super().__init__()
        self.model_name = model_name
        self.seed = seed
        self.seed_idx = seed_idx
        self.total_seeds = total_seeds
        self.total_epochs = total_epochs
        self.epoch_start_time = 0.0

    def on_train_begin(self, logs: Optional[dict] = None):
        border = "=" * 60
        banner = (
            f"\n{border}\n"
            f"  {self.model_name:<18} | Seed {self.seed_idx}/{self.total_seeds} (seed={self.seed:<4})\n"
            f"{border}"
        )
        print(banner, flush=True)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time
        loss = logs.get("loss", logs.get("output_loss", 0.0))
        val_loss = logs.get("val_loss", logs.get("val_output_loss", 0.0))
        val_acc = logs.get("val_accuracy", logs.get("val_output_accuracy", 0.0))
        macro_f1 = logs.get("val_macro_f1", 0.0)
        
        lr = float(self.model.optimizer.learning_rate)

        metrics_str = (
            f"  Epoch {epoch+1:02d}/{self.total_epochs:02d} [{elapsed:.1f}s] | "
            f"loss={loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc*100:.2f}%  macro_F1={macro_f1:.4f}  "
            f"LR={lr:.1e}"
        )
        print(metrics_str, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Importance Printing Helper
# ═══════════════════════════════════════════════════════════════════════════

def print_feature_importance(model, feature_names: List[str], top_n: int = 20):
    """Prints top N feature importances for tree-based models (ASCII box)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "booster_"):
        importances = model.booster_.feature_importance(importance_type="gain")
    else:
        return

    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {min(top_n, len(indices))} Feature Importances:")
    print(f"  +------------------------------------------+--------------+")
    print(f"  | {'Feature':<40} | {'Importance':<12} |")
    print(f"  +------------------------------------------+--------------+")
    for idx in indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        val = importances[idx]
        print(f"  | {name[:40]:<40} | {val:<12.4f} |")
    print(f"  +------------------------------------------+--------------+\n", flush=True)
