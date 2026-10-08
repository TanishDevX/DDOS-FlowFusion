"""
src/utils/gpu.py
-----------------
GPU setup helpers for TensorFlow, XGBoost, and LightGBM.

Compatible with:
  - Lightning AI T4 GPU Studio (CUDA 12.x, Python 3.10)
  - Local CPU-only environments (auto-fallback)

Call `setup_gpu()` once at the start of any TF training script.
Call `get_xgb_device()` / `get_lgbm_device()` to get the right
device string for tree-based models.
"""

import os
import subprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA detection helper
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_is_available() -> bool:
    """Return True if at least one CUDA-capable GPU is visible to the system."""
    # 1. Try nvidia-smi (fastest, works on all CUDA systems)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = result.stdout.strip().split("\n")
            logger.info(f"[GPU] CUDA GPUs detected via nvidia-smi: {gpu_names}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. Fall back to TensorFlow's device list (if TF is importable)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return True
    except Exception:
        pass

    return False


# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow GPU setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu(mixed_precision: bool = True) -> None:
    """
    Configure TensorFlow for GPU training.

    - Enables memory growth on all GPUs (avoids pre-allocating all VRAM).
    - Enables mixed precision (float16) for ~2× speedup on T4 / Turing GPUs.
    - Falls back gracefully to CPU if no GPU is detected.
    - Safe to call multiple times (no-op after first call).

    Args:
        mixed_precision: If True, enable float16 mixed precision. Default True.
    """
    try:
        import tensorflow as tf

        # ── Memory growth ──────────────────────────────────────────────────
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    # Memory growth must be set before GPUs have been initialized.
                    pass
            logger.info(f"[TF GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
        else:
            logger.warning("[TF GPU] No GPU detected by TensorFlow — running on CPU.")

        # ── Mixed precision (fp16) ─────────────────────────────────────────
        # T4 (Turing) has Tensor Cores — fp16 gives 2–4× throughput boost.
        if mixed_precision and gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("[TF GPU] Mixed precision enabled: float16 compute / float32 weights")
        else:
            logger.info("[TF GPU] Mixed precision NOT enabled (no GPU or disabled).")

    except ImportError:
        logger.warning("[TF GPU] TensorFlow not found — skipping GPU setup.")


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost device selector
# ─────────────────────────────────────────────────────────────────────────────

def get_xgb_device() -> str:
    """
    Return the XGBoost `device` string for the current hardware.

    Returns:
        "cuda"  — if a CUDA GPU is available (Lightning AI T4, local GPU).
        "cpu"   — fallback for CPU-only environments.

    XGBoost >= 1.7 uses `device="cuda"` with `tree_method="hist"` for GPU.
    """
    if _cuda_is_available():
        logger.info("[XGB] CUDA GPU detected — using device='cuda'")
        return "cuda"
    logger.info("[XGB] No CUDA GPU — using device='cpu'")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM device selector
# ─────────────────────────────────────────────────────────────────────────────

def get_lgbm_device() -> str:
    """
    Return the LightGBM `device` string for the current hardware.
    On this environment, LightGBM is not compiled with CUDA support.
    Always use CPU for LightGBM to avoid crash.
    """
    logger.info("[LGBM] CUDA build not found — using device='cpu'")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience summary
# ─────────────────────────────────────────────────────────────────────────────

def log_hardware_summary() -> None:
    """Print a concise hardware summary to the logger. Call once at script start."""
    cuda = _cuda_is_available()
    logger.info("=" * 50)
    logger.info(f"  CUDA available : {cuda}")
    logger.info(f"  XGBoost device : {'cuda' if cuda else 'cpu'}")
    logger.info(f"  LightGBM device: {'cuda' if cuda else 'cpu'}")
    logger.info(f"  TF mixed prec  : {'enabled (fp16)' if cuda else 'disabled'}")
    logger.info("=" * 50)
