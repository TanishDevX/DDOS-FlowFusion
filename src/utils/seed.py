"""
src/utils/seed.py
-----------------
Seed everything for full reproducibility.
Call set_seed(seed) at the start of every script and notebook.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow (import only if available)
    # NOTE: TF_DETERMINISTIC_OPS=1 is intentionally NOT set here — it blocks
    # GPU-optimised cuDNN kernels and can halve throughput on CUDA devices.
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # PyTorch (import only if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[seed] All random seeds set to {seed}")
