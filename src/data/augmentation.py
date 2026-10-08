"""
src/data/augmentation.py
--------------------------
Data augmentation utilities for DDoS flow detection.

Provides two augmentation strategies targeted at the BENIGN class,
which suffers the most from cross-session distribution shift:

1. **Gaussian noise augmentation** — adds small i.i.d. Gaussian noise to
   BENIGN samples, simulating natural variation in benign traffic patterns.
   Applied only to training samples; controlled by sigma (default 0.01).

2. **MixUp augmentation** — linearly interpolates between random pairs of
   BENIGN samples (and their labels), smoothing the decision boundary.
   Applied only within BENIGN samples (class-conditional MixUp).

Both augmentations:
  - Operate on already-scaled (StandardScaler) float32 arrays.
  - Are applied ONLY to training data — never val or test.
  - Preserve the original class distribution ratios (BENIGN augmentation
    adds extra BENIGN samples on top of existing ones).

Usage:
    from src.data.augmentation import augment_benign

    X_aug, y_aug = augment_benign(
        X_train, y_train,
        benign_class_idx=0,
        gaussian_sigma=0.01,
        mixup_alpha=0.2,
        n_gaussian=5000,
        n_mixup=5000,
        seed=42,
    )
"""

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def gaussian_noise_augment(
    X_benign: np.ndarray,
    sigma: float = 0.01,
    n_samples: int = 5000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate Gaussian-noise augmented BENIGN samples.

    Each augmented sample is: x_orig + N(0, sigma^2).
    The original array X_benign must be StandardScaled (mean ~0, std ~1),
    so sigma=0.01 corresponds to 1% of feature scale — imperceptibly small
    noise that preserves class membership but increases distributional coverage.

    Args:
        X_benign:  (n, d) float32 array of BENIGN training samples (scaled).
        sigma:     Std dev of Gaussian noise. Default 0.01.
        n_samples: Number of augmented samples to generate.
        rng:       numpy random Generator. Created if None.

    Returns:
        X_aug: (n_samples, d) float32 array of augmented BENIGN samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Sample base rows (with replacement)
    indices = rng.integers(0, len(X_benign), size=n_samples)
    X_base  = X_benign[indices].astype(np.float32)

    noise   = rng.normal(0.0, sigma, size=X_base.shape).astype(np.float32)
    X_aug   = X_base + noise

    logger.info(
        f"Gaussian noise augment: {n_samples:,} BENIGN samples generated "
        f"(sigma={sigma}, base_pool={len(X_benign):,})"
    )
    return X_aug


def mixup_augment(
    X_benign: np.ndarray,
    alpha: float = 0.2,
    n_samples: int = 5000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Class-conditional MixUp for BENIGN samples.

    Each augmented sample is a convex combination of two random BENIGN samples:
        x_mix = lambda * x_i + (1 - lambda) * x_j,  lambda ~ Beta(alpha, alpha)

    Since both x_i and x_j are BENIGN, x_mix is BENIGN (y_mix = 0).
    This smooths the high-dimensional BENIGN manifold and encourages
    more robust boundaries against distribution shift.

    Args:
        X_benign:  (n, d) float32 array of BENIGN training samples.
        alpha:     Beta distribution concentration. alpha=0.2 gives mild mixing.
        n_samples: Number of MixUp samples to generate.
        rng:       numpy random Generator.

    Returns:
        X_mix: (n_samples, d) float32 array of MixUp BENIGN samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    idx_i = rng.integers(0, len(X_benign), size=n_samples)
    idx_j = rng.integers(0, len(X_benign), size=n_samples)
    lambdas = rng.beta(alpha, alpha, size=n_samples).astype(np.float32)

    X_i = X_benign[idx_i].astype(np.float32)
    X_j = X_benign[idx_j].astype(np.float32)

    # Broadcast lambda: (n_samples,) -> (n_samples, 1)
    l = lambdas[:, None]
    X_mix = l * X_i + (1.0 - l) * X_j

    logger.info(
        f"MixUp augment: {n_samples:,} BENIGN samples generated "
        f"(alpha={alpha}, base_pool={len(X_benign):,})"
    )
    return X_mix


def augment_benign(
    X_train: np.ndarray,
    y_train: np.ndarray,
    benign_class_idx: int = 0,
    gaussian_sigma: float = 0.01,
    mixup_alpha: float = 0.2,
    n_gaussian: int = 5000,
    n_mixup: int = 5000,
    seed: int = 42,
) -> tuple:
    """
    Apply both Gaussian noise and MixUp augmentation to the BENIGN class.

    The augmented samples are appended to the original training data and the
    combined dataset is shuffled. The label array y_train must use integer
    class indices.

    Args:
        X_train:          (n, d) float32 array, StandardScaled training features.
        y_train:          (n,)  int array of class labels.
        benign_class_idx: Integer label for the BENIGN class (default 0).
        gaussian_sigma:   Noise std dev for Gaussian augmentation.
        mixup_alpha:      Beta concentration for MixUp.
        n_gaussian:       Number of Gaussian-noise samples to generate.
        n_mixup:          Number of MixUp samples to generate.
        seed:             Random seed for reproducibility.

    Returns:
        X_aug: (n + n_gaussian + n_mixup, d) float32 array.
        y_aug: (n + n_gaussian + n_mixup,)  int   array.
    """
    rng = np.random.default_rng(seed)

    benign_mask = y_train == benign_class_idx
    X_benign    = X_train[benign_mask]
    n_benign    = len(X_benign)

    if n_benign == 0:
        logger.warning(
            f"No samples found for BENIGN class (idx={benign_class_idx}). "
            "Augmentation skipped."
        )
        return X_train, y_train

    logger.info(
        f"BENIGN augmentation: {n_benign:,} original BENIGN samples, "
        f"adding {n_gaussian:,} Gaussian + {n_mixup:,} MixUp samples."
    )

    aug_X_list = [X_train]
    aug_y_list = [y_train]

    if n_gaussian > 0:
        X_gauss = gaussian_noise_augment(X_benign, sigma=gaussian_sigma,
                                          n_samples=n_gaussian, rng=rng)
        y_gauss = np.full(n_gaussian, benign_class_idx, dtype=y_train.dtype)
        aug_X_list.append(X_gauss)
        aug_y_list.append(y_gauss)

    if n_mixup > 0:
        X_mix = mixup_augment(X_benign, alpha=mixup_alpha,
                               n_samples=n_mixup, rng=rng)
        y_mix = np.full(n_mixup, benign_class_idx, dtype=y_train.dtype)
        aug_X_list.append(X_mix)
        aug_y_list.append(y_mix)

    X_aug = np.concatenate(aug_X_list, axis=0).astype(np.float32)
    y_aug = np.concatenate(aug_y_list, axis=0)

    # Shuffle combined dataset
    shuffle_idx = rng.permutation(len(X_aug))
    X_aug = X_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]

    logger.info(
        f"Augmentation complete: {len(X_train):,} → {len(X_aug):,} training samples "
        f"(+{len(X_aug)-len(X_train):,} BENIGN augmented)"
    )
    return X_aug, y_aug
