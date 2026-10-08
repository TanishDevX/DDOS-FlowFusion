"""
src/data/feature_audit.py
--------------------------
Feature leakage detection and removal.

Usage:
    from src.data.feature_audit import audit_features, leakage_check
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Features to drop — documented reasons
LEAKY_FEATURES = {
    "Unnamed: 0":      "Leftover row-index from CICFlowMeter CSV export — not a feature",
    "Flow ID":         "Unique identifier per flow — model would memorize IDs, not learn patterns",
    "Source IP":       "IP address — model learns attacker IPs, not attack behavior",
    "Destination IP":  "IP address — same concern as Source IP",
    "Source Port":     "Port number — partially encodes attack type (ephemeral vs well-known)",
    "Timestamp":       "Temporal identifier — direct temporal leakage",
    "SimillarHTTP":    "Known buggy feature in CICFlowMeter — unreliable values",
    "Inbound":         "Direction flag — encodes session topology, leaky across sessions",
}

# Also drop these if they exist (alternate column name spellings in CIC datasets)
ALTERNATE_NAMES = {
    " Flow ID",
    " Source IP",
    " Destination IP",
    " Source Port",
    " Timestamp",
    "Flow ID ",
}


def audit_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove leaky and low-quality features from a CIC-DDoS2019 DataFrame.

    Steps performed:
      1. Strip whitespace from column names
      2. Drop known leaky features (documented above)
      3. Drop zero-variance columns (single unique value)
      4. Drop columns with >50% NaN/Inf values
      5. Log a summary of every dropped column and reason

    Args:
        df:      Raw DataFrame loaded from CIC CSV.
        verbose: If True, log each dropped column with its reason.

    Returns:
        Cleaned DataFrame with leaky/low-quality features removed.
    """
    original_cols = set(df.columns)

    # ── Step 1: Strip whitespace from column names ─────────────────────────
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")

    dropped = {}  # {col_name: reason}

    # ── Step 2: Drop known leaky features ─────────────────────────────────
    for col, reason in LEAKY_FEATURES.items():
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped[col] = reason

    # ── Step 3: Drop zero-variance columns ────────────────────────────────
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique(dropna=True) <= 1:
            df = df.drop(columns=[col])
            dropped[col] = "Zero variance — single unique value, carries no information"

    # ── Step 4: Drop high-NaN/Inf columns ─────────────────────────────────
    # Replace inf with NaN first for counting purposes
    df_numeric = df.select_dtypes(include=[np.number])
    inf_mask   = np.isinf(df_numeric)
    nan_counts = df_numeric.isna().sum() + inf_mask.sum()
    nan_ratio  = nan_counts / len(df)

    high_nan_cols = nan_ratio[nan_ratio > 0.5].index.tolist()
    for col in high_nan_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped[col] = f"High NaN/Inf rate: {nan_ratio[col]:.1%} of values invalid"

    # ── Logging summary ────────────────────────────────────────────────────
    if verbose:
        logger.info(f"\n{'─'*60}")
        logger.info(f"FEATURE AUDIT SUMMARY")
        logger.info(f"{'─'*60}")
        logger.info(f"  Original columns:  {len(original_cols)}")
        logger.info(f"  Dropped columns:   {len(dropped)}")
        logger.info(f"  Remaining columns: {df.shape[1]}")
        logger.info(f"{'─'*60}")
        for col, reason in dropped.items():
            logger.info(f"  DROP [{col}]: {reason}")
        logger.info(f"{'─'*60}\n")

    return df


def leakage_check(df: pd.DataFrame, label_col: str = "Label") -> dict:
    """
    Quick leakage test: train a shallow Decision Tree (depth=3) and check
    if it achieves suspiciously high accuracy. If yes, likely leakage remains.

    Args:
        df:        DataFrame with features + label column.
        label_col: Name of the label column.

    Returns:
        Dict with {'accuracy': float, 'top_features': list, 'warning': bool}
    """
    logger.info("Running quick leakage check (Decision Tree, max_depth=3)...")

    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    y = LabelEncoder().fit_transform(df[label_col].astype(str))

    # Fill NaN/Inf for this check only
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    acc = clf.score(X, y)

    # Top features used by the tree
    importances  = clf.feature_importances_
    top_indices  = np.argsort(importances)[::-1][:5]
    top_features = [(X.columns[i], importances[i]) for i in top_indices if importances[i] > 0]

    warning = acc > 0.95
    level   = "⚠️  WARNING" if warning else "✅  OK"

    logger.info(f"  Leakage check accuracy: {acc:.4f}  {level}")
    logger.info(f"  Top features used by the 3-level tree:")
    for feat, imp in top_features:
        logger.info(f"    {feat}: importance = {imp:.4f}")

    if warning:
        logger.warning(
            "Decision Tree with depth=3 achieves >95% accuracy. "
            "Inspect the top features above — one or more may be leaking label information."
        )

    return {
        "accuracy":     acc,
        "top_features": top_features,
        "warning":      warning,
    }


def get_feature_groups() -> dict:
    """
    Return named groups of CICFlowMeter features for ablation study (E5).

    Returns:
        Dict mapping group name → list of feature name substrings to match.
    """
    return {
        "Volume": [
            "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length", "Bwd Packet Length",
            "Packet Length",
        ],
        "Rate": [
            "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
            "Fwd Packets/s", "Bwd Packets/s",
        ],
        "Flags": [
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
            "CWE Flag Count", "ECE Flag Count",
            "Fwd PSH Flags", "Bwd PSH Flags",
            "Fwd URG Flags", "Bwd URG Flags",
        ],
        "IAT": [
            "Flow IAT", "Fwd IAT", "Bwd IAT",
            "Inter Arrival Time",
        ],
        "Window": [
            "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "init_win_bytes",
        ],
    }
