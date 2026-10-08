"""
src/data/sampler.py
--------------------
Stratified sampling utilities.

Usage:
    from src.data.sampler import stratified_sample, load_and_combine_csvs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map CSV filenames to canonical class label strings.
# NOTE: 01-12 uses "DrDoS_" prefix; 03-11 does NOT — handled at load time.
# The label column values inside each CSV are the ground truth.
FILENAME_TO_LABEL = {
    # January 12 filenames (train)
    "DrDoS_DNS.csv":     "DrDoS_DNS",
    "DrDoS_LDAP.csv":    "DrDoS_LDAP",
    "DrDoS_MSSQL.csv":   "DrDoS_MSSQL",
    "DrDoS_NTP.csv":     "DrDoS_NTP",
    "DrDoS_NetBIOS.csv": "DrDoS_NetBIOS",
    "DrDoS_SNMP.csv":    "DrDoS_SNMP",
    "DrDoS_SSDP.csv":    "DrDoS_SSDP",
    "DrDoS_UDP.csv":     "DrDoS_UDP",
    "Syn.csv":           "Syn",
    "TFTP.csv":          "TFTP",
    "UDPLag.csv":        "UDPLag",
    # March 11 filenames (test) — no DrDoS_ prefix
    "LDAP.csv":          "LDAP",
    "MSSQL.csv":         "MSSQL",
    "NetBIOS.csv":       "NetBIOS",
    "UDP.csv":           "UDP",
    "Portmap.csv":       "Portmap",
}


def load_and_combine_csvs(
    data_dir: str,
    target_classes: List[str],
    label_col: str = "Label",
) -> pd.DataFrame:
    """
    Load all CSVs in data_dir, filter to target_classes, and combine.

    Args:
        data_dir:        Directory containing CIC-DDoS2019 CSV files.
        target_classes:  List of class label strings to keep.
        label_col:       Name of the label column.

    Returns:
        Combined DataFrame with all target-class rows.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for csv_path in csv_files:
        logger.info(f"Loading: {csv_path.name} ...")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = df.columns.str.strip()

            # Normalize label column
            if label_col not in df.columns:
                alt = f" {label_col}"
                if alt in df.columns:
                    df = df.rename(columns={alt: label_col})
                else:
                    logger.warning(f"  No label column found in {csv_path.name} — skipping")
                    continue

            df[label_col] = df[label_col].str.strip()
            rows_before = len(df)
            df = df[df[label_col].isin(target_classes)]
            logger.info(f"  {rows_before:,} rows → {len(df):,} rows after class filter")

            if len(df) > 0:
                dfs.append(df)

        except Exception as e:
            logger.error(f"  Failed to load {csv_path.name}: {e}")
            continue

    if not dfs:
        raise ValueError(f"No data found for classes {target_classes} in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nCombined: {len(combined):,} rows across {len(dfs)} files")
    _log_class_distribution(combined, label_col)
    return combined


def stratified_sample(
    df: pd.DataFrame,
    n_per_class: int,
    label_col: str = "Label",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample exactly n_per_class rows from each class.
    If a class has fewer than n_per_class rows, take ALL rows and log a warning.

    Args:
        df:           DataFrame to sample from.
        n_per_class:  Target number of samples per class.
        label_col:    Name of the label column.
        seed:         Random seed for reproducibility.

    Returns:
        Sampled and shuffled DataFrame (balanced).
    """
    logger.info(f"\nStratified sampling: {n_per_class:,} per class (seed={seed})")
    sampled_dfs = []

    for cls in df[label_col].unique():
        cls_df = df[df[label_col] == cls]
        available = len(cls_df)

        if available >= n_per_class:
            sample = cls_df.sample(n=n_per_class, random_state=seed)
            logger.info(f"  [{cls}]: sampled {n_per_class:,} from {available:,} available")
        else:
            sample = cls_df
            logger.warning(
                f"  [{cls}]: only {available:,} samples available — using ALL "
                f"(shortfall: {n_per_class - available:,})"
            )

        sampled_dfs.append(sample)

    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"\nSampling complete: {len(result):,} total rows")
    _log_class_distribution(result, label_col)
    return result


def _log_class_distribution(df: pd.DataFrame, label_col: str) -> None:
    """Log class distribution as counts and percentages."""
    counts = df[label_col].value_counts()
    total  = len(df)
    logger.info("Class distribution:")
    for cls, count in counts.items():
        logger.info(f"  {cls:<25} {count:>8,}  ({count/total*100:5.1f}%)")
