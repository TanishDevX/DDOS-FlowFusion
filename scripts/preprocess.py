"""
scripts/preprocess.py
---------------------
Day 3 — End-to-end preprocessing pipeline.

What this script does:
  1. Load config from config/config.yaml
  2. TRAIN  (01-12): Read each CSV, pool BENIGN rows from all files,
                     sample 25k per class, run feature audit, fit
                     RobustScaler + LabelEncoder on train data only.
  3. TEST   (03-11): Read each CSV, remap test labels → train labels,
                     transform using the FITTED preprocessor (no re-fitting).
  4. Save processed arrays to data/processed/{train,test}/:
       X.npy, y.npy, label_classes.npy, feature_names.npy
  5. Save fitted preprocessor to models/preprocessor.joblib

Run from the project root:
    python scripts/preprocess.py

Expected runtime: 5-15 minutes (reading ~20 GB of CSV data)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Make src/ importable ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.flow_preprocessor import FlowPreprocessor
from src.data.feature_audit import leakage_check

logger = get_logger("preprocess")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# CSV Loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_label_col(csv_path: Path) -> str:
    """Return the exact label column name used in this CSV (may have leading space)."""
    sample = pd.read_csv(csv_path, nrows=2)
    for col in sample.columns:
        if col.strip().lower() == "label":
            return col
    raise ValueError(f"No label column found in {csv_path.name}. "
                     f"Columns: {list(sample.columns[:8])}")


def load_train_data(
    train_dir: Path,
    target_classes: list,
    n_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """
    Load training CSVs and return a balanced, stratified-sampled DataFrame.

    Strategy:
      - Attack classes: each class lives mainly in one file → sample n_per_class
        rows directly from that file (fast, avoids loading full file).
      - BENIGN class: tiny minority in every file → pool ALL BENIGN rows from
        every CSV, then sample n_per_class from the pool.

    Returns: combined DataFrame with exactly n_per_class rows per class.
    """
    logger.info("=" * 60)
    logger.info("LOADING TRAIN DATA  (01-12)")
    logger.info("=" * 60)

    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")

    attack_classes = [c for c in target_classes if c != "BENIGN"]
    sampled_parts  = []   # one df per class
    benign_pool    = []   # BENIGN rows pooled from all files

    # ── Map canonical class name → file ───────────────────────────────────
    # We'll scan all files and accumulate by class name found in Label column
    class_dfs = {cls: [] for cls in attack_classes}

    for csv_path in csv_files:
        logger.info(f"\nScanning: {csv_path.name}")
        try:
            label_col = _read_label_col(csv_path)

            # Read only the label column first to see what's inside
            df_labels = pd.read_csv(
                csv_path, usecols=[label_col], low_memory=False
            )
            df_labels.columns = df_labels.columns.str.strip()
            unique_labels = df_labels["Label"].str.strip().unique()
            logger.info(f"  Labels found: {list(unique_labels)}")

            # Check if any of our target classes appear in this file
            relevant_labels = [
                lbl for lbl in unique_labels
                if lbl in target_classes
            ]
            if not relevant_labels:
                logger.info(f"  -> No target classes here - skipping full load")
                continue

            # Now load the full file (only if we need it)
            logger.info(f"  -> Loading full file ...")
            t0  = time.time()
            df  = pd.read_csv(csv_path, low_memory=False)
            df.columns = df.columns.str.strip()
            df["Label"] = df["Label"].str.strip()
            logger.info(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

            # Collect BENIGN rows
            benign_rows = df[df["Label"] == "BENIGN"]
            if len(benign_rows) > 0:
                benign_pool.append(benign_rows)
                logger.info(f"  BENIGN: collected {len(benign_rows):,} rows")

            # Collect attack class rows
            for cls in attack_classes:
                rows = df[df["Label"] == cls]
                if len(rows) > 0:
                    class_dfs[cls].append(rows)
                    logger.info(f"  {cls}: collected {len(rows):,} rows")

        except MemoryError:
            logger.error(f"  OOM reading {csv_path.name} — skipping")
            continue
        except Exception as e:
            logger.error(f"  Failed {csv_path.name}: {e}")
            continue

    # ── Sample each attack class ───────────────────────────────────────────
    for cls in attack_classes:
        rows_list = class_dfs[cls]
        if not rows_list:
            logger.warning(f"WARNING: No rows found for class '{cls}' — SKIPPING")
            continue

        combined = pd.concat(rows_list, ignore_index=True)
        available = len(combined)

        if available >= n_per_class:
            sample = combined.sample(n=n_per_class, random_state=seed)
            logger.info(f"[{cls}] Sampled {n_per_class:,} / {available:,} rows")
        else:
            sample = combined
            logger.warning(
                f"[{cls}] Only {available:,} rows available "
                f"(target={n_per_class:,}) — using ALL"
            )
        sampled_parts.append(sample)

    # ── Sample BENIGN pool ─────────────────────────────────────────────────
    if not benign_pool:
        raise ValueError("No BENIGN rows found in any training CSV!")

    benign_combined = pd.concat(benign_pool, ignore_index=True)
    benign_available = len(benign_combined)

    if benign_available >= n_per_class:
        benign_sample = benign_combined.sample(n=n_per_class, random_state=seed)
        logger.info(f"[BENIGN] Sampled {n_per_class:,} / {benign_available:,} rows")
    else:
        benign_sample = benign_combined
        logger.warning(
            f"[BENIGN] Only {benign_available:,} rows available "
            f"(target={n_per_class:,}) — using ALL"
        )
    sampled_parts.append(benign_sample)

    # ── Combine and shuffle ────────────────────────────────────────────────
    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"\nTrain combined: {len(result):,} rows total")
    _log_class_dist(result, "Label")
    return result


def process_test_data(
    test_dir: Path,
    label_map: dict,
    target_classes: list,
    preprocessor,
    seed: int,
) -> tuple:
    """
    Load, remap, and transform test CSVs **one file at a time**.

    Each CSV is:
      1. Loaded into RAM
      2. Labels remapped (LDAP -> DrDoS_LDAP, etc.)
      3. Rows outside target_classes dropped
      4. Transformed via the FITTED preprocessor
      5. DataFrame freed from memory

    Resulting numpy arrays are concatenated and shuffled.
    Peak RAM = one file at a time (not all 20M rows at once).
    """
    logger.info("=" * 60)
    logger.info("LOADING + TRANSFORMING TEST DATA  (03-11)")
    logger.info("=" * 60)

    csv_files = sorted(test_dir.glob("*.csv"))
    X_parts = []
    y_parts = []

    for csv_path in csv_files:
        logger.info(f"\nLoading: {csv_path.name}")
        try:
            label_col = _read_label_col(csv_path)
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = df.columns.str.strip()
            df["Label"] = df["Label"].str.strip()

            before = len(df)
            # Remap test labels -> canonical train labels
            df["Label"] = df["Label"].map(label_map)
            df = df.dropna(subset=["Label"])
            df = df[df["Label"].isin(target_classes)]
            logger.info(f"  {before:,} rows -> {len(df):,} after label remap + filter")

            if len(df) == 0:
                logger.info(f"  No target-class rows in {csv_path.name} - skipping")
                continue

            _log_class_dist(df, "Label", indent=4)

            # Transform using TRAIN-fitted preprocessor (no re-fitting!)
            X_file, y_file = preprocessor.transform(df)
            X_parts.append(X_file)
            y_parts.append(y_file)
            logger.info(f"  Transformed: X{X_file.shape}")

            # Explicitly free the DataFrame to recover RAM
            del df
            import gc; gc.collect()

        except MemoryError:
            logger.error(f"  OOM reading {csv_path.name} - skipping")
            continue
        except Exception as e:
            logger.error(f"  Failed {csv_path.name}: {e}")
            continue

    if not X_parts:
        raise ValueError("No test data loaded - check label_map and test directory")

    # Concatenate arrays (much cheaper than concatenating DataFrames)
    X_test = np.concatenate(X_parts, axis=0)
    y_test = np.concatenate(y_parts, axis=0)

    # Shuffle arrays cheaply on numpy
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_test))
    X_test, y_test = X_test[idx], y_test[idx]

    logger.info(f"\nTest combined: X{X_test.shape}  y{y_test.shape}")
    return X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
# Save helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_split(out_dir: Path, X: np.ndarray, y: np.ndarray,
               feature_names: list, label_classes: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "feature_names.npy", np.array(feature_names))
    np.save(out_dir / "label_classes.npy", label_classes)
    logger.info(f"Saved to {out_dir}/  ->  X{X.shape}  y{y.shape}")


def _log_class_dist(df: pd.DataFrame, label_col: str, indent: int = 2) -> None:
    pad = " " * indent
    counts = df[label_col].value_counts()
    total = len(df)
    for cls, cnt in counts.items():
        logger.info(f"{pad}{cls:<25} {cnt:>8,}  ({cnt/total*100:5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    cfg = load_config()
    seed = cfg["data"]["random_state"]
    set_seed(seed)

    # ── Paths ──────────────────────────────────────────────────────────────
    train_dir     = ROOT / cfg["data"]["train_dir"]
    test_dir      = ROOT / cfg["data"]["test_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    models_dir    = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    target_classes  = cfg["data"]["classes"]
    n_per_class     = cfg["data"]["samples_per_class"]
    test_label_map  = cfg["data"]["test_label_map"]

    logger.info(f"Target classes: {target_classes}")
    logger.info(f"Samples per class: {n_per_class:,}")

    # ── Step 1: Load raw train data ────────────────────────────────────────
    train_df = load_train_data(train_dir, target_classes, n_per_class, seed)

    # ── Step 2: Quick leakage sanity check on raw (sampled) data ──────────
    logger.info("\nRunning leakage check on raw training sample …")
    leakage_result = leakage_check(train_df, label_col="Label")
    if leakage_result["warning"]:
        logger.warning("Leakage check triggered — inspect top features above!")
    else:
        logger.info("Leakage check OK — no obvious leakage detected.")

    # ── Step 3: Fit preprocessor on TRAIN, then transform ─────────────────
    preprocessor = FlowPreprocessor(cfg)
    X_train, y_train = preprocessor.fit_transform(train_df)

    # Free train DataFrame — no longer needed
    del train_df
    import gc; gc.collect()

    # ── Step 4 & 5: Load + transform TEST one file at a time (memory safe) ─
    X_test, y_test = process_test_data(
        test_dir, test_label_map, target_classes, preprocessor, seed
    )

    # ── Step 6: Save processed arrays ─────────────────────────────────────
    label_classes = preprocessor.label_enc.classes_
    feature_names = preprocessor.feature_cols_

    save_split(processed_dir / "train", X_train, y_train, feature_names, label_classes)
    save_split(processed_dir / "test",  X_test,  y_test,  feature_names, label_classes)

    # ── Step 7: Save fitted preprocessor ──────────────────────────────────
    preprocessor_path = models_dir / "preprocessor.joblib"
    preprocessor.save(str(preprocessor_path))

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Train:  X{X_train.shape}  y{y_train.shape}")
    logger.info(f"  Test:   X{X_test.shape}   y{y_test.shape}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Classes:  {list(zip(label_classes, range(len(label_classes))))}")
    logger.info(f"  Preprocessor -> {preprocessor_path}")
    logger.info(f"  Total time: {elapsed/60:.1f} min")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
