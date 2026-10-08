"""
src/data/flow_preprocessor.py
------------------------------
End-to-end flow feature preprocessing pipeline.

Usage:
    from src.data.flow_preprocessor import FlowPreprocessor
    
    # Fit on training data, transform both train and test
    preprocessor = FlowPreprocessor(config)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test,  y_test  = preprocessor.transform(test_df)
    preprocessor.save("models/scaler.joblib")
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from src.data.feature_audit import audit_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FlowPreprocessor:
    """
    Preprocessing pipeline for CIC-DDoS2019 flow features.

    Pipeline (in order):
        1. Audit and drop leaky/low-quality features
        2. Filter to target classes only
        3. Replace ±inf with NaN
        4. Fill NaN with column median (fit on train only)
        5. Apply RobustScaler  (fit on train only)
        6. Encode labels to integers

    Critical: fit() must be called on training data ONLY.
    transform() is called on both train and test.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict loaded from config.yaml.
                    Uses config['data'] and config['preprocessing'] sections.
        """
        self.config      = config
        self.classes     = config["data"]["classes"]
        self.label_col   = "Label"
        self.scaler      = RobustScaler()
        self.label_enc   = LabelEncoder()
        self.medians_     = None   # Per-feature medians (fitted on train)
        self.feature_cols_= None   # Final feature columns after audit
        self._fitted      = False

    # ── Public API ─────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor on training data and return transformed arrays.

        Args:
            df: Raw DataFrame loaded from training CSVs.

        Returns:
            (X, y) — float32 feature array, int label array
        """
        logger.info("=== fit_transform (TRAINING DATA) ===")
        df = self._clean(df)

        # Fit medians on training data
        self.medians_ = df[self.feature_cols_].median()

        # Fill NaN/Inf with training medians
        df = self._fill_nan(df)

        X = df[self.feature_cols_].values.astype(np.float32)
        y = self.label_enc.fit_transform(df[self.label_col].astype(str))

        # Fit and transform scaler
        X = self.scaler.fit_transform(X).astype(np.float32)

        self._fitted = True
        logger.info(f"fit_transform complete: X={X.shape}, y={y.shape}")
        logger.info(f"Classes encoded: {dict(zip(self.label_enc.classes_, self.label_enc.transform(self.label_enc.classes_)))}")
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform test data using parameters fitted on training data.

        Args:
            df: Raw DataFrame loaded from test CSVs.

        Returns:
            (X, y) — float32 feature array, int label array
        """
        assert self._fitted, "Call fit_transform() on training data before transform()."
        logger.info("=== transform (TEST DATA) ===")
        df = self._clean(df, fit=False)
        df = self._fill_nan(df)

        X = df[self.feature_cols_].values.astype(np.float32)
        y = self.label_enc.transform(df[self.label_col].astype(str))

        X = self.scaler.transform(X).astype(np.float32)
        logger.info(f"transform complete: X={X.shape}, y={y.shape}")
        return X, y

    def save(self, path: str) -> None:
        """Save the fitted preprocessor to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> "FlowPreprocessor":
        """Load a fitted preprocessor from disk."""
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded ← {path}")
        return preprocessor

    # ── Private helpers ────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Run feature audit, filter classes, identify feature columns.
        """
        # 1. Strip column names and run audit
        df = audit_features(df, verbose=fit)

        # 2. Normalize label column name
        df.columns = df.columns.str.strip()
        if "Label" not in df.columns and " Label" in df.columns:
            df = df.rename(columns={" Label": "Label"})

        # 3. Filter to target classes
        original_len = len(df)
        df[self.label_col] = df[self.label_col].str.strip()
        df = df[df[self.label_col].isin(self.classes)].copy()
        logger.info(f"Class filter: {original_len:,} → {len(df):,} rows "
                    f"(kept classes: {self.classes})")

        # 4. Identify feature columns
        if fit:
            self.feature_cols_ = [
                c for c in df.columns
                if c != self.label_col and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            ]
            logger.info(f"Feature columns identified: {len(self.feature_cols_)}")

        return df

    def _fill_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace ±inf with NaN, then fill NaN with training medians."""
        df[self.feature_cols_] = df[self.feature_cols_].replace(
            [np.inf, -np.inf], np.nan
        )
        nan_before = df[self.feature_cols_].isna().sum().sum()
        df[self.feature_cols_] = df[self.feature_cols_].fillna(self.medians_)
        nan_after  = df[self.feature_cols_].isna().sum().sum()
        logger.info(f"NaN fill: {nan_before:,} NaN/Inf values filled → {nan_after} remaining")
        return df
