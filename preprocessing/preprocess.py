"""
preprocessing/preprocess.py
============================
Optimised data loading, cleaning, feature engineering, encoding, PCA, and splitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
PII_COLUMNS = [
    'first', 'last', 'street', 'trans_num', 'cc_num', 'dob',
    'trans_date_trans_time', 'merchant', 'city', 'zip',
    'unix_time', 'city_pop', 'Unnamed: 0', 'index', 'id',
]

CATEGORICAL_COLUMNS = ['category', 'gender', 'state', 'job']
NUMERIC_COLUMNS     = [
    'amt', 'lat', 'long', 'merch_lat', 'merch_long',
    'hour', 'day_of_week', 'month', 'age', 'distance', 'amt_log',
]
TARGET_COLUMN = 'is_fraud'

# Explicit dtype map — skips Pandas type-inference on 22 columns
_DTYPE_MAP = {
    'amt':        'float32',
    'lat':        'float32',
    'long':       'float32',
    'merch_lat':  'float32',
    'merch_long': 'float32',
    'zip':        'str',
    'city_pop':   'int32',
    'is_fraud':   'int8',
    'unix_time':  'int64',
}


class FraudPreprocessor:
    """End-to-end preprocessing pipeline — optimised for large datasets."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler       = StandardScaler()
        self.pca          = None
        self.pca_info     = {}
        self.feature_names: list[str] = []
        self.is_fitted    = False

    # ------------------------------------------------------------------
    # 1. Load — fast dtype-declared read
    # ------------------------------------------------------------------
    def load_data(
        self,
        filepath: str,
        max_rows: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Load CSV with explicit dtypes.
        If max_rows is set, only that many rows are read (for EDA preview).
        """
        df = pd.read_csv(
            filepath,
            dtype=_DTYPE_MAP,
            low_memory=False,
            nrows=max_rows,
        )

        total_rows   = self._count_csv_rows(filepath) if max_rows else len(df)
        analysis_rows = len(df)
        is_sampled   = bool(max_rows and total_rows > analysis_rows)

        info = {
            'rows':          total_rows,
            'total_rows':    total_rows,
            'analysis_rows': analysis_rows,
            'is_sampled':    is_sampled,
            'columns':       len(df.columns),
            'column_names':  list(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'memory_mb':     round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
            'has_target':    TARGET_COLUMN in df.columns,
            'fraud_count':   int(df[TARGET_COLUMN].sum())         if TARGET_COLUMN in df.columns else None,
            'legit_count':   int((df[TARGET_COLUMN] == 0).sum())  if TARGET_COLUMN in df.columns else None,
        }
        return df, info

    @staticmethod
    def _count_csv_rows(filepath: str) -> int:
        """Count newlines via 1 MB chunks."""
        count = 0
        with open(filepath, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                count += chunk.count(b'\n')
        return max(count - 1, 0)

    # ------------------------------------------------------------------
    # 2. Clean — vectorised, single-pass
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop PII and index columns
        df.drop(columns=[c for c in PII_COLUMNS if c in df.columns], inplace=True, errors='ignore')
        df.dropna(axis=1, how='all', inplace=True)

        # Vectorised numeric fill
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            medians = df[num_cols].median()
            df[num_cols] = df[num_cols].fillna(medians).fillna(0)

        # Vectorised categorical fill
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df[col].fillna(fill_val, inplace=True)

        return df

    # ------------------------------------------------------------------
    # 3. Feature Engineering — fully vectorised
    # ------------------------------------------------------------------
    def engineer_features(
        self,
        df: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df  = df.copy()
        src = raw_df if raw_df is not None else df

        # ---- Temporal --------------------------------------------------
        if 'trans_date_trans_time' in src.columns:
            dt = pd.to_datetime(src['trans_date_trans_time'], errors='coerce')
            df['hour']        = dt.dt.hour.fillna(0).astype('int8')
            df['day_of_week'] = dt.dt.dayofweek.fillna(0).astype('int8')
            df['month']       = dt.dt.month.fillna(1).astype('int8')
        else:
            for col in ('hour', 'day_of_week', 'month'):
                if col not in df.columns:
                    df[col] = np.int8(0)

        # ---- Age -------------------------------------------------------
        if 'dob' in src.columns:
            ref = pd.Timestamp('today')
            dob = pd.to_datetime(src['dob'], errors='coerce')
            df['age'] = ((ref - dob).dt.days / 365.25).fillna(35).astype('int16')
        elif 'age' not in df.columns:
            df['age'] = np.int16(35)

        # ---- Haversine distance ----------------------------------------
        lat_cols = ('lat', 'long', 'merch_lat', 'merch_long')
        src_for_dist = df if all(c in df.columns for c in lat_cols) else src
        if all(c in src_for_dist.columns for c in lat_cols):
            df['distance'] = self._haversine(
                src_for_dist['lat'].to_numpy(dtype='float32'),
                src_for_dist['long'].to_numpy(dtype='float32'),
                src_for_dist['merch_lat'].to_numpy(dtype='float32'),
                src_for_dist['merch_long'].to_numpy(dtype='float32'),
            )
        else:
            df['distance'] = np.float32(0.0)

        # ---- Log-amount ------------------------------------------------
        if 'amt' in df.columns:
            df['amt_log'] = np.log1p(np.maximum(df['amt'].to_numpy(dtype='float32'), 0))
        else:
            df['amt_log'] = np.float32(0.0)

        return df

    @staticmethod
    def _haversine(
        lat1: np.ndarray, lon1: np.ndarray,
        lat2: np.ndarray, lon2: np.ndarray,
    ) -> np.ndarray:
        R     = np.float32(6371.0)
        dlat  = np.radians(lat2 - lat1)
        dlon  = np.radians(lon2 - lon1)
        rlat1 = np.radians(lat1)
        rlat2 = np.radians(lat2)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(np.maximum(a, 0)), np.sqrt(np.maximum(1 - a, 0)))

    # ------------------------------------------------------------------
    # 4. Encode & Scale
    # ------------------------------------------------------------------
    def encode_and_scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()

        # Label-encode categoricals
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            col_str = df[col].astype(str)
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(col_str)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    known = set(le.classes_)
                    fallback = le.classes_[0]
                    col_mapped = col_str.where(col_str.isin(known), fallback)
                    df[col] = le.transform(col_mapped)
                else:
                    df[col] = 0

        # Standard scale numerics
        num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
        if num_cols:
            if fit:
                df[num_cols] = self.scaler.fit_transform(df[num_cols].astype('float32'))
                self.is_fitted = True
            else:
                df[num_cols] = self.scaler.transform(df[num_cols].astype('float32'))

        return df

    # ------------------------------------------------------------------
    # PCA Feature Identification & Dimensionality Reduction
    # ------------------------------------------------------------------
    def perform_pca(self, X: pd.DataFrame, n_components: float | int = 0.95) -> tuple[np.ndarray, dict]:
        """
        Principal Component Analysis (PCA) for feature identification & dimensionality reduction.
        Retains variance threshold (e.g., 95%) and identifies top contributing features per PC.
        """
        if X.empty:
            return np.array([]), {}

        # Ensure no residual NaNs or Infs
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_clean)
        self.pca = pca

        exp_var = pca.explained_variance_ratio_.tolist()
        cum_var = np.cumsum(pca.explained_variance_ratio_).tolist()

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=X.columns
        )

        top_features_per_pc = {}
        for col in loadings.columns:
            top_feats = loadings[col].abs().sort_values(ascending=False).head(5)
            top_features_per_pc[col] = [
                {'feature': feat, 'weight': round(float(loadings.loc[feat, col]), 4)}
                for feat in top_feats.index
            ]

        self.pca_info = {
            'n_components': int(pca.n_components_),
            'explained_variance_ratio': [round(v, 4) for v in exp_var],
            'cumulative_variance_ratio': [round(v, 4) for v in cum_var],
            'total_variance_explained': round(float(sum(exp_var)) * 100, 2),
            'top_features_per_pc': top_features_per_pc,
            'components_summary': [
                {
                    'pc': f'PC{i+1}',
                    'explained_var_pct': round(exp_var[i] * 100, 2),
                    'cum_var_pct': round(cum_var[i] * 100, 2),
                    'top_feature': top_features_per_pc[f'PC{i+1}'][0]['feature'] if top_features_per_pc.get(f'PC{i+1}') else 'N/A'
                }
                for i in range(len(exp_var))
            ]
        }
        return X_pca, self.pca_info

    # ------------------------------------------------------------------
    # 5. Split X / y
    # ------------------------------------------------------------------
    def get_X_y(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN].copy()
            X = df.drop(columns=[TARGET_COLUMN])
        else:
            y = None
            X = df.copy()

        # Drop any remaining non-numeric or index columns
        X = X.select_dtypes(include=[np.number])
        X = X.drop(columns=[c for c in PII_COLUMNS if c in X.columns], errors='ignore')

        self.feature_names = list(X.columns)
        return X, y

    # ------------------------------------------------------------------
    # 6. Stratified split
    # ------------------------------------------------------------------
    @staticmethod
    def split_data(
        X: pd.DataFrame, y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # ------------------------------------------------------------------
    # Full pipeline convenience method
    # ------------------------------------------------------------------
    def full_pipeline(self, filepath: str) -> dict:
        """Run the complete preprocessing pipeline on a CSV file."""
        raw_df, info  = self.load_data(filepath)
        cleaned       = self.clean_data(raw_df)
        engineered    = self.engineer_features(cleaned, raw_df)
        encoded       = self.encode_and_scale(engineered, fit=True)
        X, y          = self.get_X_y(encoded)

        # Perform PCA feature identification
        _, pca_info   = self.perform_pca(X, n_components=0.95)

        if y is not None:
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'feature_names': self.feature_names,
                'pca_info': pca_info,
                'info': info,
            }
        return {
            'X': X,
            'feature_names': self.feature_names,
            'pca_info': pca_info,
            'info': info,
        }
