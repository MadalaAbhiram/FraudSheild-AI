"""
models/predict.py
=================
Prediction utilities — load the saved model and run inference on
single transactions, DataFrames, or raw CSV files.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from preprocessing.preprocess import FraudPreprocessor

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')


class FraudPredictor:
    """Load a saved fraud-detection model and run predictions."""

    def __init__(self):
        self.model = None
        self.preprocessor: FraudPreprocessor | None = None
        self.feature_names: list[str] = []
        self.model_name: str = ''
        self._loaded = False

    # ------------------------------------------------------------------
    def load_model(self) -> bool:
        """Load the saved model, preprocessor, and metadata. Returns True on success."""
        try:
            if not os.path.exists(BEST_MODEL_PATH):
                return False

            self.model = joblib.load(BEST_MODEL_PATH)

            if os.path.exists(PREPROCESSOR_PATH):
                self.preprocessor = joblib.load(PREPROCESSOR_PATH)

            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH) as f:
                    meta = json.load(f)
                self.feature_names = meta.get('feature_names', [])
                self.model_name = meta.get('best_model_name', 'Unknown')

            self._loaded = True
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if not self._loaded:
            if not self.load_model():
                raise RuntimeError(
                    'Model not found. Please train a model first via the /train route.'
                )

    # ------------------------------------------------------------------
    def predict_single(self, transaction: dict) -> dict:
        """Predict fraud probability for a single transaction dict."""
        self._ensure_loaded()
        df = pd.DataFrame([transaction])

        # If preprocessor exists, apply feature engineering & scaling (fit=False)
        if self.preprocessor is not None:
            cleaned = self.preprocessor.clean_data(df)
            engineered = self.preprocessor.engineer_features(cleaned, df)
            encoded = self.preprocessor.encode_and_scale(engineered, fit=False)
            X, _ = self.preprocessor.get_X_y(encoded)
        else:
            X = df.select_dtypes(include=[np.number])

        aligned = self._align_features(X)
        pred = int(self.model.predict(aligned)[0])
        prob = float(self.model.predict_proba(aligned)[0][1]) if hasattr(self.model, 'predict_proba') else float(pred)

        return {
            'prediction': pred,
            'fraud_probability': round(prob * 100, 2),
            'label': 'FRAUD' if pred == 1 else 'LEGITIMATE',
            'risk_level': self._risk_level(prob),
        }

    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch prediction on a preprocessed DataFrame."""
        self._ensure_loaded()
        aligned = self._align_features(df)
        preds = self.model.predict(aligned)
        probs = (
            self.model.predict_proba(aligned)[:, 1]
            if hasattr(self.model, 'predict_proba')
            else preds.astype(float)
        )
        result = df.copy()
        result['prediction'] = preds
        result['fraud_probability'] = np.round(probs * 100, 2)
        result['label'] = ['FRAUD' if p == 1 else 'LEGITIMATE' for p in preds]
        return result

    # ------------------------------------------------------------------
    def predict_from_file(self, filepath: str) -> pd.DataFrame:
        """Full pipeline: load CSV → preprocess using saved preprocessor (fit=False) → predict."""
        self._ensure_loaded()

        proc = self.preprocessor if self.preprocessor is not None else FraudPreprocessor()
        raw_df, _ = proc.load_data(filepath)
        cleaned = proc.clean_data(raw_df)
        engineered = proc.engineer_features(cleaned, raw_df)
        encoded = proc.encode_and_scale(engineered, fit=False)

        X, y = proc.get_X_y(encoded)
        results = self.predict_batch(X)

        # Re-attach original human-readable columns if present
        for col in ['amt', 'category', 'trans_date_trans_time', 'is_fraud', 'first', 'last']:
            if col in raw_df.columns:
                results[col] = raw_df[col].values

        return results

    # ------------------------------------------------------------------
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame columns to match training feature order."""
        import re
        def _sanitise(name: str) -> str:
            return re.sub(r'[^A-Za-z0-9_]', '_', str(name))

        df_copy = df.copy()
        df_copy.columns = [_sanitise(c) for c in df_copy.columns]

        for col in self.feature_names:
            if col not in df_copy.columns:
                df_copy[col] = 0.0

        # Replace infs/nans
        df_copy = df_copy[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df_copy

    # ------------------------------------------------------------------
    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.30:
            return 'LOW'
        elif prob < 0.60:
            return 'MEDIUM'
        elif prob < 0.80:
            return 'HIGH'
        else:
            return 'CRITICAL'
