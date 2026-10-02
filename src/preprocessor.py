import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    DROP_COLUMNS, METADATA_COLUMNS, ALL_FEATURES,
    MODELS_DIR
)

class IDSPreprocessor:
    """
    Leak-free Preprocessor for NSL-KDD Network Intrusion Detection System.
    - Fits OneHotEncoder strictly on training split data.
    - Handles unseen test-time categorical categories cleanly via handle_unknown='ignore'.
    - Leaves numerical features unscaled so XGBoost preserves natural split thresholds.
    - Drops zero-variance feature (num_outbound_cmds) and benchmark metadata (difficulty_level).
    """
    def __init__(self):
        self.categorical_features = CATEGORICAL_FEATURES
        self.numerical_features = NUMERICAL_FEATURES
        self.drop_columns = DROP_COLUMNS + METADATA_COLUMNS
        self.column_transformer = None
        self.feature_names_out_ = None
        self.categories_ = {}

    def fit(self, X):
        """
        Fits the preprocessor strictly on the training data.
        """
        print(f"Fitting preprocessor on {len(X)} training samples...")
        # Save categories present during training
        for col in self.categorical_features:
            self.categories_[col] = sorted(list(X[col].dropna().unique()))

        ohe = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('cat', ohe, self.categorical_features),
                ('num', 'passthrough', self.numerical_features)
            ],
            remainder='drop'
        )

        self.column_transformer.fit(X)
        
        # Build human-readable output feature names
        cat_encoder = self.column_transformer.named_transformers_['cat']
        cat_feature_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names_out_ = cat_feature_names + self.numerical_features
        print(f"Preprocessor fitted. Transformed feature dimension: {len(self.feature_names_out_)} features.")
        return self

    def transform(self, X):
        """
        Transforms input DataFrame into a 2D numpy array using fitted encoders.
        """
        if self.column_transformer is None:
            raise ValueError("Preprocessor has not been fitted yet!")
        
        # If X is a single dictionary or series, convert to DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = pd.DataFrame([X])

        # Ensure required columns exist in input
        missing_cols = [c for c in ALL_FEATURES if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required features: {missing_cols}")

        X_transformed = self.column_transformer.transform(X)
        return X_transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        return self.feature_names_out_

    def save(self, filepath=None):
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No preprocessor file found at {filepath}")
        return joblib.load(filepath)
