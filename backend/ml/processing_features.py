"""Processing condition feature encoder for polymer data"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def encode_processing_conditions(
    df: pd.DataFrame,
    numeric_cols: list[str]
) -> tuple[np.ndarray, list[str], SimpleImputer | None]:
    """
    Extract and standardize user-provided processing condition columns.

    Returns:
        X: (n_samples, n_numeric_features) feature matrix
        feature_names: column names
        imputer: fitted imputer for later use
    """
    if not numeric_cols:
        return np.array([]).reshape(len(df), 0), [], None

    X = df[numeric_cols].copy()

    # Convert any non-numeric to NaN
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Do not fit an imputer here: preprocessing must be fitted inside each
    # training/CV fold. The returned matrix intentionally preserves NaNs.
    return X.values.astype(np.float64), numeric_cols, None
