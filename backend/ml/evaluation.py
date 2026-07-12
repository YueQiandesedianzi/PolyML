"""Model evaluation metrics and utilities."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, pearsonr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Pearson correlation
    if len(y_true) > 2:
        pearson_r, _ = pearsonr(y_true, y_pred)
    else:
        pearson_r = 0.0

    return {
        "mse": round(float(mse), 6),
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 6),
        "mae": round(float(mae), 6),
        "pearson_r": round(float(pearson_r), 6),
    }


def prepare_parity_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Prepare parity plot data for frontend consumption."""
    metrics = compute_metrics(y_true, y_pred)
    return {
        "y_test": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "min_val": float(min(y_true.min(), y_pred.min())),
        "max_val": float(max(y_true.max(), y_pred.max())),
        **metrics,
    }
