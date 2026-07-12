"""SHAP explainability utilities for trained models."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap


def compute_shap(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    max_test_samples: int = 200,
    max_background: int = 100,
) -> dict:
    """
    Compute SHAP values for the estimator inside the pipeline.

    Returns dict with 'values', 'base_values', 'feature_names'.
    """
    # Separate preprocessor and estimator
    preprocessor = Pipeline(pipeline.steps[:-1]) if len(pipeline.steps) > 1 else None
    estimator = pipeline.steps[-1][1]

    if preprocessor:
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)
    else:
        X_train_t = X_train
        X_test_t = X_test

    # Use stratified sample for background
    if len(X_train_t) > max_background:
        idx = np.random.choice(len(X_train_t), max_background, replace=False)
        background = X_train_t[idx]
    else:
        background = X_train_t

    # Choose appropriate SHAP explainer
    if isinstance(estimator, (RandomForestRegressor, xgb.XGBRegressor)):
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.KernelExplainer(estimator.predict, background)

    # Limit test samples for performance
    X_explain = X_test_t[:max_test_samples]
    shap_values = explainer(X_explain)

    # Convert to serializable format
    return {
        "values": shap_values.values.tolist(),
        "base_values": (
            shap_values.base_values.tolist()
            if hasattr(shap_values.base_values, 'tolist')
            else [float(shap_values.base_values)]
        ),
        "feature_names": feature_names[:shap_values.values.shape[1]],
        "n_test_samples": X_explain.shape[0],
    }


def compute_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
) -> dict:
    """
    Compute feature importance from the trained pipeline.
    Uses model-native importance or permutation importance as fallback.
    """
    estimator = pipeline.steps[-1][1]

    importance = None

    # Try model-native importance first
    if hasattr(estimator, 'feature_importances_'):
        importance = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        importance = np.abs(estimator.coef_)
    elif hasattr(estimator, 'alpha') and hasattr(estimator, 'coef_'):
        importance = np.abs(estimator.coef_)

    if importance is None or len(importance) != len(feature_names):
        importance = np.zeros(len(feature_names))

    # Sort by importance descending
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx].tolist()

    return {
        "features": sorted_features,
        "importance": sorted_importance,
    }
