"""Single prediction API routes."""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from routers._utils import get_project_dir, load_meta

router = APIRouter(prefix="/api/projects/{project_id}/predict", tags=["prediction"])


class PredictionRequest(CamelModel):
    smiles: str
    processing_params: dict[str, float] = {}
    model_name: str | None = None


def _load_pipeline(project_id: str, model_name: str | None = None):
    """Load a trained pipeline. If model_name given, load saved model; else load latest run's best."""
    proj_dir = get_project_dir(project_id)

    if model_name:
        # Load named saved model
        model_path = proj_dir / "models" / f"{model_name}.joblib"
        if model_path.exists():
            return joblib.load(model_path), model_name

    # Load latest run's best model pipeline
    runs_file = proj_dir / "training_runs.json"
    if runs_file.exists():
        with open(runs_file, encoding="utf-8") as f:
            runs = json.load(f)
        if runs:
            latest_run = runs[-1]
            best_model = latest_run.get("best_model", "ridge")
            run_id = latest_run.get("run_id", "")
            pipeline_path = proj_dir / f"pipeline_{best_model}_{run_id}.joblib"
            if pipeline_path.exists():
                return joblib.load(pipeline_path), best_model

    return None, None


@router.post("")
async def predict(project_id: str, request: PredictionRequest):
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"
    mapping_file = proj_dir / "column_mapping.json"

    if not features_path.exists():
        raise HTTPException(status_code=400, detail="Model not trained yet")

    # Load training features to know which columns to keep
    data = np.load(features_path, allow_pickle=True)
    X_train_raw = data["X"]
    y_train = data["y"]
    train_feature_names = data["feature_names"].tolist()

    # Load column mapping
    if mapping_file.exists():
        with open(mapping_file, encoding="utf-8") as f:
            column_mapping = json.load(f)
        numeric_cols = [k for k, v in column_mapping.items() if v == "numeric"]
    else:
        numeric_cols = list(request.processing_params.keys())

    # Build single-row DataFrame
    row_data = {col: request.processing_params.get(col, 0.0) for col in numeric_cols}
    row_data["_smiles"] = request.smiles
    df = pd.DataFrame([row_data])

    # Run feature engineering
    from services.feature_engineering import engineer_features
    result = engineer_features(
        df=df,
        smiles_col="_smiles",
        numeric_cols=numeric_cols,
        target_col=numeric_cols[0] if numeric_cols else None,
        include_descriptors=True,
        include_van_krevelen=True,
        include_3d=False,
    )

    X_new_raw = result.X
    pred_feature_names = result.feature_names

    # Align prediction features to training feature set
    X_new_aligned = np.full((1, len(train_feature_names)), np.nan)
    for i, fname in enumerate(train_feature_names):
        if fname in pred_feature_names:
            j = pred_feature_names.index(fname)
            X_new_aligned[0, i] = X_new_raw[0, j]

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_new_imputed = imputer.transform(X_new_aligned)

    # Try to load a trained pipeline
    pipeline, model_used = _load_pipeline(project_id, request.model_name)

    if pipeline is not None:
        # Use the trained pipeline (includes scaler if needed)
        prediction = float(pipeline.predict(X_new_imputed)[0])
        # Load test residuals for uncertainty
        runs_file = proj_dir / "training_runs.json"
        residual_std = 0.0
        if runs_file.exists():
            with open(runs_file, encoding="utf-8") as f:
                runs = json.load(f)
            if runs:
                latest = runs[-1]
                best_model = latest.get("best_model", "ridge")
                results = latest.get("results", {})
                if best_model in results:
                    residual_std = results[best_model].get("test_rmse", 0.0)
    else:
        # Fallback: train a fresh Ridge model
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_new_scaled = scaler.transform(X_new_imputed)

        model = Ridge()
        model.fit(X_train_scaled, y_train)
        prediction = float(model.predict(X_new_scaled)[0])
        y_pred_train = model.predict(X_train_scaled)
        residual_std = float(np.std(y_train - y_pred_train))
        model_used = "Ridge (default)"

    return {
        "prediction": round(prediction, 4) if np.isfinite(prediction) else 0.0,
        "uncertainty": round(residual_std, 4) if np.isfinite(residual_std) else 0.0,
        "units": "Same units as target variable",
        "model_used": model_used,
    }
