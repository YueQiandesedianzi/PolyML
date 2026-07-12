"""Prediction API that only uses feature-complete ModelBundleV2 artifacts."""

import json

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import Field

from ml.artifacts import ModelBundleV2, load_model_bundle
from routers._utils import get_project_dir, safe_child_path, validate_id
from schemas.base import CamelModel
from services.feature_engineering import transform_features

router = APIRouter(prefix="/api/projects/{project_id}/predict", tags=["prediction"])


class PredictionRequest(CamelModel):
    smiles: str = ""
    processing_params: dict[str, float] = Field(default_factory=dict)
    model_name: str | None = None


def _load_bundle(project_id: str, model_name: str | None) -> ModelBundleV2:
    proj_dir = get_project_dir(project_id)
    if model_name:
        name = validate_id(model_name, "model_name")
        path = safe_child_path(proj_dir / "models", f"{name}.joblib")
    else:
        runs_file = proj_dir / "training_runs.json"
        if not runs_file.exists():
            raise HTTPException(status_code=404, detail="No trained model bundle found")
        with open(runs_file, encoding="utf-8") as f:
            runs = json.load(f)
        if not runs:
            raise HTTPException(status_code=404, detail="No trained model bundle found")
        path = proj_dir / f"bundle_{runs[-1].get('run_id', '')}.joblib"
    if not path.exists():
        raise HTTPException(
            status_code=409,
            detail="Model is legacy or incomplete; recompute features and retrain",
        )
    try:
        bundle = load_model_bundle(path)
    except ValueError:
        raise HTTPException(status_code=409, detail="Unsupported legacy model artifact")
    return bundle


@router.post("")
async def predict(project_id: str, request: PredictionRequest):
    bundle = _load_bundle(project_id, request.model_name)
    spec = bundle.feature_spec
    smiles_col = spec.get("smilesColumn") or "_smiles"
    if spec.get("smilesColumn") and not request.smiles.strip():
        raise HTTPException(status_code=400, detail="SMILES is required for this model")
    numeric_cols = list(spec.get("numericColumns", []))
    row = {smiles_col: request.smiles.strip()}
    warnings: list[str] = []
    for col in numeric_cols:
        if col in request.processing_params:
            row[col] = request.processing_params[col]
        else:
            row[col] = np.nan
            warnings.append(f"Missing processing parameter '{col}' was imputed by the trained pipeline")
    unknown = sorted(set(request.processing_params) - set(numeric_cols))
    if unknown:
        warnings.append(f"Ignored unknown processing parameters: {', '.join(unknown)}")

    X, missing_features = transform_features(pd.DataFrame([row]), spec)
    if missing_features:
        warnings.append(f"Unavailable derived features were imputed: {', '.join(missing_features[:10])}")
    prediction, uncertainty, uncertainty_kind = bundle.predict_matrix(X)
    applicability = bundle.applicability(X)
    if applicability.get("inside") is False:
        warnings.append("Prediction is outside the training applicability domain")
    value = float(prediction[0])
    error = float(uncertainty[0])
    if not np.isfinite(value):
        raise HTTPException(status_code=422, detail="Prediction was not finite")

    return {
        "prediction": round(value, 6),
        "uncertainty": round(error, 6) if np.isfinite(error) else 0.0,
        "uncertainty_kind": uncertainty_kind,
        "units": bundle.target_unit or "Same units as target variable",
        "target_name": bundle.target_name,
        "target_unit": bundle.target_unit,
        "model_id": bundle.model_id,
        "model_used": bundle.model_type,
        "warnings": warnings,
        "applicability_domain": applicability,
    }
