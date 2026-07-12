"""Candidate-set-driven active learning API routes."""

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ml.active_learning import BayesianOptimizer
from routers._utils import get_project_dir, validate_id
from schemas.base import CamelModel
from services.feature_engineering import transform_features

router = APIRouter(prefix="/api/projects/{project_id}/active-learning", tags=["active-learning"])


class SuggestRequest(CamelModel):
    candidate_set_id: str
    acquisition: str = "ei"
    n_suggestions: int = 5
    xi: float = 0.01
    beta: float = 2.0
    smiles_template: str | None = None


def _load_project_data(project_id: str):
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"
    spec_path = proj_dir / "feature-spec.json"
    if not features_path.exists() or not spec_path.exists():
        raise HTTPException(status_code=404, detail="Features and feature specification are required")
    data = np.load(features_path, allow_pickle=True)
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    return proj_dir, data, spec


def _candidate_dataframe(proj_dir, spec: dict, rows: list[dict], smiles_template: str | None):
    candidates = pd.DataFrame(rows)
    smiles_col = spec.get("smilesColumn")
    if smiles_col and smiles_col not in candidates.columns:
        template = smiles_template
        if not template:
            raw = pd.read_csv(proj_dir / "imported_data.csv")
            unique = raw[smiles_col].dropna().astype(str).unique() if smiles_col in raw.columns else []
            if len(unique) == 1:
                template = unique[0]
        if not template:
            raise HTTPException(
                status_code=400,
                detail="Candidate set needs a SMILES template because the project contains multiple structures",
            )
        candidates[smiles_col] = template
    return candidates


def _signature(row: pd.Series | dict, columns: list[str]) -> tuple:
    values = []
    for column in columns:
        value = row.get(column, np.nan)
        values.append(None if pd.isna(value) else str(value))
    return tuple(values)


@router.post("/suggest")
async def suggest_next_experiments(project_id: str, req: SuggestRequest):
    candidate_set_id = validate_id(req.candidate_set_id, "candidate_set_id")
    if req.acquisition not in {"ei", "ucb", "pi"}:
        raise HTTPException(status_code=400, detail="Unknown acquisition function")
    if not 1 <= req.n_suggestions <= 100:
        raise HTTPException(status_code=400, detail="n_suggestions must be between 1 and 100")

    proj_dir, data, spec = _load_project_data(project_id)
    candidate_path = proj_dir / "candidate_sets" / f"{candidate_set_id}.json"
    if not candidate_path.exists():
        raise HTTPException(status_code=404, detail="Candidate set not found")
    with open(candidate_path, encoding="utf-8") as f:
        candidate_set = json.load(f)
    rows = candidate_set.get("design_matrix", [])
    if not rows:
        raise HTTPException(status_code=400, detail="Candidate set is empty")

    candidates = _candidate_dataframe(proj_dir, spec, rows, req.smiles_template)
    signature_columns = [
        *([spec.get("smilesColumn")] if spec.get("smilesColumn") else []),
        *spec.get("numericColumns", []),
    ]
    raw = pd.read_csv(proj_dir / "imported_data.csv")
    row_indices = data["row_indices"] if "row_indices" in data else np.arange(len(data["y"]))
    labeled = raw.iloc[row_indices]
    observed = {_signature(row, signature_columns) for _, row in labeled.iterrows()}
    keep = [
        index for index, (_, row) in enumerate(candidates.iterrows())
        if _signature(row, signature_columns) not in observed
    ]
    if not keep:
        raise HTTPException(status_code=400, detail="All candidates are already present in labeled data")
    candidates = candidates.iloc[keep].reset_index(drop=True)
    kept_rows = [rows[index] for index in keep]

    X_candidates, missing = transform_features(candidates, spec)
    try:
        optimizer = BayesianOptimizer(data["X"], data["y"], data["feature_names"].tolist())
        suggestions = optimizer.suggest_next(
            X_candidates,
            acquisition=req.acquisition,
            n_suggestions=min(req.n_suggestions, len(X_candidates)),
            xi=req.xi,
            beta=req.beta,
        )
        for suggestion in suggestions:
            suggestion["factors"] = kept_rows[suggestion["index"]]
            suggestion["candidate_set_id"] = candidate_set_id
            suggestion.pop("features", None)
        return {
            "suggestions": suggestions,
            "feature_importance": optimizer.feature_importance(),
            "acquisition": req.acquisition,
            "candidate_set_id": candidate_set_id,
            "candidate_count": len(X_candidates),
            "missing_features": missing,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/evaluate-gpr")
async def evaluate_gpr(project_id: str):
    _, data, _ = _load_project_data(project_id)
    if len(data["y"]) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 labeled samples")
    try:
        optimizer = BayesianOptimizer(data["X"], data["y"], data["feature_names"].tolist())
        return {
            "cv_results": optimizer.cross_validate(),
            "feature_importance": optimizer.feature_importance(),
            "n_features": data["X"].shape[1],
            "n_samples": len(data["y"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/status")
async def active_learning_status(project_id: str):
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"
    if not features_path.exists():
        return {"ready": False, "reason": "Features not computed"}
    data = np.load(features_path, allow_pickle=True)
    candidate_dir = proj_dir / "candidate_sets"
    candidate_sets = [path.stem for path in candidate_dir.glob("*.json")] if candidate_dir.exists() else []
    return {
        "ready": True,
        "n_samples": len(data["y"]),
        "n_features": data["X"].shape[1],
        "feature_names": data["feature_names"].tolist(),
        "candidate_sets": candidate_sets,
    }
