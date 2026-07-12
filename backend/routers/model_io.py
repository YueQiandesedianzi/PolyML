"""Versioned model bundle save/load API routes."""

import json

import joblib
from fastapi import APIRouter, HTTPException

from ml.artifacts import load_model_bundle, save_model_bundle
from routers._utils import atomic_write_json, get_project_dir, safe_child_path, validate_id
from schemas.base import CamelModel

router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["models"])


class SaveModelRequest(CamelModel):
    name: str
    run_id: str = ""


class LoadModelRequest(CamelModel):
    name: str


@router.post("/save")
async def save_model(project_id: str, body: SaveModelRequest):
    name = validate_id(body.name, "model_name")
    if body.run_id:
        validate_id(body.run_id, "run_id")
    proj_dir = get_project_dir(project_id)
    models_dir = proj_dir / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = safe_child_path(models_dir, f"{name}.joblib")
    meta_path = safe_child_path(models_dir, f"{name}_meta.json")
    if model_path.exists() or meta_path.exists():
        raise HTTPException(status_code=400, detail=f"Model '{name}' already exists")

    runs_file = proj_dir / "training_runs.json"
    if not runs_file.exists():
        raise HTTPException(status_code=400, detail="No training runs found")
    with open(runs_file, encoding="utf-8") as f:
        runs = json.load(f)
    target_run = next((r for r in runs if r.get("run_id") == body.run_id), None) if body.run_id else None
    if target_run is None and not body.run_id and runs:
        target_run = runs[-1]
    if target_run is None:
        raise HTTPException(status_code=404, detail="Training run not found")

    run_id = target_run["run_id"]
    bundle_path = proj_dir / f"bundle_{run_id}.joblib"
    if not bundle_path.exists():
        raise HTTPException(
            status_code=409,
            detail="Legacy run has no feature-complete bundle; recompute features and retrain",
        )
    try:
        bundle = load_model_bundle(bundle_path)
    except ValueError:
        raise HTTPException(status_code=409, detail="Unsupported legacy model artifact")
    bundle.model_id = name
    save_model_bundle(bundle, model_path)

    metadata = {
        "schema_version": 2,
        "name": name,
        "run_id": run_id,
        "model_type": bundle.model_type,
        "target_name": bundle.target_name,
        "target_unit": bundle.target_unit,
        "metrics": bundle.metrics,
        "model_card": bundle.model_card,
        "legacy": False,
    }
    atomic_write_json(meta_path, metadata)
    return {"saved": True, "path": f"models/{name}.joblib", "model": metadata}


@router.get("")
async def list_models(project_id: str):
    models_dir = get_project_dir(project_id) / "models"
    if not models_dir.exists():
        return []
    models = []
    known_artifacts = set()
    for meta_file in models_dir.glob("*_meta.json"):
        try:
            with open(meta_file, encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        name = metadata.get("name", meta_file.name.removesuffix("_meta.json"))
        known_artifacts.add(name)
        artifact = models_dir / f"{name}.joblib"
        metadata["legacy"] = metadata.get("schema_version") != 2 or not artifact.exists()
        models.append(metadata)
    for artifact in models_dir.glob("*.joblib"):
        if artifact.stem not in known_artifacts:
            models.append({"name": artifact.stem, "legacy": True, "model_type": "unknown"})
    return models


@router.post("/load")
async def load_model(project_id: str, body: LoadModelRequest):
    name = validate_id(body.name, "model_name")
    models_dir = get_project_dir(project_id) / "models"
    meta_path = safe_child_path(models_dir, f"{name}_meta.json")
    artifact_path = safe_child_path(models_dir, f"{name}.joblib")
    if not meta_path.exists() or not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("schema_version") != 2:
        raise HTTPException(status_code=409, detail="Legacy model must be retrained")
    return {"loaded": True, "model": metadata}
