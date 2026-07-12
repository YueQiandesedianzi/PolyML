"""Model save/load API routes."""

import json
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from routers._utils import get_project_dir

router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["models"])


class SaveModelRequest(CamelModel):
    name: str
    run_id: str = ""


class LoadModelRequest(CamelModel):
    name: str


@router.post("/save")
async def save_model(project_id: str, body: SaveModelRequest):
    proj_dir = get_project_dir(project_id)
    models_dir = proj_dir / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"{body.name}.joblib"

    if model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model '{body.name}' already exists")

    # Find the pipeline from the specified run or latest run
    runs_file = proj_dir / "training_runs.json"
    if not runs_file.exists():
        raise HTTPException(status_code=400, detail="No training runs found")

    with open(runs_file, encoding="utf-8") as f:
        runs = json.load(f)

    # Find the target run
    target_run = None
    if body.run_id:
        target_run = next((r for r in runs if r["run_id"] == body.run_id), None)
    if not target_run and runs:
        target_run = runs[-1]

    if not target_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    run_id = target_run["run_id"]
    best_model = target_run.get("best_model", "ridge")

    # Load the actual pipeline from disk
    pipeline_path = proj_dir / f"pipeline_{best_model}_{run_id}.joblib"
    if not pipeline_path.exists():
        raise HTTPException(status_code=404, detail=f"Pipeline file not found for model '{best_model}'")

    import joblib
    pipeline = joblib.load(pipeline_path)

    # Save as named model
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "name": body.name,
        "run_id": run_id,
        "model_type": best_model,
        "metrics": target_run.get("results", {}).get(best_model, {}),
    }

    with open(models_dir / f"{body.name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {"saved": True, "path": f"models/{body.name}.joblib"}


@router.get("")
async def list_models(project_id: str):
    proj_dir = get_project_dir(project_id)
    models_dir = proj_dir / "models"

    if not models_dir.exists():
        return []

    models = []
    for meta_file in models_dir.glob("*_meta.json"):
        with open(meta_file, encoding="utf-8") as f:
            models.append(json.load(f))

    return models


@router.post("/load")
async def load_model(project_id: str, body: LoadModelRequest):
    proj_dir = get_project_dir(project_id)
    models_dir = proj_dir / "models"
    meta_path = models_dir / f"{body.name}_meta.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{body.name}' not found")

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Verify the joblib file exists
    joblib_path = models_dir / f"{body.name}.joblib"
    if not joblib_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file '{body.name}.joblib' not found")

    return {"loaded": True, "model": metadata}
