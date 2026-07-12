"""AutoML training API routes with SSE progress streaming."""

import uuid
import json
import threading
import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.base import CamelModel
from ml.training import run_automl_pipeline
from ml.shap_utils import compute_shap, compute_feature_importance
from routers._utils import get_project_dir, load_meta, validate_id

router = APIRouter(prefix="/api/projects/{project_id}/automl", tags=["automl"])


class TrainConfig(CamelModel):
    models: list[str] = ["ridge", "lasso", "random_forest", "xgboost", "svm", "gaussian_process"]
    cv_folds: int = 5
    cv_method: str = "kfold"  # kfold, loocv, repeated_kfold
    n_trials: int = 50
    test_size: float = 0.2

    def model_post_init(self, __context):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")


# Active training cancellation flags keyed by run_id; secondary index by project_id
_cancel_events: dict[str, threading.Event] = {}
_active_runs: dict[str, str] = {}  # project_id → run_id


@router.post("/train")
async def train(project_id: str, config: TrainConfig):
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"

    if not features_path.exists():
        raise HTTPException(status_code=404, detail="Features not computed yet")

    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"].tolist()

    cancel_event = threading.Event()
    run_id = str(uuid.uuid4())
    _cancel_events[run_id] = cancel_event
    _active_runs[project_id] = run_id

    def generate():
        result_events = []
        best_result = None

        try:
            for event in run_automl_pipeline(
                X, y,
                selected_models=config.models,
                cv_folds=config.cv_folds,
                cv_method=config.cv_method,
                n_trials=config.n_trials,
                test_size=config.test_size,
                cancel_event=cancel_event,
                project_dir=str(proj_dir),
                run_id=run_id,
            ):
                result_events.append(event)

                if event["type"] == "all_complete":
                    best_key = event["data"]["best_model"]

                    # Compute SHAP and feature importance for best model
                    try:
                        split_path = proj_dir / f"split_{run_id}.npz"
                        if split_path.exists():
                            split_data = np.load(split_path)
                            X_train_split = split_data["X_train"]
                            X_test_split = split_data["X_test"]
                            y_test_split = split_data["y_test"]

                            best_pipeline_path = proj_dir / f"pipeline_{best_key}_{run_id}.joblib"
                            if best_pipeline_path.exists():
                                import joblib
                                best_pipeline = joblib.load(best_pipeline_path)

                                # Parity data (y_test vs y_pred)
                                try:
                                    y_pred_split = best_pipeline.predict(X_test_split)
                                    from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
                                    parity_result = {
                                        "y_test": y_test_split.tolist(),
                                        "y_pred": y_pred_split.tolist(),
                                        "r2": float(r2_score(y_test_split, y_pred_split)),
                                        "rmse": float(root_mean_squared_error(y_test_split, y_pred_split)),
                                        "mae": float(mean_absolute_error(y_test_split, y_pred_split)),
                                    }
                                    parity_path = proj_dir / f"parity_{run_id}.json"
                                    with open(parity_path, "w", encoding="utf-8") as f:
                                        json.dump(parity_result, f, ensure_ascii=False)
                                except Exception as p_err:
                                    yield f"event: error\ndata: {json.dumps({'message': f'Parity data failed: {p_err}'})}\n\n"

                                # Feature importance
                                fi_result = compute_feature_importance(best_pipeline, feature_names)
                                fi_path = proj_dir / f"feature_importance_{run_id}.json"
                                with open(fi_path, "w", encoding="utf-8") as f:
                                    json.dump(fi_result, f, ensure_ascii=False)

                                # SHAP (with safety limits for GP)
                                max_test = 50 if best_key == "gaussian_process" else 200
                                max_bg = 20 if best_key == "gaussian_process" else 100
                                shap_result = compute_shap(
                                    best_pipeline, X_train_split, X_test_split, feature_names,
                                    max_test_samples=max_test, max_background=max_bg,
                                )
                                shap_path = proj_dir / f"shap_{run_id}.json"
                                with open(shap_path, "w", encoding="utf-8") as f:
                                    json.dump(shap_result, f, ensure_ascii=False)
                    except Exception as shap_err:
                        # SHAP failure should not break training
                        yield f"event: error\ndata: {json.dumps({'message': f'SHAP computation failed: {shap_err}'})}\n\n"

                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"

            # Save run results to project
            _save_run(project_id, run_id, config, result_events)

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            _cancel_events.pop(run_id, None)
            _active_runs.pop(project_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/cancel")
async def cancel(project_id: str):
    active_run_id = _active_runs.get(project_id)
    if active_run_id:
        event = _cancel_events.get(active_run_id)
        if event:
            event.set()
            return {"cancelled": True}
    return {"cancelled": False, "message": "No active training"}


@router.get("/results")
async def get_results(project_id: str):
    proj_dir = get_project_dir(project_id)
    runs_file = proj_dir / "training_runs.json"

    if not runs_file.exists():
        return []

    with open(runs_file, encoding="utf-8") as f:
        runs = json.load(f)

    return runs


@router.get("/parity-data")
async def get_parity_data(project_id: str, run_id: str = ""):
    if run_id:
        validate_id(run_id, "run_id")
    proj_dir = get_project_dir(project_id)
    parity_file = proj_dir / f"parity_{run_id}.json"

    if parity_file.exists():
        with open(parity_file, encoding="utf-8") as f:
            return json.load(f)

    # Compute on the fly from saved split + pipeline
    split_path = proj_dir / f"split_{run_id}.npz"
    if not split_path.exists():
        raise HTTPException(status_code=404, detail="Parity data not found")

    split_data = np.load(split_path)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # Find any pipeline for this run
    import joblib, glob
    pipeline_files = list(proj_dir.glob(f"pipeline_*_{run_id}.joblib"))
    if not pipeline_files:
        raise HTTPException(status_code=404, detail="Pipeline not found for this run")

    pipeline = joblib.load(pipeline_files[0])
    y_pred = pipeline.predict(X_test)

    from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
    parity = {
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

    # Cache for next time
    with open(parity_file, "w", encoding="utf-8") as f:
        json.dump(parity, f, ensure_ascii=False)

    return parity


@router.get("/feature-importance")
async def get_feature_importance(project_id: str, run_id: str = ""):
    if run_id:
        validate_id(run_id, "run_id")
    proj_dir = get_project_dir(project_id)
    fi_file = proj_dir / f"feature_importance_{run_id}.json"

    if not fi_file.exists():
        raise HTTPException(status_code=404, detail="Feature importance not found")

    with open(fi_file, encoding="utf-8") as f:
        return json.load(f)


@router.delete("/runs/{run_id}")
async def delete_run(project_id: str, run_id: str):
    """Delete a training run and all its associated artifacts."""
    validate_id(run_id, "run_id")
    proj_dir = get_project_dir(project_id)
    runs_file = proj_dir / "training_runs.json"

    if not runs_file.exists():
        raise HTTPException(status_code=404, detail="No training runs")

    with open(runs_file, encoding="utf-8") as f:
        runs = json.load(f)

    # Find and remove the run
    new_runs = [r for r in runs if r.get("run_id") != run_id]
    if len(new_runs) == len(runs):
        raise HTTPException(status_code=404, detail="Run not found")

    # Delete associated artifact files
    artifacts = [
        f"parity_{run_id}.json",
        f"feature_importance_{run_id}.json",
        f"shap_{run_id}.json",
        f"split_{run_id}.npz",
    ]
    # Delete all pipeline files for this run
    for f_path in proj_dir.iterdir():
        if f_path.name.startswith("pipeline_") and f_path.name.endswith(f"_{run_id}.joblib"):
            artifacts.append(f_path.name)

    for fname in artifacts:
        fpath = proj_dir / fname
        if fpath.exists():
            fpath.unlink()

    # Save updated runs list
    with open(runs_file, "w", encoding="utf-8") as f:
        json.dump(new_runs, f, indent=2, ensure_ascii=False)

    return {"deleted": run_id, "remaining": len(new_runs)}


def _save_run(project_id: str, run_id: str, config: TrainConfig, events: list[dict]):
    """Save training run results to project directory."""
    proj_dir = get_project_dir(project_id)
    runs_file = proj_dir / "training_runs.json"

    runs = []
    if runs_file.exists():
        with open(runs_file, encoding="utf-8") as f:
            runs = json.load(f)

    # Extract model results from events
    model_results = {}
    best_model = None
    for event in events:
        if event["type"] == "all_complete":
            best_model = event["data"].get("best_model")
            model_results = event["data"].get("results", {})

    run_data = {
        "run_id": run_id,
        "config": config.model_dump(),
        "best_model": best_model,
        "results": model_results,
        "timestamp": str(__import__('datetime').datetime.now(__import__('datetime').timezone.utc)),
    }

    runs.append(run_data)

    with open(runs_file, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)
