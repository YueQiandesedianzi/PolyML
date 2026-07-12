"""Active Learning API routes."""

import json
import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from ml.active_learning import BayesianOptimizer
from routers._utils import get_project_dir

router = APIRouter(prefix="/api/projects/{project_id}/active-learning", tags=["active-learning"])


class SuggestRequest(CamelModel):
    acquisition: str = "ei"  # ei, ucb, pi
    n_suggestions: int = 5
    xi: float = 0.01
    beta: float = 2.0


def _load_project_data(project_id: str):
    """Load features, labels, and feature names from project."""
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"

    if not features_path.exists():
        raise HTTPException(status_code=404, detail="Features not computed yet")

    data = np.load(features_path, allow_pickle=True)
    return data["X"], data["y"], data["feature_names"].tolist()


@router.post("/suggest")
async def suggest_next_experiments(project_id: str, req: SuggestRequest):
    """Use Bayesian optimization to suggest next experiments."""
    X, y, feature_names = _load_project_data(project_id)

    if len(y) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 samples for active learning")

    try:
        optimizer = BayesianOptimizer(X, y, feature_names)
        suggestions = optimizer.suggest_next(
            X,
            acquisition=req.acquisition,
            n_suggestions=min(req.n_suggestions, len(X)),
            xi=req.xi,
            beta=req.beta,
        )

        # Also get feature importance from kernel
        fi = optimizer.feature_importance()

        return {
            "suggestions": suggestions,
            "feature_importance": fi,
            "acquisition": req.acquisition,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-gpr")
async def evaluate_gpr(project_id: str):
    """Cross-validate the GPR surrogate model."""
    X, y, feature_names = _load_project_data(project_id)

    if len(y) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 samples")

    try:
        optimizer = BayesianOptimizer(X, y, feature_names)
        cv_result = optimizer.cross_validate()
        fi = optimizer.feature_importance()

        return {
            "cv_results": cv_result,
            "feature_importance": fi,
            "n_features": len(feature_names),
            "n_samples": len(y),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def active_learning_status(project_id: str):
    """Check if active learning can be run (features computed)."""
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"

    if not features_path.exists():
        return {"ready": False, "reason": "Features not computed"}

    data = np.load(features_path, allow_pickle=True)
    n_samples = len(data["y"])
    n_features = data["X"].shape[1]

    return {
        "ready": True,
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_names": data["feature_names"].tolist(),
    }
