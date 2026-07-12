import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from config import settings
from main import app
from routers.automl import _active_runs


def test_project_to_saved_bundle_prediction(tmp_path):
    settings.app_data_path = str(tmp_path / "app-data")
    frame = pd.DataFrame({
        "temperature": np.linspace(20, 100, 24),
        "strength": np.linspace(5, 25, 24) + np.sin(np.arange(24)) * 0.1,
    })
    csv_bytes = frame.to_csv(index=False).encode("utf-8")

    with TestClient(app) as client:
        project = client.post("/api/projects", json={"name": "e2e", "description": ""})
        assert project.status_code == 200
        project_id = project.json()["id"]

        imported = client.post(
            f"/api/projects/{project_id}/data/import",
            files={"file": ("training.csv", csv_bytes, "text/csv")},
        )
        assert imported.status_code == 200
        assert (tmp_path / "app-data" / "projects" / project_id / "source" / "source_data.csv").exists()

        mapped = client.post(
            f"/api/projects/{project_id}/data/map-columns",
            json={"mapping": {"temperature": "numeric", "strength": "target"}},
        )
        assert mapped.status_code == 200
        features = client.post(
            f"/api/projects/{project_id}/features/engineer",
            json={"includeDescriptors": False, "includeVanKrevelen": False},
        )
        assert features.status_code == 200, features.text

        _active_runs[project_id] = "already-running"
        conflict = client.post(
            f"/api/projects/{project_id}/automl/train",
            json={"models": ["ridge"], "cvFolds": 3, "nTrials": 1},
        )
        assert conflict.status_code == 409
        _active_runs.pop(project_id, None)

        trained = client.post(
            f"/api/projects/{project_id}/automl/train",
            json={"models": ["ridge"], "cvFolds": 3, "nTrials": 1, "testSize": 0.2},
        )
        assert trained.status_code == 200
        assert "event: all_complete" in trained.text

        runs = client.get(f"/api/projects/{project_id}/automl/results").json()
        assert runs and runs[-1]["best_model"] == "ridge"
        run_id = runs[-1]["run_id"]
        saved = client.post(
            f"/api/projects/{project_id}/models/save",
            json={"name": "ridge-e2e", "runId": run_id},
        )
        assert saved.status_code == 200, saved.text

        predicted = client.post(
            f"/api/projects/{project_id}/predict",
            json={"processingParams": {"temperature": 60}, "modelName": "ridge-e2e"},
        )
        assert predicted.status_code == 200, predicted.text
        payload = predicted.json()
        assert payload["model_id"] == "ridge-e2e"
        assert payload["target_name"] == "strength"
        assert np.isfinite(payload["prediction"])
