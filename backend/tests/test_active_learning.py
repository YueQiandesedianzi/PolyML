import json

import numpy as np
import pandas as pd
import pytest

from config import settings
from routers.active_learning import SuggestRequest, suggest_next_experiments


@pytest.mark.asyncio
async def test_suggestions_come_only_from_unlabeled_candidate_set(tmp_path):
    settings.app_data_path = str(tmp_path / "app")
    project_id = "candidate-test"
    project = settings.projects_path / project_id
    (project / "candidate_sets").mkdir(parents=True)
    X = np.arange(1.0, 7.0).reshape(-1, 1)
    y = np.array([1.0, 1.8, 3.2, 4.1, 5.2, 6.4])
    np.savez_compressed(
        project / "features.npz", X=X, y=y,
        feature_names=np.array(["x"], dtype=object), row_indices=np.arange(6),
    )
    (project / "feature-spec.json").write_text(json.dumps({
        "schemaVersion": 2, "descriptors": False, "vanKrevelen": False,
        "smilesColumn": None, "numericColumns": ["x"], "customRules": [],
        "featureNames": ["x"], "nFeatures": 1,
    }), encoding="utf-8")
    pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": y}).to_csv(project / "imported_data.csv", index=False)
    (project / "candidate_sets" / "next.json").write_text(json.dumps({
        "candidate_set_id": "next",
        "design_matrix": [{"x": 6}, {"x": 7}, {"x": 8}],
    }), encoding="utf-8")

    response = await suggest_next_experiments(
        project_id, SuggestRequest(candidate_set_id="next", n_suggestions=2)
    )
    suggested = {item["factors"]["x"] for item in response["suggestions"]}
    assert suggested <= {7, 8}
    assert 6 not in suggested
