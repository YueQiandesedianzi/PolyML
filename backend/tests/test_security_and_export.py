import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ml.artifacts import ModelBundleV2, save_model_bundle
from routers._utils import safe_child_path, validate_id
from routers.code_export import _build_script


def test_path_traversal_is_rejected(tmp_path):
    with pytest.raises(HTTPException):
        validate_id("../model", "model_name")
    with pytest.raises(HTTPException):
        safe_child_path(tmp_path, "../model.joblib")


def test_exported_script_executes_and_matches_pipeline(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("estimator", Ridge(alpha=1e-6)),
    ]).fit(X, y)
    spec = {
        "schemaVersion": 2, "descriptors": False, "vanKrevelen": False,
        "smilesColumn": None, "numericColumns": ["x"], "customRules": [],
        "featureNames": ["x"], "nFeatures": 1,
    }
    bundle = ModelBundleV2(
        model_id="linear", run_id="run", model_type="ridge", pipeline=pipeline,
        feature_spec=spec, metrics={"test_rmse": 0.0}, target_name="y",
    )
    save_model_bundle(bundle, tmp_path / "model_bundle.joblib")
    pd.DataFrame({"x": [5.0]}).to_csv(tmp_path / "data.csv", index=False)
    script = _build_script(spec, {}, True)
    script_path = tmp_path / "reproduce.py"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(script_path), "--bundle", "model_bundle.joblib", "--input", "data.csv"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    exported = pd.read_csv(tmp_path / "predictions.csv")["y_predicted"].iloc[0]
    assert exported == pytest.approx(float(pipeline.predict([[5.0]])[0]))
