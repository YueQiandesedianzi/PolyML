import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ml.artifacts import ModelBundleV2, load_model_bundle, save_model_bundle
from services.feature_engineering import engineer_features, transform_features


def test_feature_spec_round_trip_and_bundle(tmp_path):
    df = pd.DataFrame({
        "SMILES": ["CC", "CCC", "CCCC", "CCO", "CCN"],
        "temperature": [20.0, np.nan, 40.0, 50.0, 60.0],
        "target": [1.0, 1.5, 2.0, 2.4, 2.8],
    })
    result = engineer_features(
        df, "SMILES", ["temperature"], "target",
        include_descriptors=True, include_van_krevelen=False,
    )
    assert result.X.shape[0] == len(df)
    assert result.feature_spec["featureNames"] == result.feature_names
    transformed, missing = transform_features(df.iloc[:2], result.feature_spec)
    assert transformed.shape == (2, result.X.shape[1])
    assert missing == []
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("estimator", Ridge()),
    ]).fit(result.X, result.y)
    bundle = ModelBundleV2(
        model_id="test-model", run_id="run-1", model_type="ridge",
        pipeline=pipeline, feature_spec=result.feature_spec,
        metrics={"test_rmse": 0.2}, target_name="target", target_unit="MPa",
    )
    path = tmp_path / "bundle.joblib"
    save_model_bundle(bundle, path)
    raw_payload = joblib.load(path)
    assert raw_payload["artifactType"] == "PolyMLModelBundle"
    loaded = load_model_bundle(path)
    prediction, uncertainty, kind = loaded.predict_matrix(transformed)
    assert len(prediction) == 2
    assert np.allclose(uncertainty, 0.2)
    assert kind == "validation_rmse"


def test_bin_rule_persists_edges():
    rules = [{"name": "temp_bin", "ruleType": "bin", "expression": "temperature", "params": {"n_bins": 3}}]
    df = pd.DataFrame({"temperature": [10, 20, 30, 40, 50], "target": [1, 2, 3, 4, 5]})
    result = engineer_features(
        df, None, ["temperature"], "target",
        include_descriptors=False, include_van_krevelen=False, custom_rules=rules,
    )
    assert rules[0]["params"]["bin_edges"]
    transformed, missing = transform_features(df.iloc[:1], result.feature_spec)
    assert transformed.shape[1] == len(result.feature_names)
    assert missing == []
