"""Versioned, auditable model artifacts used by training and prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class ModelBundleV2:
    """Self-contained model bundle with the exact feature contract used to train it."""

    model_id: str
    run_id: str
    model_type: str
    pipeline: Any
    feature_spec: dict
    metrics: dict
    target_name: str
    target_unit: str = ""
    model_card: dict = field(default_factory=dict)
    schema_version: int = 2
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def predict_matrix(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Return prediction, uncertainty/error estimate, and its semantics."""
        if self.model_type == "gaussian_process":
            if isinstance(self.pipeline, Pipeline) and len(self.pipeline.steps) > 1:
                pre = Pipeline(self.pipeline.steps[:-1])
                X_model = pre.transform(X)
                estimator = self.pipeline.steps[-1][1]
            else:
                X_model = X
                estimator = self.pipeline
            mean, std = estimator.predict(X_model, return_std=True)
            return np.asarray(mean), np.asarray(std), "gpr_posterior_std"

        pred = np.asarray(self.pipeline.predict(X))
        if self.metrics.get("conformal_90_radius") is not None:
            radius = float(self.metrics["conformal_90_radius"])
            return pred, np.full(pred.shape, radius), "conformal_90_half_width"
        validation_error = float(self.metrics.get("test_rmse", 0.0) or 0.0)
        return pred, np.full(pred.shape, validation_error), "validation_rmse"

    def applicability(self, X: np.ndarray) -> dict:
        domain = self.model_card.get("applicabilityDomain", {})
        if not domain:
            return {"available": False, "inside": None, "distance": None, "threshold": None}
        imputer = self.pipeline.named_steps.get("imputer") if isinstance(self.pipeline, Pipeline) else None
        X_ready = imputer.transform(X) if imputer is not None else X
        center = np.asarray(domain["center"], dtype=float)
        scale = np.asarray(domain["scale"], dtype=float)
        distance = float(np.sqrt(np.mean(((X_ready[0] - center) / scale) ** 2)))
        threshold = float(domain["threshold"])
        return {"available": True, "inside": distance <= threshold, "distance": distance, "threshold": threshold}

    def to_payload(self) -> dict:
        """Use a plain payload so exported joblib files do not depend on this class."""
        return {
            "artifactType": "PolyMLModelBundle",
            "schemaVersion": self.schema_version,
            "modelId": self.model_id,
            "runId": self.run_id,
            "modelType": self.model_type,
            "pipeline": self.pipeline,
            "featureSpec": self.feature_spec,
            "metrics": self.metrics,
            "targetName": self.target_name,
            "targetUnit": self.target_unit,
            "modelCard": self.model_card,
            "createdAt": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ModelBundleV2":
        if payload.get("artifactType") != "PolyMLModelBundle" or payload.get("schemaVersion") != 2:
            raise ValueError("Unsupported model bundle")
        return cls(
            model_id=payload["modelId"],
            run_id=payload["runId"],
            model_type=payload["modelType"],
            pipeline=payload["pipeline"],
            feature_spec=payload["featureSpec"],
            metrics=payload.get("metrics", {}),
            target_name=payload.get("targetName", "target"),
            target_unit=payload.get("targetUnit", ""),
            model_card=payload.get("modelCard", {}),
            schema_version=2,
            created_at=payload.get("createdAt", datetime.now(timezone.utc).isoformat()),
        )


def save_model_bundle(bundle: ModelBundleV2, path: str | Path):
    joblib.dump(bundle.to_payload(), path)


def load_model_bundle(path: str | Path) -> ModelBundleV2:
    payload = joblib.load(path)
    if isinstance(payload, ModelBundleV2):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Unsupported model artifact")
    return ModelBundleV2.from_payload(payload)
