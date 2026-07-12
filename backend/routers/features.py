"""Feature engineering API routes."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from services.feature_engineering import engineer_features
from routers._utils import get_project_dir, load_meta, save_meta

router = APIRouter(prefix="/api/projects/{project_id}/features", tags=["features"])


class FeatureConfig(CamelModel):
    include_descriptors: bool = True
    include_van_krevelen: bool = True
    include_3d: bool = False
    custom_rules: list[dict] | None = None


@router.post("/engineer")
async def engineer(project_id: str, config: FeatureConfig):
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"
    mapping_file = proj_dir / "column_mapping.json"

    if not raw_csv.exists():
        raise HTTPException(status_code=404, detail="No data imported")
    if not mapping_file.exists():
        raise HTTPException(status_code=400, detail="Column mapping not configured")

    df = pd.read_csv(raw_csv)

    with open(mapping_file, encoding="utf-8") as f:
        column_mapping = json.load(f)

    smiles_col = None
    target_col = None
    numeric_cols = []

    for col, ctype in column_mapping.items():
        if ctype == "smiles":
            smiles_col = col
        elif ctype == "target":
            target_col = col
        elif ctype == "numeric":
            numeric_cols.append(col)

    if not target_col:
        raise HTTPException(status_code=400, detail="No target column defined")

    # Validate target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' is not numeric. Please select a numeric column as target."
        )
    if df[target_col].nunique() <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' has only {df[target_col].nunique()} unique value(s). Please select a column with more variation."
        )

    # If no SMILES column or it's mapped to a non-SMILES column, skip molecular features
    has_smiles = smiles_col is not None and smiles_col in df.columns
    if has_smiles and not config.include_descriptors and not config.include_van_krevelen:
        pass  # User explicitly disabled molecular features
    elif not has_smiles:
        config.include_descriptors = False
        config.include_van_krevelen = False
        smiles_col = None

    result = engineer_features(
        df=df,
        smiles_col=smiles_col,
        numeric_cols=numeric_cols,
        target_col=target_col,
        include_descriptors=config.include_descriptors,
        include_van_krevelen=config.include_van_krevelen,
        include_3d=config.include_3d,
        custom_rules=config.custom_rules,
    )

    # Save feature matrix and target
    np.savez_compressed(
        proj_dir / "features.npz",
        X=result.X,
        y=result.y,
        feature_names=np.array(result.feature_names, dtype=object),
    )

    # Update metadata
    meta = load_meta(project_id)
    meta["feature_count"] = result.X.shape[1]
    meta["n_descriptors"] = result.n_descriptors
    meta["n_van_krevelen"] = result.n_van_krevelen
    meta["n_processing"] = result.n_processing
    meta["rdkit_failures"] = result.rdkit_failures
    save_meta(project_id, meta)

    return {
        "X_shape": list(result.X.shape),
        "feature_names": result.feature_names,
        "n_descriptors": result.n_descriptors,
        "n_van_krevelen": result.n_van_krevelen,
        "n_processing": result.n_processing,
        "n_custom": getattr(result, "n_custom", 0),
        "n_smiles_failed": len(result.rdkit_failures),
        "n_dropped_low_variance": len(result.dropped_low_variance),
    }


@router.get("/summary")
async def get_feature_summary(project_id: str):
    proj_dir = get_project_dir(project_id)
    features_path = proj_dir / "features.npz"

    if not features_path.exists():
        raise HTTPException(status_code=404, detail="Features not computed yet")

    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    feature_names = data["feature_names"].tolist()

    return {
        "feature_count": X.shape[1],
        "sample_count": X.shape[0],
        "names": feature_names[:50],  # First 50 for preview
    }
