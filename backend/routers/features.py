"""Feature engineering API routes."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from services.feature_engineering import engineer_features
from routers._utils import atomic_write_json, get_project_dir, load_meta, save_meta

router = APIRouter(prefix="/api/projects/{project_id}/features", tags=["features"])


class FeatureConfig(CamelModel):
    include_descriptors: bool = True
    include_van_krevelen: bool = True
    include_3d: bool = False
    custom_rules: list[dict] | None = None
    target_unit: str = ""


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

    # Validate and isolate labeled rows. DOE/pending rows remain in the project
    # dataset but never enter supervised training until a measured target exists.
    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    if target_numeric.notna().sum() == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' is not numeric. Please select a numeric column as target."
        )
    labeled_mask = target_numeric.notna()
    labeled_indices = np.flatnonzero(labeled_mask.to_numpy())
    df_labeled = df.loc[labeled_mask].copy()
    df_labeled[target_col] = target_numeric.loc[labeled_mask]
    if df_labeled[target_col].nunique() <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' has only {df_labeled[target_col].nunique()} unique value(s). Please select a column with more variation."
        )
    if len(df_labeled) < 3:
        raise HTTPException(status_code=400, detail="At least 3 labeled rows are required")

    # If no SMILES column or it's mapped to a non-SMILES column, skip molecular features
    has_smiles = smiles_col is not None and smiles_col in df.columns
    if has_smiles and not config.include_descriptors and not config.include_van_krevelen:
        pass  # User explicitly disabled molecular features
    elif not has_smiles:
        config.include_descriptors = False
        config.include_van_krevelen = False
        smiles_col = None

    result = engineer_features(
        df=df_labeled,
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
        row_indices=labeled_indices,
    )

    meta = load_meta(project_id)
    result.feature_spec.update({
        "targetUnit": config.target_unit or meta.get("target_unit", ""),
        "dataRevision": meta.get("data_revision", 1),
        "labeledRowCount": len(df_labeled),
        "excludedUnlabeledRows": int((~labeled_mask).sum()),
    })
    atomic_write_json(proj_dir / "feature-spec.json", result.feature_spec)
    atomic_write_json(proj_dir / "features_config.json", config.model_dump())

    # Update metadata
    meta["feature_count"] = result.X.shape[1]
    meta["n_descriptors"] = result.n_descriptors
    meta["n_van_krevelen"] = result.n_van_krevelen
    meta["n_processing"] = result.n_processing
    meta["rdkit_failures"] = result.rdkit_failures
    meta["target_unit"] = result.feature_spec["targetUnit"]
    meta["labeled_row_count"] = len(df_labeled)
    meta["excluded_unlabeled_rows"] = int((~labeled_mask).sum())
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
        "labeled_rows": len(df_labeled),
        "excluded_unlabeled_rows": int((~labeled_mask).sum()),
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
