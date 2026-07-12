"""Data import and column mapping API routes."""

import json
import shutil
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.data_import import parse_file, detect_column_types, get_data_summary
from routers._utils import get_project_dir, load_meta, save_meta

router = APIRouter(prefix="/api/projects/{project_id}/data", tags=["data"])


class ColumnMapRequest(BaseModel):
    mapping: dict[str, str]  # {col_name: "smiles"|"numeric"|"target"|"ignore"}


@router.post("/import")
async def import_data(project_id: str, file: UploadFile = File(...)):
    proj_dir = get_project_dir(project_id)
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Save uploaded file
    filename = file.filename or "data.csv"
    suffix = Path(filename).suffix.lower()
    save_path = proj_dir / f"imported_data{suffix}"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Parse file
    df = parse_file(str(save_path), file_content=content)

    # Detect column types
    detected = detect_column_types(df)

    # Save raw data as CSV (normalized)
    raw_csv_path = proj_dir / "imported_data.csv"
    df.to_csv(raw_csv_path, index=False)

    # Update project metadata
    meta = load_meta(project_id)
    meta["data_filename"] = filename
    meta["data_row_count"] = len(df)
    save_meta(project_id, meta)

    # Return preview (first 100 rows)
    preview = df.head(100).fillna("").to_dict(orient="records")

    return {
        "columns": list(df.columns),
        "preview": preview,
        "detected_types": detected,
        "row_count": len(df),
    }


@router.post("/map-columns")
async def map_columns(project_id: str, body: ColumnMapRequest):
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"
    if not raw_csv.exists():
        raise HTTPException(status_code=404, detail="No data imported")

    df = pd.read_csv(raw_csv)

    # Validate that all mapped columns exist in the CSV
    csv_columns = set(df.columns)
    unknown = [k for k in body.mapping if k not in csv_columns]
    if unknown:
        raise HTTPException(status_code=400, detail=f"CSV 中不存在的列: {', '.join(unknown)}")

    # Determine smiles, numeric, target columns
    smiles_cols = [k for k, v in body.mapping.items() if v == "smiles"]
    target_cols = [k for k, v in body.mapping.items() if v == "target"]
    numeric_cols = [k for k, v in body.mapping.items() if v == "numeric"]

    # Save mapping
    mapping_path = proj_dir / "column_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(body.mapping, f, indent=2, ensure_ascii=False)

    # Update metadata
    meta = load_meta(project_id)
    meta["smiles_column"] = smiles_cols[0] if smiles_cols else None
    meta["target_column"] = target_cols[0] if target_cols else None
    meta["numeric_columns"] = numeric_cols
    save_meta(project_id, meta)

    return {"mapped": True, "row_count": len(df)}


@router.get("/preview")
async def get_preview(project_id: str, page: int = 1, limit: int = 100):
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"
    if not raw_csv.exists():
        raise HTTPException(status_code=404, detail="No data imported")

    df = pd.read_csv(raw_csv)
    start = (page - 1) * limit
    end = start + limit

    rows = df.iloc[start:end].fillna("").to_dict(orient="records")
    return {"rows": rows, "total": len(df)}


@router.get("/reload")
async def reload_project_data(project_id: str):
    """Reload saved project data: preview, columns, detected types, and mapping."""
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"
    if not raw_csv.exists():
        return {"has_data": False}

    df = pd.read_csv(raw_csv)
    columns = list(df.columns)

    # Load saved mapping
    mapping_path = proj_dir / "column_mapping.json"
    saved_mapping = {}
    mapping_done = False
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            saved_mapping = json.load(f)
        mapping_done = True

    # Detect types (may differ from saved mapping if data changed)
    detected = detect_column_types(df)

    # Preview first 100 rows
    preview = df.head(100).fillna("").to_dict(orient="records")

    return {
        "has_data": True,
        "columns": columns,
        "preview": preview,
        "detected_types": detected,
        "mapping": saved_mapping,
        "mapping_done": mapping_done,
        "row_count": len(df),
    }


@router.get("/summary")
async def get_summary(project_id: str):
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"
    if not raw_csv.exists():
        raise HTTPException(status_code=404, detail="No data imported")

    df = pd.read_csv(raw_csv)
    return get_data_summary(df)
