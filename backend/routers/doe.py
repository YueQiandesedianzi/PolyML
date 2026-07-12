"""DOE (Design of Experiments) API routes."""

import uuid
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from ml.doe import (
    DOEFactor,
    DOEConstraint,
    generate_full_factorial,
    generate_fractional_factorial,
    generate_lhs,
    generate_box_behnken,
    generate_ccd,
    estimate_experiment_count,
    apply_constraints,
)
from routers._utils import atomic_write_json, get_project_dir

router = APIRouter(prefix="/api/projects/{project_id}/doe", tags=["doe"])


class DOEFactorInput(CamelModel):
    name: str
    low: float
    high: float
    center: float | None = None


class DOEConstraintInput(CamelModel):
    type: str
    factor_names: list[str]
    value: float
    relation: Literal["eq", "lte", "gte"] = "eq"


class DOEDesignRequest(CamelModel):
    method: str
    factors: list[DOEFactorInput]
    n_samples: int | None = None
    resolution: int = 3
    seed: int = 42
    constraints: list[DOEConstraintInput] | None = None


class DOEApplyRequest(CamelModel):
    mode: Literal["pending", "prediction"] = "pending"
    design_matrix: list[dict]
    smiles_template: str = "*"
    fill_values: dict = {}
    candidate_set_id: str | None = None


@router.get("/factors")
async def get_factors(project_id: str):
    """Get available numeric factors from imported data with min/max/mean statistics."""
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

    # Get numeric columns (not target, not smiles, not ignore)
    numeric_cols = [
        col for col, ctype in column_mapping.items()
        if ctype == "numeric" and col in df.columns
    ]

    factors = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        factors.append({
            "name": col,
            "type": "numeric",
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "current_low": float(series.min()),
            "current_high": float(series.max()),
        })

    return {"factors": factors}


@router.post("/generate")
async def generate_design(project_id: str, request: DOEDesignRequest):
    """Generate a DOE design matrix."""
    proj_dir = get_project_dir(project_id)
    if not (proj_dir / "imported_data.csv").exists():
        raise HTTPException(status_code=404, detail="No data imported")

    # Convert to DOEFactor objects
    factors = []
    for f in request.factors:
        if not f.name or f.low >= f.high:
            raise HTTPException(status_code=400, detail=f"Factor '{f.name}' must have low < high")
        center = f.center if f.center is not None else (f.low + f.high) / 2
        if not f.low <= center <= f.high:
            raise HTTPException(status_code=400, detail=f"Factor '{f.name}' center must be within bounds")
        factors.append(DOEFactor(
            name=f.name,
            low=f.low,
            high=f.high,
            center=center,
        ))

    if len(factors) < 2:
        raise HTTPException(status_code=400, detail="At least 2 factors required")
    if len(factors) > 12:
        raise HTTPException(status_code=400, detail="At most 12 factors are supported")
    if len({factor.name for factor in factors}) != len(factors):
        raise HTTPException(status_code=400, detail="Factor names must be unique")

    # Generate design based on method
    try:
        if request.method == "full_factorial":
            matrix = generate_full_factorial(factors)
        elif request.method == "fractional_factorial":
            matrix = generate_fractional_factorial(factors, resolution=request.resolution)
        elif request.method == "lhs":
            n = request.n_samples or 10
            matrix = generate_lhs(factors, n_samples=n, seed=request.seed)
        elif request.method == "box_behnken":
            matrix = generate_box_behnken(factors)
        elif request.method == "ccd":
            matrix = generate_ccd(factors)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Apply constraints if provided
    n_before = len(matrix)
    if request.constraints:
        constraint_objs = []
        for c in request.constraints:
            unknown = sorted(set(c.factor_names) - {factor.name for factor in factors})
            if unknown:
                raise HTTPException(status_code=400, detail=f"Unknown constraint factors: {', '.join(unknown)}")
            if c.type == "ratio" and len(c.factor_names) != 2:
                raise HTTPException(status_code=400, detail="Ratio constraints require exactly two factors")
            constraint_objs.append(DOEConstraint(
                constraint_type=c.type,
                factor_names=c.factor_names,
                value=c.value,
                relation=c.relation,
            ))
        matrix = apply_constraints(matrix, constraint_objs)

    # Build response
    factor_names = [f.name for f in factors]
    levels = {f.name: [f.low, f.high] for f in factors}

    # Save design to project directory for later apply
    design_id = str(uuid.uuid4())[:8]
    candidate_dir = proj_dir / "candidate_sets"
    design_path = candidate_dir / f"{design_id}.json"
    atomic_write_json(design_path, {
        "schema_version": 2,
        "candidate_set_id": design_id,
        "method": request.method,
        "factor_names": factor_names,
        "design_matrix": matrix,
        "source": "doe",
    })

    return {
        "design_id": design_id,
        "candidate_set_id": design_id,
        "method": request.method,
        "n_experiments": len(matrix),
        "n_before_constraints": n_before,
        "design_matrix": matrix,
        "factor_names": factor_names,
        "levels": levels,
        "constraints_applied": len(request.constraints or []) > 0,
    }


@router.post("/apply")
async def apply_design(project_id: str, request: DOEApplyRequest):
    """Apply a DOE design: append to dataset or save as prediction set."""
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"
    mapping_file = proj_dir / "column_mapping.json"

    if not raw_csv.exists():
        raise HTTPException(status_code=404, detail="No data imported")

    df = pd.read_csv(raw_csv)

    if mapping_file.exists():
        with open(mapping_file, encoding="utf-8") as f:
            column_mapping = json.load(f)
        smiles_col = next((col for col, ct in column_mapping.items() if ct == "smiles"), None)
        target_col = next((col for col, ct in column_mapping.items() if ct == "target"), None)
    else:
        smiles_col = None
        target_col = None

    # Build new rows from design matrix
    new_rows = []
    for row in request.design_matrix:
        new_row = row.copy()
        if smiles_col:
            new_row[smiles_col] = request.smiles_template
        if target_col and target_col not in new_row:
            new_row[target_col] = request.fill_values.get(target_col, np.nan)
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    if request.mode == "pending":
        new_df["candidate_set_id"] = request.candidate_set_id or "manual"
        new_df["experiment_status"] = "pending"
        queue_path = proj_dir / "experiment_queue.csv"
        existing = pd.read_csv(queue_path) if queue_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        combined.to_csv(queue_path, index=False)
        return {
            "applied": True,
            "rows_added": len(new_rows),
            "total_rows": len(combined),
            "destination": "experiment_queue.csv",
        }
    else:
        # Save as prediction set
        pred_path = proj_dir / "doe_predict_data.csv"
        new_df.to_csv(pred_path, index=False)
        return {
            "applied": True,
            "rows_added": len(new_rows),
            "total_rows": len(new_rows),
            "destination": "doe_predict_data.csv",
        }
