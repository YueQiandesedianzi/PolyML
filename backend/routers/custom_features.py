"""Custom feature engineering API routes."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel
from ml.custom_features import (
    CustomFeatureRule,
    SafeFormulaEvaluator,
    evaluate_custom_features,
    get_available_domain_formulas,
)
from routers._utils import get_project_dir

router = APIRouter(prefix="/api/projects/{project_id}/custom-features", tags=["custom-features"])


class ValidateRequest(CamelModel):
    rule_type: str
    expression: str
    available_columns: list[str] = []


class SaveRulesRequest(CamelModel):
    rules: list[dict]


@router.post("/validate")
async def validate_expression(project_id: str, body: ValidateRequest):
    """Validate a custom feature expression."""
    if body.rule_type == "formula":
        evaluator = SafeFormulaEvaluator(body.available_columns)
        valid, error = evaluator.validate(body.expression)
        return {"valid": valid, "error": error if not valid else None}

    elif body.rule_type == "substructure":
        # Validate SMARTS pattern
        try:
            from rdkit import Chem
            pat = Chem.MolFromSmarts(body.expression)
            if pat is None:
                return {"valid": False, "error": "无效的 SMARTS 模式"}
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": f"SMARTS 验证失败: {e}"}

    elif body.rule_type in ("bin", "interaction", "domain"):
        # Basic validation
        return {"valid": True, "error": None}

    return {"valid": False, "error": f"未知的规则类型: {body.rule_type}"}


@router.post("/preview")
async def preview_features(project_id: str, body: SaveRulesRequest):
    """Preview custom features on first 5 rows of data."""
    proj_dir = get_project_dir(project_id)
    raw_csv = proj_dir / "imported_data.csv"

    if not raw_csv.exists():
        raise HTTPException(status_code=404, detail="No data imported")

    df = pd.read_csv(raw_csv)
    preview_df = df.head(5).copy()

    # Get smiles column name
    mapping_file = proj_dir / "column_mapping.json"
    smiles_col = None
    if mapping_file.exists():
        with open(mapping_file, encoding="utf-8") as f:
            column_mapping = json.load(f)
        smiles_col = next((col for col, ct in column_mapping.items() if ct == "smiles"), None)

    rules = []
    for r in body.rules:
        rules.append(CustomFeatureRule(
            name=r.get("name", ""),
            rule_type=r.get("ruleType", r.get("rule_type", "formula")),
            expression=r.get("expression", ""),
            params=r.get("params", {}),
        ))

    try:
        X_preview, feature_names = evaluate_custom_features(preview_df, rules, smiles_col)
        preview_data = X_preview.tolist() if X_preview.size > 0 else []
        return {
            "preview": preview_data,
            "feature_names": feature_names,
            "n_features": len(feature_names),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save")
async def save_rules(project_id: str, body: SaveRulesRequest):
    """Save custom feature rules to project directory."""
    proj_dir = get_project_dir(project_id)

    # Convert camelCase to snake_case for storage
    normalized_rules = []
    for r in body.rules:
        normalized_rules.append({
            "name": r.get("name", ""),
            "rule_type": r.get("ruleType", r.get("rule_type", "formula")),
            "expression": r.get("expression", ""),
            "params": r.get("params", {}),
        })

    rules_path = proj_dir / "custom_feature_rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(normalized_rules, f, indent=2, ensure_ascii=False)

    return {"saved": True, "count": len(normalized_rules)}


@router.get("/domain-formulas")
async def get_domain_formulas():
    """List available domain-specific polymer formulas."""
    return {"formulas": get_available_domain_formulas()}
