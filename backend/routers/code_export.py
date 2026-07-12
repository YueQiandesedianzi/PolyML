"""Code Export API routes — generates reproducible Python scripts."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from schemas.base import CamelModel
from routers._utils import get_project_dir, load_meta

router = APIRouter(prefix="/api/projects/{project_id}/code-export", tags=["code-export"])


class ExportRequest(CamelModel):
    pipeline: str = "full"  # full, features_only, training_only
    include_comments: bool = True


@router.post("/generate")
async def generate_code(project_id: str, req: ExportRequest):
    """Generate a standalone Python script reproducing the pipeline."""
    proj_dir = get_project_dir(project_id)
    meta = load_meta(project_id)
    col_mapping = meta.get("column_mapping", {})

    # Gather project state
    features_config = {}
    features_path = proj_dir / "features_config.json"
    if features_path.exists():
        with open(features_path, encoding="utf-8") as f:
            features_config = json.load(f)

    training_runs = []
    runs_path = proj_dir / "training_runs.json"
    if runs_path.exists():
        with open(runs_path, encoding="utf-8") as f:
            training_runs = json.load(f)

    custom_rules = []
    rules_path = proj_dir / "custom_feature_rules.json"
    if rules_path.exists():
        with open(rules_path, encoding="utf-8") as f:
            custom_rules = json.load(f)

    script = _build_script(meta, features_config, training_runs, custom_rules, req)

    return {"script": script, "filename": f"polyml_reproduce_{project_id[:8]}.py"}


@router.get("/download")
async def download_script(project_id: str):
    """Download generated Python script."""
    proj_dir = get_project_dir(project_id)
    meta = load_meta(project_id)

    features_config = {}
    features_path = proj_dir / "features_config.json"
    if features_path.exists():
        with open(features_path, encoding="utf-8") as f:
            features_config = json.load(f)

    training_runs = []
    runs_path = proj_dir / "training_runs.json"
    if runs_path.exists():
        with open(runs_path, encoding="utf-8") as f:
            training_runs = json.load(f)

    custom_rules = []
    rules_path = proj_dir / "custom_feature_rules.json"
    if rules_path.exists():
        with open(rules_path, encoding="utf-8") as f:
            custom_rules = json.load(f)

    req = ExportRequest()
    script = _build_script(meta, features_config, training_runs, custom_rules, req)

    return PlainTextResponse(
        script,
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename=polyml_reproduce_{project_id[:8]}.py"},
    )


def _build_script(meta: dict, features_config: dict, training_runs: list, custom_rules: list, req: ExportRequest) -> str:
    """Build the full Python script."""
    lines = []
    smi = meta.get("smiles_column", "SMILES")
    target = meta.get("target_column", "target")
    col_mapping = meta.get("column_mapping", {})

    # Header
    lines.append('"""')
    lines.append(f"PolyML — Reproducible Pipeline Script")
    lines.append(f"Project: {meta.get('name', 'unnamed')}")
    lines.append(f"SMILES column: {smi}")
    lines.append(f"Target column: {target}")
    lines.append('"""')
    lines.append("")

    # Imports
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("from sklearn.model_selection import train_test_split, cross_val_score, KFold")
    lines.append("from sklearn.preprocessing import StandardScaler")
    lines.append("from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error")
    lines.append("from sklearn.pipeline import Pipeline")

    # Feature engineering imports
    use_rdkit = features_config.get("use_rdkit", True)
    use_van_krevelen = features_config.get("use_van_krevelen", False)
    use_3d = features_config.get("use_3d", False)

    if use_rdkit:
        lines.append("from rdkit import Chem")
        lines.append("from rdkit.Chem import Descriptors, AllChem")
    if use_van_krevelen:
        lines.append("# Van Krevelen group contribution method")
    if custom_rules:
        lines.append("import re  # For substructure matching")

    lines.append("")

    # Step 1: Load data
    if req.include_comments:
        lines.append("# ═══════════════════════════════════════════")
        lines.append("# Step 1: Load Data")
        lines.append("# ═══════════════════════════════════════════")
    lines.append(f'df = pd.read_csv("data.csv")')
    lines.append(f"smiles_col = '{smi}'")
    lines.append(f"target_col = '{target}'")
    lines.append("")

    # Step 2: Feature engineering
    if req.include_comments:
        lines.append("# ═══════════════════════════════════════════")
        lines.append("# Step 2: Feature Engineering")
        lines.append("# ═══════════════════════════════════════════")

    if use_rdkit:
        lines.append(_rdkit_feature_code())

    if use_van_krevelen:
        lines.append(_van_krevelen_code())

    if custom_rules:
        lines.append(_custom_rules_code(custom_rules))

    lines.append("")
    lines.append("# Combine all features")
    lines.append("all_features = np.hstack([f for f in [")
    if use_rdkit:
        lines.append("    rdkit_features,")
    if use_van_krevelen:
        lines.append("    vk_features,")
    if custom_rules:
        lines.append("    custom_features,")
    lines.append("] if f is not None and len(f) > 0])")
    lines.append("feature_names = [")
    if use_rdkit:
        lines.append("    [n for n in RDKIT_DESCRIPTOR_NAMES],")
    if use_van_krevelen:
        lines.append("    [n for n in VK_GROUP_NAMES],")
    if custom_rules:
        lines.append("    [n for n in CUSTOM_FEATURE_NAMES],")
    lines.append("]")
    lines.append("feature_names = [n for sublist in feature_names for n in sublist]")
    lines.append("")
    lines.append("X = all_features")
    lines.append(f"y = df[target_col].values")
    lines.append("print(f'Feature matrix: {X.shape}')")
    lines.append("")

    # Step 3: Training
    if req.pipeline in ("full", "training_only") and training_runs:
        lines.append(_training_code(training_runs, req.include_comments))

    return "\n".join(lines)


def _rdkit_feature_code() -> str:
    return '''
# RDKit molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

def compute_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 112
    descriptor_names = [n for n, _ in Descriptors.descList]
    values = []
    for name in descriptor_names:
        try:
            fn = Descriptors.descList[name][1] if isinstance(Descriptors.descList[name], tuple) else getattr(Descriptors, name, None)
            if fn:
                values.append(float(fn(mol)))
            else:
                values.append(np.nan)
        except Exception:
            values.append(np.nan)
    return values

RDKIT_DESCRIPTOR_NAMES = [n for n, _ in Descriptors.descList]
rdikit_features = np.array([compute_rdkit_descriptors(s) for s in df[smiles_col]])
print(f"RDKit features: {rdikit_features.shape[1]}")
'''


def _van_krevelen_code() -> str:
    return '''
# Van Krevelen group contribution method (28 groups)
import re
from rdkit import Chem

VK_SMARTS = [
    ("CH3", "[CH3]"), ("CH2", "[CH2]"), ("CH", "[CH]"), ("C_quat", "[C]"),
    ("CH2_olefin", "[CH2]=[CH]"), ("CH_olefin", "[CH]=[CH]"),
    ("CH2_ar", "c[CH2]"), ("CH_ar", "c[CH]"), ("C_quat_ar", "c[C]"),
    ("CH_alcohol", "[CH2][OH]"), ("CH_ether", "[CH2]O[CH2]"),
    ("CH2_amine", "[CH2][NH2]"), ("CH_amide", "[CH2][NH][C](=O)"),
    ("CH2_halide", "[CH2][F,Cl,Br,I]"),
    ("CH3_methyl", "C([CH3])([CH3])"),
    ("CH2_acid", "[CH2][C](=O)[OH]"),
    ("CH_ester", "[CH][C](=O)O"),
    ("C_carbonyl", "[C]=[O]"),
    ("CH_siloxane", "[CH2][Si]"),
    ("C_fluoro", "[C][F]"),
    ("CH_nitrile", "[CH2][C]#N"),
    ("CH_nitro", "[CH2][N+](=O)[O-]"),
    ("O_ether", "[O]([CH2])"),
    ("NH_primary", "[NH2]"),
    ("NH_secondary", "[NH]([CH2])"),
    ("N_tertiary", "[N]([CH2])([CH2])"),
    ("S_sulfide", "[S]([CH2])"),
    ("C_ring", "c1ccccc1"),
]

VK_GROUP_NAMES = [name for name, _ in VK_SMARTS]

def compute_van_krevelen(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * len(VK_SMARTS)
    counts = []
    for name, smarts in VK_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            counts.append(len(mol.GetSubstructMatches(pattern)))
        else:
            counts.append(0)
    return counts

vk_features = np.array([compute_van_krevelen(s) for s in df[smiles_col]])
print(f"Van Krevelen features: {vk_features.shape[1]}")
'''


def _custom_rules_code(rules: list) -> str:
    """Generate code for custom feature rules."""
    lines = ["# Custom feature engineering"]
    for rule in rules:
        if rule.get("rule_type") == "formula":
            expr = rule.get("expression", "").replace("^", "**")
            name = rule.get("name", "custom")
            lines.append(f"# Custom: {name} = {expr}")
            lines.append(f"# Apply via: df['{name}'] = df.eval('{expr}')")

    lines.append("")
    lines.append("CUSTOM_FEATURE_NAMES = [" + ", ".join(f'"{r.get("name", f"custom_{i}")}"' for i, r in enumerate(rules)) + "]")
    lines.append("custom_features = np.zeros((len(df), len(CUSTOM_FEATURE_NAMES)))")
    lines.append("# TODO: Implement custom feature evaluation based on rules above")

    return "\n".join(lines)


def _training_code(runs: list, include_comments: bool) -> str:
    """Generate training code from saved run data."""
    lines = []

    if include_comments:
        lines.append("")
        lines.append("# ═══════════════════════════════════════════")
        lines.append("# Step 3: Model Training & Evaluation")
        lines.append("# ═══════════════════════════════════════════")

    # Use the most recent run
    last_run = runs[-1]
    config = last_run.get("config", {})
    models = config.get("models", ["ridge"])
    cv_folds = config.get("cv_folds", 5)
    test_size = config.get("test_size", 0.2)
    cv_method = config.get("cv_method", "kfold")

    lines.append(f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)")
    lines.append("")

    # Model imports
    model_imports = set()
    for m in models:
        if m == "ridge":
            model_imports.add("from sklearn.linear_model import Ridge")
        elif m == "lasso":
            model_imports.add("from sklearn.linear_model import Lasso")
        elif m == "elasticnet":
            model_imports.add("from sklearn.linear_model import ElasticNet")
        elif m == "pls":
            model_imports.add("from sklearn.cross_decomposition import PLSCanonical")
        elif m == "knn":
            model_imports.add("from sklearn.neighbors import KNeighborsRegressor")
        elif m == "kernel_ridge":
            model_imports.add("from sklearn.kernel_ridge import KernelRidge")
        elif m == "random_forest":
            model_imports.add("from sklearn.ensemble import RandomForestRegressor")
        elif m == "gradient_boosting":
            model_imports.add("from sklearn.ensemble import GradientBoostingRegressor")
        elif m == "xgboost":
            model_imports.add("import xgboost as xgb")
        elif m == "svm":
            model_imports.add("from sklearn.svm import SVR")
        elif m == "gaussian_process":
            model_imports.add("from sklearn.gaussian_process import GaussianProcessRegressor")
            model_imports.add("from sklearn.gaussian_process.kernels import Matern")
        elif m == "mlp":
            model_imports.add("from sklearn.neural_network import MLPRegressor")

    for imp in sorted(model_imports):
        lines.append(imp)
    lines.append("")

    # Training loop
    lines.append("models_results = {}")
    lines.append("")

    # CV method
    if cv_method == "loocv":
        lines.append("from sklearn.model_selection import LeaveOneOut")
        lines.append("cv = LeaveOneOut()")
    elif cv_method == "repeated_kfold":
        lines.append("from sklearn.model_selection import RepeatedKFold")
        lines.append(f"cv = RepeatedKFold(n_splits={cv_folds}, n_repeats=3, random_state=42)")
    else:
        lines.append(f"cv = KFold(n_splits={cv_folds}, shuffle=True, random_state=42)")
    lines.append("")

    # Results from the run
    results = last_run.get("results", {})
    for model_key in models:
        if model_key in results:
            r = results[model_key]
            lines.append(f"# {model_key}: R²={r.get('test_r2', 'N/A')}, RMSE={r.get('test_rmse', 'N/A')}")

    lines.append("")

    return "\n".join(lines)
