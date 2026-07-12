"""
Feature engineering orchestrator for polymer materials data.
Combines RDKit descriptors, Van Krevelen group contributions,
and processing condition features.
"""

from typing import Optional
import pickle
import numpy as np
import pandas as pd
from ml.descriptors import RDKitDescriptorCalculator
from ml.van_krevelen import VanKrevelenEngine
from ml.processing_features import encode_processing_conditions


class FeatureEngineeringResult:
    """Container for feature engineering output."""

    def __init__(self):
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.feature_names: list[str] = []
        self.n_descriptors: int = 0
        self.n_van_krevelen: int = 0
        self.n_processing: int = 0
        self.dropped_low_variance: list[str] = []
        self.dropped_inf_nan: list[str] = []
        self.rdkit_failures: list[int] = []
        self.preprocessor_bytes: Optional[bytes] = None  # serialized preprocessor
        self.feature_spec: dict = {}


def engineer_features(
    df: pd.DataFrame,
    smiles_col: str,
    numeric_cols: list[str],
    target_col: str | None,
    include_descriptors: bool = True,
    include_van_krevelen: bool = True,
    include_3d: bool = False,
    custom_rules: list[dict] | None = None,
) -> FeatureEngineeringResult:
    """
    Full feature engineering pipeline for polymer data.

    Args:
        df: DataFrame with SMILES, numeric, and target columns
        smiles_col: name of column containing SMILES strings
        numeric_cols: list of numeric feature column names (processing params)
        target_col: name of the target variable column
        include_descriptors: compute RDKit molecular descriptors
        include_van_krevelen: compute Van Krevelen group contributions
        include_3d: compute 3D descriptors (slow, not recommended for MVP)

    Returns:
        FeatureEngineeringResult with feature matrix, names, and metadata
    """
    result = FeatureEngineeringResult()

    smiles_list = df[smiles_col].astype(str).tolist() if smiles_col and smiles_col in df.columns else []

    # Validate SMILES: if less than 20% parse, skip molecular features
    if smiles_list and (include_descriptors or include_van_krevelen):
        try:
            from rdkit import Chem
            sample = [str(v) for v in smiles_list[:20] if v and str(v) != 'nan']
            valid = sum(1 for s in sample if Chem.MolFromSmiles(s) is not None)
            if valid < max(1, len(sample) * 0.2):
                print(f"[FeatureEngineering] SMILES column '{smiles_col}' has no valid SMILES ({valid}/{len(sample)} parsed). Skipping molecular features.")
                smiles_list = []
                include_descriptors = False
                include_van_krevelen = False
        except ImportError:
            pass

    if target_col:
        result.y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)

    all_features = []
    all_names = []
    feature_groups: dict[str, list[str]] = {
        "molecular_structure": [],
        "group_counts": [],
        "composition_process": [],
        "theory_derived": [],
    }

    # Branch 1: RDKit molecular descriptors
    if include_descriptors and smiles_list:
        calc = RDKitDescriptorCalculator()
        X_desc, desc_names, failed = calc.compute(smiles_list)
        result.rdkit_failures = failed

        # Keep the registry-defined descriptor contract stable. Missing values are
        # imputed inside each CV fold by the model pipeline.
        X_desc[~np.isfinite(X_desc)] = np.nan
        desc_names_final = desc_names
        result.n_descriptors = len(desc_names_final)
        all_features.append(X_desc)
        all_names.extend(desc_names_final)
        feature_groups["molecular_structure"].extend(desc_names_final)

    # Branch 2: Van Krevelen group contributions
    if include_van_krevelen and smiles_list:
        vk = VanKrevelenEngine()
        X_vk, vk_names = vk.compute_features(smiles_list)
        # Group-contribution Yg sums are not generally valid without repeat-unit
        # normalization and a target-specific unit contract. Use auditable group
        # counts (plus repeat-unit molecular weight) as generic descriptors.
        vk_names = [name for name in vk_names if name.startswith("vk_count_") or name == "vk_M0"]
        X_vk_arr = X_vk[vk_names].values.astype(np.float64)
        result.n_van_krevelen = len(vk_names)
        all_features.append(X_vk_arr)
        all_names.extend(vk_names)
        feature_groups["group_counts"].extend(vk_names)

    # Branch 3: Processing conditions
    if numeric_cols:
        X_proc, proc_names, _ = encode_processing_conditions(df, numeric_cols)
        result.n_processing = len(proc_names)
        if X_proc.shape[1] > 0:
            all_features.append(X_proc)
            all_names.extend(proc_names)
            feature_groups["composition_process"].extend(proc_names)

    # Branch 4: Custom features
    result.n_custom = 0
    if custom_rules:
        try:
            from ml.custom_features import CustomFeatureRule, evaluate_custom_features
            rules = []
            for r in custom_rules:
                rules.append(CustomFeatureRule(
                    name=r.get("name", ""),
                    rule_type=r.get("rule_type", r.get("ruleType", "formula")),
                    expression=r.get("expression", ""),
                    params=r.get("params", {}),
                ))
            X_custom, custom_names = evaluate_custom_features(df, rules, smiles_col)
            if X_custom.shape[1] > 0:
                result.n_custom = len(custom_names)
                all_features.append(X_custom)
                all_names.extend(custom_names)
                feature_groups["theory_derived"].extend(custom_names)
        except Exception as e:
            print(f"[FeatureEngineering] Custom features failed: {e}")

    # Combine all feature branches
    if not all_features:
        result.X = np.empty((len(df), 0))
        result.feature_names = []
        return result

    # Handle mismatched row counts (due to RDKit failures)
    n_rows = len(df)
    aligned_features = []
    for X_feat in all_features:
        if X_feat.shape[0] == n_rows:
            aligned_features.append(X_feat)
        else:
            # Pad with zeros
            padded = np.zeros((n_rows, X_feat.shape[1]))
            padded[:X_feat.shape[0], :] = X_feat
            aligned_features.append(padded)

    X_combined = np.hstack(aligned_features)

    # Preserve missing values. The fitted model pipeline owns imputation so the
    # training fold, validation fold, and later predictions share one contract.
    X_combined[~np.isfinite(X_combined)] = np.nan

    result.X = X_combined
    result.feature_names = all_names

    result.feature_spec = {
        "schemaVersion": 2,
        "descriptors": include_descriptors,
        "vanKrevelen": include_van_krevelen,
        "include3d": include_3d,
        "smilesColumn": smiles_col,
        "numericColumns": numeric_cols,
        "targetColumn": target_col,
        "customRules": custom_rules or [],
        "featureNames": all_names,
        "nFeatures": X_combined.shape[1],
        "featureGroups": feature_groups,
        "vanKrevelenMode": "counts_only",
        "vanKrevelenReference": "D.W. van Krevelen, Properties of Polymers, 4th ed.",
    }
    result.preprocessor_bytes = pickle.dumps(result.feature_spec)

    return result


def transform_features(df: pd.DataFrame, feature_spec: dict) -> tuple[np.ndarray, list[str]]:
    """Transform rows using a persisted feature contract without fitting state."""
    smiles_col = feature_spec.get("smilesColumn")
    numeric_cols = list(feature_spec.get("numericColumns", []))
    requested_names = list(feature_spec.get("featureNames", []))
    custom_rules = feature_spec.get("customRules", [])

    branches: list[np.ndarray] = []
    names: list[str] = []
    smiles_list = (
        df[smiles_col].astype(str).tolist()
        if smiles_col and smiles_col in df.columns
        else []
    )

    if feature_spec.get("descriptors") and smiles_list:
        calc = RDKitDescriptorCalculator()
        X_desc, desc_names, _ = calc.compute(smiles_list)
        X_desc[~np.isfinite(X_desc)] = np.nan
        branches.append(X_desc)
        names.extend(desc_names)

    if feature_spec.get("vanKrevelen") and smiles_list:
        vk = VanKrevelenEngine()
        X_vk, vk_names = vk.compute_features(smiles_list)
        branches.append(X_vk.to_numpy(dtype=np.float64))
        names.extend(vk_names)

    if numeric_cols:
        numeric = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            numeric[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        branches.append(numeric.to_numpy(dtype=np.float64))
        names.extend(numeric_cols)

    if custom_rules:
        from ml.custom_features import CustomFeatureRule, evaluate_custom_features

        rules = [
            CustomFeatureRule(
                name=r.get("name", ""),
                rule_type=r.get("rule_type", r.get("ruleType", "formula")),
                expression=r.get("expression", ""),
                params=r.get("params", {}),
            )
            for r in custom_rules
        ]
        X_custom, custom_names = evaluate_custom_features(df, rules, smiles_col)
        if X_custom.shape[1]:
            branches.append(X_custom)
            names.extend(custom_names)

    raw = np.hstack(branches) if branches else np.empty((len(df), 0))
    raw[~np.isfinite(raw)] = np.nan
    index = {name: i for i, name in enumerate(names)}
    aligned = np.full((len(df), len(requested_names)), np.nan, dtype=np.float64)
    missing: list[str] = []
    for out_idx, name in enumerate(requested_names):
        source_idx = index.get(name)
        if source_idx is None:
            missing.append(name)
        else:
            aligned[:, out_idx] = raw[:, source_idx]
    return aligned, missing
