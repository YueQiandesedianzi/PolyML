"""
Feature engineering orchestrator for polymer materials data.
Combines RDKit descriptors, Van Krevelen group contributions,
and processing condition features.
"""

from typing import Optional
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


def engineer_features(
    df: pd.DataFrame,
    smiles_col: str,
    numeric_cols: list[str],
    target_col: str,
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

    y = df[target_col].values.astype(np.float64)
    result.y = y

    all_features = []
    all_names = []

    # Branch 1: RDKit molecular descriptors
    if include_descriptors and smiles_list:
        calc = RDKitDescriptorCalculator()
        X_desc, desc_names, failed = calc.compute(smiles_list)
        result.rdkit_failures = failed

        # Clean up inf/nan
        X_desc, desc_names_cleaned, dropped_inf = calc.remove_inf_nan_columns(X_desc, desc_names)
        result.dropped_inf_nan.extend(dropped_inf)

        # Filter low variance
        X_desc, desc_names_final, dropped_var = calc.filter_low_variance(X_desc, desc_names_cleaned)
        result.dropped_low_variance.extend(dropped_var)

        result.n_descriptors = len(desc_names_final)
        all_features.append(X_desc)
        all_names.extend(desc_names_final)

    # Branch 2: Van Krevelen group contributions
    if include_van_krevelen and smiles_list:
        vk = VanKrevelenEngine()
        X_vk, vk_names = vk.compute_features(smiles_list)
        X_vk_arr = X_vk.values.astype(np.float64)
        result.n_van_krevelen = len(vk_names)
        all_features.append(X_vk_arr)
        all_names.extend(vk_names)

    # Branch 3: Processing conditions
    if numeric_cols:
        X_proc, proc_names, proc_imputer = encode_processing_conditions(df, numeric_cols)
        result.n_processing = len(proc_names)
        if X_proc.shape[1] > 0:
            all_features.append(X_proc)
            all_names.extend(proc_names)

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

    # Fill any remaining NaN with 0
    X_combined = np.nan_to_num(X_combined, nan=0.0)

    result.X = X_combined
    result.feature_names = all_names

    # Save preprocessor state for later prediction
    result.preprocessor_bytes = pickle.dumps({
        "descriptors": include_descriptors,
        "van_krevelen": include_van_krevelen,
        "numeric_cols": numeric_cols,
        "feature_names": all_names,
        "n_features": X_combined.shape[1],
    })

    return result
