"""RDKit molecular descriptor calculator — robust version using individual functions."""

import numpy as np
from typing import List, Tuple
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# Suppress RDKit warnings globally
RDLogger.logger().setLevel(RDLogger.ERROR)

# Safe individual descriptor functions — avoids MoleculeDescriptors.MolecularDescriptorCalculator crash
DESCRIPTOR_FUNCTIONS = {
    # Constitutional
    "MolWt": Descriptors.MolWt,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "ExactMolWt": Descriptors.ExactMolWt,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    # Electronic / EState
    "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex,
    "MaxEStateIndex": Descriptors.MaxEStateIndex,
    "MinAbsEStateIndex": Descriptors.MinAbsEStateIndex,
    "MinEStateIndex": Descriptors.MinEStateIndex,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
    "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
    # Topological
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "Chi0": Descriptors.Chi0,
    "Chi0n": Descriptors.Chi0n,
    "Chi0v": Descriptors.Chi0v,
    "Chi1": Descriptors.Chi1,
    "Chi1n": Descriptors.Chi1n,
    "Chi1v": Descriptors.Chi1v,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "LabuteASA": Descriptors.LabuteASA,
    "PEOE_VSA1": Descriptors.PEOE_VSA1,
    "PEOE_VSA2": Descriptors.PEOE_VSA2,
    "PEOE_VSA3": Descriptors.PEOE_VSA3,
    "PEOE_VSA4": Descriptors.PEOE_VSA4,
    "PEOE_VSA5": Descriptors.PEOE_VSA5,
    "PEOE_VSA6": Descriptors.PEOE_VSA6,
    "PEOE_VSA7": Descriptors.PEOE_VSA7,
    "PEOE_VSA8": Descriptors.PEOE_VSA8,
    "PEOE_VSA9": Descriptors.PEOE_VSA9,
    "PEOE_VSA10": Descriptors.PEOE_VSA10,
    "PEOE_VSA11": Descriptors.PEOE_VSA11,
    "PEOE_VSA12": Descriptors.PEOE_VSA12,
    "PEOE_VSA13": Descriptors.PEOE_VSA13,
    "PEOE_VSA14": Descriptors.PEOE_VSA14,
    "SMR_VSA1": Descriptors.SMR_VSA1,
    "SMR_VSA2": Descriptors.SMR_VSA2,
    "SMR_VSA3": Descriptors.SMR_VSA3,
    "SMR_VSA4": Descriptors.SMR_VSA4,
    "SMR_VSA5": Descriptors.SMR_VSA5,
    "SMR_VSA6": Descriptors.SMR_VSA6,
    "SMR_VSA7": Descriptors.SMR_VSA7,
    "SMR_VSA8": Descriptors.SMR_VSA8,
    "SMR_VSA9": Descriptors.SMR_VSA9,
    "SMR_VSA10": Descriptors.SMR_VSA10,
    "SlogP_VSA1": Descriptors.SlogP_VSA1,
    "SlogP_VSA2": Descriptors.SlogP_VSA2,
    "SlogP_VSA3": Descriptors.SlogP_VSA3,
    "SlogP_VSA4": Descriptors.SlogP_VSA4,
    "SlogP_VSA5": Descriptors.SlogP_VSA5,
    "SlogP_VSA6": Descriptors.SlogP_VSA6,
    "SlogP_VSA7": Descriptors.SlogP_VSA7,
    "SlogP_VSA8": Descriptors.SlogP_VSA8,
    "SlogP_VSA9": Descriptors.SlogP_VSA9,
    "SlogP_VSA10": Descriptors.SlogP_VSA10,
    "SlogP_VSA11": Descriptors.SlogP_VSA11,
    "SlogP_VSA12": Descriptors.SlogP_VSA12,
    "EState_VSA1": Descriptors.EState_VSA1,
    "EState_VSA2": Descriptors.EState_VSA2,
    "EState_VSA3": Descriptors.EState_VSA3,
    "EState_VSA4": Descriptors.EState_VSA4,
    "EState_VSA5": Descriptors.EState_VSA5,
    "EState_VSA6": Descriptors.EState_VSA6,
    "EState_VSA7": Descriptors.EState_VSA7,
    "EState_VSA8": Descriptors.EState_VSA8,
    "EState_VSA9": Descriptors.EState_VSA9,
    "EState_VSA10": Descriptors.EState_VSA10,
    "EState_VSA11": Descriptors.EState_VSA11,
    "VSA_EState1": Descriptors.VSA_EState1,
    "VSA_EState2": Descriptors.VSA_EState2,
    "VSA_EState3": Descriptors.VSA_EState3,
    "VSA_EState4": Descriptors.VSA_EState4,
    "VSA_EState5": Descriptors.VSA_EState5,
    "VSA_EState6": Descriptors.VSA_EState6,
    "VSA_EState7": Descriptors.VSA_EState7,
    "VSA_EState8": Descriptors.VSA_EState8,
    "VSA_EState9": Descriptors.VSA_EState9,
    "VSA_EState10": Descriptors.VSA_EState10,
    # Lipinski / MOE
    "TPSA": Descriptors.TPSA,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    # Fingerprint density
    "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
    "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
    "FpDensityMorgan3": Descriptors.FpDensityMorgan3,
}


class RDKitDescriptorCalculator:
    """Compute molecular descriptors for a list of SMILES strings, one function at a time."""

    def __init__(self):
        self.descriptor_names = list(DESCRIPTOR_FUNCTIONS.keys())

    def compute(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Compute descriptors row by row, function by function (avoids MoleculeDescriptors crash).

        Returns:
            X: (n_samples, n_descriptors) feature matrix
            names: descriptor names
            failed_indices: indices of rows where SMILES parsing failed
        """
        n_desc = len(self.descriptor_names)
        n_rows = len(smiles_list)
        X = np.full((n_rows, n_desc), np.nan, dtype=np.float64)
        failed_indices = []

        # Parse all SMILES first
        mols = []
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                failed_indices.append(idx)
                mols.append(None)
            else:
                mols.append(mol)

        # Compute each descriptor across all valid molecules
        for d_idx, (desc_name, desc_fn) in enumerate(DESCRIPTOR_FUNCTIONS.items()):
            for r_idx, mol in enumerate(mols):
                if mol is None:
                    continue
                try:
                    val = desc_fn(mol)
                    if np.isfinite(val):
                        X[r_idx, d_idx] = val
                except Exception:
                    pass  # leave as NaN

        return X, self.descriptor_names, failed_indices

    @staticmethod
    def remove_inf_nan_columns(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Remove columns that are all NaN or all inf."""
        n_rows = X.shape[0]
        valid_mask = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            col = X[:, i]
            valid_col = col[np.isfinite(col)]
            # Drop if: all NaN, or zero variance among valid values
            if len(valid_col) == 0:
                valid_mask[i] = False
            elif len(valid_col) > 1 and np.std(valid_col) < 1e-10:
                valid_mask[i] = False

        dropped = [n for n, m in zip(names, ~valid_mask) if not m]
        filtered_names = [n for n, m in zip(names, valid_mask) if m]
        return X[:, valid_mask], filtered_names, dropped

    @staticmethod
    def filter_low_variance(X: np.ndarray, names: List[str], threshold: float = 1e-4) -> Tuple[np.ndarray, List[str], List[str]]:
        """Remove near-zero variance columns (ignoring NaN)."""
        variances = np.nanvar(X, axis=0)
        mask = variances > threshold
        dropped = [n for n, m in zip(names, ~mask) if not m]
        filtered_names = [n for n, m in zip(names, mask) if m]
        return X[:, mask], filtered_names, dropped
