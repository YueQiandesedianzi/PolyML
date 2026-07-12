"""Van Krevelen group contribution engine for polymer Tg prediction"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# Default group contribution table (SMARTS → Yg value)
# Reference: D.W. van Krevelen, "Properties of Polymers", 4th ed., Elsevier (2009)
DEFAULT_VK_GROUPS: Dict[str, Dict[str, Any]] = {
    "methylene_CH2": {
        "smarts": "[CH2;D2;!$(C-C(=O))]",
        "Yg": 2700,
        "name": "亚甲基 -CH2-",
        "description": "Chain methylene group"
    },
    "methine_CH": {
        "smarts": "[CH;D3]([#6])[#6]",
        "Yg": 8000,
        "name": "次甲基 >CH-",
        "description": "Methine group"
    },
    "quaternary_C": {
        "smarts": "[C;D4]([#6])([#6])[#6]",
        "Yg": 14000,
        "name": "季碳 >C<",
        "description": "Quaternary carbon"
    },
    "p_phenylene": {
        "smarts": "c1ccccc1",
        "Yg": 32000,
        "name": "对苯基 -C6H4-",
        "description": "p-Phenylene ring"
    },
    "ether_O": {
        "smarts": "[OD2]([#6])[#6]",
        "Yg": 4000,
        "name": "醚键 -O-",
        "description": "Ether oxygen"
    },
    "ester_COO": {
        "smarts": "[C](=[O])[O][#6]",
        "Yg": 12000,
        "name": "酯基 -COO-",
        "description": "Ester group"
    },
    "ketone_C=O": {
        "smarts": "[C](=[O])[#6]",
        "Yg": 17000,
        "name": "酮基 >C=O",
        "description": "Ketone group"
    },
    "amide_NH": {
        "smarts": "[NH1][#6]",
        "Yg": 8000,
        "name": "酰胺 NH",
        "description": "Amide N-H"
    },
    "tertiary_N": {
        "smarts": "[ND3]([#6])[#6]",
        "Yg": 10000,
        "name": "叔胺 >N-",
        "description": "Tertiary amine"
    },
    "CF2": {
        "smarts": "[C;D2]([F])[F]",
        "Yg": 12000,
        "name": "二氟亚甲基 -CF2-",
        "description": "Difluoromethylene"
    },
    "methyl_side": {
        "smarts": "[CH3;D1][#6]",
        "Yg": 2000,
        "name": "甲基侧基 -CH3",
        "description": "Methyl side group"
    },
    "chloro": {
        "smarts": "[Cl]",
        "Yg": 5000,
        "name": "氯原子 -Cl",
        "description": "Chlorine substituent"
    },
    "cyano_CN": {
        "smarts": "[C]#N",
        "Yg": 25000,
        "name": "氰基 -C≡N",
        "description": "Cyano group"
    },
    "carboxylic_acid": {
        "smarts": "[C](=[O])[OH]",
        "Yg": 20000,
        "name": "羧基 -COOH",
        "description": "Carboxylic acid"
    },
    "hydroxyl_OH": {
        "smarts": "[OH]",
        "Yg": 15000,
        "name": "羟基 -OH",
        "description": "Hydroxyl group"
    },
    "imidazole": {
        "smarts": "c1cncn1",
        "Yg": 30000,
        "name": "咪唑环",
        "description": "Imidazole ring"
    },
    "sulfone_SO2": {
        "smarts": "[S](=[O])(=O)",
        "Yg": 16000,
        "name": "砜基 -SO2-",
        "description": "Sulfone group"
    },
    "thioether_S": {
        "smarts": "[SD2]([#6])[#6]",
        "Yg": 8000,
        "name": "硫醚 -S-",
        "description": "Thioether"
    },
    "anhydride": {
        "smarts": "[C](=O)[O][C](=O)",
        "Yg": 18000,
        "name": "酸酐 -CO-O-CO-",
        "description": "Anhydride"
    },
    "isocyanate": {
        "smarts": "N=C=O",
        "Yg": 22000,
        "name": "异氰酸酯 -NCO",
        "description": "Isocyanate"
    },
    "siloxane": {
        "smarts": "[Si]([O])([O])",
        "Yg": 6000,
        "name": "硅氧烷 -Si-O-",
        "description": "Siloxane linkage"
    },
    "urethane": {
        "smarts": "[NH][C](=O)[O]",
        "Yg": 15000,
        "name": "氨基甲酸酯 -NHCOO-",
        "description": "Urethane group"
    },
    "urea": {
        "smarts": "[NH][C](=O)[NH]",
        "Yg": 16000,
        "name": "脲基 -NHCONH-",
        "description": "Urea group"
    },
    "methylene_bridged": {
        "smarts": "[CH2]([#6])([#6])([#6])",
        "Yg": 2700,
        "name": "桥接亚甲基",
        "description": "Branched methylene"
    },
    "epoxide": {
        "smarts": "C1OC1",
        "Yg": 20000,
        "name": "环氧基",
        "description": "Epoxide ring"
    },
    "trifluoromethyl": {
        "smarts": "C(F)(F)F",
        "Yg": 18000,
        "name": "三氟甲基 -CF3",
        "description": "Trifluoromethyl"
    },
    "furan": {
        "smarts": "c1ccoc1",
        "Yg": 28000,
        "name": "呋喃环",
        "description": "Furan ring"
    },
    "pyridine": {
        "smarts": "c1ccncc1",
        "Yg": 32000,
        "name": "吡啶环",
        "description": "Pyridine ring"
    },
}


class VanKrevelenEngine:
    """
    Van Krevelen group contribution feature extractor for polymers.
    Computes group counts and derived Yg-based features.
    """

    def __init__(self, groups_db: Optional[Dict[str, Dict[str, Any]]] = None):
        self.groups = groups_db or DEFAULT_VK_GROUPS
        self._compiled = {}
        for key, info in self.groups.items():
            mol = Chem.MolFromSmarts(info["smarts"])
            if mol is not None:
                self._compiled[key] = mol

    def compute_features(self, smiles_list: List[str]) -> tuple[pd.DataFrame, List[str]]:
        """
        Compute Van Krevelen group contribution features.

        Returns:
            DataFrame with group counts and Yg-derived features
            List of feature names
        """
        rows = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            row = {}

            if mol is None:
                # All zeros for failed SMILES
                for key in self._compiled:
                    row[f"vk_count_{key}"] = 0
                    row[f"vk_Yg_{key}"] = 0.0
                row["vk_M0"] = 0.0
                row["vk_Yg_total"] = 0.0
                rows.append(row)
                continue

            Yg_total = 0.0

            for key, pattern in self._compiled.items():
                try:
                    matches = mol.GetSubstructMatches(pattern, uniquify=True)
                    count = len(matches)
                    Yg = self.groups[key]["Yg"]
                    row[f"vk_count_{key}"] = count
                    row[f"vk_Yg_{key}"] = float(count * Yg)
                    Yg_total += count * Yg
                except Exception:
                    row[f"vk_count_{key}"] = 0
                    row[f"vk_Yg_{key}"] = 0.0

            try:
                M0 = Descriptors.MolWt(mol)
            except Exception:
                M0 = 0.0

            row["vk_M0"] = M0
            row["vk_Yg_total"] = Yg_total
            rows.append(row)

        df = pd.DataFrame(rows)
        return df, list(df.columns)

    @classmethod
    def load_from_json(cls, json_path: str) -> "VanKrevelenEngine":
        """Load a custom group contribution table from JSON file."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(groups_db=data)

    def export_to_json(self, json_path: str):
        """Export current group table to JSON."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.groups, f, indent=2, ensure_ascii=False)
