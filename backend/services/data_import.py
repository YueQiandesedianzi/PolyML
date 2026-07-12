"""CSV/Excel parsing and column type detection service"""

import re
import io
from pathlib import Path
from typing import Optional
import pandas as pd

# Heuristic patterns
SMILES_NAME_HINTS = re.compile(r'smiles|molecule|structure|formula|compound', re.IGNORECASE)
TARGET_NAME_HINTS = re.compile(r'target|property|result|y\b|output|response', re.IGNORECASE)
NUMERIC_NAMES = re.compile(r'mn|mw|mz|pdi|molecular.?weight|temperature|temp|pressure|time|ratio|fraction|concentration|percent|density|thickness', re.IGNORECASE)
# Patterns that indicate non-target columns (IDs, labels, categories)
NON_TARGET_HINTS = re.compile(r'id$|_id|label|quality|category|type|name|code|group|class', re.IGNORECASE)


def parse_file(file_path: str, file_content: Optional[bytes] = None) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if file_content:
        buf = io.BytesIO(file_content)
        if suffix == '.xlsx':
            return pd.read_excel(buf, engine='openpyxl')
        if suffix == '.xls':
            return pd.read_excel(buf)
        else:
            return pd.read_csv(buf)

    if suffix == '.xlsx':
        return pd.read_excel(str(path), engine='openpyxl')
    if suffix == '.xls':
        return pd.read_excel(str(path))
    else:
        return pd.read_csv(str(path))


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Return a dict of {col_name: type} where type is one of:
    'smiles', 'numeric', 'target', 'ignore'
    """
    result: dict[str, str] = {}

    for col in df.columns:
        col_str = str(col).lower()

        # 1. Check name hint for SMILES
        if SMILES_NAME_HINTS.search(col_str):
            result[col] = 'smiles'
            continue

        # 2. Check name hint for target — only if numeric with >1 unique value
        if TARGET_NAME_HINTS.search(col_str):
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
                result[col] = 'target'
            else:
                result[col] = 'ignore'
            continue

        # 3. Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check name hint for processing params
            if NUMERIC_NAMES.search(col_str):
                result[col] = 'numeric'
                continue

            # If low cardinality with numeric, it could be categorical
            n_unique = df[col].nunique()
            if n_unique <= 10 and n_unique < len(df) * 0.1:
                result[col] = 'ignore'
                continue

            result[col] = 'numeric'
            continue

        # 4. Check if string values look like SMILES
        sample = df[col].dropna().head(20).astype(str)
        smiles_count = 0
        for val in sample:
            smiles_count += _looks_like_smiles(val)

        if smiles_count > len(sample) * 0.5:
            result[col] = 'smiles'
            continue

        # 5. Otherwise ignore
        result[col] = 'ignore'

    return result


def _looks_like_smiles(s: str) -> bool:
    """Very rough heuristic for SMILES-like strings."""
    if not s or len(s) < 2:
        return False
    # SMILES-specific characters (beyond alphanumeric): =, #, @, [, ], (, ), /, \
    smiles_chars = set('=#@[]()/\\')
    has_smiles_char = any(c in smiles_chars for c in s)
    if not has_smiles_char:
        return False
    # SMILES contain: letters, digits, =, #, @, [, ], (, ), /, \, +, -
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=#@[]()/\\:;+-')
    ratio = sum(1 for c in s if c in allowed) / len(s)
    return ratio > 0.7


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return data quality summary."""
    return {
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': [
            {
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].notna().sum()),
                'missing': int(df[col].isna().sum()),
                'n_unique': int(df[col].nunique()),
            }
            for col in df.columns
        ],
        'total_missing': int(df.isna().sum().sum()),
    }
