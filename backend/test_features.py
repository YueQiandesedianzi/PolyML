import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from services.feature_engineering import engineer_features

df = pd.read_csv(Path(__file__).parent / 'test_polymer_data.csv')
print("Loaded", len(df), "rows")
print("SMILES sample:", df['SMILES'].iloc[0])

try:
    result = engineer_features(
        df=df, smiles_col='SMILES', numeric_cols=['Mn'],
        target_col='Tg_degC', include_descriptors=True,
        include_van_krevelen=True, include_3d=False
    )
    print("Feature matrix shape:", result.X.shape)
    print("Feature count:", len(result.feature_names))
    print("RDKit failures:", result.rdkit_failures)
except Exception as e:
    traceback.print_exc()
