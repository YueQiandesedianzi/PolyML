"""Quick E2E training test - runs pipeline directly (bypasses HTTP)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from pathlib import Path
from ml.training import run_automl_pipeline

# Generate synthetic test data instead of loading from a specific project path
rng = np.random.RandomState(42)
X = rng.randn(100, 20)
y = X @ rng.randn(20) + rng.randn(100) * 0.5
print(f"X: {X.shape}, y: {y.shape}")
print(f"y range: [{y.min():.1f}, {y.max():.1f}]")

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
X_clean = imp.fit_transform(X)
print(f"After imputation: {X_clean.shape}, NaN remaining: {np.isnan(X_clean).sum()}")

for event in run_automl_pipeline(
    X_clean, y,
    selected_models=["ridge", "random_forest", "xgboost"],
    cv_folds=5,
    n_trials=10,
    test_size=0.2,
):
    if event["type"] == "model_start":
        print(f"\n>> Training {event['data']['model']}...")
    elif event["type"] == "model_complete":
        d = event["data"]
        print(f"   R2={d['r2']:.3f}  RMSE={d['rmse']:.2f}  MAE={d['mae']:.2f}")
    elif event["type"] == "all_complete":
        print(f"\n>> Best model: {event['data']['best_model']}")
        for k, v in event["data"]["results"].items():
            print(f"   {k}: R2={v['test_r2']:.3f}  RMSE={v['test_rmse']:.2f}")
        break
    elif event["type"] == "error":
        print(f"   ERROR: {event['data']}")
