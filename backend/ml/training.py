"""
AutoML training pipeline with Optuna hyperparameter optimization.
K-fold cross-validation with SSE progress streaming.
"""

import time
import threading
import numpy as np
import joblib
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ml.models import MODEL_DEFINITIONS, create_estimator

# Suppress Optuna logs to stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_automl_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    selected_models: list[str],
    cv_folds: int = 5,
    cv_method: str = "kfold",
    n_trials: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    cancel_event: threading.Event | None = None,
    project_dir: str | None = None,
    run_id: str = "",
):
    """
    Run AutoML for selected models, yielding SSE event dicts.

    Events:
        model_start, trial_update, model_complete, all_complete, error, cancelled
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Persist train/test split for later SHAP and prediction use
    if project_dir and run_id:
        split_path = Path(project_dir) / f"split_{run_id}.npz"
        np.savez_compressed(split_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    results = {}
    total_start = time.time()

    for model_idx, model_key in enumerate(selected_models):
        if cancel_event and cancel_event.is_set():
            yield {"type": "cancelled", "data": {"message": "Training cancelled by user"}}
            return

        yield {"type": "model_start", "data": {
            "model": model_key,
            "total_models": len(selected_models),
            "current_model": model_idx + 1,
        }}

        defn = MODEL_DEFINITIONS[model_key]
        model_start = time.time()

        def objective(trial):
            params = defn["param_space"](trial)
            est = create_estimator(model_key, params)

            pipeline_steps = []
            if defn["needs_scaling"]:
                pipeline_steps.append(("scaler", StandardScaler()))
            pipeline_steps.append(("estimator", est))
            pipe = Pipeline(pipeline_steps)

            # Select cross-validation method
            if cv_method == "loocv":
                cv = LeaveOneOut()
            elif cv_method == "repeated_kfold":
                cv = RepeatedKFold(n_splits=cv_folds, n_repeats=3, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=1,
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except optuna.exceptions.OptunaError as e:
            yield {"type": "error", "data": {"model": model_key, "message": str(e)}}
            continue

        yield {"type": "trial_update", "data": {
            "model": model_key,
            "trial": n_trials,
            "total_trials": n_trials,
            "best_mse": study.best_value,
            "elapsed_sec": round(time.time() - model_start, 1),
        }}

        # Train final model with best params on all training data
        try:
            best_est = create_estimator(model_key, study.best_params)

            pipeline_steps = []
            if defn["needs_scaling"]:
                pipeline_steps.append(("scaler", StandardScaler()))
            pipeline_steps.append(("estimator", best_est))
            final_pipeline = Pipeline(pipeline_steps)
            final_pipeline.fit(X_train, y_train)
            y_pred = final_pipeline.predict(X_test)
        except Exception as e:
            yield {"type": "error", "data": {"model": model_key, "message": f"Final training failed: {e}"}}
            continue

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[model_key] = {
            "model_name": defn["name"],
            "best_params": study.best_params,
            "cv_mse": study.best_value,
            "test_rmse": round(float(rmse), 6),
            "test_r2": round(float(r2), 6),
            "test_mae": round(float(mae), 6),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "pipeline": final_pipeline,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
        }

        # Persist pipeline to disk for later use by prediction and SHAP
        if project_dir and run_id:
            pipeline_path = Path(project_dir) / f"pipeline_{model_key}_{run_id}.joblib"
            joblib.dump(final_pipeline, pipeline_path)

        yield {"type": "model_complete", "data": {
            "model": model_key,
            "rmse": round(float(rmse), 6),
            "r2": round(float(r2), 6),
            "mae": round(float(mae), 6),
            "duration_sec": round(time.time() - model_start, 1),
        }}

    # Select best model by test R2
    if not results:
        yield {"type": "error", "data": {"message": "All models failed"}}
        return

    best_key = max(results, key=lambda k: results[k]["test_r2"])
    best_result = results[best_key]

    # Extract feature names for SHAP
    best_pipeline = best_result["pipeline"]

    yield {"type": "all_complete", "data": {
        "best_model": best_key,
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("pipeline", "y_test", "y_pred", "X_train", "y_train", "X_test")}
            for k, v in results.items()
        },
        "total_duration_sec": round(time.time() - total_start, 1),
    }}
