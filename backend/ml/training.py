"""AutoML training with fold-local preprocessing and an untouched outer test set."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    TimeSeriesSplit,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.models import MODEL_DEFINITIONS, create_estimator

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _outer_indices(
    n_rows: int,
    test_size: float,
    random_state: int,
    split_strategy: str,
    groups: np.ndarray | None,
    time_values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_rows)
    if split_strategy == "group":
        if groups is None or len(np.unique(groups)) < 2:
            raise ValueError("group split requires at least two distinct groups")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(splitter.split(indices, groups=groups))
    if split_strategy == "time":
        order = np.argsort(time_values, kind="stable") if time_values is not None else indices
        n_test = max(1, int(np.ceil(n_rows * test_size)))
        if n_rows - n_test < 2:
            raise ValueError("time split leaves too few training rows")
        return order[:-n_test], order[-n_test:]
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _inner_cv(
    cv_method: str,
    cv_folds: int,
    random_state: int,
    split_strategy: str,
    groups_train: np.ndarray | None,
    n_train: int,
):
    if split_strategy == "group":
        n_groups = len(np.unique(groups_train)) if groups_train is not None else 0
        if n_groups < 2:
            raise ValueError("group CV requires at least two training groups")
        return GroupKFold(n_splits=min(cv_folds, n_groups))
    if split_strategy == "time":
        return TimeSeriesSplit(n_splits=min(cv_folds, max(2, n_train - 1)))
    if cv_method == "loocv":
        return LeaveOneOut()
    if cv_method == "repeated_kfold":
        return RepeatedKFold(
            n_splits=min(cv_folds, n_train), n_repeats=3, random_state=random_state
        )
    return KFold(
        n_splits=min(cv_folds, n_train), shuffle=True, random_state=random_state
    )


def _scoring(selection_metric: str) -> str:
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }[selection_metric]


def _metric_from_loss(selection_metric: str, loss: float) -> float:
    return -loss if selection_metric == "r2" else loss


def _build_pipeline(model_key: str, params: dict) -> Pipeline:
    definition = MODEL_DEFINITIONS[model_key]
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if definition["needs_scaling"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", create_estimator(model_key, params)))
    return Pipeline(steps)


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
    split_strategy: str = "random",
    groups: np.ndarray | None = None,
    time_values: np.ndarray | None = None,
    selection_metric: str = "rmse",
):
    """Tune on the training partition, select by inner CV, then test once."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y dimensions do not match")
    if len(y) < 5:
        raise ValueError("At least 5 labeled rows are required for AutoML")
    if not np.isfinite(y).all():
        raise ValueError("Target contains missing or non-finite values")
    if selection_metric not in {"rmse", "mae", "r2"}:
        raise ValueError("selection_metric must be rmse, mae, or r2")
    invalid_models = sorted(set(selected_models) - set(MODEL_DEFINITIONS))
    if invalid_models:
        raise ValueError(f"Unknown models: {', '.join(invalid_models)}")

    train_idx, test_idx = _outer_indices(
        len(y), test_size, random_state, split_strategy, groups, time_values
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx] if groups is not None else None

    if project_dir and run_id:
        np.savez_compressed(
            Path(project_dir) / f"split_{run_id}.npz",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_indices=train_idx,
            test_indices=test_idx,
        )

    results: dict[str, dict] = {}
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
        definition = MODEL_DEFINITIONS[model_key]
        model_start = time.time()
        cv = _inner_cv(
            cv_method, cv_folds, random_state, split_strategy, groups_train, len(y_train)
        )

        def objective(trial):
            if cancel_event and cancel_event.is_set():
                raise optuna.TrialPruned()
            params = definition["param_space"](trial)
            if model_key == "pls":
                params["n_components"] = min(
                    int(params["n_components"]), max(1, min(X_train.shape[1], len(y_train) - 1))
                )
            if model_key == "knn":
                min_fold_train = max(1, len(y_train) - int(np.ceil(len(y_train) / max(2, cv_folds))))
                params["n_neighbors"] = min(int(params["n_neighbors"]), min_fold_train)
            pipe = _build_pipeline(model_key, params)
            scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=cv,
                groups=groups_train if split_strategy == "group" else None,
                scoring=_scoring(selection_metric),
                n_jobs=1,
                error_score="raise",
            )
            mean_score = float(np.mean(scores))
            return -mean_score

        study = optuna.create_study(direction="minimize")
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            if not study.trials or study.best_trial is None:
                raise ValueError("No successful optimization trials")
            best_params = dict(study.best_params)
            if model_key == "pls":
                best_params["n_components"] = min(
                    int(best_params["n_components"]), max(1, min(X_train.shape[1], len(y_train) - 1))
                )
            if model_key == "knn":
                best_params["n_neighbors"] = min(int(best_params["n_neighbors"]), max(1, len(y_train) - 1))
            pipeline = _build_pipeline(model_key, best_params)
            pipeline.fit(X_train, y_train)
        except Exception as exc:
            yield {"type": "error", "data": {"model": model_key, "message": str(exc)}}
            continue

        cv_value = _metric_from_loss(selection_metric, float(study.best_value))
        results[model_key] = {
            "model_name": definition["name"],
            "best_params": best_params,
            "selection_metric": selection_metric,
            "cv_score": cv_value,
            "cv_loss": float(study.best_value),
            "pipeline": pipeline,
            "duration_sec": round(time.time() - model_start, 1),
        }
        if project_dir and run_id:
            joblib.dump(pipeline, Path(project_dir) / f"pipeline_{model_key}_{run_id}.joblib")

        yield {"type": "model_complete", "data": {
            "model": model_key,
            "selection_metric": selection_metric,
            "cv_score": round(cv_value, 6),
            "duration_sec": results[model_key]["duration_sec"],
        }}

    if not results:
        yield {"type": "error", "data": {"message": "All models failed"}}
        return

    best_key = min(results, key=lambda key: results[key]["cv_loss"])
    best = results[best_key]
    if split_strategy != "time":
        if split_strategy == "group":
            calibration_cv = GroupKFold(n_splits=min(cv_folds, len(np.unique(groups_train))))
        else:
            calibration_cv = KFold(
                n_splits=min(cv_folds, len(y_train)), shuffle=True, random_state=random_state
            )
        calibration_pipeline = _build_pipeline(best_key, best["best_params"])
        oof_pred = np.asarray(cross_val_predict(
            calibration_pipeline,
            X_train,
            y_train,
            cv=calibration_cv,
            groups=groups_train if split_strategy == "group" else None,
            n_jobs=1,
        )).reshape(-1)
        best["conformal_90_radius"] = float(
            np.quantile(np.abs(y_train - oof_pred), 0.9, method="higher")
        )
        if project_dir and run_id:
            np.savez_compressed(
                Path(project_dir) / f"oof_{run_id}.npz",
                row_indices=train_idx,
                y_true=y_train,
                y_pred=oof_pred,
                residual=y_train - oof_pred,
            )
    y_pred = np.asarray(best["pipeline"].predict(X_test)).reshape(-1)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    best.update({
        "test_rmse": test_rmse,
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "outer_test_size": len(y_test),
    })

    yield {"type": "all_complete", "data": {
        "best_model": best_key,
        "selection_metric": selection_metric,
        "results": {
            key: {k: v for k, v in value.items() if k != "pipeline"}
            for key, value in results.items()
        },
        "outer_test": {
            "r2": best["test_r2"],
            "rmse": best["test_rmse"],
            "mae": best["test_mae"],
            "n": len(y_test),
        },
        "total_duration_sec": round(time.time() - total_start, 1),
    }}
