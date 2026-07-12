import numpy as np

from ml.training import run_automl_pipeline


def test_best_model_is_selected_by_inner_cv_and_test_is_reported_once():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.05, size=40)
    events = list(run_automl_pipeline(
        X, y, ["ridge", "random_forest"], cv_folds=3, n_trials=2,
        test_size=0.2, selection_metric="rmse", random_state=11,
    ))
    complete = next(event for event in events if event["type"] == "all_complete")
    results = complete["data"]["results"]
    best = complete["data"]["best_model"]
    assert best == min(results, key=lambda key: results[key]["cv_loss"])
    assert "test_r2" in results[best]
    assert sum("test_r2" in result for result in results.values()) == 1


def test_group_split_keeps_groups_disjoint(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=30)
    groups = np.repeat(np.arange(10), 3)
    list(run_automl_pipeline(
        X, y, ["ridge"], cv_folds=3, n_trials=1, test_size=0.3,
        split_strategy="group", groups=groups, project_dir=str(tmp_path), run_id="grouped",
    ))
    split = np.load(tmp_path / "split_grouped.npz")
    assert set(groups[split["train_indices"]]).isdisjoint(set(groups[split["test_indices"]]))
