"""
Active Learning with Bayesian Optimization for polymer materials.
Uses GPR surrogate model with acquisition functions (EI, UCB, PI).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import norm


class BayesianOptimizer:
    """Bayesian optimization for active learning — suggests next experiments to run."""

    def __init__(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        feature_names: list[str],
        kernel=None,
        random_state: int = 42,
    ):
        self.X_raw = np.asarray(X_labeled, dtype=float)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.X_labeled = self.scaler.fit_transform(self.imputer.fit_transform(self.X_raw))
        self.y_labeled = y_labeled
        self.feature_names = feature_names
        self.kernel = kernel or Matern(
            length_scale=np.ones(self.X_labeled.shape[1]),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        self.rng = np.random.default_rng(random_state)

        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=0,
            normalize_y=True,
            random_state=random_state,
        )
        self.gpr.fit(self.X_labeled, self.y_labeled)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions."""
        X_scaled = self.scaler.transform(self.imputer.transform(X))
        return self.gpr.predict(X_scaled, return_std=True)

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        X_scaled = self.scaler.transform(self.imputer.transform(X))
        mu, sigma = self.gpr.predict(X_scaled, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        y_best = np.max(self.y_labeled)
        z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def upper_confidence_bound(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """UCB acquisition function."""
        X_scaled = self.scaler.transform(self.imputer.transform(X))
        mu, sigma = self.gpr.predict(X_scaled, return_std=True)
        return mu + beta * sigma

    def probability_of_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """PI acquisition function."""
        X_scaled = self.scaler.transform(self.imputer.transform(X))
        mu, sigma = self.gpr.predict(X_scaled, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        y_best = np.max(self.y_labeled)
        z = (mu - y_best - xi) / sigma
        return norm.cdf(z)

    def suggest_next(
        self,
        X_candidates: np.ndarray,
        acquisition: str = "ei",
        n_suggestions: int = 1,
        xi: float = 0.01,
        beta: float = 2.0,
    ) -> list[dict]:
        """Suggest next experiments from a candidate pool."""
        if acquisition == "ei":
            scores = self.expected_improvement(X_candidates, xi=xi)
        elif acquisition == "ucb":
            scores = self.upper_confidence_bound(X_candidates, beta=beta)
        elif acquisition == "pi":
            scores = self.probability_of_improvement(X_candidates, xi=xi)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")

        top_idx = np.argsort(scores)[-n_suggestions:][::-1]

        suggestions = []
        for idx in top_idx:
            suggestions.append({
                "index": int(idx),
                "features": X_candidates[idx].tolist(),
                "acquisition_score": float(scores[idx]),
                "predicted_y": float(
                    self.gpr.predict(
                        self.scaler.transform(self.imputer.transform(X_candidates[idx:idx+1]))
                    )[0]
                ),
                "predicted_std": float(
                    self.gpr.predict(
                        self.scaler.transform(self.imputer.transform(X_candidates[idx:idx+1])),
                        return_std=True,
                    )[1][0]
                ),
            })

        return suggestions

    def cross_validate(self, n_folds: int = 5) -> dict:
        """Simple CV to evaluate GPR quality."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=min(n_folds, len(self.X_raw)), shuffle=True, random_state=42)
        y_true_all, y_pred_all = [], []

        for train_idx, val_idx in kf.split(self.X_raw):
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(imputer.fit_transform(self.X_raw[train_idx]))
            X_val = scaler.transform(imputer.transform(self.X_raw[val_idx]))
            gpr = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, random_state=42)
            gpr.fit(X_train, self.y_labeled[train_idx])
            y_pred = gpr.predict(X_val)
            y_true_all.extend(self.y_labeled[val_idx])
            y_pred_all.extend(y_pred)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        return {
            "r2": round(float(r2_score(y_true_all, y_pred_all)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))), 4),
            "mae": round(float(mean_absolute_error(y_true_all, y_pred_all)), 4),
            "n_samples": len(self.X_raw),
        }

    def feature_importance(self) -> list[dict]:
        """Estimate feature importance from GPR kernel lengthscales."""
        lengthscales = self.gpr.kernel_.length_scale
        if np.isscalar(lengthscales):
            lengthscales = np.full(len(self.feature_names), lengthscales)

        # Shorter lengthscale = more important
        importance = 1.0 / lengthscales
        importance = importance / importance.sum()

        return sorted(
            [{"name": n, "importance": round(float(v), 4)}
             for n, v in zip(self.feature_names, importance)],
            key=lambda x: x["importance"],
            reverse=True,
        )
