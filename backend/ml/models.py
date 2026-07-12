"""
Model definitions for polymer AutoML with Optuna parameter spaces.
11 models suited for small-sample polymer data.
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSCanonical
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb


# Each model definition: name, estimator class, param_space_fn, needs_scaling
MODEL_DEFINITIONS = {
    "ridge": {
        "name": "Ridge Regression",
        "description": "线性回归 + L2正则化，适合小样本",
        "tags": ["线性", "小样本", "快速"],
        "estimator_class": Ridge,
        "default_params": {},
        "param_space": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        },
        "needs_scaling": True,
    },
    "lasso": {
        "name": "LASSO",
        "description": "线性回归 + L1正则化，自动特征选择",
        "tags": ["线性", "特征选择"],
        "estimator_class": Lasso,
        "default_params": {"max_iter": 5000},
        "param_space": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
        },
        "needs_scaling": True,
    },
    "elasticnet": {
        "name": "ElasticNet",
        "description": "L1+L2混合正则化，平衡特征选择与稳定性",
        "tags": ["线性", "特征选择"],
        "estimator_class": ElasticNet,
        "default_params": {"max_iter": 5000},
        "param_space": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        },
        "needs_scaling": True,
    },
    "pls": {
        "name": "PLS",
        "description": "偏最小二乘法，适合高维小样本",
        "tags": ["线性", "小样本", "高维"],
        "estimator_class": PLSCanonical,
        "default_params": {},
        "param_space": lambda trial: {
            "n_components": trial.suggest_int("n_components", 2, 10),
        },
        "needs_scaling": True,
    },
    "knn": {
        "name": "KNN",
        "description": "K近邻回归，非参数方法",
        "tags": ["非参数", "简单"],
        "estimator_class": KNeighborsRegressor,
        "default_params": {},
        "param_space": lambda trial: {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 15),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        },
        "needs_scaling": True,
    },
    "kernel_ridge": {
        "name": "Kernel Ridge",
        "description": "核岭回归，结合SVM和Ridge的优点",
        "tags": ["核方法", "小样本", "不确定性"],
        "estimator_class": KernelRidge,
        "default_params": {},
        "param_space": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "polynomial"]),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        },
        "needs_scaling": True,
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "随机森林集成，鲁棒性好",
        "tags": ["集成", "鲁棒", "可解释"],
        "estimator_class": RandomForestRegressor,
        "default_params": {"random_state": 42, "n_jobs": 1},
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        },
        "needs_scaling": False,
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "梯度提升集成，表格数据表现优异",
        "tags": ["集成", "强学习器"],
        "estimator_class": GradientBoostingRegressor,
        "default_params": {"random_state": 42},
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        },
        "needs_scaling": False,
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "极端梯度提升，竞赛常胜将军",
        "tags": ["集成", "强学习器"],
        "estimator_class": xgb.XGBRegressor,
        "default_params": {"random_state": 42, "verbosity": 0, "n_jobs": 1},
        "param_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        },
        "needs_scaling": False,
    },
    "svm": {
        "name": "SVM (RBF Kernel)",
        "description": "支持向量机，适合中小样本非线性问题",
        "tags": ["核方法", "非线性"],
        "estimator_class": SVR,
        "default_params": {},
        "param_space": lambda trial: {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1e0, log=True),
        },
        "needs_scaling": True,
    },
    "gaussian_process": {
        "name": "Gaussian Process",
        "description": "高斯过程回归，自带不确定性估计，小样本首选",
        "tags": ["小样本", "不确定性", "贝叶斯"],
        "estimator_class": GaussianProcessRegressor,
        "default_params": {"random_state": 42, "normalize_y": True},
        "param_space": lambda trial: {
            "kernel_type": trial.suggest_categorical("kernel_type", ["RBF", "Matern", "RQ"]),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-2, log=True),
        },
        "needs_scaling": True,
    },
    "mlp": {
        "name": "MLP",
        "description": "多层感知机神经网络",
        "tags": ["神经网络", "非线性"],
        "estimator_class": MLPRegressor,
        "default_params": {"random_state": 42, "max_iter": 1000},
        "param_space": lambda trial: {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [
                (32,), (64,), (32, 16), (64, 32), (64, 32, 16)
            ]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        },
        "needs_scaling": True,
    },
}


def create_estimator(model_key: str, params: dict):
    """
    Instantiate an estimator given model_key and hyperparameters.
    For Gaussian Process, resolves kernel from params.
    """
    defn = MODEL_DEFINITIONS[model_key]
    merged = {**defn["default_params"], **params}

    if model_key == "gaussian_process":
        kernel_type = merged.pop("kernel_type", "RBF")
        if kernel_type == "RBF":
            kernel = RBF()
        elif kernel_type == "Matern":
            kernel = Matern()
        else:
            kernel = RationalQuadratic()
        merged["kernel"] = kernel

    return defn["estimator_class"](**merged)


# Mapping of model keys to model names for display
MODEL_DISPLAY_NAMES = {k: v["name"] for k, v in MODEL_DEFINITIONS.items()}
