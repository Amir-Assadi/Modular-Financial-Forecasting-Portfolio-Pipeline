import os
import joblib
import yaml
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, make_scorer
)
from typing import Any, Dict, Tuple, Optional
from utils.logger import get_logger

# XGBoost support
import xgboost as xgb

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "ridge": {
        "regression": Ridge,
        "classification": LogisticRegression
    },
    "random_forest": {
        "regression": RandomForestRegressor,
        "classification": RandomForestClassifier
    },
    "gradient_boosting": {
        "regression": GradientBoostingRegressor,
        "classification": GradientBoostingClassifier
    },
    "svm": {
        "regression": SVR,
        "classification": SVC
    },
    "xgboost": {
        "regression": xgb.XGBRegressor,
        "classification": xgb.XGBClassifier
    }
}

DEFAULT_RANDOM_STATE = 42
HYPERPARAM_GRIDS = {
    "ridge": {
        "regression": {"alpha": [0.01, 0.1, 1, 10]},
        "classification": {"C": [0.01, 0.1, 1, 10], "random_state": [DEFAULT_RANDOM_STATE]}
    },
    "random_forest": {
        "regression": {"n_estimators": [100, 200], "max_depth": [3, 5, 10], "random_state": [DEFAULT_RANDOM_STATE]},
        "classification": {"n_estimators": [100, 200], "max_depth": [3, 5, 10], "random_state": [DEFAULT_RANDOM_STATE]}
    },
    "gradient_boosting": {
        "regression": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "random_state": [DEFAULT_RANDOM_STATE]},
        "classification": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "random_state": [DEFAULT_RANDOM_STATE]}
    },
    "svm": {
        "regression": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "classification": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "probability": [True]}
    },
    "xgboost": {
        "regression": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "random_state": [DEFAULT_RANDOM_STATE]
        },
        "classification": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "random_state": [DEFAULT_RANDOM_STATE]
        }
    }
}

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_config(config: dict):
    required_keys = ["model_params", "hyperparam_grid", "cv"]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            raise KeyError(f"Missing required config key: {key}")

def assert_metric_model_compatibility(model_name, task_type, metrics):
    if task_type == "regression" and any(m in ["roc_auc", "accuracy", "f1", "precision", "recall"] for m in metrics):
        logger.error(f"Metric(s) {metrics} not compatible with regression models.")
        raise ValueError(f"Metric(s) {metrics} not compatible with regression models.")
    if model_name == "svm" and task_type == "regression" and "roc_auc" in metrics:
        logger.error("SVR does not support probability estimation required for ROC AUC.")
        raise ValueError("SVR does not support probability estimation required for ROC AUC.")

def get_model(model_name: str, task_type: str, config: dict) -> Any:
    model_name = model_name.lower()
    task_type = task_type.lower()
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Model '{model_name}' not found in registry.")
        raise ValueError(f"Model '{model_name}' not found in registry.")
    if task_type not in MODEL_REGISTRY[model_name]:
        logger.error(f"Task type '{task_type}' not supported for model '{model_name}'.")
        raise ValueError(f"Task type '{task_type}' not supported for model '{model_name}'.")
    model_class = MODEL_REGISTRY[model_name][task_type]
    model_params = config.get("model_params", {}).get(model_name, {}).get(task_type, {}).copy()
    if "random_state" in model_class().get_params().keys() and "random_state" not in model_params:
        model_params["random_state"] = DEFAULT_RANDOM_STATE
    if model_name == "svm" and task_type == "classification":
        model_params.setdefault("probability", True)
    return model_class(**model_params)

def get_hyperparam_grid(model_name: str, task_type: str, config: dict) -> dict:
    model_name = model_name.lower()
    task_type = task_type.lower()
    grid = HYPERPARAM_GRIDS.get(model_name, {}).get(task_type, {}).copy()
    override = config.get("hyperparam_grid", {}).get(model_name, {}).get(task_type, {})
    if override:
        grid = deep_update(grid, override)
    return grid

def get_cv_strategy(config: dict, y=None) -> Any:
    cv_cfg = config.get("cv", {})
    strategy = cv_cfg.get("strategy", "timeseriessplit").lower()
    n_splits = cv_cfg.get("n_splits", 5)
    if strategy == "timeseriessplit":
        return TimeSeriesSplit(n_splits=n_splits)
    elif strategy == "stratifiedkfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=cv_cfg.get("shuffle", False), random_state=cv_cfg.get("random_state", DEFAULT_RANDOM_STATE))
    elif strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=cv_cfg.get("shuffle", False), random_state=cv_cfg.get("random_state", DEFAULT_RANDOM_STATE))
    elif strategy == "walkforward":
        return TimeSeriesSplit(n_splits=n_splits)
    elif strategy == "purgedkfold":
        try:
            from purgedkfold import PurgedKFold
            return PurgedKFold(n_splits=n_splits)
        except ImportError:
            logger.error("PurgedKFold is not installed. Please install or implement it.")
            raise NotImplementedError("PurgedKFold is not installed. Please install or implement it.")
    else:
        logger.error(f"Unknown CV strategy: {strategy}")
        raise ValueError(f"Unknown CV strategy: {strategy}")

def roc_auc_scorer_func(y_true, y_score, **kwargs):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)

def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_scorer(config: dict, task_type: str):
    metrics = config.get("metrics", {}).get(task_type, [])
    if not metrics:
        metrics = ["neg_mean_squared_error"] if task_type == "regression" else ["accuracy"]
    main_metric = metrics[0]
    if main_metric == "roc_auc":
        from sklearn.metrics import make_scorer
        return make_scorer(roc_auc_scorer_func, needs_proba=True)
    elif main_metric == "neg_mean_squared_error":
        from sklearn.metrics import mean_squared_error, make_scorer
        return make_scorer(mean_squared_error, greater_is_better=False)
    elif main_metric == "mean_absolute_error" or main_metric == "mae":
        from sklearn.metrics import mean_absolute_error, make_scorer
        return make_scorer(mean_absolute_error, greater_is_better=False)
    elif main_metric == "r2":
        from sklearn.metrics import r2_score, make_scorer
        return make_scorer(r2_score)
    elif main_metric == "rmse":
        return make_scorer(rmse_func, greater_is_better=False)
    elif main_metric == "accuracy":
        from sklearn.metrics import accuracy_score, make_scorer
        return make_scorer(accuracy_score)
    elif main_metric == "f1" or main_metric == "f1_score":
        from sklearn.metrics import f1_score, make_scorer
        return make_scorer(f1_score, average="macro")
    elif main_metric == "precision":
        from sklearn.metrics import precision_score, make_scorer
        return make_scorer(precision_score, average="macro")
    elif main_metric == "recall":
        from sklearn.metrics import recall_score, make_scorer
        return make_scorer(recall_score, average="macro")
    else:
        logger.error(f"Unknown main metric: {main_metric}")
        raise ValueError(f"Unknown main metric: {main_metric}")


def get_scorers_dict(config: dict, task_type: str) -> dict:
    metrics = config.get("metrics", {}).get(task_type, [])
    if not metrics:
        metrics = ["neg_mean_squared_error"] if task_type == "regression" else ["accuracy"]
    scorers = {}
    from sklearn.metrics import make_scorer
    for metric in metrics:
        if metric == "roc_auc":
            scorers[metric] = make_scorer(roc_auc_scorer_func, needs_proba=True)
        elif metric == "neg_mean_squared_error":
            from sklearn.metrics import mean_squared_error
            scorers[metric] = make_scorer(mean_squared_error, greater_is_better=False)
        elif metric == "mean_absolute_error" or metric == "mae":
            from sklearn.metrics import mean_absolute_error
            scorers[metric] = make_scorer(mean_absolute_error, greater_is_better=False)
        elif metric == "r2":
            from sklearn.metrics import r2_score
            scorers[metric] = make_scorer(r2_score)
        elif metric == "rmse":
            scorers[metric] = make_scorer(rmse_func, greater_is_better=False)
        elif metric == "accuracy":
            from sklearn.metrics import accuracy_score
            scorers[metric] = make_scorer(accuracy_score)
        elif metric == "f1" or metric == "f1_score":
            from sklearn.metrics import f1_score
            scorers[metric] = make_scorer(f1_score, average="macro")
        elif metric == "precision":
            from sklearn.metrics import precision_score
            scorers[metric] = make_scorer(precision_score, average="macro")
        elif metric == "recall":
            from sklearn.metrics import recall_score
            scorers[metric] = make_scorer(recall_score, average="macro")
        else:
            logger.error(f"Unknown metric: {metric}")
            raise ValueError(f"Unknown metric: {metric}")
    return scorers

def save_model(model, scaler=None, output_dir="models", model_name="model", extra: Optional[dict]=None):
    os.makedirs(output_dir, exist_ok=True)
    bundle = {"model": model}
    if scaler is not None:
        bundle["scaler"] = scaler
    if extra is not None:
        bundle["extra"] = extra
    path = os.path.join(output_dir, f"{model_name}_bundle.pkl")
    joblib.dump(bundle, path)
    logger.info(f"Model bundle saved to {path}")
    return path

def load_model(path: str):
    bundle = joblib.load(path)
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    extra = bundle.get("extra")
    logger.info(f"Model bundle loaded from {path}")
    return model, scaler, extra

def get_model_output_dir(base_dir, model_name, task_type, run_id=None):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, "models", f"{model_name}_{task_type}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Model output directory: {out_dir}")
    return out_dir

def run_grid_search(X, y, config, task_type, model_name=None, groups=None, multi_metric=False, refit=None):
    if model_name is None:
        model_name = config.get("model", None)
        if model_name is None:
            logger.error("Model name must be provided either as argument or in config['model']")
            raise ValueError("Model name must be provided either as argument or in config['model']")
    metrics = config.get("metrics", {}).get(task_type, [])
    assert_metric_model_compatibility(model_name, task_type, metrics)
    model = get_model(model_name, task_type, config)
    param_grid = get_hyperparam_grid(model_name, task_type, config)
    cv = get_cv_strategy(config, y)
    if multi_metric:
        scorers = get_scorers_dict(config, task_type)
        refit_metric = refit if refit is not None else config.get("refit_metric", list(scorers.keys())[0])
        if refit_metric not in scorers:
            logger.error(f"refit_metric '{refit_metric}' not in scorers {list(scorers.keys())}")
            raise ValueError(f"refit_metric '{refit_metric}' not in scorers {list(scorers.keys())}")
        search_kwargs = {
            "estimator": model,
            "param_grid": param_grid,
            "scoring": scorers,
            "refit": refit_metric,
            "cv": cv,
            "n_jobs": config.get("n_jobs", -1),
            "verbose": config.get("grid_search_verbose", 1),
            "return_train_score": True
        }
    else:
        scorer = get_scorer(config, task_type)
        search_kwargs = {
            "estimator": model,
            "param_grid": param_grid,
            "scoring": scorer,
            "refit": True,
            "cv": cv,
            "n_jobs": config.get("n_jobs", -1),
            "verbose": config.get("grid_search_verbose", 1),
            "return_train_score": True
        }
    if groups is not None:
        search_kwargs["groups"] = groups
    logger.info(f"Running GridSearchCV for model={model_name}, task={task_type}, multi_metric={multi_metric}")
    grid_search = GridSearchCV(**search_kwargs)
    grid_search.fit(X, y)
    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_}")
    return grid_search.best_estimator_, grid_search

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model utilities test CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to train_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. ridge, random_forest, xgboost)")
    parser.add_argument("--task", type=str, required=True, help="Task type (regression or classification)")
    parser.add_argument("--multi-metric", action="store_true", help="Enable multi-metric scoring")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    validate_config(config)
    model = get_model(args.model, args.task, config)
    grid = get_hyperparam_grid(args.model, args.task, config)
    cv = get_cv_strategy(config)
    if args.multi_metric:
        scorers = get_scorers_dict(config, args.task)
        print(f"Scorers: {scorers}")
    else:
        scorer = get_scorer(config, args.task)
        print(f"Scorer: {scorer}")
    print(f"Model: {model}")
    print(f"Hyperparameter grid: {grid}")
    print(f"CV: {cv}")