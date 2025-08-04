import os
import time
import pandas as pd
import numpy as np
import yaml
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from forcasting_ml.model_utils import (
    load_yaml_config, validate_config, get_model, get_hyperparam_grid, get_cv_strategy,
    run_grid_search, save_model, get_logger
)
from utils.logger import get_logger

# Set global random seed for reproducibility
np.random.seed(42)

# Load configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "train_config.yaml")
CONFIG_DATA_PATH = os.path.join(BASE_DIR, "..", "Data", "config_data.yaml")
FEATURES_DIR = os.path.join(BASE_DIR, "feature_engineering_output")
TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

# Initialize logger
logger = get_logger(__name__)

def load_tickers(config_data_path):
    with open(config_data_path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data.get("tickers", [])

def get_metrics_dict(task_type):
    if task_type == "regression":
        return {
            "r2": r2_score,
            "mae": mean_absolute_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        }
    elif task_type == "classification":
        return {
            "accuracy": accuracy_score,
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
            "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
            "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan,
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def main():
    """
    Main training loop: loads configs, features, splits data, trains and evaluates models for each ticker and each target variable (horizon).
    """
    config = load_yaml_config(CONFIG_PATH)
    validate_config(config)
    tickers = load_tickers(CONFIG_DATA_PATH)
    training_cfg = config.get("training", {})
    test_size = training_cfg.get("test_size", 0.2)
    time_index = training_cfg.get("time_index", "Date")
    scale_features = training_cfg.get("scale_features", True)
    target_cfg = config["target"]
    # Support both 'variables' (list of horizons) and single 'variable'
    target_variables = target_cfg.get("variables", [target_cfg.get("variable")])
    task_type = target_cfg["type"]
    metrics_list = config["metrics"][task_type]
    refit_metric = config.get("refit_metric", metrics_list[0])
    model_name = config["model"]

    # Optionally, get explicit feature list from config
    explicit_features = config.get("feature_cols", None)

    for ticker in tickers:
        try:
            logger.info(f"=== Training for {ticker} ===")
            feature_path = os.path.join(FEATURES_DIR, f"features_{ticker}.csv")
            if not os.path.exists(feature_path):
                logger.warning(f"Feature file not found for {ticker}, skipping.")
                continue

            df = pd.read_csv(feature_path)
            if time_index in df.columns:
                df = df.sort_values(time_index)
            else:
                logger.warning(f"Time index {time_index} not found in features for {ticker}, using default order.")

            for target_variable in target_variables:
                if target_variable not in df.columns:
                    logger.warning(f"Target variable {target_variable} not found for {ticker}, skipping.")
                    continue

                logger.info(f"Training for target: {target_variable}")

                # Prepare features and target
                if explicit_features:
                    feature_cols = [col for col in explicit_features if col in df.columns]
                else:
                    feature_cols = [col for col in df.columns if col not in [time_index] + target_variables]

                # Drop all-NaN, all-zero, and zero-variance columns from features
                dropped_nan = [col for col in feature_cols if df[col].isna().all()]
                dropped_zero = [col for col in feature_cols if (df[col] == 0).all()]
                dropped_const = [col for col in feature_cols if df[col].nunique(dropna=False) <= 1]
                dropped = set(dropped_nan + dropped_zero + dropped_const)
                if dropped:
                    logger.warning(f"Dropping features for {ticker} ({target_variable}): {dropped}")
                    feature_cols = [col for col in feature_cols if col not in dropped]

                # Only use numeric columns
                non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
                if non_numeric:
                    logger.warning(f"Non-numeric features dropped for {ticker} ({target_variable}): {non_numeric}")
                    feature_cols = [col for col in feature_cols if col not in non_numeric]

                # Log if no features remain
                if not feature_cols:
                    logger.error(f"No usable features remain for {ticker} ({target_variable}), skipping.")
                    continue

                # Log if target is all-NaN or constant
                if np.isnan(df[target_variable]).all():
                    logger.error(f"Target variable {target_variable} is all-NaN for {ticker}, skipping.")
                    continue
                if df[target_variable].nunique(dropna=False) <= 1:
                    logger.error(f"Target variable {target_variable} is constant for {ticker}, skipping.")
                    continue

                X = df[feature_cols].values
                y = df[target_variable].values

                # Check for missing data
                if np.isnan(X).any() or np.isnan(y).any():
                    if training_cfg.get("impute_missing", False):
                        logger.info("Imputing missing values with column means.")
                        from sklearn.impute import SimpleImputer
                        imp = SimpleImputer(strategy="mean")
                        X = imp.fit_transform(X)
                        if np.isnan(y).any():
                            y = np.where(np.isnan(y), np.nanmedian(y), y)
                    else:
                        logger.error("Missing values detected in features or target. Skipping.")
                        continue

                # Train/test split
                if time_index in df.columns:
                    split_idx = int(len(df) * (1 - test_size))
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)

                logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

                # Feature scaling
                scaler = None
                if scale_features:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                # --- Filter out NaNs in y_test after scaling ---
                if np.isnan(y_test).any():
                    logger.warning(f"NaNs detected in y_test for {ticker} ({target_variable}). Filtering out these rows for evaluation.")
                    mask = ~np.isnan(y_test)
                    X_test = X_test[mask]
                    y_test = y_test[mask]

                model = get_model(model_name, task_type, config)
                param_grid = get_hyperparam_grid(model_name, task_type, config)
                cv = get_cv_strategy(config, y_train)
                logger.info(f"Model instance: {model}")
                logger.info(f"Hyperparameter grid: {param_grid}")
                logger.info(f"CV strategy: {cv}")

                # Model training with hyperparameter tuning
                start_time = time.time()
                best_model, grid_search = run_grid_search(
                    X_train, y_train, config, task_type, model_name=model_name, multi_metric=True, refit=refit_metric
                )
                duration = time.time() - start_time

                # Evaluation
                y_pred = best_model.predict(X_test)
                metrics_dict = get_metrics_dict(task_type)
                results = {}
                for metric in metrics_list:
                    try:
                        if metric == "roc_auc" and hasattr(best_model, "predict_proba"):
                            y_score = best_model.predict_proba(X_test)
                            if y_score.shape[1] == 2:
                                y_score = y_score[:, 1]
                            results[metric] = roc_auc_score(y_test, y_score)
                        elif metric == "roc_auc" and hasattr(best_model, "decision_function"):
                            y_score = best_model.decision_function(X_test)
                            results[metric] = roc_auc_score(y_test, y_score)
                        else:
                            results[metric] = metrics_dict[metric](y_test, y_pred)
                    except Exception as e:
                        logger.warning(f"Could not compute {metric} for {ticker} ({target_variable}): {e}")
                        results[metric] = np.nan

                # Logging results
                logger.info(f"Training duration for {ticker} ({target_variable}): {duration:.2f} seconds")
                logger.info(f"Best hyperparameters for {ticker} ({target_variable}): {grid_search.best_params_}")
                logger.info(f"Evaluation metrics for {ticker} ({target_variable}): {results}")

                # Save metrics and params to JSON
                metrics_save_path = os.path.join(
                    TRAINED_MODEL_DIR, f"{ticker}_{target_variable}_{model_name}_{task_type}_metrics.json"
                )
                with open(metrics_save_path, "w") as f:
                    json.dump({
                        "ticker": ticker,
                        "target_variable": target_variable,
                        "model": model_name,
                        "task_type": task_type,
                        "best_params": grid_search.best_params_,
                        "metrics": results,
                        "duration_sec": duration,
                        "feature_cols": feature_cols
                    }, f, indent=2)

                # Serialize model and scaler
                save_model(
                    best_model,
                    scaler=scaler,
                    output_dir=TRAINED_MODEL_DIR,
                    model_name=f"{ticker}_{target_variable}_{model_name}_{task_type}",
                    extra={
                        "feature_cols": feature_cols,
                    }

                )
                logger.info(f"Saved trained model and scaler for {ticker} ({target_variable}) to {TRAINED_MODEL_DIR}")

        except Exception as e:
            logger.error(f"Training failed for {ticker}: {e}")

if __name__ == "__main__":
    main()