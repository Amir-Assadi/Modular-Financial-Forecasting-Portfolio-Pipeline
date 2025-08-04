import os
import pandas as pd
from forcasting_ml.model_utils import load_model, load_yaml_config
from forcasting_ml.live_data_feature_engineering import engineer_features_for_live
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # --- CONFIGURATION ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "data", "config_data.yaml")
    train_config_path = os.path.join(base_dir, "train_config.yaml")
    config = load_yaml_config(config_path)
    train_config = load_yaml_config(train_config_path)
    tickers = config["tickers"]

    # Get target info from train_config.yaml
    target_cfg = train_config.get("target", {})
    prediction_horizons = target_cfg.get("horizons", [1])
    target_variables = target_cfg.get("variables", [])
    task_type = target_cfg.get("type", "classification")
    model_name = train_config.get("model", "xgboost")

    markov_output_dir = os.path.join(base_dir, "..", "regime_detection", "markov_output")
    markov_prob_output_dir = os.path.join(base_dir, "..", "regime_detection", "markov_prob_output")
    changepoint_output_dir = os.path.join(base_dir, "..", "regime_detection", "changepoint_output")
    macro_path = os.path.join(base_dir, "..", "data", "historical_data", "macro_data.csv")
    trend_path = os.path.join(base_dir, "..", "data", "historical_data", "google_trend.csv")

    for ticker in tickers:
        logger.info(f"--- Inference for {ticker} ---")
        live_data_path = os.path.join(base_dir, "..", "data", "live_data", f"{ticker}.csv")
        if not os.path.exists(live_data_path):
            logger.error(f"No live data found for {ticker} at {live_data_path}")
            continue
        live_df = pd.read_csv(live_data_path)
        if "Date" not in live_df.columns:
            if "timestamp" in live_df.columns:
                live_df["Date"] = pd.to_datetime(live_df["timestamp"]).dt.date
            else:
                logger.error(f"No 'Date' or 'timestamp' column in live data for {ticker}.")
                continue

        for target_variable in target_variables:
            # Build model filename dynamically
            suffix = "classification" if task_type == "classification" or target_variable.endswith("_class") else "regression"
            model_path = os.path.join(
                base_dir, "trained_model",
                f"{ticker}_{target_variable}_{model_name}_{suffix}_bundle.pkl"
            )

            # --- LOAD MODEL, SCALER, FEATURE COLS ---
            try:
                model, scaler, extra = load_model(model_path)
            except Exception as e:
                logger.error(f"Could not load model for {ticker}: {e}")
                continue
            if extra and "feature_cols" in extra:
                feature_cols = extra["feature_cols"]
            else:
                logger.error(f"Feature columns not found in model bundle for {ticker}.")
                continue

            # --- FEATURE ENGINEERING FOR LIVE DATA ---
            try:
                live_features_scaled = engineer_features_for_live(
                    live_df,
                    ticker,
                    markov_output_dir,
                    markov_prob_output_dir,
                    changepoint_output_dir,
                    macro_path,
                    trend_path,
                    train_config_path,
                    scaler,
                    feature_cols
                )
            except Exception as e:
                logger.error(f"Feature engineering failed for {ticker}: {e}")
                continue

            # --- INFERENCE ---
            try:
                y_pred = model.predict(live_features_scaled)
                logger.info(f"Prediction for {ticker} ({target_variable}, latest row): {y_pred[0]}")
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")

if __name__ == "__main__":
    main()