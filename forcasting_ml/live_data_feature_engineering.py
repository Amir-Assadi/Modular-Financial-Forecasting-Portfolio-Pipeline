import os
import pandas as pd
from forcasting_ml.feature_engineering import (
    load_yaml_config, load_csv, add_lagged_features, compute_drawdown,
    compute_rolling_sharpe, compute_entropy, compute_time_since_last_changepoint
)
from utils.logger import get_logger

logger = get_logger(__name__)

def engineer_features_for_live(
    df,
    ticker,
    markov_output_dir,
    markov_prob_output_dir,
    changepoint_output_dir,
    macro_path,
    trend_path,
    config_path,
    scaler,
    feature_cols
):
    """
    Engineer features for live data inference.
    df: DataFrame with at least a 'Date' column (should be cleaned and have all price columns).
    scaler: The fitted scaler from training.
    feature_cols: List of feature columns used in training (order matters).
    Returns: DataFrame with one row, ready for model inference.
    """
    logger.info(f"Starting feature engineering for live data: {ticker}")

    config = load_yaml_config(config_path)
    features_cfg = config["features"]
    macro_features = features_cfg.get("macro_features", [])
    feature_flags = config.get("feature_flags", {})
    enable_drawdown = feature_flags.get("enable_drawdown", True)
    enable_entropy = feature_flags.get("enable_entropy", True)
    lookback_window = config["training"].get("lookback_window", 60)

    # Ensure Date is datetime for merging
    df["Date"] = pd.to_datetime(df["Date"])

    # Merge regime (markov) and regime probabilities
    markov_path = os.path.join(markov_output_dir, f"markov_output_{ticker}.csv")
    prob_path = os.path.join(markov_prob_output_dir, f"markov_prob_output_{ticker}.csv")
    if os.path.exists(markov_path) and os.path.exists(prob_path):
        logger.info("Merging regime and regime probability features.")
        markov_df = pd.read_csv(markov_path, parse_dates=["Date"])
        prob_df = pd.read_csv(prob_path, parse_dates=["Date"])
        markov_df["Date"] = pd.to_datetime(markov_df["Date"])
        prob_df["Date"] = pd.to_datetime(prob_df["Date"])
        df = df.merge(markov_df, on="Date", how="left")
        df = df.merge(prob_df, on="Date", how="left")
    else:
        logger.warning("Markov or probability files not found. Skipping regime features.")

    # Merge changepoint info
    changepoint_path = os.path.join(changepoint_output_dir, f"changepoints_{ticker}.csv")
    changepoint_df = pd.read_csv(changepoint_path) if os.path.exists(changepoint_path) else None
    if changepoint_df is not None and "Date" in changepoint_df.columns:
        changepoint_df["Date"] = pd.to_datetime(changepoint_df["Date"])
    df = compute_time_since_last_changepoint(df, changepoint_df)
    logger.info("Merged changepoint features.")

    # Merge macro/trend
    macro_df = load_csv(macro_path, parse_dates=["Date"])
    if macro_df is not None and "Date" in macro_df.columns:
        logger.info("Merging macro features.")
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        df = df.merge(macro_df, on="Date", how="left")
    else:
        logger.warning("No macro data found to merge.")
    trend_df = load_csv(trend_path, parse_dates=["Date"])
    if trend_df is not None and "Date" in trend_df.columns:
        logger.info("Merging trend features.")
        trend_df["Date"] = pd.to_datetime(trend_df["Date"])
        df = df.merge(trend_df, on="Date", how="left")
    else:
        logger.warning("No trend data found to merge.")

    # Fill missing macro/trend
    macro_trend_cols = [col for col in (macro_features + [c for c in df.columns if c.startswith("trend_")]) if col in df.columns]
    if macro_trend_cols:
        df[macro_trend_cols] = df[macro_trend_cols].ffill().bfill()
        logger.info("Filled missing macro/trend values.")

    # Add technical features (SMA, MACD, RSI, etc.)
    if "Close" in df.columns:
        logger.info("Adding technical indicators.")
        df["sma_10"] = df["Close"].rolling(window=10).mean()
        df["sma_50"] = df["Close"].rolling(window=50).mean()
        df["sma_150"] = df["Close"].rolling(window=150).mean()
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        delta = df["Close"].diff()
        gain_14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs_14 = gain_14 / loss_14
        df["rsi_14"] = 100 - (100 / (1 + rs_14))

    # Add lagged features for returns, momentum, vix, regime probs, etc.
    lag_map = {
        "returns": [1, 3, 5],
        "momentum": [1, 3, 5],
        "vix_index": [1, 3, 5]
    }
    for col, lags in lag_map.items():
        if col in df.columns:
            df = add_lagged_features(df, [col], lags)
            logger.info(f"Added lagged features for {col}.")

    # Rolling volatility
    if "returns" in df.columns:
        df["rolling_volatility_5d"] = df["returns"].rolling(5).std()
        df["rolling_volatility_10d"] = df["returns"].rolling(10).std()

    # Rolling momentum
    if "returns" in df.columns:
        df["momentum"] = df["returns"].rolling(10).mean()

    # Rolling Sharpe
    if "returns" in df.columns:
        df["rolling_sharpe"] = compute_rolling_sharpe(df["returns"], lookback_window)

    # Drawdown (configurable)
    if enable_drawdown and "Close" in df.columns:
        df["drawdown"] = compute_drawdown(df["Close"])

    # Z-score of returns
    if "returns" in df.columns:
        df["zscore_returns"] = (df["returns"] - df["returns"].rolling(lookback_window).mean()) / df["returns"].rolling(lookback_window).std()

    # Lagged regime probabilities
    prob_cols = [col for col in df.columns if col.startswith("prob_state_")]
    for col in prob_cols:
        df = add_lagged_features(df, [col], [1, 3, 5])
        logger.info(f"Added lagged regime probabilities for {col}.")

    # Regime confidence (max prob) and entropy (configurable)
    if prob_cols:
        df["regime_confidence"] = df[prob_cols].max(axis=1)
        logger.info("Regime confidence feature added.")
        if enable_entropy:
            df["regime_entropy"] = compute_entropy(df[prob_cols].values)
            logger.info("Regime entropy feature added.")

    # Time since last changepoint 
    df = compute_time_since_last_changepoint(df, changepoint_df)
    logger.info("Time since last changepoint feature added.")

    # Ensure all required feature columns exist, even if NaN
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' missing in engineered DataFrame for {ticker}, filling with NaN.")
            df[col] = float('nan')

    # Remove any extra columns not in feature_cols
    df = df[[col for col in df.columns if col in feature_cols or col == "Date"]]

    # Select only the columns used for inference, in the correct order
    try:
        live_features = df[feature_cols].tail(1)
        logger.info(f"live_features shape: {live_features.shape}, scaler expects: {scaler.mean_.shape}")
        # Scale using the scaler from training
        live_features_scaled = scaler.transform(live_features)
        logger.info("Feature engineering for live data complete.")
        return live_features_scaled
    except Exception as e:
        logger.error(f"Error during feature selection or scaling: {e}")
        raise

if __name__ == "__main__":
    print("This module is for import only. Use engineer_features_for_live()")