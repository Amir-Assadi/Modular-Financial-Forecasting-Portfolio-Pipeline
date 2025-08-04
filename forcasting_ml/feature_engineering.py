import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
import argparse

logger = get_logger(__name__)

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def add_lagged_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def compute_drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max) / roll_max

def compute_rolling_sharpe(returns, window):
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return mean / std

def compute_entropy(probs):
    return -np.nansum(probs * np.log(probs + 1e-12), axis=1)

def compute_time_since_last_changepoint(df, changepoint_df):
    # Add time since last changepoint feature
    df = df.copy()
    df["time_since_changepoint"] = np.nan
    if changepoint_df is not None and not changepoint_df.empty:
        changepoint_dates = pd.to_datetime(changepoint_df["end_time"])
        last_cp_idx = 0
        for i, date in enumerate(df["Date"]):
            while last_cp_idx + 1 < len(changepoint_dates) and date >= changepoint_dates.iloc[last_cp_idx + 1]:
                last_cp_idx += 1
            df.at[i, "time_since_changepoint"] = (date - changepoint_dates.iloc[last_cp_idx]).days if last_cp_idx < len(changepoint_dates) else np.nan
    return df

def load_csv(path, parse_dates=None):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=parse_dates)
            if df.empty or (parse_dates and any(col not in df.columns for col in parse_dates)):
                logger.warning(f"{path} is empty or missing required columns. Skipping.")
                return None
            return df
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}. Skipping.")
            return None
    return None

def main(
    ticker,
    markov_output_dir,
    markov_prob_output_dir,
    changepoint_output_dir,
    macro_path,
    trend_path,
    config_path,
    output_path
):
    config = load_yaml_config(config_path)
    features_cfg = config["features"]
    price_features = features_cfg.get("price_features", [])
    regime_features = features_cfg.get("regime_features", [])
    composite_features = features_cfg.get("composite_features", [])
    macro_features = features_cfg.get("macro_features", [])
    feature_flags = config.get("feature_flags", {})
    enable_drawdown = feature_flags.get("enable_drawdown", True)
    enable_entropy = feature_flags.get("enable_entropy", True)
    lookback_window = config.get("training", {}).get("lookback_window", 60)
    target_cfg = config.get("target", {})
    target_variable = target_cfg.get("variables", ["future_return_1d_class"])
    target_cols = target_variable if isinstance(target_variable, list) else [target_variable]

    markov_path = os.path.join(markov_output_dir, f"{ticker}_regimes.csv")
    prob_path = os.path.join(markov_prob_output_dir, f"{ticker}_regime_probs.csv")
    if not os.path.exists(markov_path) or not os.path.exists(prob_path):
        logger.error(f"Markov output or probability file missing for {ticker}. Skipping.")
        return
    df = pd.read_csv(markov_path, parse_dates=["Date"])
    prob_df = pd.read_csv(prob_path, parse_dates=["Date"])
    df = df.merge(prob_df, on="Date", how="left")

    changepoint_path = os.path.join(changepoint_output_dir, f"changepoints_{ticker}.csv")
    changepoint_df = pd.read_csv(changepoint_path) if os.path.exists(changepoint_path) else None

    macro_df = load_csv(macro_path, parse_dates=["Date"])
    trend_df = load_csv(trend_path, parse_dates=["Date"])

    # Merge macro and trend data
    for merge_df in [macro_df, trend_df]:
        if merge_df is not None:
            df = df.merge(merge_df, on="Date", how="left")

    # Fill missing macro features
    for col_group in [macro_features]:
        cols = [col for col in col_group if col in df.columns]
        if cols:
            df[cols] = df[cols].ffill().bfill()

    # Add technical indicators
    if "Close" in df.columns:
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

    lag_map = {
        "returns": [1, 3, 5],
        "momentum": [1, 3, 5],
        "vix_index": [1, 3, 5]
    }
    for col, lags in lag_map.items():
        if col in df.columns:
            df = add_lagged_features(df, [col], lags)

    if "returns" in df.columns:
        df["rolling_volatility_5d"] = df["returns"].rolling(5).std()
        df["rolling_volatility_10d"] = df["returns"].rolling(10).std()
        df["momentum"] = df["returns"].rolling(10).mean()
        df["rolling_sharpe"] = compute_rolling_sharpe(df["returns"], lookback_window)
        df["zscore_returns"] = (df["returns"] - df["returns"].rolling(lookback_window).mean()) / df["returns"].rolling(lookback_window).std()

    if enable_drawdown and "Close" in df.columns:
        df["drawdown"] = compute_drawdown(df["Close"])

    prob_cols = [col for col in df.columns if col.startswith("prob_state_")]
    for col in prob_cols:
        df = add_lagged_features(df, [col], [1, 3, 5])

    if prob_cols:
        df["regime_confidence"] = df[prob_cols].max(axis=1)
        if enable_entropy:
            df["regime_entropy"] = compute_entropy(df[prob_cols].values)

    df = compute_time_since_last_changepoint(df, changepoint_df)

        # --- Dynamically generate target columns based on config ---
    if "Close" in df.columns and "target" in config:
        horizons = config["target"].get("horizons", [1])
        variables = config["target"].get("variables", [])
        for h in horizons:
            # Regression target
            col_name = f"future_return_{h}d"
            df[col_name] = df["Close"].shift(-h) / df["Close"] - 1
            # Classification target
            class_col_name = f"future_return_{h}d_class"
            df[class_col_name] = (df[col_name] > 0).astype(int)
        # Only keep variables specified in config
        for var in variables:
            if var not in df.columns:
                logger.warning(f"Target variable {var} not found in dataframe after generation.")

    # Select features to save
    all_features = ["Date"]
    for feature_list in [price_features, regime_features, composite_features, macro_features]:
        for feature in feature_list:
            if feature in df.columns and feature not in all_features:
                all_features.append(feature)
    trend_cols = [c for c in df.columns if c.startswith("trend_")]
    for trend_col in trend_cols:
        if trend_col not in all_features:
            all_features.append(trend_col)
    engineered_features = ["time_since_changepoint"]
    for eng_feature in engineered_features:
        if eng_feature in df.columns and eng_feature not in all_features:
            all_features.append(eng_feature)
    for target_col in target_cols:
        if target_col in df.columns and target_col not in all_features:
            all_features.append(target_col)
    if 'future_return_1d_class' in df.columns and 'future_return_1d_class' not in all_features:
        all_features.append('future_return_1d_class')
    features_to_save = [col for col in all_features if col in df.columns]
    leakage_patterns = ["future_return_"]
    features_to_save = [col for col in features_to_save if not (
        any(col.startswith(pattern) for pattern in leakage_patterns) and col not in target_cols
        or (col.endswith('_class') and col not in target_cols)
    ) or col == "Date"]

    target_columns = target_cols + (['future_return_1d_class'] if 'future_return_1d_class' in df.columns else [])
    feature_columns = [col for col in features_to_save if col not in ["Date"] + target_columns]

    # Drop columns with all NaN, all zero, or zero variance
    exclude_cols = ["Date"] + target_columns
    feature_cols_for_cleaning = [col for col in feature_columns if col not in exclude_cols]
    all_nan_cols = [col for col in feature_cols_for_cleaning if col in df.columns and df[col].isna().all()]
    if all_nan_cols:
        df.drop(columns=all_nan_cols, inplace=True)
        feature_cols_for_cleaning = [col for col in feature_cols_for_cleaning if col in df.columns]
        feature_columns = [col for col in feature_columns if col in df.columns]
    all_zero_cols = [col for col in feature_cols_for_cleaning if col in df.columns and (df[col] == 0).all()]
    if all_zero_cols:
        df.drop(columns=all_zero_cols, inplace=True)
        feature_cols_for_cleaning = [col for col in feature_cols_for_cleaning if col in df.columns]
        feature_columns = [col for col in feature_columns if col in df.columns]
    zero_var_cols = [col for col in feature_cols_for_cleaning if col in df.columns and df[col].nunique(dropna=False) <= 1]
    if zero_var_cols:
        df.drop(columns=zero_var_cols, inplace=True)
        feature_cols_for_cleaning = [col for col in feature_cols_for_cleaning if col in df.columns]
        feature_columns = [col for col in feature_columns if col in df.columns]

    # Impute missing values with median
    imputed_cols = []
    for col in feature_columns:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            imputed_cols.append(col)
    if config.get("training", {}).get("scale_features", True) and feature_columns:
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

    features_to_save = [col for col in features_to_save if col in df.columns]
    df[features_to_save].to_csv(output_path, index=False)
    logger.info(f"Saved {len(features_to_save)} features to {output_path}")
    logger.info(f"Final dataset shape: {df[features_to_save].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--ticker", type=str, default=None)
    args, unknown = parser.parse_known_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    markov_output_dir = os.path.join(base_dir, "..", "regime_detection", "markov_output")
    markov_prob_output_dir = os.path.join(base_dir, "..", "regime_detection", "markov_prob_output")
    changepoint_output_dir = os.path.join(base_dir, "..", "regime_detection", "changepoint_output")
    macro_path = os.path.join(base_dir, "..", "data", "historical_data", "macro_data.csv")
    trend_path = os.path.join(base_dir, "..", "data", "historical_data", "google_trend.csv")
    config_path = os.path.join(base_dir, "train_config.yaml")
    config_data_path = os.path.join(base_dir, "..", "data", "config_data.yaml")

    if args.ticker and args.output:
        ticker = args.ticker
        output_path = args.output
        logger.info(f"Starting feature engineering for {ticker} (CLI mode)")
        main(
            ticker,
            markov_output_dir,
            markov_prob_output_dir,
            changepoint_output_dir,
            macro_path,
            trend_path,
            config_path,
            output_path
        )
        logger.info(f"Finished feature engineering for {ticker}")
    else:
        output_dir = os.path.join(base_dir, "feature_engineering_output")
        os.makedirs(output_dir, exist_ok=True)
        with open(config_data_path, "r") as f:
            config_data = yaml.safe_load(f)
        tickers = config_data.get("tickers", [])
        for ticker in tickers:
            output_path = os.path.join(output_dir, f"features_{ticker}.csv")
            logger.info(f"Starting feature engineering for {ticker}")
            main(
                ticker,
                markov_output_dir,
                markov_prob_output_dir,
                changepoint_output_dir,
                macro_path,
                trend_path,
                config_path,
                output_path
            )
            logger.info(f"Finished feature engineering for {ticker}")