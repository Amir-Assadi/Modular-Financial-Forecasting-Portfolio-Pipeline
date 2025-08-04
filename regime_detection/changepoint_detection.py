import os
import pandas as pd
import numpy as np
import yaml
from utils.logger import get_logger
from sklearn.preprocessing import StandardScaler
from data.macro_data import fetch_macro_from_fred

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_markov_output(ticker, markov_output_dir):
    # Check if Markov output file exists and is valid
    if not os.path.exists(markov_output_dir):
        logger.warning(f"Markov output directory not found: {markov_output_dir}")
        return None

    # Ensure Markov output directory exists
    os.makedirs(markov_output_dir, exist_ok=True)

    file_path = os.path.join(markov_output_dir, f"{ticker}_regimes.csv")
    if not os.path.exists(file_path):
        logger.warning(f"Markov output for {ticker} not found at {file_path}. Creating placeholder file.")
        placeholder_data = pd.DataFrame(columns=["Date", "regime"])
        placeholder_data.to_csv(file_path, index=False)
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
        if df.empty:
            logger.warning(f"Markov output file for {ticker} is empty: {file_path}. Creating placeholder data.")
            placeholder_data = pd.DataFrame(columns=["Date", "regime"])
            placeholder_data.to_csv(file_path, index=False)
            return None
    except Exception as e:
        logger.error(f"Failed to parse Markov output file for {ticker}: {e}. Creating placeholder file.")
        placeholder_data = pd.DataFrame(columns=["Date", "regime"])
        placeholder_data.to_csv(file_path, index=False)
        return None

    return df

def detect_changepoints(feature_series, method="pelt", penalty=10, min_size=None):
    try:
        import ruptures as rpt
        feature_series = pd.Series(feature_series).dropna()
        feature_series_1d = feature_series.values.flatten()

        algo = {
            "pelt": rpt.Pelt(model="rbf", min_size=min_size) if min_size else rpt.Pelt(model="rbf"),
            "binseg": rpt.Binseg(model="rbf"),
            "window": rpt.Window(model="rbf")
        }.get(method)

        if not algo:
            logger.error(f"Unknown changepoint method: {method}")
            return []

        algo = algo.fit(feature_series_1d)
        changepoints = algo.predict(pen=penalty) if method == "pelt" else algo.predict(n_bkps=penalty)
        return changepoints
    except Exception as e:
        logger.error(f"Changepoint detection failed: {e}")
        return []

def save_changepoints(ticker, changepoints, feature_series, date_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"changepoints_{ticker}.csv")
    logger.info(f"Saving changepoints to {output_path}")
    rows = []
    prev_idx = 0
    for idx in changepoints:
        segment = feature_series[prev_idx:idx]
        rows.append({
            "start_idx": prev_idx,
            "end_idx": idx,
            "start_time": date_index[prev_idx],
            "end_time": date_index[idx-1] if idx-1 < len(date_index) else date_index[-1],
            "mean": segment.mean(),
            "std": segment.std()
        })
        prev_idx = idx
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f"Successfully saved changepoints for {ticker} to {output_path}")
    else:
        logger.warning(f"No rows to save for changepoints of {ticker}")

def get_params_for_freq(freq):
    """Return penalty and min_size based on frequency."""
    if freq == "1h":
        return 30, 24   # penalty, min_size (1 day for 1h data)
    else:
        return 10, 5    # penalty, min_size (5 days for 1d data)

def preprocess_macro_features(ticker, start_date, end_date):
    """Fetch and preprocess macro features."""
    macro_features = ["interest_rate", "inflation", "vix_index"]
    macro_data = fetch_macro_from_fred(macro_features, start_date)

    # Ensure Date column exists in macro data
    if 'Date' not in macro_data.columns:
        macro_data['Date'] = pd.to_datetime(macro_data.index)

    return macro_data

# Update changepoint detection logic to use macro features only
def detect_changepoints_with_macro(ticker, start_date, end_date, method="pelt", penalty=10):
    macro_data = preprocess_macro_features(ticker, start_date, end_date)

    for column in macro_data.columns:
        feature_series = macro_data[column]
        changepoints = detect_changepoints(feature_series, method=method, penalty=penalty)
        logger.info(f"Detected changepoints for {column}: {changepoints}")

    return changepoints

# Move pipeline termination logic inside a function
def terminate_pipeline_if_no_changepoints(ticker, changepoints):
    if not changepoints:
        logger.info(f"No changepoints detected for {ticker}, terminating pipeline.")
        return True
    return False

if __name__ == "__main__":
    # Set frequency here: "1d" or "1h"
    freq = "1d"  # Change to "1h" for hourly data

    # Load tickers from config
    config_path = os.path.join(os.path.dirname(__file__), "..", "data", "config_data.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    tickers = config["tickers"]

    markov_output_dir = os.path.join(os.path.dirname(__file__), "markov_output")
    output_dir = os.path.join(os.path.dirname(__file__), "changepoint_output")

    # Choose which feature to use: 'returns', 'volatility', 'momentum', 'composite', 'fundamental_composite'
    feature_choice = "composite"  # or "returns", "volatility", "fundamental_composite", etc.

    penalty, min_size = get_params_for_freq(freq)

    for ticker in tickers:
        df = load_markov_output(ticker, markov_output_dir)
        if df is None or df.empty:
            continue
        # Ensure Date column is datetime64[ns] before merging
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # --- Merge macro, trend data only (skip fundamental CSV, as Markov output already has all features) ---
        base_dir = os.path.dirname(__file__)
        macro_path = os.path.join(base_dir, "..", "data", "historical_data", "macro_data.csv")
        trend_path = os.path.join(base_dir, "..", "data", "historical_data", "google_trend.csv")
        # Only merge macro, trend data
        if os.path.exists(macro_path):
            macro_df = pd.read_csv(macro_path, parse_dates=["Date"])
            df = df.merge(macro_df, on="Date", how="left")
            logger.info(f"Merged macro data for {ticker}")
        if os.path.exists(trend_path) and os.path.getsize(trend_path) > 0:
            try:
                trend_df = pd.read_csv(trend_path, parse_dates=["Date"])
                df = df.merge(trend_df, on="Date", how="left")
                logger.info(f"Merged trend data for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to load trend data: {e}")
    
        # Drop features with excessive missingness
        threshold = 0.5  # Drop columns with more than 50% missing values
        df = df.dropna(axis=1, thresh=int(threshold * len(df)))
        logger.info(f"Dropped columns with excessive missingness for {ticker}. Remaining columns: {df.columns.tolist()}")

        # Ensure no all-zeros columns remain
        def is_all_zeros(series):
            return (series == 0).all()
        all_zero_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and is_all_zeros(df[col])]
        if all_zero_columns:
            logger.warning(f"Columns with all zeros detected for {ticker}: {all_zero_columns}. Dropping these columns.")
            df = df.drop(columns=all_zero_columns)

        # Only fill NaNs in numeric columns that will be used for the composite feature
        exclude_cols = ['Date', 'regime']
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols]
        valid_cols = [col for col in numeric_cols if not df[col].isna().all() and not (df[col] == 0).all()]

        # Fill NaNs in these columns with the column median (only if there are any NaNs)
        for col in valid_cols:
            if df[col].isna().any():
                col_median = df[col].median()
                df[col].fillna(col_median, inplace=True)

        if not valid_cols:
            logger.warning(f"No valid numeric columns for composite feature for {ticker}. Skipping changepoint detection.")
            continue

        # Standardize columns
        standardized = df[valid_cols].apply(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))
        # Drop columns with zero variance after standardization
        zero_var_cols = [col for col in standardized.columns if standardized[col].std() == 0]
        if zero_var_cols:
            logger.warning(f"Dropping features with zero variance after standardization for {ticker}: {zero_var_cols}")
            standardized = standardized.drop(columns=zero_var_cols)
        if standardized.shape[1] == 0:
            logger.warning(f"No valid standardized columns for composite feature for {ticker}. Skipping changepoint detection.")
            continue

        composite = standardized.sum(axis=1)
        feature_series = composite.dropna()
        if feature_series.empty:
            logger.warning(f"Composite feature is empty for {ticker}. Skipping changepoint detection.")
            continue

        changepoints = detect_changepoints(feature_series, method="pelt", penalty=penalty)
        logger.info(f"Detected changepoints for {ticker}: {changepoints}")
        save_changepoints(ticker, changepoints, feature_series, df["Date"].values, output_dir)