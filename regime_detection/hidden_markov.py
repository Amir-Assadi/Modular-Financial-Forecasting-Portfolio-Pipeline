import os
import pandas as pd
import numpy as np
import yaml
import joblib
from hmmlearn.hmm import GaussianHMM
from utils.logger import get_logger
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_features(df, freq="1h"):
    """
    Add features for regime detection (HMM, changepoint, etc.).
    freq: "1h" for hourly, "1d" for daily.
    """
    df = df.copy()
    freq_params = {
        "1h": {"sma_1": 24, "sma_2": 72, "rsi_n": 7, "vol_win": 24, "mom_win": 24},
        "1d": {"sma_1": 50, "sma_2": 150, "rsi_n": 14, "vol_win": 10, "mom_win": 10}
    }
    params = freq_params.get(freq, freq_params["1d"])

    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=params["vol_win"]).std()
    df["momentum"] = df["Close"] - df["Close"].shift(params["mom_win"])
    df["sma_50"] = df["Close"].rolling(window=params["sma_1"]).mean()
    df["sma_150"] = df["Close"].rolling(window=params["sma_2"]).mean()
    df["macd"] = (
        df["Close"].ewm(span=12, adjust=False).mean() -
        df["Close"].ewm(span=26, adjust=False).mean()
    )
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(params["rsi_n"]).mean()
    roll_down = down.rolling(params["rsi_n"]).mean()
    rs = roll_up / roll_down
    df["rsi_14"] = 100 - (100 / (1 + rs))
    feature_cols = ["returns", "volatility", "momentum", "sma_50", "sma_150", "macd", "rsi_14"]
    df = df.dropna(subset=feature_cols)
    return df

def train_hmm_on_ticker(ticker, data_dir, n_states=3, covariance_type='full', n_iter=500):
    # Output directories
    markov_output_dir = os.path.join(os.path.dirname(__file__), "markov_output")
    markov_prob_output_dir = os.path.join(os.path.dirname(__file__), "markov_prob_output")
    hmm_models_dir = os.path.join(os.path.dirname(__file__), "hmm_models")
    os.makedirs(markov_output_dir, exist_ok=True)
    os.makedirs(markov_prob_output_dir, exist_ok=True)
    os.makedirs(hmm_models_dir, exist_ok=True)

    file_path = os.path.join(data_dir, f"{ticker}.csv")
    base_dir = os.path.dirname(__file__)

    if not os.path.exists(file_path):
        logger.error(f"Data file for {ticker} not found at {file_path}, skipping.")
        return

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"Data file for {ticker} is empty: {file_path}. Creating placeholder data.")
            placeholder_data = pd.DataFrame(columns=["Date", "Close", "High", "Low", "Open", "Volume"])
            placeholder_data.to_csv(file_path, index=False)
            return
    except Exception as e:
        logger.error(f"Failed to parse data file for {ticker}: {e}. Creating placeholder file.")
        placeholder_data = pd.DataFrame(columns=["Date", "Close", "High", "Low", "Open", "Volume"])
        placeholder_data.to_csv(file_path, index=False)
        return

    # Merge macro data only (macro_data.csv contains all macro/trend/VIX)
    macro_path = os.path.join(base_dir, "..", "data", "historical_data", "macro_data.csv")
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path, parse_dates=["Date"])
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        if "Date" in macro_df.columns:
            macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        df = df.merge(macro_df, on="Date", how="left")
        logger.info(f"Merged macro data for HMM on {ticker}")

    # Drop all-zero columns
    def is_all_zeros(series):
        return (series == 0).all()
    all_zero_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and is_all_zeros(df[col])]
    if all_zero_columns:
        logger.warning(f"Columns with all zeros detected for {ticker} after consolidation: {all_zero_columns}. Dropping these columns.")
        df = df.drop(columns=all_zero_columns)

    # Fill NaNs with column median for numeric columns (except 'Date' and 'regime')
    exclude_cols = ['Date', 'regime']
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols]
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Build feature list: numeric, not all-NaN, not all-zero, not datetime, not 'Date'/'regime'
    cleaned_numeric_features = [
        col for col in numeric_cols
        if not df[col].isna().all() and not (df[col] == 0).all() and not np.issubdtype(df[col].dtype, np.datetime64)
    ]
    feature_cols = cleaned_numeric_features

    if not feature_cols:
        logger.error(f"No valid features for HMM on {ticker}. Skipping.")
        return

    features = df[feature_cols].values
    logger.info(f"Final numeric features for HMM on {ticker}: {feature_cols}")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Drop features with zero variance after scaling
    zero_var_indices = [i for i, std in enumerate(features_scaled.std(axis=0)) if std == 0]
    if zero_var_indices:
        dropped_features = [feature_cols[i] for i in zero_var_indices]
        logger.warning(f"Dropping features with zero variance after scaling: {dropped_features}")
        keep_indices = [i for i in range(len(feature_cols)) if i not in zero_var_indices]
        features_scaled = features_scaled[:, keep_indices]
        feature_cols = [feature_cols[i] for i in keep_indices]

    valid_rows = ~np.isnan(features_scaled).any(axis=1)
    if not valid_rows.any():
        logger.error("No valid rows available for HMM training due to NaN values after final cleaning.")
        df['regime'] = np.nan
        regime_path = os.path.join(markov_output_dir, f"{ticker}_regimes.csv")
        prob_path = os.path.join(markov_prob_output_dir, f"{ticker}_regime_probs.csv")
        df[['Date', 'regime']].to_csv(regime_path, index=False)
        df[['Date']].to_csv(prob_path, index=False)
        return

    # Use function arguments for HMM parameters
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=42)
    model.fit(features_scaled[valid_rows])
    logger.info(f"Converged: {model.monitor_.converged}, Iterations: {model.monitor_.iter}")

    regimes = np.full(len(df), np.nan)
    regimes[valid_rows] = model.predict(features_scaled[valid_rows])
    df['regime'] = regimes

    # Save regime output
    cols_to_save = ['Date'] + feature_cols + ['regime']
    regime_path = os.path.join(markov_output_dir, f"{ticker}_regimes.csv")
    df[cols_to_save].to_csv(regime_path, index=False)

    # Save regime probabilities
    prob_path = os.path.join(markov_prob_output_dir, f"{ticker}_regime_probs.csv")
    probs = np.full((len(df), n_states), np.nan)
    probs_valid = model.predict_proba(features_scaled[valid_rows])
    probs[valid_rows, :] = probs_valid
    prob_cols = [f"regime_{i}_prob" for i in range(n_states)]
    prob_df = pd.DataFrame(probs, columns=prob_cols)
    prob_df.insert(0, 'Date', df['Date'].values)
    prob_df.to_csv(prob_path, index=False)

    # Save model and scaler
    hmm_model_path = os.path.join(hmm_models_dir, f"{ticker}_hmm_model.pkl")
    joblib.dump({"model": model, "scaler": scaler}, hmm_model_path)

    logger.info(f"HMM regime detection complete for {ticker}. Outputs saved.")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "data", "config_data.yaml")
    config = load_config(config_path)
    tickers = config["tickers"]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "historical_data")

    # Set your HMM parameters here if you want to change them
    n_states = 10
    covariance_type = 'full'
    n_iter = 2000

    for ticker in tickers:
        try:
            train_hmm_on_ticker(ticker, data_dir, n_states=n_states, covariance_type=covariance_type, n_iter=n_iter)
        except Exception as e:
            logger.error(f"Error training HMM on {ticker}: {e}")