import os
import pandas as pd
import yaml
from utils.logger import get_logger
from pandas_datareader import data as pdr
from pytrends.request import TrendReq
from urllib3.util.retry import Retry

logger = get_logger(__name__)

FRED_CODES = {
    "interest_rate": "FEDFUNDS",
    "inflation": "CPIAUCSL",
    "vix_index": "VIXCLS"
}

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_macro_from_fred(features, start_date="2000-01-01"):
    df = pd.DataFrame()
    for feature in features:
        fred_code = FRED_CODES.get(feature)
        if fred_code is None:
            logger.warning(f"No FRED code for macro feature: {feature}")
            continue
        try:
            logger.info(f"Fetching {feature} ({fred_code}) from FRED starting {start_date}...")
            series = pdr.DataReader(fred_code, "fred", start=start_date)
            series = series.rename(columns={fred_code: feature})
            if df.empty:
                df = series
            else:
                df = df.join(series, how="outer")
        except Exception as e:
            logger.error(f"Failed to fetch {feature} from FRED: {e}")
    df = df.reset_index().rename(columns={"DATE": "Date"})
    return df

def fetch_trends(keywords, start_date, end_date, geo="US"):
    """Fetch Google Trends data with robust error handling"""
    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10,25), retries=0, backoff_factor=0.1)
        timeframe = f"{start_date} {end_date}"
        all_trends = pd.DataFrame()

        for kw in keywords:
            try:
                logger.info(f"Fetching Google Trends for: {kw}")
                pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo=geo, gprop="")
                trend = pytrends.interest_over_time()

                if not trend.empty:
                    trend = trend.drop(columns=["isPartial"], errors="ignore")
                    trend = trend.rename(columns={kw: f"trend_{kw}"})
                    all_trends = pd.concat([all_trends, trend], axis=1)
            except Exception as e:
                logger.error(f"Failed to fetch trends for {kw}: {e}")

        if not all_trends.empty:
            all_trends = all_trends.reset_index().rename(columns={"date": "Date"})
        return all_trends

    except Exception as e:
        logger.error(f"Google Trends fetch failed: {e}")
        return pd.DataFrame()

def get_macro_data(config_path=None):
    """
    Load macro data and Google Trends data with robust error handling
    
    Args:
        config_path (str): Path to config file, if None uses default
        
    Returns:
        tuple: (macro_df, trends_df) - DataFrames with macro and trends data
    """
    try:
        # Determine paths
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
        else:
            base_dir = os.path.dirname(config_path)
            
        macro_output_path = os.path.join(base_dir, "data", "historical_data", "macro_data.csv")
        google_trend_path = os.path.join(base_dir, "data", "historical_data", "google_trend.csv")
        
        # Load macro data
        macro_df = pd.DataFrame()
        if os.path.exists(macro_output_path):
            try:
                macro_df = pd.read_csv(macro_output_path)
                macro_df['Date'] = pd.to_datetime(macro_df['Date'])
                logger.info(f"Loaded macro data: {macro_df.shape}")
            except Exception as e:
                logger.error(f"Error loading macro data: {e}")
        else:
            logger.warning(f"Macro data file not found: {macro_output_path}")
        
        # Load Google Trends data with validation
        trends_df = pd.DataFrame()
        if os.path.exists(google_trend_path):
            try:
                trends_df = pd.read_csv(google_trend_path)
                
                # Validate Date column exists
                if 'Date' not in trends_df.columns:
                    logger.warning("Google Trends CSV missing 'Date' column, attempting to fix")
                    # Try to use first column as Date if it looks like a date
                    if len(trends_df.columns) > 0:
                        first_col = trends_df.columns[0]
                        try:
                            pd.to_datetime(trends_df[first_col])
                            trends_df = trends_df.rename(columns={first_col: 'Date'})
                            logger.info("Fixed Google Trends Date column")
                        except:
                            logger.error("Could not fix Google Trends Date column")
                            trends_df = pd.DataFrame()
                
                if not trends_df.empty and 'Date' in trends_df.columns:
                    trends_df['Date'] = pd.to_datetime(trends_df['Date'])
                    logger.info(f"Loaded Google Trends data: {trends_df.shape}")
                else:
                    logger.warning("Google Trends data is empty or invalid")
                    trends_df = pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error loading Google Trends data: {e}")
                trends_df = pd.DataFrame()
        else:
            logger.warning(f"Google Trends file not found: {google_trend_path}")
        
        return macro_df, trends_df
        
    except Exception as e:
        logger.error(f"Error in get_macro_data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, "forcasting_ml", "train_config.yaml")
    macro_config_path = os.path.join(base_dir, "data", "config_data.yaml")
    macro_output_path = os.path.join(base_dir, "data", "historical_data", "macro_data.csv")
    google_trend_path = os.path.join(base_dir, "data", "historical_data", "google_trend.csv")

    # Load macro features and tickers from config
    config = load_yaml_config(config_path)
    macro_config = load_yaml_config(macro_config_path)
    macro_features = config.get("features", {}).get("macro_features", [])
    tickers = macro_config.get("tickers", [])
    start_date = macro_config.get("start_date", "2000-01-01")
    end_date = macro_config.get("end_date", "2025-05-29")
    logger.info(f"Macro features requested: {macro_features}")
    logger.info(f"Tickers for trends: {tickers}")
    logger.info(f"Using start_date: {start_date}, end_date: {end_date}")

    # Fetch macro data from FRED
    macro_df = fetch_macro_from_fred(macro_features, start_date=start_date)
    logger.info(f"Fetched macro data shape: {macro_df.shape}")

    # Save macro data only (no trends)
    macro_df.to_csv(macro_output_path, index=False)
    logger.info(f"Saved macro data to {macro_output_path}")

    # Fetch and save Google Trends for macro features and tickers
    search_terms = macro_features + tickers
    trends_df = fetch_trends(search_terms, start_date, end_date)
    logger.info(f"Fetched Google Trends data shape: {trends_df.shape}")

    # Ensure at least a header is written even if DataFrame is empty
    if trends_df.empty:
        columns = ["Date"] + [f"trend_{kw}" for kw in search_terms]
        trends_df = pd.DataFrame(columns=columns)
    trends_df.to_csv(google_trend_path, index=False)
    logger.info(f"Saved Google Trends data to {google_trend_path}")
if __name__ == "__main__":
    main()