import yfinance as yf
import pandas as pd
import yaml
import os

from utils.logger import get_logger
from data.data_cleaning import clean_dataframe

logger = get_logger(__name__)  # This will use the central logger setup

def download_historical_data(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_data.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return

    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    interval = config["interval"]

    save_dir = os.path.join(os.path.dirname(__file__), "historical_data")
    os.makedirs(save_dir, exist_ok=True)

    for ticker in tickers:
        try:
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            if not data.empty:
                data = data.reset_index()
                # Simplify logic for dropping rows and resetting index
                data = data.iloc[3:].reset_index(drop=True)
                # Set columns to expected
                data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
                data = clean_dataframe(data)
                file_path = os.path.join(save_dir, f"{ticker}.csv")
                data.to_csv(file_path, index=False)
                logger.info(f"Saved data for {ticker} to {file_path}")
            else:
                logger.warning(f"No data was returned for {ticker}.")
        except Exception as e:
            logger.error(f"yfinance connection failed for {ticker}: {e}")

def main():
    download_historical_data()

if __name__ == "__main__":
    main()