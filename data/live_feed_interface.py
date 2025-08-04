import os
import time
import yaml
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import get_logger
from data.data_cleaning import clean_dataframe

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_yfinance_data(symbol, interval, start=None, end=None):
    import yfinance as yf
    kwargs = {
        "tickers": symbol,
        "interval": interval,
        "progress": False
    }
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    df = yf.download(**kwargs)
    if df.empty:
        return df
    df = df.reset_index()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns.values]
    # Normalize all columns to uppercase
    df.columns = [col.upper() for col in df.columns]
    # Rename timestamp
    if "DATETIME" in df.columns:
        df = df.rename(columns={"DATETIME": "TIMESTAMP"})
    elif "DATE" in df.columns:
        df = df.rename(columns={"DATE": "TIMESTAMP"})
    # Remove ticker suffixes from price columns
    suffix = f"_{symbol.upper()}"
    rename_map = {}
    for col in df.columns:
        if col.endswith(suffix):
            rename_map[col] = col.replace(suffix, "")
    df = df.rename(columns=rename_map)
    expected_cols = ["TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        logger.error(f"{symbol}: Missing columns after fetch: {missing}")
        return df
    df = df[expected_cols]
    df = df.rename(columns={
        "TIMESTAMP": "Date",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "VOLUME": "Volume"
    })
    # Ensure correct column order
    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]
    return df

def save_live_data(ticker, df, save_dir):
    file_path = os.path.join(save_dir, f"{ticker}.csv")
    df = df[df["Date"].notnull()]
    df = df[df["Date"] != ticker]
    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]
    df["Date"] = df["Date"].astype(str)
    df = df.sort_values("Date")
    df.to_csv(file_path, index=False)

def backfill_data(tickers, interval, end_date, save_dir):
    now = datetime.now()
    # Remove old files before backfill
    for ticker in tickers:
        file_path = os.path.join(save_dir, f"{ticker}.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
    for ticker in tickers:
        try:
            logger.info(f"Backfilling {ticker} from {end_date} to {now.date()}")
            df = fetch_yfinance_data(ticker, interval, start=end_date, end=now.strftime("%Y-%m-%d"))
            if df.empty:
                logger.warning(f"No backfill data for {ticker}")
                continue
            try:
                df = clean_dataframe(df)
            except Exception as e:
                logger.error(f"Cleaning failed for {ticker} during backfill: {e}")
                continue
            # Ensure Date is string and filter
            df["Date"] = df["Date"].astype(str)
            df = df[df["Date"] >= end_date]
            save_live_data(ticker, df, save_dir)
            logger.info(f"Backfilled and saved data for {ticker}")
        except Exception as e:
            logger.error(f"Backfill error for {ticker}: {e}")

def get_sleep_seconds(interval):
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    elif interval.endswith("h"):
        return int(interval[:-1]) * 3600
    elif interval.endswith("d"):
        return int(interval[:-1]) * 86400
    else:
        return 60  # default 1 minute

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../Data/config_data.yaml")
    config = load_config(config_path)
    tickers = config["tickers"]
    interval = config["live_data"]["interval"]
    end_date = config["end_date"]

    save_dir = os.path.join(os.path.dirname(__file__), "live_data")
    os.makedirs(save_dir, exist_ok=True)

    # --- Backfill before starting live feed ---
    if end_date:
        backfill_data(tickers, interval, end_date, save_dir)

    logger.info("Starting live data feed. Press Ctrl+C to stop.")
    try:
        while True:
            for ticker in tickers:
                try:
                    file_path = os.path.join(save_dir, f"{ticker}.csv")
                    # Only fetch new data since last date in file
                    if os.path.exists(file_path):
                        existing_df = pd.read_csv(file_path)
                        if not existing_df.empty:
                            last_date = existing_df["Date"].max()
                            fetch_start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        else:
                            fetch_start = end_date
                    else:
                        fetch_start = end_date

                    df = fetch_yfinance_data(ticker, interval, start=fetch_start)
                    if df.empty:
                        logger.warning(f"No new data for {ticker}")
                        continue
                    try:
                        df = clean_dataframe(df)
                    except Exception as e:
                        logger.error(f"Cleaning failed for {ticker}: {e}")
                        continue
                    df["Date"] = df["Date"].astype(str)
                    df = df[df["Date"] >= fetch_start]
                    # Append only new rows
                    if os.path.exists(file_path):
                        existing_df = pd.read_csv(file_path)
                        combined = pd.concat([existing_df, df]).drop_duplicates(subset="Date").sort_values("Date")
                        combined.to_csv(file_path, index=False)
                    else:
                        df.to_csv(file_path, index=False)
                    logger.info(f"Fetched, cleaned, and saved live data for {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
            sleep_time = get_sleep_seconds(interval)
            logger.info(f"Sleeping for {sleep_time} seconds to sync with interval.")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Live data feed stopped by user.")

if __name__ == "__main__":
    main()