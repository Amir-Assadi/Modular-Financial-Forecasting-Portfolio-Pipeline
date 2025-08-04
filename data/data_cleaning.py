import os
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = ["Date", "Close", "High", "Low", "Open", "Volume"]

def clean_dataframe(df):
    # Ensure all columns are strings
    df.columns = [str(col) for col in df.columns]
    # Standardize column names (case-insensitive)
    col_map = {c.lower(): c for c in EXPECTED_COLUMNS}
    df = df.rename(columns={c: col_map.get(str(c).lower(), c) for c in df.columns})
    # Handle 'timestamp' column from live data
    if "timestamp" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"timestamp": "Date"})
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    # Drop rows with zero volume
    df = df[df["Volume"] != 0]
    # Forward fill missing values
    df[["Close", "High", "Low", "Open", "Volume"]] = df[["Close", "High", "Low", "Open", "Volume"]].ffill()
    # Drop any remaining NaNs
    df = df.dropna(subset=["Close", "High", "Low", "Open", "Volume"])
    # Ensure correct column order
    df = df[EXPECTED_COLUMNS]
    return df

def clean_data(data_dir="historical_data"):
    """
    Clean all CSVs in a directory (for historical or live data).
    Always outputs with 'Date' as the date column.
    """
    data_dir = os.path.join(os.path.dirname(__file__), data_dir)
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            try:
                # Attempt to read the CSV file
                df = pd.read_csv(file_path, skiprows=3, header=None)
                df.columns = EXPECTED_COLUMNS

                # Clean the DataFrame
                df = clean_dataframe(df)

                # Save cleaned data with the correct header and column order
                df.to_csv(file_path, index=False, columns=EXPECTED_COLUMNS)
                logger.info(f"Cleaned {filename}")
            except Exception as e:
                logger.error(f"Failed to clean {filename}: {e}")
