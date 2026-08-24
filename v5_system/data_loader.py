"""
DataLoader Module (Stage 1)
---------------------------
Handles loading raw Dukascopy XAUUSD M1 data, column standardization,
timezone conversion from UTC/GMT to Vietnam Time (UTC+7 / Asia/Ho_Chi_Minh),
and data cleaning without forward-filling across non-trading days.
"""

import os
import glob
import logging
from typing import List, Union, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class M1DataLoader:
    """Loads and preprocesses Dukascopy M1 CSV data for XAUUSD strategy development."""

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, target_tz: str = "Asia/Ho_Chi_Minh"):
        """
        Args:
            target_tz: Target timezone string (default: Asia/Ho_Chi_Minh for UTC+7).
        """
        self.target_tz = target_tz

    def load_single_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads a single M1 CSV file, standardizes schema, and converts timezone to UTC+7.

        Args:
            file_path: Absolute or relative path to CSV file.

        Returns:
            Preprocessed pandas DataFrame with standardized columns and datetime index.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Loading M1 data from: {file_path}")
        
        try:
            # Read CSV efficiently
            df = pd.read_csv(file_path)
            
            # Standardize column headers (lowercase, strip whitespace)
            df.columns = df.columns.str.strip().str.lower()

            # Verify required columns exist
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(f"File {file_path} is missing required columns: {missing_cols}")

            # Parse Timestamp column
            # Dukascopy data usually stores timestamp in epoch milliseconds or ISO format
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                # Determine unit: milliseconds (13 digits) or seconds (10 digits)
                sample_val = df["timestamp"].iloc[0]
                unit = "ms" if sample_val > 1e11 else "s"
                df["dt_utc"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
            else:
                df["dt_utc"] = pd.to_datetime(df["timestamp"], utc=True)

            # Convert to target timezone (UTC+7 Vietnam Time)
            df["dt_vn"] = df["dt_utc"].dt.tz_convert(self.target_tz)

            # Extract convenience time component columns
            df["date_vn"] = df["dt_vn"].dt.date
            df["time_vn"] = df["dt_vn"].dt.time
            df["hour_vn"] = df["dt_vn"].dt.hour
            df["minute_vn"] = df["dt_vn"].dt.minute
            df["dayofweek"] = df["dt_vn"].dt.dayofweek  # 0 = Monday, 4 = Friday

            # Drop duplicates and sort chronologically
            df = df.drop_duplicates(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)

            # Basic data sanity checks
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=numeric_cols).reset_index(drop=True)

            # Sanity check High >= Low, High >= Open, High >= Close
            invalid_rows = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
            if invalid_rows.any():
                logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships in {file_path}. Filtering out.")
                df = df[~invalid_rows].reset_index(drop=True)

            logger.info(f"Successfully loaded {len(df):,} M1 bars. Range: {df['dt_vn'].min()} to {df['dt_vn'].max()}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise

    def load_multiple_csvs(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Loads and concatenates multiple M1 CSV files chronologically.

        Args:
            file_paths: List of CSV file paths.

        Returns:
            Combined pandas DataFrame.
        """
        dfs = []
        for fp in sorted(file_paths):
            df = self.load_single_csv(fp)
            dfs.append(df)

        if not dfs:
            raise ValueError("No CSV files loaded.")

        combined_df = pd.concat(dfs, ignore_index=True)
        # Final deduplication across boundaries
        combined_df = combined_df.drop_duplicates(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)
        logger.info(f"Total combined M1 dataset: {len(combined_df):,} bars from {combined_df['date_vn'].min()} to {combined_df['date_vn'].max()}")
        return combined_df

    def load_directory(self, directory_path: str, pattern: str = "XAUUSD_*_m1.csv") -> pd.DataFrame:
        """
        Scans a directory for matching CSV files and loads them.

        Args:
            directory_path: Directory path.
            pattern: File matching glob pattern.

        Returns:
            Combined pandas DataFrame.
        """
        search_path = os.path.join(directory_path, pattern)
        matching_files = glob.glob(search_path)

        if not matching_files:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {directory_path}")

        logger.info(f"Found {len(matching_files)} CSV files: {[os.path.basename(f) for f in matching_files]}")
        return self.load_multiple_csvs(matching_files)


if __name__ == "__main__":
    # Test DataLoader independently
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader = M1DataLoader()
    try:
        sample_file = os.path.join(base_dir, "XAUUSD_2020_m1.csv")
        if os.path.exists(sample_file):
            df_test = loader.load_single_csv(sample_file)
            print(df_test.head())
        else:
            print("Sample file XAUUSD_2020_m1.csv not found in base dir.")
    except Exception as err:
        print(f"DataLoader test failed: {err}")
