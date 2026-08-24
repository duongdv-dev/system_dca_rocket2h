"""
DataLoader Module (Giai đoạn 1)
-------------------------------
Đọc dữ liệu XAUUSD M1 thô từ Dukascopy, chuẩn hóa schema,
chuyển đổi múi giờ từ UTC/GMT sang Giờ Việt Nam (UTC+7 / Asia/Ho_Chi_Minh),
và làm sạch dữ liệu mà không forward-fill qua các ngày nghỉ giao dịch.
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
    """Đọc và tiền xử lý dữ liệu M1 CSV từ Dukascopy cho chiến lược XAUUSD."""

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, target_tz: str = "Asia/Ho_Chi_Minh"):
        """
        Args:
            target_tz: Chuỗi múi giờ đích (mặc định: Asia/Ho_Chi_Minh cho UTC+7).
        """
        self.target_tz = target_tz

    def load_single_csv(self, file_path: str) -> pd.DataFrame:
        """
        Đọc một file CSV M1, chuẩn hóa schema và chuyển đổi múi giờ sang UTC+7.

        Args:
            file_path: Đường dẫn tuyệt đối hoặc tương đối tới file CSV.

        Returns:
            DataFrame pandas đã xử lý với các cột chuẩn hóa và index thời gian.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file CSV: {file_path}")

        logger.info(f"Đang đọc dữ liệu M1 từ file: {file_path}")
        
        try:
            # Đọc CSV hiệu quả
            df = pd.read_csv(file_path)
            
            # Chuẩn hóa tên cột (viết thường, xóa khoảng trắng thừa)
            df.columns = df.columns.str.strip().str.lower()

            # Kiểm tra các cột bắt buộc
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(f"File {file_path} thiếu các cột bắt buộc: {missing_cols}")

            # Parse cột Timestamp
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                sample_val = df["timestamp"].iloc[0]
                unit = "ms" if sample_val > 1e11 else "s"
                df["dt_utc"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
            else:
                df["dt_utc"] = pd.to_datetime(df["timestamp"], utc=True)

            # Chuyển đổi sang múi giờ Việt Nam (UTC+7)
            df["dt_vn"] = df["dt_utc"].dt.tz_convert(self.target_tz)

            # Trích xuất các cột thời gian tiện ích
            df["date_vn"] = df["dt_vn"].dt.date
            df["time_vn"] = df["dt_vn"].dt.time
            df["hour_vn"] = df["dt_vn"].dt.hour
            df["minute_vn"] = df["dt_vn"].dt.minute
            df["dayofweek"] = df["dt_vn"].dt.dayofweek  # 0 = Thứ 2, 4 = Thứ 6

            # Lọc trùng lặp và sắp xếp theo thời gian
            df = df.drop_duplicates(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)

            # Kiểm tra kiểu dữ liệu số
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=numeric_cols).reset_index(drop=True)

            # Lọc các dòng lỗi nến OHLC (High < Low, High < Open, High < Close)
            invalid_rows = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
            if invalid_rows.any():
                logger.warning(f"Phát hiện {invalid_rows.sum()} dòng có quan hệ OHLC không hợp lệ trong {file_path}. Đã loại bỏ.")
                df = df[~invalid_rows].reset_index(drop=True)

            logger.info(f"Tải thành công {len(df):,} nến M1. Khoảng thời gian: {df['dt_vn'].min()} đến {df['dt_vn'].max()}")
            return df

        except Exception as e:
            logger.error(f"Lỗi khi đọc file CSV {file_path}: {e}")
            raise

    def load_multiple_csvs(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Đọc và gộp nhiều file CSV M1 theo thứ tự thời gian.

        Args:
            file_paths: Danh sách đường dẫn file CSV.

        Returns:
            DataFrame gộp hoàn chỉnh.
        """
        dfs = []
        for fp in sorted(file_paths):
            df = self.load_single_csv(fp)
            dfs.append(df)

        if not dfs:
            raise ValueError("Không có file CSV nào được tải.")

        combined_df = pd.concat(dfs, ignore_index=True)
        # Khử trùng lặp lần cuối giữa các file
        combined_df = combined_df.drop_duplicates(subset=["dt_utc"]).sort_values("dt_utc").reset_index(drop=True)
        logger.info(f"Tổng hợp dataset M1: {len(combined_df):,} nến từ {combined_df['date_vn'].min()} đến {combined_df['date_vn'].max()}")
        return combined_df

    def load_directory(self, directory_path: str, pattern: str = "XAUUSD_*_m1.csv", years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Quét thư mục tìm các file CSV phù hợp và nạp dữ liệu (hỗ trợ lọc theo năm).

        Args:
            directory_path: Đường dẫn thư mục.
            pattern: Mẫu glob tìm file.
            years: Danh sách năm cần lọc (Ví dụ: [2020, 2021]).

        Returns:
            DataFrame gộp hoàn chỉnh.
        """
        search_path = os.path.join(directory_path, pattern)
        matching_files = glob.glob(search_path)

        if not matching_files:
            raise FileNotFoundError(f"Không tìm thấy file nào khớp với mẫu '{pattern}' trong {directory_path}")

        # Lọc file theo danh sách năm nếu được chỉ định
        if years:
            years_str = [str(y) for y in years]
            filtered_files = [
                f for f in matching_files 
                if any(y_str in os.path.basename(f) for y_str in years_str)
            ]
            if filtered_files:
                matching_files = filtered_files
                logger.info(f"Đã lọc lấy các file CSV thuộc năm {years}: {[os.path.basename(f) for f in matching_files]}")
            else:
                logger.warning(f"Không tìm thấy file CSV nào khớp với các năm {years}. Sẽ dùng toàn bộ file tìm thấy.")

        logger.info(f"Tìm thấy {len(matching_files)} file CSV: {[os.path.basename(f) for f in matching_files]}")
        combined_df = self.load_multiple_csvs(matching_files)

        if years:
            combined_df = combined_df[combined_df["dt_vn"].dt.year.isin(years)].reset_index(drop=True)
            logger.info(f"Đã lọc dữ liệu theo năm {years}: {len(combined_df):,} nến M1.")

        return combined_df



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader = M1DataLoader()
    try:
        sample_file = os.path.join(base_dir, "XAUUSD_2020_m1.csv")
        if os.path.exists(sample_file):
            df_test = loader.load_single_csv(sample_file)
            print(df_test.head())
        else:
            print("Không tìm thấy file mẫu XAUUSD_2020_m1.csv trong thư mục gốc.")
    except Exception as err:
        print(f"Kiểm tra DataLoader thất bại: {err}")

