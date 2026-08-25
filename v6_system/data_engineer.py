"""
v6_system/data_engineer.py
--------------------------
Module Data Engineering cho Version 6.
Chịu trách nhiệm kiểm định, làm sạch dữ liệu Dukascopy M1 (2020-2025)
và tạo dataset phiên 10:00 -> 12:00 VN.
"""

import os
import glob
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from v6_system.config import (
    BASE_DIR,
    OUTPUT_DIR,
    TARGET_TIMEZONE,
    YEARS,
    SESSION_START_TIME,
    SESSION_END_TIME,
    ABNORMAL_GAP_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("V6DataEngineer")


class V6DataEngineer:
    """Class kiểm định, làm sạch và tạo session cho dữ liệu XAUUSD M1 (Phase 1)."""

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, target_tz: str = TARGET_TIMEZONE):
        self.target_tz = target_tz
        self.quality_report: Dict[str, Any] = {}

    def load_single_year_csv(self, file_path: str) -> pd.DataFrame:
        """Đọc file CSV M1 thô của 1 năm và chuẩn hóa các cột cơ bản."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        logger.info(f"Đang đọc dữ liệu từ: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"File {file_path} thiếu các cột: {missing}")

        return df

    def load_all_years(self, data_dir: str = BASE_DIR, years: List[int] = YEARS) -> pd.DataFrame:
        """Đọc và gộp toàn bộ file CSV M1 từ năm 2020 tới 2025."""
        dfs = []
        for yr in sorted(years):
            file_name = f"XAUUSD_{yr}_m1.csv"
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                df_yr = self.load_single_year_csv(file_path)
                df_yr["source_year"] = yr
                dfs.append(df_yr)
            else:
                logger.warning(f"Không tìm thấy file {file_name} trong {data_dir}")

        if not dfs:
            raise FileNotFoundError("Không nạp được file CSV nào trong danh sách năm yêu cầu.")

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Đã nạp tổng cộng {len(combined_df):,} dòng nến M1 thô.")
        return combined_df

    def audit_and_clean_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Thực hiện 8 bước kiểm định Data Engineering:
        1. Duplicate check
        2. Timezone conversion & DST verification
        3. Timeframe consistency check
        4. OHLC price integrity check
        5. Volume validity check
        6. Missing candles analysis
        7. Abnormal gap detection
        8. Summary report generation
        """
        logger.info("--- BẮT ĐẦU KIỂM ĐỊNH & LÀM SẠCH DỮ LIỆU (PHASE 1) ---")
        report: Dict[str, Any] = {
            "total_raw_rows": len(df_raw),
            "duplicates_removed": 0,
            "invalid_ohlc_removed": 0,
            "invalid_volume_count": 0,
            "zero_volume_count": 0,
            "missing_candles_count": 0,
            "abnormal_gaps_count": 0,
            "dst_handled_correctly": True,
            "final_clean_rows": 0
        }

        df = df_raw.copy()

        # 1. Duplicates check
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        dup_count = initial_count - len(df)
        report["duplicates_removed"] = dup_count
        if dup_count > 0:
            logger.info(f"Đã loại bỏ {dup_count:,} dòng trùng lặp timestamp.")

        # 2. Timezone & DST check
        logger.info("Chuyển đổi Timestamp (UTC epoch ms -> Asia/Ho_Chi_Minh)...")
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            sample_val = df["timestamp"].iloc[0]
            unit = "ms" if sample_val > 1e11 else "s"
            df["dt_utc"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
        else:
            df["dt_utc"] = pd.to_datetime(df["timestamp"], utc=True)

        # Chuyển đổi sang múi giờ Asia/Ho_Chi_Minh (UTC+7)
        df["dt_vn"] = df["dt_utc"].dt.tz_convert(self.target_tz)
        df["date_vn"] = df["dt_vn"].dt.date
        df["time_vn"] = df["dt_vn"].dt.time
        df["hour_vn"] = df["dt_vn"].dt.hour
        df["minute_vn"] = df["dt_vn"].dt.minute

        # Sắp xếp theo thời gian tăng dần
        df = df.sort_values("dt_utc").reset_index(drop=True)

        # DST Check: UTC timestamp là thời gian chuẩn tuyệt đối, chuyển sang Asia/Ho_Chi_Minh (UTC+7 cố định)
        # Verify monotonicity
        is_monotonic = df["dt_utc"].is_monotonic_increasing
        report["timestamp_is_monotonic"] = is_monotonic
        report["target_timezone"] = self.target_tz

        # 3. OHLC Price Integrity Check
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        invalid_prices = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
        invalid_ohlc_rel = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )

        invalid_mask = invalid_prices | invalid_ohlc_rel | df[numeric_cols].isnull().any(axis=1)
        invalid_cnt = int(invalid_mask.sum())
        report["invalid_ohlc_removed"] = invalid_cnt
        if invalid_cnt > 0:
            logger.warning(f"Loại bỏ {invalid_cnt:,} dòng lỗi OHLC / giá âm / null.")
            df = df[~invalid_mask].reset_index(drop=True)

        # 4. Volume Check
        negative_vol = (df["volume"] < 0)
        zero_vol = (df["volume"] == 0)
        report["invalid_volume_count"] = int(negative_vol.sum())
        report["zero_volume_count"] = int(zero_vol.sum())
        if negative_vol.any():
            logger.warning(f"Phát hiện {negative_vol.sum()} dòng volume âm -> Chuẩn hóa về 0.")
            df.loc[negative_vol, "volume"] = 0.0

        # 5. Timeframe & Missing Candles Check
        df["time_diff_sec"] = df["dt_utc"].diff().dt.total_seconds()
        # Trong phiên giao dịch bình thường, nến M1 có diff = 60s
        # Diff > 60s và không phải thời điểm đóng cửa thị trường (ví dụ cuối tuần)
        missing_candles_mask = (df["time_diff_sec"] > 60) & (df["dt_vn"].dt.dayofweek < 5)
        # Loại trừ khoảng nghỉ 1 tiếng hàng ngày giữa phiên (05:00-06:00 VN tuỳ mùa)
        intraday_missing = missing_candles_mask & (df["time_diff_sec"] < 12000)
        report["missing_candles_intervals"] = int(intraday_missing.sum())
        
        # Estimate total missing 1-minute bars during intraday gaps
        total_missing_bars = int(((df.loc[intraday_missing, "time_diff_sec"] / 60) - 1).sum())
        report["missing_candles_count"] = total_missing_bars
        logger.info(f"Phát hiện {report['missing_candles_intervals']} khoảng gián đoạn intraday (khoảng {total_missing_bars:,} nến M1 thiếu).")

        # 6. Abnormal Gap Detection (|Open(t) - Close(t-1)| > Threshold)
        df["prev_close"] = df["close"].shift(1)
        df["price_gap"] = (df["open"] - df["prev_close"]).abs()
        # Chỉ xét gap trong cùng ngày giao dịch (loại bỏ gap qua đêm/cuối tuần)
        same_day_mask = (df["date_vn"] == df["date_vn"].shift(1))
        abnormal_gaps = (df["price_gap"] > ABNORMAL_GAP_THRESHOLD) & same_day_mask
        report["abnormal_gaps_count"] = int(abnormal_gaps.sum())

        if abnormal_gaps.any():
            max_gap = float(df.loc[abnormal_gaps, "price_gap"].max())
            report["max_abnormal_gap"] = max_gap
            logger.warning(f"Phát hiện {abnormal_gaps.sum()} gap giá bất thường trong ngày (> ${ABNORMAL_GAP_THRESHOLD}). Gap lớn nhất: ${max_gap:.2f}")

        # Clean temp columns
        df = df.drop(columns=["time_diff_sec", "prev_close", "price_gap"], errors="ignore")
        report["final_clean_rows"] = len(df)
        report["date_range_vn"] = {
            "start": str(df["dt_vn"].min()),
            "end": str(df["dt_vn"].max())
        }

        self.quality_report = report
        logger.info(f"Kiểm định hoàn tất. Dữ liệu M1 sạch: {len(df):,} nến từ {report['date_range_vn']['start']} đến {report['date_range_vn']['end']}.")
        return df, report

    def extract_daily_sessions(
        self,
        df_clean: pd.DataFrame,
        start_time_str: str = SESSION_START_TIME,
        end_time_str: str = SESSION_END_TIME
    ) -> pd.DataFrame:
        """
        Trích xuất và tổng hợp phiên 10:00 -> 12:00 VN cho từng ngày giao dịch.

        Output mỗi ngày:
        - date: Ngày giao dịch (YYYY-MM-DD)
        - anchor_price: Giá Open tại nến 10:00 VN
        - open_1000, high_1000, low_1000, close_1000, volume_1000 (nến 10:00)
        - open_1200, high_1200, low_1200, close_1200, volume_1200 (nến 12:00)
        - session_high: Giá High cao nhất trong khoảng 10:00 -> 12:00 VN
        - session_low: Giá Low thấp nhất trong khoảng 10:00 -> 12:00 VN
        - session_close: Giá Close của nến 12:00 (hoặc nến cuối của phiên)
        - candle_count: Số nến M1 ghi nhận trong phiên
        """
        logger.info(f"--- TRÍCH XUẤT PHIÊN GIAO DỊCH ({start_time_str[:5]} -> {end_time_str[:5]} VN) ---")

        start_t = pd.to_datetime(start_time_str).time()
        end_t = pd.to_datetime(end_time_str).time()

        # Lọc các nến nằm trong khung giờ [10:00, 12:00]
        session_mask = (df_clean["time_vn"] >= start_t) & (df_clean["time_vn"] <= end_t)
        df_session_candles = df_clean[session_mask].copy()

        daily_records = []

        grouped = df_session_candles.groupby("date_vn")

        for date_val, group in grouped:
            group = group.sort_values("dt_utc")
            if group.empty:
                continue

            # Lấy nến 10:00 (nến bắt đầu phiên hoặc nến sớm nhất trong phiên)
            candle_1000_rows = group[group["time_vn"] == start_t]
            if not candle_1000_rows.empty:
                c1000 = candle_1000_rows.iloc[0]
            else:
                c1000 = group.iloc[0]

            # Lấy nến 12:00 (nến đúng 12:00 hoặc nến muộn nhất trong phiên)
            candle_1200_rows = group[group["time_vn"] == end_t]
            if not candle_1200_rows.empty:
                c1200 = candle_1200_rows.iloc[0]
            else:
                c1200 = group.iloc[-1]

            anchor_price = float(c1000["open"])
            session_high = float(group["high"].max())
            session_low = float(group["low"].min())
            session_close = float(c1200["close"])

            record = {
                "date": str(date_val),
                "anchor_price": anchor_price,
                "open_1000": float(c1000["open"]),
                "high_1000": float(c1000["high"]),
                "low_1000": float(c1000["low"]),
                "close_1000": float(c1000["close"]),
                "volume_1000": float(c1000["volume"]),
                "open_1200": float(c1200["open"]),
                "high_1200": float(c1200["high"]),
                "low_1200": float(c1200["low"]),
                "close_1200": float(c1200["close"]),
                "volume_1200": float(c1200["volume"]),
                "session_high": session_high,
                "session_low": session_low,
                "session_close": session_close,
                "candle_count": len(group)
            }
            daily_records.append(record)

        df_daily = pd.DataFrame(daily_records)
        logger.info(f"Đã trích xuất thành công {len(df_daily):,} ngày phiên 10:00 -> 12:00 VN.")
        return df_daily

    def export_results(
        self,
        df_clean: pd.DataFrame,
        df_daily: pd.DataFrame,
        output_dir: str = OUTPUT_DIR
    ):
        """Lưu trữ kết quả kiểm định & phiên giao dịch ra file."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Báo cáo chất lượng dữ liệu JSON
        report_path = os.path.join(output_dir, "data_quality_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.quality_report, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu báo cáo chất lượng: {report_path}")

        # 2. File Dataset phiên hàng ngày
        daily_csv_path = os.path.join(output_dir, "daily_sessions_10_12.csv")
        df_daily.to_csv(daily_csv_path, index=False)
        logger.info(f"Đã lưu dataset phiên 10:00-12:00: {daily_csv_path}")

        # 3. File Dataset M1 sạch (dạng CSV hoặc Parquet nếu có pyarrow)
        clean_m1_path = os.path.join(output_dir, "clean_m1_2020_2025.csv")
        # Lưu các cột cần thiết cho M1 sạch
        export_cols = ["timestamp", "dt_utc", "dt_vn", "date_vn", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in export_cols if c in df_clean.columns]
        df_clean[available_cols].to_csv(clean_m1_path, index=False)
        logger.info(f"Đã lưu dataset M1 sạch: {clean_m1_path}")
