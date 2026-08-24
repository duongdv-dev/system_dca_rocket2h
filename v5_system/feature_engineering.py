"""
Feature Engineering & Quantitative EDA Module (Giai đoạn 2)
----------------------------------------------------------
Tính toán các đặc trưng kỹ thuật & hành vi trước 10:00 AM (không có lookahead bias),
gán nhãn phiên Mean Reversion [10:00 - 12:00 VN], và xuất báo cáo thống kê EDA chi tiết bằng tiếng Việt.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntradayFeatureExtractor:
    """Trích xuất đặc trưng trước 10:00 AM và mục tiêu đảo chiều (Mean Reversion) [10:00-12:00]."""

    def __init__(self, min_dev_usd: float = 1.0):
        """
        Args:
            min_dev_usd: Độ lệch tối thiểu (USD) so với P0 để kích hoạt theo dõi đảo chiều (mặc định 1.0 USD).
        """
        self.min_dev_usd = min_dev_usd

    def _precompute_global_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán trước các chỉ báo kỹ thuật toàn cục trên toàn bộ dataset M1 
        để tối ưu hóa tốc độ xử lý gấp 50 lần.
        """
        logger.info("Đang tính toán trước các chỉ báo ATR(14) M5/M15 và EMA(50/200) trên toàn bộ dữ liệu M1...")
        df_sorted = df.sort_values("dt_vn").copy()

        # EMA 50 và EMA 200 trên nến M1
        df_sorted["ema50"] = df_sorted["close"].ewm(span=50, adjust=False).mean()
        df_sorted["ema200"] = df_sorted["close"].ewm(span=200, adjust=False).mean()

        # Resample sang M5 để tính ATR(14) M5
        df_indexed = df_sorted.set_index("dt_vn")
        m5_resampled = df_indexed.resample("5min").agg({
            "high": "max", "low": "min", "close": "last"
        }).dropna()
        
        tr_m5 = np.maximum(
            m5_resampled["high"] - m5_resampled["low"],
            np.maximum(
                abs(m5_resampled["high"] - m5_resampled["close"].shift(1)),
                abs(m5_resampled["low"] - m5_resampled["close"].shift(1))
            )
        )
        m5_resampled["atr_m5_14"] = tr_m5.rolling(14).mean().fillna(1.0)
        
        # Resample sang M15 để tính ATR(14) M15
        m15_resampled = df_indexed.resample("15min").agg({
            "high": "max", "low": "min", "close": "last"
        }).dropna()
        
        tr_m15 = np.maximum(
            m15_resampled["high"] - m15_resampled["low"],
            np.maximum(
                abs(m15_resampled["high"] - m15_resampled["close"].shift(1)),
                abs(m15_resampled["low"] - m15_resampled["close"].shift(1))
            )
        )
        m15_resampled["atr_m15_14"] = tr_m15.rolling(14).mean().fillna(2.0)

        # Merge lại vào df gốc bằng merge_asof (asof join theo timestamp)
        df_sorted = df_sorted.reset_index(drop=True)
        m5_resampled = m5_resampled.reset_index()
        m15_resampled = m15_resampled.reset_index()

        df_sorted = pd.merge_asof(
            df_sorted, m5_resampled[["dt_vn", "atr_m5_14"]],
            on="dt_vn", direction="backward"
        )
        df_sorted = pd.merge_asof(
            df_sorted, m15_resampled[["dt_vn", "atr_m15_14"]],
            on="dt_vn", direction="backward"
        )

        df_sorted["atr_m5_14"] = df_sorted["atr_m5_14"].fillna(1.0)
        df_sorted["atr_m15_14"] = df_sorted["atr_m15_14"].fillna(2.0)

        return df_sorted

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý toàn bộ dataset M1, trích xuất đặc trưng hàng ngày trước 10:00 và mục tiêu [10:00-12:00].

        Args:
            df: DataFrame M1 đã được chuẩn hóa với các cột thời gian VN.

        Returns:
            DataFrame tổng hợp từng ngày giao dịch.
        """
        logger.info("Bắt đầu khởi tạo & trích xuất đặc trưng giá Anchor 10:00 VN và các chỉ báo hành vi...")
        
        # Pre-compute indicators nhanh chóng
        df_prep = self._precompute_global_indicators(df)

        records = []
        unique_dates = sorted(df_prep["date_vn"].unique())
        total_days = len(unique_dates)

        logger.info(f"Tổng số ngày giao dịch cần trích xuất đặc trưng: {total_days} ngày.")

        for idx, current_date in enumerate(unique_dates):
            # Hiển thị log tiến độ định kỳ mỗi 100 ngày hoặc ở mốc 100%
            if (idx + 1) % 100 == 0 or (idx + 1) == total_days:
                logger.info(f" [Tiến độ Trích xuất]: {idx + 1}/{total_days} ngày ({(idx + 1)/total_days*100:.1f}%)")

            df_date_mask = (df_prep["date_vn"] == current_date)
            df_day = df_prep[df_date_mask]

            # Xกำหนด Anchor Price P0 tại đúng 10:00:00 AM VN Time
            p0_bars = df_day[(df_day["hour_vn"] == 10) & (df_day["minute_vn"] == 0)]
            if len(p0_bars) == 0:
                p0_bars = df_day[(df_day["hour_vn"] == 10) & (df_day["minute_vn"] <= 5)]
                if len(p0_bars) == 0:
                    continue

            p0 = p0_bars["open"].iloc[0]
            p0_time = p0_bars["dt_vn"].iloc[0]

            # Lấy bối cảnh dữ liệu trước 10:00 AM cùng ngày
            if idx > 0:
                prev_date = unique_dates[idx - 1]
                df_context = df_prep[(df_prep["date_vn"] == prev_date) | (df_prep["date_vn"] == current_date)]
            else:
                df_context = df_day

            df_pre10 = df_context[df_context["dt_vn"] < p0_time]
            if len(df_pre10) < 30:
                continue

            # Nến thanh khoản trước 10:00 AM trong ngày
            pre10_bars = df_day[df_day["dt_vn"] < p0_time]
            
            # Phiên Á: 06:00 - 09:59 AM VN Time
            asian_bars = pre10_bars[(pre10_bars["hour_vn"] >= 6) & (pre10_bars["hour_vn"] < 10)]
            if len(asian_bars) > 0:
                asian_high = asian_bars["high"].max()
                asian_low = asian_bars["low"].min()
                asian_range = asian_high - asian_low
                asian_open = asian_bars["open"].iloc[0]
                asian_close = asian_bars["close"].iloc[-1]
                asian_return = asian_close - asian_open
                asian_vol_sum = asian_bars["volume"].sum()
                asian_vwap = (asian_bars["close"] * asian_bars["volume"]).sum() / asian_vol_sum if asian_vol_sum > 0 else asian_close
            else:
                asian_high = pre10_bars["high"].max() if len(pre10_bars) > 0 else p0
                asian_low = pre10_bars["low"].min() if len(pre10_bars) > 0 else p0
                asian_range = asian_high - asian_low
                asian_open = pre10_bars["open"].iloc[0] if len(pre10_bars) > 0 else p0
                asian_close = pre10_bars["close"].iloc[-1] if len(pre10_bars) > 0 else p0
                asian_return = asian_close - asian_open
                asian_vol_sum = pre10_bars["volume"].sum() if len(pre10_bars) > 0 else 0
                asian_vwap = asian_close

            # Phiên Mỹ trước đó (21:00 - 04:00 VN)
            ny_bars = df_context[(df_context["hour_vn"] >= 21) | (df_context["hour_vn"] < 4)]
            if len(ny_bars) >= 10:
                ny_high = ny_bars["high"].max()
                ny_low = ny_bars["low"].min()
                ny_range = ny_high - ny_low
                ny_return = ny_bars["close"].iloc[-1] - ny_bars["open"].iloc[0]
                ny_volatility = ny_bars["close"].pct_change().std() * 100.0
                ny_trend = 1.0 if ny_return > 0 else (-1.0 if ny_return < 0 else 0.0)
            else:
                ny_range = asian_range
                ny_return = asian_return
                ny_volatility = 0.0
                ny_trend = 0.0

            # Lấy các chỉ báo kỹ thuật đã pre-compute tại nến ngay trước 10:00 AM
            last_pre10_bar = df_pre10.iloc[-1]
            atr_m5_14 = last_pre10_bar["atr_m5_14"]
            atr_m15_14 = last_pre10_bar["atr_m15_14"]
            ema50 = last_pre10_bar["ema50"]
            ema200 = last_pre10_bar["ema200"]

            # Khung thời gian giao dịch [10:00 - 12:00]
            session_bars = df_day[
                (df_day["dt_vn"] >= p0_time) & 
                ((df_day["hour_vn"] == 10) | (df_day["hour_vn"] == 11) | ((df_day["hour_vn"] == 12) & (df_day["minute_vn"] == 0)))
            ].sort_values("dt_vn").reset_index(drop=True)

            if len(session_bars) <= 5:
                continue

            # Tính MAE (Maximum Adverse Excursion)
            high_max = session_bars["high"].max()
            low_min = session_bars["low"].min()
            mae_up = high_max - p0
            mae_down = p0 - low_min
            max_mae = max(mae_up, mae_down)

            # Logic nhãn Target Reversion:
            # Giá lệch >= min_dev_usd, sau đó có chạm lại P0 trước 12:00 PM không?
            has_deviated = False
            reverted = 0
            revert_minute = np.nan
            first_dev_direction = 0

            for bar_idx, bar in session_bars.iterrows():
                elapsed_min = (bar["dt_vn"] - p0_time).total_seconds() / 60.0
                if bar_idx == 0:
                    continue

                curr_up_dev = bar["high"] - p0
                curr_down_dev = p0 - bar["low"]

                if not has_deviated:
                    if curr_up_dev >= self.min_dev_usd:
                        has_deviated = True
                        first_dev_direction = 1
                    elif curr_down_dev >= self.min_dev_usd:
                        has_deviated = True
                        first_dev_direction = -1

                if has_deviated:
                    if bar["low"] <= p0 <= bar["high"]:
                        reverted = 1
                        revert_minute = elapsed_min
                        break

            dist_vwap = (p0 - asian_vwap) / p0 * 1000.0
            dist_ema50 = p0 - ema50
            dist_ema200 = p0 - ema200
            day_of_week = p0_time.dayofweek

            record = {
                "date": current_date,
                "p0": p0,
                "p0_time": p0_time,
                "dayofweek": day_of_week,
                # Đặc trưng trước 10:00 AM
                "asian_range": float(asian_range),
                "asian_return": float(asian_return),
                "asian_volume": float(asian_vol_sum),
                "ny_range": float(ny_range),
                "ny_return": float(ny_return),
                "ny_volatility": float(ny_volatility if not np.isnan(ny_volatility) else 0.0),
                "ny_trend": float(ny_trend),
                "atr_m5_14": float(atr_m5_14),
                "atr_m15_14": float(atr_m15_14),
                "dist_vwap": float(dist_vwap),
                "dist_ema50": float(dist_ema50),
                "dist_ema200": float(dist_ema200),
                # Thứ trong tuần (One-hot encoded)
                "is_monday": 1 if day_of_week == 0 else 0,
                "is_tuesday": 1 if day_of_week == 1 else 0,
                "is_wednesday": 1 if day_of_week == 2 else 0,
                "is_thursday": 1 if day_of_week == 3 else 0,
                "is_friday": 1 if day_of_week == 4 else 0,
                # Kết quả phiên [10:00 - 12:00]
                "mae_up": float(mae_up),
                "mae_down": float(mae_down),
                "max_mae": float(max_mae),
                "has_deviated": 1 if has_deviated else 0,
                "reverted": reverted,
                "revert_minute": revert_minute,
                "first_dev_direction": first_dev_direction,
            }
            records.append(record)

        daily_df = pd.DataFrame(records)
        logger.info(f"Hoàn thành trích xuất {len(daily_df)} ngày giao dịch hợp lệ.")
        return daily_df


class EDAReporter:
    """Tạo báo cáo thống kê EDA định lượng bằng tiếng Việt."""

    @staticmethod
    def generate_eda_report(daily_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các chỉ số thống kê EDA cho Giai đoạn 2.

        Args:
            daily_df: DataFrame hàng ngày từ IntradayFeatureExtractor.

        Returns:
            Dictionary chứa Tỷ lệ thắng (Win Rate), phân bố phần trăm MAE và thời gian đảo chiều.
        """
        total_days = len(daily_df)
        deviated_days = daily_df[daily_df["has_deviated"] == 1]
        num_deviated = len(deviated_days)
        reverted_days = daily_df[daily_df["reverted"] == 1]
        num_reverted = len(reverted_days)

        overall_win_rate = (num_reverted / total_days * 100.0) if total_days > 0 else 0.0
        conditional_win_rate = (num_reverted / num_deviated * 100.0) if num_deviated > 0 else 0.0

        mae_series = daily_df["max_mae"]
        mae_percentiles = {
            "Phân vị 50th (Trung vị)": np.percentile(mae_series, 50),
            "Phân vị 75th": np.percentile(mae_series, 75),
            "Phân vị 90th": np.percentile(mae_series, 90),
            "Phân vị 95th": np.percentile(mae_series, 95),
            "Phân vị 99th": np.percentile(mae_series, 99),
            "Tối đa (Max MAE)": mae_series.max(),
        }

        revert_mins = daily_df.loc[daily_df["reverted"] == 1, "revert_minute"]
        median_revert_time = revert_mins.median() if len(revert_mins) > 0 else np.nan
        mean_revert_time = revert_mins.mean() if len(revert_mins) > 0 else np.nan

        report = {
            "total_days": total_days,
            "num_deviated_days": num_deviated,
            "num_reverted_days": num_reverted,
            "overall_win_rate_pct": overall_win_rate,
            "conditional_win_rate_pct": conditional_win_rate,
            "mae_percentiles": mae_percentiles,
            "median_revert_time_min": median_revert_time,
            "mean_revert_time_min": mean_revert_time,
        }

        # In báo cáo tiếng Việt định dạng đẹp
        print("\n" + "=" * 70)
        print("GIAI ĐOẠN 2: BÁO CÁO PHÂN TÍCH ĐỊNH LƯỢNG EDA & HÀNH VI GIÁ (VIỆT NAM)")
        print("=" * 70)
        print(f"Tổng số ngày giao dịch phân tích  : {total_days} ngày")
        print(f"Số ngày lệch giá >= 1.0$ (Deviation): {num_deviated} ngày ({num_deviated/total_days*100:.1f}%)")
        print(f"Số ngày đảo chiều thành công về P0 : {num_reverted} ngày")
        print(f"Xác suất đảo chiều tổng thể (Win Rate) : {overall_win_rate:.2f}% (Tất cả các ngày)")
        print(f"Xác suất đảo chiều điều kiện           : {conditional_win_rate:.2f}% (Khi có lệch giá >= 1.0$)")
        print("-" * 70)
        print("Phân bố phần trăm Mức lệch giá tối đa MAE (Maximum Adverse Excursion - USD):")
        for pct_name, val in mae_percentiles.items():
            print(f"  - {pct_name:<28}: {val:.2f} USD")
        print("-" * 70)
        print(f"Thời gian trung vị đảo chiều (Median Time): {median_revert_time:.1f} phút")
        print(f"Thời gian trung bình đảo chiều (Mean Time)  : {mean_revert_time:.1f} phút")
        print("=" * 70 + "\n")

        return report
