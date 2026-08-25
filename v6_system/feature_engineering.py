"""
v6_system/feature_engineering.py
--------------------------------
Module Feature Engineering cho Version 6 Phase 5 XGBoost V1.
Trích xuất 25 đặc trưng kỹ thuật và gán nhãn Target xác suất hồi về Anchor.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("V6FeatureEngineer")


class V6FeatureEngineer:
    """Class tính toán 25 đặc trưng và gán nhãn Target cho mô hình ML Gatekeeper."""

    def __init__(self, rsi_period: int = 14, atr_period: int = 14):
        self.rsi_period = rsi_period
        self.atr_period = atr_period

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo kỹ thuật trên toàn bộ dữ liệu chuỗi thời gian M1."""
        df = df.copy()

        # ATR 14
        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift(1)).abs()
        low_cp = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period, min_periods=1).mean()
        df["atr_norm"] = df["atr"] / (df["close"] + 1e-6)

        # RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period, min_periods=1).mean()
        rs = gain / (loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ADX 14 (Đơn giản hóa)
        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / (df["atr"] + 1e-6)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / (df["atr"] + 1e-6)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-6)
        df["adx"] = dx.rolling(14, min_periods=1).mean()

        # EMA 9, 21, 50
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # EMA Slope (độ dốc EMA9 5 nến)
        df["ema_slope"] = (df["ema_9"] - df["ema_9"].shift(5)) / 5.0

        # Volume MA 20
        df["vol_ma_20"] = df["volume"].rolling(20, min_periods=1).mean()
        df["vol_over_avg"] = df["volume"] / (df["vol_ma_20"] + 1e-6)

        # Returns & Volatility
        df["return_1m"] = df["close"] / (df["open"] + 1e-6) - 1.0
        df["return_5m"] = df["close"] / (df["close"].shift(5) + 1e-6) - 1.0
        df["return_15m"] = df["close"] / (df["close"].shift(15) + 1e-6) - 1.0
        df["volatility"] = df["return_1m"].rolling(15, min_periods=1).std()

        # Candle structure
        df["candle_body"] = (df["close"] - df["open"]).abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        return df

    def extract_session_features_and_targets(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất 25 đặc trưng và gán nhãn Target xác suất hồi Anchor cho phiên 10:00 -> 12:00 VN.
        """
        logger.info("Bắt đầu trích xuất 25 Features và gán nhãn Target cho Phase 5...")
        
        # Tính toán chỉ báo trên chuỗi thời gian liên tục
        df_indicators = self.compute_technical_indicators(df_m1)

        start_t = pd.to_datetime("10:00:00").time()
        end_t = pd.to_datetime("12:00:00").time()
        session_mask = (df_indicators["time_vn"] >= start_t) & (df_indicators["time_vn"] <= end_t)
        df_session_candles = df_indicators[session_mask].copy()

        feature_records = []

        grouped = df_session_candles.groupby("date_vn")

        for date_val, group in grouped:
            group = group.sort_values("dt_utc").reset_index(drop=True)
            if group.empty:
                continue

            anchor_price = float(group.iloc[0]["open"])
            total_session_minutes = len(group)

            # Tính VWAP lũy tiến trong phiên
            group["pv"] = group["close"] * group["volume"]
            group["cum_pv"] = group["pv"].cumsum()
            group["cum_vol"] = group["volume"].cumsum()
            group["vwap"] = group["cum_pv"] / (group["cum_vol"] + 1e-6)
            group["dist_to_vwap"] = group["close"] - group["vwap"]

            # Session Cumulative High / Low
            group["session_high"] = group["high"].cummax()
            group["session_low"] = group["low"].cummin()

            for i in range(len(group)):
                row = group.iloc[i]
                current_price = float(row["close"])

                # Khoảng cách tới Anchor
                dist_anchor = current_price - anchor_price
                dist_anchor_over_atr = dist_anchor / (float(row["atr"]) + 1e-6)

                # Thời gian
                time_since_10 = i  # Số phút từ 10:00
                time_remaining_12 = total_session_minutes - 1 - i

                # Gán nhãn TARGET `y`: Liệu trong tương lai [i+1, end] giá có chạm Anchor không?
                future_candles = group.iloc[i + 1:]
                target_revert = 0

                if future_candles.empty:
                    target_revert = 1 if abs(current_price - anchor_price) < 0.20 else 0
                else:
                    if current_price > anchor_price:
                        # Cần có nến tương lai với Low <= Anchor
                        if (future_candles["low"] <= anchor_price).any():
                            target_revert = 1
                    elif current_price < anchor_price:
                        # Cần có nến tương lai với High >= Anchor
                        if (future_candles["high"] >= anchor_price).any():
                            target_revert = 1
                    else:
                        target_revert = 1

                rec = {
                    "date": str(date_val),
                    "dt_vn": str(row["dt_vn"]),
                    "anchor_price": anchor_price,
                    "close": current_price,
                    # 25 Features
                    "distance_from_anchor": dist_anchor,
                    "dist_anchor_over_atr": dist_anchor_over_atr,
                    "atr": float(row["atr"]),
                    "atr_norm": float(row["atr_norm"]),
                    "rsi": float(row["rsi"]),
                    "adx": float(row["adx"]),
                    "ema_9": float(row["ema_9"]),
                    "ema_21": float(row["ema_21"]),
                    "ema_50": float(row["ema_50"]),
                    "ema_slope": float(row["ema_slope"]),
                    "volume": float(row["volume"]),
                    "vol_over_avg": float(row["vol_over_avg"]),
                    "return_1m": float(row["return_1m"]),
                    "return_5m": float(row["return_5m"]),
                    "return_15m": float(row["return_15m"]),
                    "volatility": float(row["volatility"]),
                    "time_since_10": time_since_10,
                    "time_remaining_12": time_remaining_12,
                    "session_high": float(row["session_high"]),
                    "session_low": float(row["session_low"]),
                    "candle_body": float(row["candle_body"]),
                    "upper_wick": float(row["upper_wick"]),
                    "lower_wick": float(row["lower_wick"]),
                    "vwap": float(row["vwap"]),
                    "dist_to_vwap": float(row["dist_to_vwap"]),
                    # Target
                    "target_revert": target_revert
                }
                feature_records.append(rec)

        df_feat = pd.DataFrame(feature_records)
        logger.info(f"Hoàn thành trích xuất {len(df_feat):,} bản ghi đặc trưng cho Phase 5.")
        return df_feat
