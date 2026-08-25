"""
v6_system/feature_engineering.py
--------------------------------
Module Feature Engineering cho Version 6 Phase 5 XGBoost V1 (Vectorized & Fast).
Trích xuất 25 đặc trưng kỹ thuật và gán nhãn Target xác suất hồi về Anchor.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("V6FeatureEngineer")


class V6FeatureEngineer:
    """Class tính toán 25 đặc trưng và gán nhãn Target cho mô hình ML Gatekeeper (Vectorized)."""

    def __init__(self, rsi_period: int = 14, atr_period: int = 14):
        self.rsi_period = rsi_period
        self.atr_period = atr_period

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo kỹ thuật bằng các thao tác numpy/pandas vectorized hiệu năng cao."""
        df = df.copy()

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        open_p = df["open"].values
        volume = df["volume"].values

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        # ATR 14
        high_low = high - low
        high_cp = np.abs(high - prev_close)
        low_cp = np.abs(low - prev_close)
        tr = np.maximum(high_low, np.maximum(high_cp, low_cp))
        
        df["atr"] = pd.Series(tr, index=df.index).rolling(self.atr_period, min_periods=1).mean().values
        df["atr_norm"] = df["atr"].values / (close + 1e-6)

        # RSI 14
        delta = pd.Series(close, index=df.index).diff().values
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain, index=df.index).rolling(self.rsi_period, min_periods=1).mean().values
        roll_loss = pd.Series(loss, index=df.index).rolling(self.rsi_period, min_periods=1).mean().values
        rs = roll_gain / (roll_loss + 1e-6)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # ADX 14
        prev_high = np.roll(high, 1); prev_high[0] = high[0]
        prev_low = np.roll(low, 1); prev_low[0] = low[0]
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        atr_vals = df["atr"].values + 1e-6
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean().values / atr_vals
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean().values / atr_vals
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        df["adx"] = pd.Series(dx, index=df.index).rolling(14, min_periods=1).mean().values

        # EMAs
        close_s = pd.Series(close, index=df.index)
        df["ema_9"] = close_s.ewm(span=9, adjust=False).mean().values
        df["ema_21"] = close_s.ewm(span=21, adjust=False).mean().values
        df["ema_50"] = close_s.ewm(span=50, adjust=False).mean().values

        # EMA Slope
        ema9_s = pd.Series(df["ema_9"].values, index=df.index)
        df["ema_slope"] = ((ema9_s - ema9_s.shift(5)) / 5.0).fillna(0.0).values

        # Volume ratio
        vol_s = pd.Series(volume, index=df.index)
        df["vol_ma_20"] = vol_s.rolling(20, min_periods=1).mean().values
        df["vol_over_avg"] = volume / (df["vol_ma_20"].values + 1e-6)

        # Returns & Volatility
        df["return_1m"] = close / (open_p + 1e-6) - 1.0
        df["return_5m"] = (close_s / (close_s.shift(5) + 1e-6) - 1.0).fillna(0.0).values
        df["return_15m"] = (close_s / (close_s.shift(15) + 1e-6) - 1.0).fillna(0.0).values
        df["volatility"] = pd.Series(df["return_1m"].values, index=df.index).rolling(15, min_periods=1).std().fillna(0.0).values

        # Candle structure
        df["candle_body"] = np.abs(close - open_p)
        df["upper_wick"] = high - np.maximum(open_p, close)
        df["lower_wick"] = np.minimum(open_p, close) - low

        return df

    def extract_session_features_and_targets(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất 25 đặc trưng và gán nhãn Target bằng thuật toán Vectorized.
        """
        logger.info("Bắt đầu trích xuất 25 Features và gán nhãn Target (Vectorized Optimized)...")

        df_ind = self.compute_technical_indicators(df_m1)

        start_t = pd.to_datetime("10:00:00").time()
        end_t = pd.to_datetime("12:00:00").time()
        session_mask = (df_ind["time_vn"] >= start_t) & (df_ind["time_vn"] <= end_t)
        df_session = df_ind[session_mask].copy().reset_index(drop=True)

        session_dfs = []

        grouped = df_session.groupby("date_vn")

        for date_val, group in grouped:
            if group.empty:
                continue

            group = group.sort_values("dt_utc").copy()
            n = len(group)

            group["date"] = str(date_val)

            anchor_price = float(group.iloc[0]["open"])
            group["anchor_price"] = anchor_price

            close_vals = group["close"].values
            high_vals = group["high"].values
            low_vals = group["low"].values
            vol_vals = group["volume"].values

            group["distance_from_anchor"] = close_vals - anchor_price
            group["dist_anchor_over_atr"] = group["distance_from_anchor"].values / (group["atr"].values + 1e-6)

            pv = close_vals * vol_vals
            cum_pv = np.cumsum(pv)
            cum_vol = np.cumsum(vol_vals)
            group["vwap"] = cum_pv / (cum_vol + 1e-6)
            group["dist_to_vwap"] = close_vals - group["vwap"].values

            group["session_high"] = np.maximum.accumulate(high_vals)
            group["session_low"] = np.minimum.accumulate(low_vals)

            group["time_since_10"] = np.arange(n)
            group["time_remaining_12"] = n - 1 - np.arange(n)

            rev_low = low_vals[::-1]
            rev_high = high_vals[::-1]

            cum_min_low = np.minimum.accumulate(rev_low)[::-1]
            cum_max_high = np.maximum.accumulate(rev_high)[::-1]

            future_min_low = np.roll(cum_min_low, -1)
            future_min_low[-1] = low_vals[-1]

            future_max_high = np.roll(cum_max_high, -1)
            future_max_high[-1] = high_vals[-1]

            above_mask = close_vals > anchor_price
            below_mask = close_vals < anchor_price
            equal_mask = ~above_mask & ~below_mask

            target = np.zeros(n, dtype=int)
            target[above_mask & (future_min_low <= anchor_price)] = 1
            target[below_mask & (future_max_high >= anchor_price)] = 1
            target[equal_mask] = 1

            group["target_revert"] = target
            session_dfs.append(group)

        df_final = pd.concat(session_dfs, ignore_index=True)
        logger.info(f"Hoàn thành trích xuất {len(df_final):,} bản ghi đặc trưng cho Phase 5 (Vectorized).")
        return df_final
