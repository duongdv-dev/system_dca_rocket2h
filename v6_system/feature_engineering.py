"""
v6_system/feature_engineering.py
--------------------------------
Module Feature Engineering cho Version 6 Phase 5 XGBoost V1 (Ultra Fast & Memory-Efficient).
Trích xuất 25 đặc trưng kỹ thuật và gán nhãn Target bằng 100% Vectorized pandas/numpy operations.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("V6FeatureEngineer")


class V6FeatureEngineer:
    """Class tính toán 25 đặc trưng và gán nhãn Target siêu nhanh, tối ưu bộ nhớ RAM."""

    def __init__(self, rsi_period: int = 14, atr_period: int = 14):
        self.rsi_period = rsi_period
        self.atr_period = atr_period

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán chỉ báo kỹ thuật với kiểu dữ liệu float32 tiết kiệm bộ nhớ."""
        high = df["high"].values.astype(np.float32)
        low = df["low"].values.astype(np.float32)
        close = df["close"].values.astype(np.float32)
        open_p = df["open"].values.astype(np.float32)
        volume = df["volume"].values.astype(np.float32)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        # ATR 14
        high_low = high - low
        high_cp = np.abs(high - prev_close)
        low_cp = np.abs(low - prev_close)
        tr = np.maximum(high_low, np.maximum(high_cp, low_cp))
        
        atr = pd.Series(tr, index=df.index).rolling(self.atr_period, min_periods=1).mean().values.astype(np.float32)
        df["atr"] = atr
        df["atr_norm"] = atr / (close + 1e-6)

        # RSI 14
        delta = pd.Series(close, index=df.index).diff().values
        gain = np.where(delta > 0, delta, 0.0).astype(np.float32)
        loss = np.where(delta < 0, -delta, 0.0).astype(np.float32)
        roll_gain = pd.Series(gain, index=df.index).rolling(self.rsi_period, min_periods=1).mean().values
        roll_loss = pd.Series(loss, index=df.index).rolling(self.rsi_period, min_periods=1).mean().values
        rs = roll_gain / (roll_loss + 1e-6)
        df["rsi"] = (100.0 - (100.0 / (1.0 + rs))).astype(np.float32)

        # ADX 14
        prev_high = np.roll(high, 1); prev_high[0] = high[0]
        prev_low = np.roll(low, 1); prev_low[0] = low[0]
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0).astype(np.float32)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0).astype(np.float32)
        
        atr_vals = atr + 1e-6
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean().values / atr_vals
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean().values / atr_vals
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        df["adx"] = pd.Series(dx, index=df.index).rolling(14, min_periods=1).mean().values.astype(np.float32)

        # EMAs
        close_s = pd.Series(close, index=df.index)
        df["ema_9"] = close_s.ewm(span=9, adjust=False).mean().values.astype(np.float32)
        df["ema_21"] = close_s.ewm(span=21, adjust=False).mean().values.astype(np.float32)
        df["ema_50"] = close_s.ewm(span=50, adjust=False).mean().values.astype(np.float32)

        # EMA Slope
        ema9_s = pd.Series(df["ema_9"].values, index=df.index)
        df["ema_slope"] = ((ema9_s - ema9_s.shift(5)) / 5.0).fillna(0.0).values.astype(np.float32)

        # Volume ratio
        vol_s = pd.Series(volume, index=df.index)
        vol_ma_20 = vol_s.rolling(20, min_periods=1).mean().values.astype(np.float32)
        df["vol_over_avg"] = volume / (vol_ma_20 + 1e-6)

        # Returns & Volatility
        df["return_1m"] = (close / (open_p + 1e-6) - 1.0).astype(np.float32)
        df["return_5m"] = ((close_s / (close_s.shift(5) + 1e-6) - 1.0).fillna(0.0).values).astype(np.float32)
        df["return_15m"] = ((close_s / (close_s.shift(15) + 1e-6) - 1.0).fillna(0.0).values).astype(np.float32)
        df["volatility"] = (pd.Series(df["return_1m"].values, index=df.index).rolling(15, min_periods=1).std().fillna(0.0).values).astype(np.float32)

        # Candle structure
        df["candle_body"] = np.abs(close - open_p).astype(np.float32)
        df["upper_wick"] = (high - np.maximum(open_p, close)).astype(np.float32)
        df["lower_wick"] = (np.minimum(open_p, close) - low).astype(np.float32)

        return df

    def extract_session_features_and_targets(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất 25 đặc trưng và gán nhãn Target bằng 100% Vectorized pandas transform (Không dùng loop).
        """
        logger.info("Bắt đầu trích xuất 25 Features và gán nhãn Target (100% Vectorized Optimized)...")

        df_ind = self.compute_technical_indicators(df_m1)

        start_t = pd.to_datetime("10:00:00").time()
        end_t = pd.to_datetime("12:00:00").time()
        session_mask = (df_ind["time_vn"] >= start_t) & (df_ind["time_vn"] <= end_t)
        df_session = df_ind[session_mask].copy().reset_index(drop=True)

        date_col = "date_vn" if "date_vn" in df_session.columns else "date"
        df_session["date"] = df_session[date_col].astype(str)

        # Vectorized Anchor & Session High/Low
        df_session["anchor_price"] = df_session.groupby(date_col)["open"].transform("first")
        close_vals = df_session["close"].values
        anchor_vals = df_session["anchor_price"].values

        df_session["distance_from_anchor"] = close_vals - anchor_vals
        df_session["dist_anchor_over_atr"] = df_session["distance_from_anchor"].values / (df_session["atr"].values + 1e-6)

        # Vectorized VWAP
        df_session["pv"] = df_session["close"] * df_session["volume"]
        df_session["cum_pv"] = df_session.groupby(date_col)["pv"].cumsum()
        df_session["cum_vol"] = df_session.groupby(date_col)["volume"].cumsum()
        df_session["vwap"] = df_session["cum_pv"] / (df_session["cum_vol"] + 1e-6)
        df_session["dist_to_vwap"] = df_session["close"] - df_session["vwap"]
        df_session.drop(columns=["pv", "cum_pv", "cum_vol"], inplace=True)

        df_session["session_high"] = df_session.groupby(date_col)["high"].cummax()
        df_session["session_low"] = df_session.groupby(date_col)["low"].cummin()

        df_session["time_since_10"] = df_session.groupby(date_col).cumcount()
        group_counts = df_session.groupby(date_col)["time_since_10"].transform("count")
        df_session["time_remaining_12"] = group_counts - 1 - df_session["time_since_10"]

        # Fast Vectorized Target Labeling per session using group transform
        session_dfs = []
        for date_val, group in df_session.groupby(date_col):
            n = len(group)
            low_vals = group["low"].values
            high_vals = group["high"].values
            c_vals = group["close"].values
            anc = group["anchor_price"].values[0]

            rev_low = low_vals[::-1]
            rev_high = high_vals[::-1]

            cum_min_low = np.minimum.accumulate(rev_low)[::-1]
            cum_max_high = np.maximum.accumulate(rev_high)[::-1]

            future_min_low = np.roll(cum_min_low, -1)
            future_min_low[-1] = low_vals[-1]

            future_max_high = np.roll(cum_max_high, -1)
            future_max_high[-1] = high_vals[-1]

            above_mask = c_vals > anc
            below_mask = c_vals < anc
            equal_mask = ~above_mask & ~below_mask

            target = np.zeros(n, dtype=np.int8)
            target[above_mask & (future_min_low <= anc)] = 1
            target[below_mask & (future_max_high >= anc)] = 1
            target[equal_mask] = 1

            group["target_revert"] = target
            session_dfs.append(group)

        df_final = pd.concat(session_dfs, ignore_index=True)
        logger.info(f"Hoàn thành trích xuất {len(df_final):,} bản ghi đặc trưng (Tối ưu RAM float32).")
        return df_final
