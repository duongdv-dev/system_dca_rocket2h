"""
Feature Engineering & Quantitative EDA Module (Stage 2)
------------------------------------------------------
Computes pre-10:00 AM features without lookahead bias, labels the 10:00-12:00 VN
Anchor Mean Reversion session, and computes comprehensive EDA statistics report.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntradayFeatureExtractor:
    """Extracts pre-10:00 AM behavioral features and 10:00-12:00 reversion targets."""

    def __init__(self, min_dev_usd: float = 1.0):
        """
        Args:
            min_dev_usd: Minimum USD deviation from P0 required to trigger reversion tracking (default 1.0 USD).
        """
        self.min_dev_usd = min_dev_usd

    def compute_indicators_pre1000(self, df_day: pd.DataFrame) -> Dict[str, float]:
        """
        Computes technical indicators strictly before 10:00 AM VN time for a single day / session context.
        
        Args:
            df_day: M1 DataFrame containing historical data up to 09:59 AM VN time.

        Returns:
            Dictionary of pre-10:00 technical features.
        """
        # Filter pre-10:00 bars for the current day (00:00 to 09:59)
        pre10_bars = df_day[df_day["hour_vn"] < 10].copy()
        
        if len(pre10_bars) == 0:
            return {}

        # Asian Session: 06:00 to 09:59 AM VN Time
        asian_bars = pre10_bars[(pre10_bars["hour_vn"] >= 6) & (pre10_bars["hour_vn"] < 10)]
        
        if len(asian_bars) > 0:
            asian_high = asian_bars["high"].max()
            asian_low = asian_bars["low"].min()
            asian_range = asian_high - asian_low
            asian_open = asian_bars["open"].iloc[0]
            asian_close = asian_bars["close"].iloc[-1]  # Price right before 10:00
            asian_return = asian_close - asian_open
            asian_vol_sum = asian_bars["volume"].sum()
            
            # Asian VWAP
            if asian_vol_sum > 0:
                asian_vwap = (asian_bars["close"] * asian_bars["volume"]).sum() / asian_vol_sum
            else:
                asian_vwap = asian_bars["close"].mean()
        else:
            asian_high = pre10_bars["high"].max()
            asian_low = pre10_bars["low"].min()
            asian_range = asian_high - asian_low
            asian_open = pre10_bars["open"].iloc[0]
            asian_close = pre10_bars["close"].iloc[-1]
            asian_return = asian_close - asian_open
            asian_vol_sum = pre10_bars["volume"].sum()
            asian_vwap = asian_close

        # Previous NY Session (21:00 - 04:00 VN Time if present in dataset)
        # Check if we have previous session bars in df_day
        ny_bars = df_day[(df_day["hour_vn"] >= 21) | (df_day["hour_vn"] < 4)]
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

        # Resample pre10_bars to M5 and M15 to calculate ATR(14)
        pre10_bars_indexed = pre10_bars.set_index("dt_vn")
        
        # M5 ATR
        m5_df = pre10_bars_indexed.resample("5min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(m5_df) >= 14:
            tr_m5 = np.maximum(
                m5_df["high"] - m5_df["low"],
                np.maximum(
                    abs(m5_df["high"] - m5_df["close"].shift(1)),
                    abs(m5_df["low"] - m5_df["close"].shift(1))
                )
            )
            atr_m5_14 = tr_m5.rolling(14).mean().iloc[-1]
        else:
            atr_m5_14 = (m5_df["high"] - m5_df["low"]).mean() if len(m5_df) > 0 else 1.0

        # M15 ATR
        m15_df = pre10_bars_indexed.resample("15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(m15_df) >= 14:
            tr_m15 = np.maximum(
                m15_df["high"] - m15_df["low"],
                np.maximum(
                    abs(m15_df["high"] - m15_df["close"].shift(1)),
                    abs(m15_df["low"] - m15_df["close"].shift(1))
                )
            )
            atr_m15_14 = tr_m15.rolling(14).mean().iloc[-1]
        else:
            atr_m15_14 = (m15_df["high"] - m15_df["low"]).mean() if len(m15_df) > 0 else 2.0

        # M1 EMA 50 & 200 calculated up to 09:59 AM
        if len(pre10_bars) >= 50:
            ema50 = pre10_bars["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        else:
            ema50 = pre10_bars["close"].mean()

        if len(pre10_bars) >= 200:
            ema200 = pre10_bars["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        else:
            ema200 = pre10_bars["close"].mean()

        return {
            "asian_range": float(asian_range),
            "asian_return": float(asian_return),
            "asian_volume": float(asian_vol_sum),
            "asian_vwap": float(asian_vwap),
            "ny_range": float(ny_range),
            "ny_return": float(ny_return),
            "ny_volatility": float(ny_volatility if not np.isnan(ny_volatility) else 0.0),
            "ny_trend": float(ny_trend),
            "atr_m5_14": float(atr_m5_14 if not np.isnan(atr_m5_14) else 1.0),
            "atr_m15_14": float(atr_m15_14 if not np.isnan(atr_m15_14) else 2.0),
            "ema50": float(ema50),
            "ema200": float(ema200),
        }

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes entire M1 dataset, extracting daily pre-10:00 features and 10:00-12:00 targets.

        Args:
            df: Cleaned M1 DataFrame with `date_vn`, `hour_vn`, `minute_vn`, `time_vn`.

        Returns:
            Daily aggregated DataFrame ready for Quantitative EDA & Machine Learning.
        """
        logger.info("Extracting daily Anchor features and 10:00-12:00 reversion targets...")
        
        records = []
        unique_dates = sorted(df["date_vn"].unique())

        for idx, current_date in enumerate(unique_dates):
            # Subset data up to current day for anti-leakage calculation of lookback windows
            # To compute pre-10:00 features cleanly, we can pass up to 2 days of M1 history
            df_date_mask = (df["date_vn"] == current_date)
            df_day = df[df_date_mask]

            # Find 10:00 AM Anchor Price P0
            # Target exact bar at hour 10, minute 0
            p0_bars = df_day[(df_day["hour_vn"] == 10) & (df_day["minute_vn"] == 0)]
            if len(p0_bars) == 0:
                # If exact 10:00:00 bar missing, take earliest bar in 10:00-10:05 window
                p0_bars = df_day[(df_day["hour_vn"] == 10) & (df_day["minute_vn"] <= 5)]
                if len(p0_bars) == 0:
                    # Skip day if no trading near 10:00 AM (e.g. holiday)
                    continue

            p0 = p0_bars["open"].iloc[0]
            p0_time = p0_bars["dt_vn"].iloc[0]

            # Include previous day data if available for continuous technical indicators
            if idx > 0:
                prev_date = unique_dates[idx - 1]
                df_context = df[(df["date_vn"] == prev_date) | (df["date_vn"] == current_date)]
            else:
                df_context = df_day

            # Pre-10:00 AM features context (bars before p0_time)
            df_pre10 = df_context[df_context["dt_vn"] < p0_time]
            if len(df_pre10) < 30:
                # Not enough history prior to 10:00 AM
                continue

            pre10_feats = self.compute_indicators_pre1000(df_pre10)
            if not pre10_feats:
                continue

            # Extract Action Window [10:00 - 12:00]
            # Inclusive of 10:00 bar up to 12:00 bar (hour 10 and 11, and 12:00 bar)
            session_bars = df_day[
                (df_day["dt_vn"] >= p0_time) & 
                ((df_day["hour_vn"] == 10) | (df_day["hour_vn"] == 11) | ((df_day["hour_vn"] == 12) & (df_day["minute_vn"] == 0)))
            ].sort_values("dt_vn").reset_index(drop=True)

            if len(session_bars) <= 5:
                # Incomplete session data
                continue

            # Compute MAE
            high_max = session_bars["high"].max()
            low_min = session_bars["low"].min()
            mae_up = high_max - p0
            mae_down = p0 - low_min
            max_mae = max(mae_up, mae_down)

            # Reversion Target Logic:
            # Did price deviate >= min_dev_usd, and then revert back to P0 before 12:00 PM?
            has_deviated = False
            reverted = 0
            revert_minute = np.nan
            first_dev_direction = 0  # +1 for up, -1 for down

            for bar_idx, bar in session_bars.iterrows():
                elapsed_min = (bar["dt_vn"] - p0_time).total_seconds() / 60.0
                if bar_idx == 0:
                    continue  # Skip anchor bar itself

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
                    # Check if price touches P0 again (Low <= P0 <= High)
                    if bar["low"] <= p0 <= bar["high"]:
                        reverted = 1
                        revert_minute = elapsed_min
                        break

            # Calculate relative distances from P0 to technical levels (Pre-10:00)
            dist_vwap = (p0 - pre10_feats["asian_vwap"]) / p0 * 1000.0  # in permille/bps
            dist_ema50 = p0 - pre10_feats["ema50"]
            dist_ema200 = p0 - pre10_feats["ema200"]
            day_of_week = p0_time.dayofweek  # 0=Mon, 4=Fri

            record = {
                "date": current_date,
                "p0": p0,
                "p0_time": p0_time,
                "dayofweek": day_of_week,
                # Pre-10:00 AM Features
                "asian_range": pre10_feats["asian_range"],
                "asian_return": pre10_feats["asian_return"],
                "asian_volume": pre10_feats["asian_volume"],
                "ny_range": pre10_feats["ny_range"],
                "ny_return": pre10_feats["ny_return"],
                "ny_volatility": pre10_feats["ny_volatility"],
                "ny_trend": pre10_feats["ny_trend"],
                "atr_m5_14": pre10_feats["atr_m5_14"],
                "atr_m15_14": pre10_feats["atr_m15_14"],
                "dist_vwap": dist_vwap,
                "dist_ema50": dist_ema50,
                "dist_ema200": dist_ema200,
                # Day of week one-hot encoded
                "is_monday": 1 if day_of_week == 0 else 0,
                "is_tuesday": 1 if day_of_week == 1 else 0,
                "is_wednesday": 1 if day_of_week == 2 else 0,
                "is_thursday": 1 if day_of_week == 3 else 0,
                "is_friday": 1 if day_of_week == 4 else 0,
                # 10:00-12:00 Session Outcomes
                "mae_up": mae_up,
                "mae_down": mae_down,
                "max_mae": max_mae,
                "has_deviated": 1 if has_deviated else 0,
                "reverted": reverted,
                "revert_minute": revert_minute,
                "first_dev_direction": first_dev_direction,
            }
            records.append(record)

        daily_df = pd.DataFrame(records)
        logger.info(f"Processed {len(daily_df)} trading days.")
        return daily_df


class EDAReporter:
    """Generates quantitative EDA summary reports and distribution metrics."""

    @staticmethod
    def generate_eda_report(daily_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes comprehensive statistical report as specified in Stage 2.

        Args:
            daily_df: Daily processed DataFrame from IntradayFeatureExtractor.

        Returns:
            Dictionary containing Win Rate, MAE percentiles, and median revert time.
        """
        total_days = len(daily_df)
        deviated_days = daily_df[daily_df["has_deviated"] == 1]
        num_deviated = len(deviated_days)
        reverted_days = daily_df[daily_df["reverted"] == 1]
        num_reverted = len(reverted_days)

        # Win Rate calculation
        overall_win_rate = (num_reverted / total_days * 100.0) if total_days > 0 else 0.0
        conditional_win_rate = (num_reverted / num_deviated * 100.0) if num_deviated > 0 else 0.0

        # MAE Percentiles
        mae_series = daily_df["max_mae"]
        mae_percentiles = {
            "50th (Median)": np.percentile(mae_series, 50),
            "75th": np.percentile(mae_series, 75),
            "90th": np.percentile(mae_series, 90),
            "95th": np.percentile(mae_series, 95),
            "99th": np.percentile(mae_series, 99),
            "Max": mae_series.max(),
        }

        # Median time to revert (in minutes) for successful reversions
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

        # Print formatted report
        print("\n" + "=" * 65)
        print("STAGE 2: QUANTITATIVE EDA & BEHAVIORAL REPORT")
        print("=" * 65)
        print(f"Total Trading Days Analyzed : {total_days}")
        print(f"Days with Deviation >= 1.0$ : {num_deviated} ({num_deviated/total_days*100:.1f}%)")
        print(f"Successful Reversions to P0 : {num_reverted}")
        print(f"Overall Reversion Probability : {overall_win_rate:.2f}% (All Days)")
        print(f"Conditional Reversion Probability: {conditional_win_rate:.2f}% (Given >=1.0$ Dev)")
        print("-" * 65)
        print("Maximum Adverse Excursion (MAE) Percentile Distribution (in USD):")
        for pct_name, val in mae_percentiles.items():
            print(f"  - {pct_name:<15}: {val:.2f} USD")
        print("-" * 65)
        print(f"Median Time to Revert       : {median_revert_time:.1f} minutes")
        print(f"Mean Time to Revert         : {mean_revert_time:.1f} minutes")
        print("=" * 65 + "\n")

        return report
