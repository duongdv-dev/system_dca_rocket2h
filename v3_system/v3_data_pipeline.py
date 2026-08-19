"""
v3_system/v3_data_pipeline.py
==============================
Module Trích Xuất Dữ Liệu & Chuẩn Hóa Đặc Trưng Thị Trường (v3 Architecture).
Được thiết kế bởi Senior Quantitative Researcher.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any

class V3DataPipeline:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def load_and_preprocess_file(self, csv_filepath: str) -> pd.DataFrame:
        """
        Đọc file CSV M1 và chuẩn hóa cột datetime.
        """
        df = pd.read_csv(csv_filepath)
        df.columns = [c.lower().strip() for c in df.columns]

        if 'time' in df.columns and 'datetime' not in df.columns:
            df.rename(columns={'time': 'datetime'}, inplace=True)

        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def compute_daily_features(self, df_m1: pd.DataFrame, target_month_str: str = "2020-01") -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Trích xuất dữ liệu cho từng ngày trong tháng target_month_str.
        Cửa sổ quan sát : 06:00:00 - 09:59:59 AM (Giờ VN)
        Cửa sổ thực thi  : 10:00:00 - 12:00:00 PM (Giờ VN)
        """
        df_m1['date_str'] = df_m1['datetime'].dt.strftime('%Y-%m-%d')
        df_m1['time_str'] = df_m1['datetime'].dt.strftime('%H:%M:%S')

        # Lọc đúng tháng yêu cầu (ví dụ: "2020-01")
        month_mask = df_m1['datetime'].dt.strftime('%Y-%m').str.startswith(target_month_str)
        month_df = df_m1[month_mask].copy()

        unique_dates = sorted(month_df['date_str'].unique())

        daily_records = []
        daily_m1_dict = {}

        for date_str in unique_dates:
            day_m1 = month_df[month_df['date_str'] == date_str].copy().reset_index(drop=True)

            # Lấy nến quan sát 06:00 - 09:59
            obs_mask = (day_m1['time_str'] >= '06:00:00') & (day_m1['time_str'] <= '09:59:59')
            exec_mask = (day_m1['time_str'] >= '10:00:00') & (day_m1['time_str'] <= '12:00:00')

            obs_m1 = day_m1[obs_mask].copy().reset_index(drop=True)
            exec_m1 = day_m1[exec_mask].copy().reset_index(drop=True)

            if len(obs_m1) < 120 or len(exec_m1) < 60:
                continue

            # Tính ATR(14) trên khung M15 của phiên sáng
            obs_m1['m15_group'] = obs_m1['datetime'].dt.floor('15min')
            m15_df = obs_m1.groupby('m15_group').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            if len(m15_df) < 5:
                continue

            high_low = m15_df['high'] - m15_df['low']
            high_cp = (m15_df['high'] - m15_df['close'].shift(1)).abs()
            low_cp = (m15_df['low'] - m15_df['close'].shift(1)).abs()

            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            atr_14_m15 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
            atr_14_m15 = max(1.0, atr_14_m15)  # Giới hạn sàn ATR = $1.00

            # Chỉ số nến sáng 06:00 - 09:59
            morning_open = obs_m1['open'].iloc[0]
            close_0959 = obs_m1['close'].iloc[-1]
            morning_high = obs_m1['high'].max()
            morning_low = obs_m1['low'].min()

            morning_range = max(0.5, morning_high - morning_low)
            morning_body = abs(close_0959 - morning_open)
            morning_momentum = morning_body / morning_range

            # Tính Daily VWAP tích lũy tới 09:59 AM
            vol_sum = obs_m1['volume'].sum()
            pv_sum = (obs_m1['close'] * obs_m1['volume']).sum()
            daily_vwap = float(pv_sum / vol_sum) if vol_sum > 0 else close_0959

            # Bollinger Bands M15
            close_m15 = m15_df['close']
            bb_mid = float(close_m15.mean())
            bb_std = float(close_m15.std()) if len(close_m15) > 1 else 1.0
            bb_zscore = float((close_0959 - bb_mid) / (bb_std + 1e-8))

            # Độ dốc Mid Band M15
            if len(close_m15) >= 3:
                bb_slope = float((close_m15.iloc[-1] - close_m15.iloc[-3]) / (2.0 * atr_14_m15))
            else:
                bb_slope = 0.0

            vwap_dist = close_0959 - daily_vwap
            vwap_dist_atr = vwap_dist / atr_14_m15

            daily_records.append({
                'date': date_str,
                'close_0959': close_0959,
                'daily_vwap': daily_vwap,
                'atr_14_m15': atr_14_m15,
                'morning_open': morning_open,
                'morning_high': morning_high,
                'morning_low': morning_low,
                'morning_range_atr': morning_range / atr_14_m15,
                'morning_body_atr': morning_body / atr_14_m15,
                'morning_momentum': morning_momentum,
                'vwap_dist_atr': vwap_dist_atr,
                'bb_zscore_m15': bb_zscore,
                'bb_slope_m15': bb_slope
            })

            daily_m1_dict[date_str] = (obs_m1, exec_m1)

        feature_df = pd.DataFrame(daily_records)
        return feature_df, daily_m1_dict


if __name__ == '__main__':
    pass
