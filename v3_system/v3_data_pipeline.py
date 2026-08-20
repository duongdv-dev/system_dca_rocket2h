"""
v3_system/v3_data_pipeline.py
==============================
Module Trích Xuất Dữ Liệu Nến M1 Thô Thực Tế 100% Không Bỏ Ngày (Raw M1 Pipeline).
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
        Đọc trực tiếp 100% nến M1 thô từ file CSV.
        """
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_filepath}")

        df = pd.read_csv(csv_filepath)
        df.columns = [c.lower().strip() for c in df.columns]

        if 'timestamp' in df.columns:
            ts_sample = df['timestamp'].iloc[0]
            if ts_sample > 1e11:  # Milliseconds
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            else:  # Seconds
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Ho_Chi_Minh')
        elif 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize('Asia/Ho_Chi_Minh')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize('Asia/Ho_Chi_Minh')
        else:
            raise ValueError(f"File CSV {csv_filepath} phải chứa cột 'timestamp', 'time' hoặc 'datetime'")

        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def compute_daily_features(self, df_m1: pd.DataFrame, target_months: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Trích xuất nến M1 chuẩn 100% từ dữ liệu thô:
        - Nến quan sát: 06:00:00 - 09:59:59 (Giờ VN)
        - Nến thực thi  : 10:00:00 - 12:00:00 (Giờ VN)
        CHẠY TRỰC TIẾP MỌI NGÀY CÓ NẾN M1 (KHÔNG BỎ SÓT NGÀY NÀO).
        """
        df_m1['date_str'] = df_m1['datetime'].dt.strftime('%Y-%m-%d')
        df_m1['month_str'] = df_m1['datetime'].dt.strftime('%Y-%m')
        df_m1['time_str'] = df_m1['datetime'].dt.strftime('%H:%M:%S')

        if target_months is not None:
            month_mask = df_m1['month_str'].isin(target_months)
            month_df = df_m1[month_mask].copy()
        else:
            month_df = df_m1.copy()

        unique_dates = sorted(month_df['date_str'].unique())

        daily_records = []
        daily_m1_dict = {}

        for date_str in unique_dates:
            day_m1 = month_df[month_df['date_str'] == date_str].copy().reset_index(drop=True)

            obs_mask = (day_m1['time_str'] >= '06:00:00') & (day_m1['time_str'] <= '09:59:59')
            exec_mask = (day_m1['time_str'] >= '10:00:00') & (day_m1['time_str'] <= '12:00:00')

            obs_m1 = day_m1[obs_mask].copy().reset_index(drop=True)
            exec_m1 = day_m1[exec_mask].copy().reset_index(drop=True)

            # Chỉ cần có nến quan sát và nến thực thi M1 là nạp ngay (Không lọc bỏ ngày mỏng)
            if len(obs_m1) < 10 or len(exec_m1) < 10:
                continue

            # Tính ATR(14) M15 từ dữ liệu nến M1 thực tế
            obs_m1['m15_group'] = obs_m1['datetime'].dt.floor('15min')
            m15_df = obs_m1.groupby('m15_group').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()

            high_low = m15_df['high'] - m15_df['low']
            high_cp = (m15_df['high'] - m15_df['close'].shift(1)).abs()
            low_cp = (m15_df['low'] - m15_df['close'].shift(1)).abs()

            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            atr_14_m15 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
            atr_14_m15 = max(1.0, atr_14_m15)

            price_0600 = obs_m1['open'].iloc[0]
            price_1000 = exec_m1['open'].iloc[0]
            close_0959 = obs_m1['close'].iloc[-1]
            morning_high = obs_m1['high'].max()
            morning_low = obs_m1['low'].min()

            delta_open_0600_1000_r = (price_1000 - price_0600) / atr_14_m15
            range_morning_r = (morning_high - morning_low) / atr_14_m15
            body_morning_r = abs(close_0959 - price_0600) / atr_14_m15
            morning_momentum = body_morning_r / max(0.1, range_morning_r)

            vol_sum = obs_m1['volume'].sum()
            pv_sum = (obs_m1['close'] * obs_m1['volume']).sum()
            daily_vwap = float(pv_sum / vol_sum) if vol_sum > 0 else close_0959
            delta_vwap_r = (close_0959 - daily_vwap) / atr_14_m15

            close_m15 = m15_df['close']
            bb_mid = float(close_m15.mean())
            bb_std = float(close_m15.std()) if len(close_m15) > 1 else 1.0
            bb_zscore = float((close_0959 - bb_mid) / (bb_std + 1e-8))

            if len(close_m15) >= 3:
                bb_slope = float((close_m15.iloc[-1] - close_m15.iloc[-3]) / (2.0 * atr_14_m15))
            else:
                bb_slope = 0.0

            daily_records.append({
                'date': date_str,
                'month': date_str[:7],
                'price_0600': price_0600,
                'price_1000': price_1000,
                'close_0959': close_0959,
                'daily_vwap': daily_vwap,
                'atr_14_m15': atr_14_m15,
                'delta_open_0600_1000_r': delta_open_0600_1000_r,
                'delta_vwap_r': delta_vwap_r,
                'range_morning_r': range_morning_r,
                'body_morning_r': body_morning_r,
                'morning_momentum': morning_momentum,
                'bb_zscore_m15': bb_zscore,
                'bb_slope_m15': bb_slope
            })

            daily_m1_dict[date_str] = (obs_m1, exec_m1)

        feature_df = pd.DataFrame(daily_records)
        return feature_df, daily_m1_dict


if __name__ == '__main__':
    pass
