"""
v2_system/data_pipeline.py
==========================
Data Pipeline & Feature Engineering Module cho Hệ Thống XAUUSD Intraday Mean-Reversion Grid/DCA (v2).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Load dữ liệu nến M1 XAUUSD (2020-2025) với múi giờ UTC+7.
2. Tách riêng từng ngày thành 2 cửa sổ:
   - Cửa sổ quan sát (Observation Window): 06:00:00 - 09:59:59 VN.
   - Cửa sổ thực thi (Execution Window): 10:00:00 - 12:00:00 VN.
3. Tính toán Ma Trận Đặc Trưng Chuẩn Hóa Dừng (Stationary Feature Matrix) chốt lúc 09:59:59:
   - morning_range_atr: Biên độ sáng chuẩn hóa theo ATR M15
   - morning_body_atr: Thân nến sáng chuẩn hóa theo ATR M15
   - morning_momentum: Động lượng sáng = Body / Range
   - vwap_dist_atr: Khoảng cách từ Close 09:59 tới Daily VWAP / ATR M15
   - bb_zscore_m15: Z-Score Bollinger Bands (20, 2) M15
   - bb_slope_m15: Độ dốc đường Mid Band M15 qua 4 nến / ATR M15
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

class DataPipeline:
    def __init__(self, data_dir: str):
        """
        Khởi tạo DataPipeline.
        :param data_dir: Thư mục chứa các file CSV nến M1 (ví dụ: XAUUSD_2020_m1.csv)
        """
        self.data_dir = data_dir

    def load_and_preprocess_year(self, filepath: str) -> pd.DataFrame:
        """
        Đọc file CSV M1 và chuẩn hóa datetime múi giờ UTC+7.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")

        df = pd.read_csv(filepath)
        
        if 'timestamp' in df.columns:
            ts_sample = df['timestamp'].iloc[0]
            if ts_sample > 1e11:  # Milliseconds
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            else:  # Seconds
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Ho_Chi_Minh')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize('Asia/Ho_Chi_Minh')
            else:
                df['datetime'] = df['datetime'].dt.tz_convert('Asia/Ho_Chi_Minh')
        else:
            raise ValueError("File CSV phải chứa cột 'timestamp' hoặc 'datetime'")

        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    def resample_m15(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """
        Resample nến M1 sang nến M15 để tính các chỉ báo khung lớn hơn.
        """
        df_temp = df_m1.copy()
        if 'datetime' in df_temp.columns:
            df_temp.set_index('datetime', inplace=True)

        resampled = df_temp.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def compute_indicators_m15(self, df_m15: pd.DataFrame) -> pd.DataFrame:
        """
        Tính ATR(14) và Bollinger Bands (20, 2) trên khung M15.
        """
        df = df_m15.copy()
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(window=14).mean()

        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + 2.0 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2.0 * df['bb_std']
        
        df['bb_zscore'] = (df['close'] - df['bb_mid']) / (df['bb_std'] + 1e-8)
        df['bb_slope'] = (df['bb_mid'] - df['bb_mid'].shift(4)) / (df['atr14'] + 1e-8)

        return df

    def extract_daily_dataset(self, file_paths: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Xử lý toàn bộ dữ liệu từ các file CSV, phân tách theo từng ngày và tạo Feature Matrix chốt lúc 09:59.
        
        :return: (feature_df, daily_m1_dict)
        """
        all_features = []
        daily_m1_dict = {}

        for fp in sorted(file_paths):
            print(f"[DataPipeline] Đang xử lý file: {os.path.basename(fp)}...")
            df_m1 = self.load_and_preprocess_year(fp)
            
            df_m15 = self.resample_m15(df_m1)
            df_m15_ind = self.compute_indicators_m15(df_m15)

            df_m1['date_str'] = df_m1['datetime'].dt.strftime('%Y-%m-%d')
            grouped = df_m1.groupby('date_str')

            for date_str, group in grouped:
                obs_mask = (group['datetime'].dt.hour >= 6) & (group['datetime'].dt.hour < 10)
                obs_df = group[obs_mask].copy()

                exec_mask = (group['datetime'].dt.hour >= 10) & (group['datetime'].dt.hour < 12)
                exec_df = group[exec_mask].copy()

                if len(obs_df) < 120 or len(exec_df) < 60:
                    continue

                m15_sub = df_m15_ind[df_m15_ind.index <= obs_df['datetime'].iloc[-1]]
                if len(m15_sub) < 20:
                    continue
                
                last_m15 = m15_sub.iloc[-1]
                atr14_m15 = last_m15['atr14']
                if pd.isna(atr14_m15) or atr14_m15 <= 0:
                    continue

                # ----- TÍNH TOÁN ĐẶC TRƯNG CHUẨN HÓA DỪNG (STATIONARY) LÚC 09:59 -----
                close_0959 = obs_df['close'].iloc[-1]
                open_0600 = obs_df['open'].iloc[0]
                high_06to10 = obs_df['high'].max()
                low_06to10 = obs_df['low'].min()

                # Biên độ sáng chuẩn hóa ATR (morning_range_atr)
                morning_range = high_06to10 - low_06to10
                if morning_range <= 0:
                    continue
                morning_range_atr = morning_range / atr14_m15

                # Thân nến sáng chuẩn hóa ATR (morning_body_atr)
                morning_body = abs(close_0959 - open_0600)
                morning_body_atr = morning_body / atr14_m15

                # Động lượng sáng (morning_momentum)
                morning_momentum = morning_body / morning_range

                # Daily VWAP & VWAP Distance chuẩn hóa ATR
                typical_price = (obs_df['high'] + obs_df['low'] + obs_df['close']) / 3.0
                vol_sum = obs_df['volume'].sum()
                if vol_sum > 0:
                    daily_vwap = (typical_price * obs_df['volume']).sum() / vol_sum
                else:
                    daily_vwap = typical_price.mean()

                vwap_dist_atr = (close_0959 - daily_vwap) / atr14_m15

                # Z-Score & Slope của Bollinger Bands M15
                bb_zscore_m15 = last_m15['bb_zscore']
                bb_slope_m15 = last_m15['bb_slope']

                if pd.isna(bb_zscore_m15) or pd.isna(bb_slope_m15):
                    continue

                feat_dict = {
                    'date': date_str,
                    'atr_14_m15': atr14_m15,
                    'morning_range_atr': morning_range_atr,
                    'morning_body_atr': morning_body_atr,
                    'morning_momentum': morning_momentum,
                    'vwap_dist_atr': vwap_dist_atr,
                    'bb_zscore_m15': bb_zscore_m15,
                    'bb_slope_m15': bb_slope_m15,
                    'close_0959': close_0959,
                    'daily_vwap': daily_vwap
                }
                all_features.append(feat_dict)
                daily_m1_dict[date_str] = (obs_df, exec_df)

        feature_df = pd.DataFrame(all_features)
        print(f"[DataPipeline] Tổng số ngày hợp lệ trích xuất đặc trưng chuẩn hóa: {len(feature_df)}")
        return feature_df, daily_m1_dict


if __name__ == '__main__':
    pass
