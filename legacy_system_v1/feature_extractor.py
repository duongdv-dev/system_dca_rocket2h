import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class FeatureExtractor:
    """
    Trích xuất đặc trưng thị trường phiên Á (08:00 - 10:00 AM VN Time / 01:00 - 03:00 UTC)
    từ dữ liệu nến M1 XAUUSD.
    """
    def __init__(self, df_m1: pd.DataFrame):
        """
        df_m1 phải chứa các cột: timestamp, open, high, low, close, volume
        timestamp có dạng Unix Milliseconds (UTC).
        """
        self.df = df_m1.copy()
        if 'datetime' not in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='ms', utc=True)
        
        self.df['date'] = self.df['datetime'].dt.strftime('%Y-%m-%d')
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute

    def extract_daily_features_and_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Trích xuất đặc trưng phiên Á và chuẩn bị dữ liệu 10:00 - 12:00 UTC (03:00 - 05:00 UTC)
        cho từng ngày giao dịch.
        
        Returns:
            features_df: DataFrame chứa đặc trưng phiên Á của từng ngày
            trading_days_data: Dict[date_str, df_trade_m1]
        """
        daily_features = []
        trading_days_data = {}

        # Nhóm theo từng ngày UTC
        grouped = self.df.groupby('date')

        for date_str, group in grouped:
            # Lọc khung quan sát phiên Á (01:00 - 03:00 UTC) -> 8:00 - 10:00 VN Time
            asian_candles = group[(group['hour'] >= 1) & (group['hour'] < 3)].sort_values('datetime')
            
            # Lọc khung giao dịch Rocket 2h (03:00 - 05:00 UTC) -> 10:00 - 12:00 VN Time
            trade_candles = group[(group['hour'] >= 3) & (group['hour'] < 5)].sort_values('datetime')

            # Đảm bảo đủ dữ liệu cả phiên Á (ít nhất 90 nến M1) và phiên giao dịch (ít nhất 90 nến M1)
            if len(asian_candles) < 90 or len(trade_candles) < 90:
                continue

            open_8h = asian_candles.iloc[0]['open']
            close_10h = asian_candles.iloc[-1]['close']
            high_asian = asian_candles['high'].max()
            low_asian = asian_candles['low'].min()
            asian_range = high_asian - low_asian

            # Tính ATR M1 trung bình phiên Á
            tr = np.maximum(
                asian_candles['high'] - asian_candles['low'],
                np.maximum(
                    abs(asian_candles['high'] - asian_candles['close'].shift(1)),
                    abs(asian_candles['low'] - asian_candles['close'].shift(1))
                )
            ).fillna(asian_candles['high'] - asian_candles['low'])
            
            asian_atr_m1 = tr.mean()
            # Quy đổi ATR M15 ước tính = ATR M1 * sqrt(15) hoặc dùng ATR nến M1
            asian_atr_m15 = asian_atr_m1 * 3.873  # approx sqrt(15)

            # Tỷ lệ thân nến phiên Á
            body_sum = (asian_candles['close'] - asian_candles['open']).abs().sum()
            range_sum = (asian_candles['high'] - asian_candles['low']).sum()
            body_ratio = body_sum / (range_sum + 1e-8)

            # Biến động return phiên Á (%)
            asian_return_pct = (close_10h - open_8h) / open_8h * 100.0

            # Điểm Anchor Price tại 10:00 AM (03:00 UTC Open)
            open_10h_anchor = trade_candles.iloc[0]['open']

            daily_features.append({
                'date': date_str,
                'open_8h': open_8h,
                'close_10h': close_10h,
                'open_10h_anchor': open_10h_anchor,
                'high_asian': high_asian,
                'low_asian': low_asian,
                'asian_range': asian_range,
                'asian_atr_m1': asian_atr_m1,
                'asian_atr_m15': asian_atr_m15,
                'asian_range_atr_ratio': asian_range / (asian_atr_m15 + 1e-8),
                'asian_return_pct': asian_return_pct,
                'asian_body_ratio': body_ratio
            })

            trading_days_data[date_str] = trade_candles

        features_df = pd.DataFrame(daily_features)
        return features_df, trading_days_data
