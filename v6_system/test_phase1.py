"""
v6_system/test_phase1.py
------------------------
Bộ kiểm thử tự động (Unit Test) cho Version 6 Phase 1 Data Engineering.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer


class TestV6Phase1DataEngineering(unittest.TestCase):
    """Test suite cho V6DataEngineer."""

    def setUp(self):
        self.engineer = V6DataEngineer()

    def test_timezone_conversion(self):
        """Kiểm tra chuyển đổi timezone từ UTC epoch ms sang Asia/Ho_Chi_Minh (UTC+7)."""
        # 1577836800000 ms = 2020-01-01 00:00:00 UTC = 2020-01-01 07:00:00 VN
        mock_data = pd.DataFrame({
            "timestamp": [1577836800000, 1577847600000],  # 00:00 UTC & 03:00 UTC
            "open": [1500.0, 1501.0],
            "high": [1505.0, 1506.0],
            "low": [1499.0, 1500.0],
            "close": [1502.0, 1505.0],
            "volume": [1.0, 2.0]
        })

        df_clean, report = self.engineer.audit_and_clean_data(mock_data)
        
        self.assertEqual(len(df_clean), 2)
        self.assertEqual(str(df_clean.iloc[0]["dt_vn"]), "2020-01-01 07:00:00+07:00")
        self.assertEqual(str(df_clean.iloc[1]["dt_vn"]), "2020-01-01 10:00:00+07:00")
        self.assertTrue(report["dst_handled_correctly"])

    def test_ohlc_integrity_validation(self):
        """Kiểm tra phát hiện và loại bỏ các nến lỗi OHLC (High < Low, Low > Close, v.v.)."""
        mock_data = pd.DataFrame({
            "timestamp": [1000000, 1060000, 1120000, 1180000],
            "open": [100.0, 100.0, -50.0, 100.0],
            "high": [105.0, 90.0, 105.0, 105.0],   # Hàng 2: High (90) < Open (100) -> Invalid
            "low": [95.0, 85.0, -60.0, 108.0],    # Hàng 3: Giá âm -> Invalid; Hàng 4: Low (108) > High (105) -> Invalid
            "close": [102.0, 88.0, 100.0, 102.0],
            "volume": [10.0, 5.0, 2.0, 1.0]
        })

        df_clean, report = self.engineer.audit_and_clean_data(mock_data)

        # Chỉ có dòng 0 là hợp lệ
        self.assertEqual(len(df_clean), 1)
        self.assertEqual(report["invalid_ohlc_removed"], 3)
        self.assertEqual(df_clean.iloc[0]["open"], 100.0)

    def test_duplicates_removal(self):
        """Kiểm tra lọc bỏ dòng trùng lặp timestamp."""
        mock_data = pd.DataFrame({
            "timestamp": [1000000, 1000000, 1060000],
            "open": [100.0, 100.0, 101.0],
            "high": [105.0, 105.0, 106.0],
            "low": [95.0, 95.0, 96.0],
            "close": [102.0, 102.0, 103.0],
            "volume": [10.0, 10.0, 11.0]
        })

        df_clean, report = self.engineer.audit_and_clean_data(mock_data)

        self.assertEqual(len(df_clean), 2)
        self.assertEqual(report["duplicates_removed"], 1)

    def test_session_10_12_aggregation(self):
        """Kiểm tra trích xuất session 10:00 -> 12:00 VN."""
        # Tạo chuỗi nến từ 09:59 đến 12:01 VN trên cùng 1 ngày (2023-05-10)
        # 10:00 VN = 03:00 UTC (Epoch timestamp = 1683687600000 ms)
        base_utc_ms = 1683687600000  # 2023-05-10 03:00:00 UTC (10:00:00 VN)

        timestamps = []
        opens, highs, lows, closes, vols = [], [], [], [], []

        # 121 nến từ 10:00:00 VN đến 12:00:00 VN (mỗi phút 1 nến)
        for i in range(121):
            ts = base_utc_ms + i * 60000
            timestamps.append(ts)
            opens.append(2000.0 + i * 0.1)
            highs.append(2000.0 + i * 0.1 + 2.0)
            lows.append(2000.0 + i * 0.1 - 1.0)
            closes.append(2000.0 + i * 0.1 + 0.5)
            vols.append(10.0 + i)

        mock_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols
        })

        df_clean, _ = self.engineer.audit_and_clean_data(mock_df)
        df_daily = self.engineer.extract_daily_sessions(df_clean)

        self.assertEqual(len(df_daily), 1)
        row = df_daily.iloc[0]

        self.assertEqual(row["date"], "2023-05-10")
        self.assertEqual(row["anchor_price"], 2000.0)  # Open of 10:00 candle
        self.assertEqual(row["open_1000"], 2000.0)
        self.assertEqual(row["open_1200"], 2000.0 + 120 * 0.1)
        self.assertEqual(row["session_high"], max(highs))
        self.assertEqual(row["session_low"], min(lows))
        self.assertEqual(row["session_close"], closes[-1])
        self.assertEqual(row["candle_count"], 121)


if __name__ == "__main__":
    unittest.main()
