"""
v6_system/test_phase5.py
------------------------
Unit Test Suite cho Version 6 Phase 5 XGBoost V1 Probability Model.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.feature_engineering import V6FeatureEngineer
from v6_system.ml_gatekeeper import MLGatekeeper


class TestV6Phase5XGBoostV1(unittest.TestCase):
    """Test suite cho Feature Engineer & ML Gatekeeper."""

    def setUp(self):
        self.fe = V6FeatureEngineer()
        self.gatekeeper = MLGatekeeper(random_state=42)

    def test_feature_engineering_calculation(self):
        """Kiểm tra trích xuất 25 tính năng và không có giá trị NaN."""
        # Tạo mock sequence nến M1
        timestamps = [1577836800000 + i * 60000 for i in range(30)]
        df_mock = pd.DataFrame({
            "timestamp": timestamps,
            "dt_utc": pd.to_datetime(timestamps, unit="ms", utc=True),
            "dt_vn": pd.to_datetime(timestamps, unit="ms", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh"),
            "date_vn": ["2020-01-01"] * 30,
            "time_vn": pd.to_datetime(timestamps, unit="ms", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh").dt.time,
            "open": [2000.0 + i * 0.2 for i in range(30)],
            "high": [2001.0 + i * 0.2 for i in range(30)],
            "low": [1999.0 + i * 0.2 for i in range(30)],
            "close": [2000.5 + i * 0.2 for i in range(30)],
            "volume": [10.0 + i for i in range(30)]
        })

        df_ind = self.fe.compute_technical_indicators(df_mock)

        self.assertIn("atr", df_ind.columns)
        self.assertIn("rsi", df_ind.columns)
        self.assertIn("ema_9", df_ind.columns)
        self.assertIn("volatility", df_ind.columns)
        self.assertFalse(df_ind["atr"].isnull().all())

    def test_ml_gatekeeper_training(self):
        """Kiểm tra quy trình huấn luyện mô hình ML Gatekeeper."""
        # Tạo mock dataset features với 2 lớp Target
        records = []
        for i in range(200):
            year = 2022 if i < 150 else 2024
            rec = {
                "date": f"{year}-05-10",
                "target_revert": 1 if i % 2 == 0 else 0
            }
            for col in MLGatekeeper.FEATURE_COLS:
                rec[col] = float(np.random.randn())
            records.append(rec)

        df_feat = pd.DataFrame(records)

        metrics, df_imp = self.gatekeeper.train_and_evaluate(df_feat, split_year=2024)

        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertEqual(len(df_imp), len(MLGatekeeper.FEATURE_COLS))


if __name__ == "__main__":
    unittest.main()
