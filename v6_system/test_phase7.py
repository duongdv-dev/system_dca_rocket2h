"""
v6_system/test_phase7.py
------------------------
Unit Test Suite cho Version 6 Phase 7 AI Filter.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.strategy_ai import StrategyAIFilter
from v6_system.run_phase7 import evaluate_system_performance


class TestV6Phase7AIFilter(unittest.TestCase):
    """Test suite cho StrategyAIFilter."""

    def test_ai_filter_blocking_low_probability(self):
        """Kiểm tra AI Filter chặn mở lệnh khi xác suất P(reversion) < Threshold."""
        # Nến chạm 2005.0 -> Tín hiệu SELL
        # Nhưng prob_revert = 0.40 < threshold 0.65 -> Phải bị CHẶN!
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 03:01:00"], utc=True),
            "date": ["2023-05-10", "2023-05-10"],
            "open": [2000.0, 2005.0],
            "high": [2005.5, 2005.0],
            "low": [1999.5, 1999.0],
            "close": [2005.0, 1999.5],
            "volume": [10.0, 15.0],
            "prob_revert": [0.40, 0.40]  # Thấp hơn threshold 0.65
        })

        strat_ai = StrategyAIFilter(step=5.0, max_dca=5, prob_threshold=0.65)
        res = strat_ai.run_daily_session_with_ai(df_session)

        # Lệnh không được mở và bị chặn bởi AI
        self.assertFalse(res["traded"])
        self.assertEqual(res["trades_count"], 0)
        self.assertGreater(res["blocked_by_ai"], 0)

    def test_ai_filter_permitting_high_probability(self):
        """Kiểm tra AI Filter cho phép mở lệnh khi xác suất P(reversion) >= Threshold."""
        # prob_revert = 0.85 >= threshold 0.65 -> Cho phép mở SELL!
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 03:01:00"], utc=True),
            "date": ["2023-05-10", "2023-05-10"],
            "open": [2000.0, 2005.0],
            "high": [2005.5, 2005.0],
            "low": [1999.5, 1999.0],
            "close": [2005.0, 1999.5],
            "volume": [10.0, 15.0],
            "prob_revert": [0.85, 0.85]  # Cao hơn threshold 0.65
        })

        strat_ai = StrategyAIFilter(step=5.0, max_dca=5, prob_threshold=0.65)
        res = strat_ai.run_daily_session_with_ai(df_session)

        # Lệnh được phép mở
        self.assertTrue(res["traded"])
        self.assertEqual(res["direction"], "SELL")
        self.assertEqual(res["trades_count"], 1)

    def test_evaluation_metrics_computation(self):
        """Kiểm tra tính toán 7 chỉ số kiểm chứng thực nghiệm."""
        df_res = pd.DataFrame([
            {"traded": True, "pnl": 10.0, "exit_reason": "TP_HIT", "max_level": 1, "date": "2023-01-01"},
            {"traded": True, "pnl": -5.0, "exit_reason": "FORCE_CLOSE_1200", "max_level": 2, "date": "2023-01-02"},
            {"traded": True, "pnl": 15.0, "exit_reason": "TP_HIT", "max_level": 1, "date": "2023-01-03"}
        ])

        perf = evaluate_system_performance(df_res, "Test System")

        self.assertEqual(perf["traded_days"], 3)
        self.assertEqual(perf["net_profit"], 20.0)
        self.assertEqual(perf["losing_days_count"], 1)
        self.assertEqual(perf["profit_factor"], 5.0)
        self.assertEqual(perf["force_close_count"], 1)


if __name__ == "__main__":
    unittest.main()
