"""
v6_system/test_phase9.py
------------------------
Unit Test Suite cho Version 6 Phase 9 AI Parameter Selection.
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.grid_selector import SafeGridSelector
from v6_system.strategy_grid_ai import StrategyGridAI


class TestV6Phase9SafeGridAI(unittest.TestCase):
    """Test suite cho SafeGridSelector & StrategyGridAI."""

    def setUp(self):
        self.selector = SafeGridSelector(skip_threshold=0.55)
        self.strategy = StrategyGridAI(selector=self.selector)

    def test_safe_grid_menu_strict_membership(self):
        """Kiểm tra tham số chọn luôn thuộc 100% danh mục Pre-defined Safe Menu."""
        cfg_high = self.selector.select_optimal_config(prob_revert=0.85, current_atr=3.0)
        self.assertFalse(cfg_high["should_skip"])
        self.assertIn("Strategy", cfg_high["name"])
        self.assertIn(cfg_high["step"], [3.0, 5.0, 7.0, 10.0])
        self.assertIn(cfg_high["multiplier"], [1.05, 1.10, 1.15])

    def test_skip_on_low_reversion_probability(self):
        """Kiểm tra AI chọn SKIP khi xác suất P < 0.55."""
        cfg_low = self.selector.select_optimal_config(prob_revert=0.40)
        self.assertTrue(cfg_low["should_skip"])
        self.assertIn("SKIP", cfg_low["name"])

    def test_strategy_grid_ai_skip_execution(self):
        """Kiểm tra chiến lược Phase 9 bỏ qua phiên khi gặp nhãn SKIP."""
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 03:01:00"], utc=True),
            "date": ["2023-05-10", "2023-05-10"],
            "open": [2000.0, 2005.0],
            "high": [2005.5, 2005.0],
            "low": [1999.5, 1999.0],
            "close": [2005.0, 1999.5],
            "volume": [10.0, 15.0],
            "prob_revert": [0.40, 0.40]  # Thấp -> Option E (SKIP)
        })

        res = self.strategy.run_daily_session_grid_ai(df_session)

        self.assertFalse(res["traded"])
        self.assertEqual(res["exit_reason"], "SKIPPED_BY_AI")


if __name__ == "__main__":
    unittest.main()
