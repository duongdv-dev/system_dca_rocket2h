"""
v6_system/test_phase8.py
------------------------
Unit Test Suite cho Version 6 Phase 8 Adaptive DCA.
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.adaptive_controller import AdaptiveDCAController
from v6_system.strategy_adaptive import StrategyAdaptiveDCA


class TestV6Phase8AdaptiveDCA(unittest.TestCase):
    """Test suite cho AdaptiveDCAController & StrategyAdaptiveDCA."""

    def setUp(self):
        self.controller = AdaptiveDCAController(high_prob_threshold=0.80, mod_prob_threshold=0.60)
        self.strategy = StrategyAdaptiveDCA(controller=self.controller)

    def test_market_regime_a_high_reversion(self):
        """Kiểm tra P >= 0.80 -> Chế độ Market A (Tấn công)."""
        cfg = self.controller.get_regime_config(prob_revert=0.85)

        self.assertFalse(cfg["should_skip"])
        self.assertEqual(cfg["step"], 4.0)
        self.assertEqual(cfg["multiplier"], 1.25)
        self.assertEqual(cfg["max_dca"], 6)
        self.assertEqual(cfg["max_distance"], 12.0)

    def test_market_regime_b_moderate_reversion(self):
        """Kiểm tra 0.60 <= P < 0.80 -> Chế độ Market B (Phòng thủ)."""
        cfg = self.controller.get_regime_config(prob_revert=0.68)

        self.assertFalse(cfg["should_skip"])
        self.assertEqual(cfg["step"], 7.0)
        self.assertEqual(cfg["multiplier"], 1.10)
        self.assertEqual(cfg["max_dca"], 4)
        self.assertEqual(cfg["max_distance"], 15.0)

    def test_market_regime_c_skip(self):
        """Kiểm tra P < 0.60 -> Chế độ Market C (Bỏ phiên SKIP)."""
        cfg = self.controller.get_regime_config(prob_revert=0.35)

        self.assertTrue(cfg["should_skip"])

    def test_strategy_skip_execution(self):
        """Kiểm tra chiến lược bỏ qua phiên khi gặp Market C."""
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 03:01:00"], utc=True),
            "date": ["2023-05-10", "2023-05-10"],
            "open": [2000.0, 2005.0],
            "high": [2005.5, 2005.0],
            "low": [1999.5, 1999.0],
            "close": [2005.0, 1999.5],
            "volume": [10.0, 15.0],
            "prob_revert": [0.35, 0.35]  # Market C
        })

        res = self.strategy.run_daily_session_adaptive(df_session)

        self.assertFalse(res["traded"])
        self.assertEqual(res["exit_reason"], "SKIPPED_BY_AI")


if __name__ == "__main__":
    unittest.main()
