"""
v6_system/test_phase3.py
------------------------
Unit Test Suite cho Version 6 Phase 3 DCA Optimization.
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.strategy_v1 import StrategyV1
from v6_system.optimizer_v3 import OptimizerV3


class TestV6Phase3DCAOptimization(unittest.TestCase):
    """Test suite cho Strategy V1 & Optimizer V3."""

    def setUp(self):
        self.optimizer = OptimizerV3(initial_capital=1000.0)

    def test_strategy_v1_multiplier_lot_sizing(self):
        """Kiểm tra tính khối lượng Lot tăng dần theo Multiplier."""
        strat = StrategyV1(step=5.0, max_dca=4, multiplier=1.20, lot_base=0.01)

        self.assertAlmostEqual(strat._get_level_lot(1), 0.0100, places=4)
        self.assertAlmostEqual(strat._get_level_lot(2), 0.0120, places=4)
        self.assertAlmostEqual(strat._get_level_lot(3), 0.0144, places=4)
        self.assertAlmostEqual(strat._get_level_lot(4), 0.0173, places=4)

    def test_optimizer_scorecard_calculation(self):
        """Kiểm tra tính toán 10 chỉ số scorecard của OptimizerV3."""
        # Nến 1: 10:00 VN Open 2000. High 2005.5 -> Touch 2005.0 -> SELL level 1 (step=5)
        # Nến 2: 12:00 VN Close 1995.0 -> TP Hit!
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 05:00:00"], utc=True),
            "date_vn": ["2023-05-10", "2023-05-10"],
            "time_vn": [pd.to_datetime("10:00:00").time(), pd.to_datetime("12:00:00").time()],
            "open": [2000.0, 2005.0],
            "high": [2005.5, 2005.0],
            "low": [1999.0, 1994.5],
            "close": [2005.0, 1995.0],
            "volume": [10.0, 15.0]
        })

        metrics = self.optimizer.evaluate_combination(df_session, step=5.0, max_dca=3, multiplier=1.10)

        self.assertEqual(metrics["step"], 5.0)
        self.assertEqual(metrics["max_dca"], 3)
        self.assertEqual(metrics["multiplier"], 1.10)
        self.assertEqual(metrics["traded_days"], 1)
        self.assertEqual(metrics["win_rate"], 100.0)
        self.assertGreater(metrics["net_profit"], 0.0)


if __name__ == "__main__":
    unittest.main()
