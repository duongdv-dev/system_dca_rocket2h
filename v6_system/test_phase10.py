"""
v6_system/test_phase10.py
-------------------------
Unit Test Suite cho Version 6 Phase 10 Walk-Forward Testing.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.walk_forward_engine import WalkForwardEngine
from v6_system.ml_gatekeeper import MLGatekeeper


class TestV6Phase10WalkForward(unittest.TestCase):
    """Test suite cho WalkForwardEngine."""

    def setUp(self):
        self.wf_engine = WalkForwardEngine(random_state=42)

    def test_folds_configuration(self):
        """Kiểm tra cấu hình 4 Folds Walk-Forward expanding windows."""
        folds = self.wf_engine.FOLDS_CONFIG
        self.assertEqual(len(folds), 4)

        # Fold 1: Train 2020-2021, Test 2022
        self.assertEqual(folds[0]["train_years"], [2020, 2021])
        self.assertEqual(folds[0]["test_year"], 2022)

        # Fold 4: Train 2020-2024, Test 2025
        self.assertEqual(folds[3]["train_years"], [2020, 2021, 2022, 2023, 2024])
        self.assertEqual(folds[3]["test_year"], 2025)

    def test_metrics_calculation(self):
        """Kiểm tra tính toán chỉ số hiệu năng OOS."""
        df_res = pd.DataFrame([
            {"traded": True, "pnl": 10.0, "exit_reason": "TP_HIT", "max_level": 1},
            {"traded": True, "pnl": -5.0, "exit_reason": "FORCE_CLOSE_1200", "max_level": 2},
            {"traded": True, "pnl": 20.0, "exit_reason": "TP_HIT", "max_level": 1}
        ])

        perf = self.wf_engine._calculate_metrics(df_res, "Test Fold")

        self.assertEqual(perf["traded_days"], 3)
        self.assertEqual(perf["net_profit"], 25.0)
        self.assertEqual(perf["win_rate"], 66.67)
        self.assertEqual(perf["profit_factor"], 6.0)


if __name__ == "__main__":
    unittest.main()
