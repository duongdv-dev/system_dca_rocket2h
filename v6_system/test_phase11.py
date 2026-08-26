"""
v6_system/test_phase11.py
-------------------------
Unit Test Suite cho Version 6 Phase 11 Monte Carlo / Stress Test.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.stress_tester import MonteCarloStressTester


class TestV6Phase11StressTest(unittest.TestCase):
    """Test suite cho MonteCarloStressTester."""

    def setUp(self):
        self.tester = MonteCarloStressTester(num_simulations=10, random_state=42)

    def test_robustness_report_evaluation(self):
        """Kiểm tra tính toán chỉ số phân phối Percentile 50th và 95th Worst Case."""
        df_runs = pd.DataFrame([
            {"sim_id": 1, "net_profit": 100.0, "profit_factor": 1.50, "max_dd_dollars": 50.0},
            {"sim_id": 2, "net_profit": 80.0, "profit_factor": 1.30, "max_dd_dollars": 70.0},
            {"sim_id": 3, "net_profit": 120.0, "profit_factor": 1.60, "max_dd_dollars": 40.0},
            {"sim_id": 4, "net_profit": 50.0, "profit_factor": 1.15, "max_dd_dollars": 100.0},
            {"sim_id": 5, "net_profit": 90.0, "profit_factor": 1.40, "max_dd_dollars": 60.0}
        ])

        report = self.tester._evaluate_robustness_report(df_runs)

        self.assertEqual(report["total_simulations"], 5)
        self.assertEqual(report["median_profit_factor"], 1.40)
        self.assertGreater(report["median_net_profit"], 0.0)
        self.assertTrue(report["robustness_passed"])


if __name__ == "__main__":
    unittest.main()
