"""
v6_system/test_phase4.py
------------------------
Unit Test Suite cho Version 6 Phase 4 Robustness Testing.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.robustness_analyzer import RobustnessAnalyzer


class TestV6Phase4RobustnessAnalyzer(unittest.TestCase):
    """Test suite cho RobustnessAnalyzer."""

    def setUp(self):
        # Tạo mock data grid 3x3
        records = []
        steps = [3.0, 5.0, 7.0]
        mults = [1.0, 1.1, 1.2]
        for st in steps:
            for m in mults:
                records.append({
                    "step": st,
                    "max_dca": 5,
                    "multiplier": m,
                    "net_profit": 500.0 if st == 5.0 and m == 1.1 else 200.0,
                    "profit_factor": 2.5 if st == 5.0 and m == 1.1 else 1.5,
                    "recovery_factor": 3.0 if st == 5.0 and m == 1.1 else 1.8
                })
        self.df_mock = pd.DataFrame(records)
        self.analyzer = RobustnessAnalyzer(self.df_mock)

    def test_2d_heatmap_creation(self):
        """Kiểm tra tạo pivot table 2D."""
        pivot = self.analyzer.build_2d_heatmap(max_dca_filter=5, metric="recovery_factor")
        self.assertEqual(pivot.shape, (3, 3))
        self.assertEqual(pivot.loc[5.0, 1.1], 3.0)

    def test_plateau_detection_and_classification(self):
        """Kiểm tra phân loại Plateau và Plateau Score."""
        df_plateau, pivot_symbols = self.analyzer.analyze_plateaus(max_dca_filter=5)
        self.assertEqual(len(df_plateau), 9)

        target_row = df_plateau[(df_plateau["step"] == 5.0) & (df_plateau["multiplier"] == 1.1)].iloc[0]
        self.assertIn(target_row["symbol"], ["+++", "++", "+"])
        self.assertGreater(target_row["plateau_score"], 0.0)

    def test_ascii_heatmap_generation(self):
        """Kiểm tra tạo chuỗi Heatmap ASCII."""
        _, pivot_symbols = self.analyzer.analyze_plateaus(max_dca_filter=5)
        ascii_text = self.analyzer.generate_ascii_heatmap(pivot_symbols, title="TEST MATRIX")
        self.assertIn("TEST MATRIX", ascii_text)
        self.assertIn("Step 5.0", ascii_text)


if __name__ == "__main__":
    unittest.main()
