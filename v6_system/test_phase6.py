"""
v6_system/test_phase6.py
------------------------
Unit Test Suite cho Version 6 Phase 6 Advanced AI Labeling System.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.label_builder import V6LabelBuilder


class TestV6Phase6LabelBuilder(unittest.TestCase):
    """Test suite cho V6LabelBuilder."""

    def setUp(self):
        self.builder = V6LabelBuilder(default_step=5.0, default_tp=2.0, max_adverse=15.0)

    def test_anchor_reversion_labeling_example(self):
        """
        Kiểm tra đúng ví dụ của User:
        - 10:15 VN: Distance = +8 (Price 2008, Anchor 2000)
        - 11:02 VN: Price quay về 2000 (Anchor)
        -> Nến 10:15 phải được gán nhãn Y = 1.
        """
        timestamps = [1577836800000 + i * 60000 for i in range(10)]
        df_mock = pd.DataFrame({
            "date": ["2020-01-01"] * 10,
            "dt_vn": [str(i) for i in range(10)],
            "anchor_price": [2000.0] * 10,
            # Nến 0: Anchor 2000
            # Nến 1 (10:15): Close 2008.0 (Distance = +8)
            # Nến 5 (11:02): Low 1999.0 (Price quay về Anchor)
            "open": [2000.0, 2008.0, 2009.0, 2006.0, 2003.0, 2000.0, 2001.0, 2002.0, 2003.0, 2004.0],
            "high": [2001.0, 2010.0, 2010.0, 2007.0, 2004.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0],
            "low": [1999.5, 2007.0, 2005.0, 2002.0, 1999.0, 1999.0, 2000.0, 2001.0, 2002.0, 2003.0],
            "close": [2000.0, 2008.0, 2006.0, 2003.0, 1999.5, 2000.0, 2001.0, 2002.0, 2003.0, 2004.0]
        })

        df_labeled = self.builder.build_all_labels(df_mock)

        # Nến index 1 (Close = 2008) có future low = 1999.0 <= Anchor 2000.0 -> Y_anchor MUST BE 1
        self.assertEqual(df_labeled.iloc[1]["y_anchor"], 1)

    def test_no_reversion_label_zero(self):
        """Kiểm tra trường hợp giá tiếp tục đi xa không về Anchor -> Y = 0."""
        timestamps = [1577836800000 + i * 60000 for i in range(5)]
        df_mock = pd.DataFrame({
            "date": ["2020-01-01"] * 5,
            "dt_vn": [str(i) for i in range(5)],
            "anchor_price": [2000.0] * 5,
            "open": [2000.0, 2005.0, 2010.0, 2015.0, 2020.0],
            "high": [2001.0, 2011.0, 2016.0, 2021.0, 2025.0],
            "low": [2000.0, 2004.0, 2009.0, 2014.0, 2019.0],
            "close": [2000.0, 2010.0, 2015.0, 2020.0, 2024.0]
        })

        df_labeled = self.builder.build_all_labels(df_mock)

        # Nến 1, 2, 3, 4 đều không bao giờ có future low <= 2000.0 -> Y_anchor MUST BE 0
        self.assertEqual(df_labeled.iloc[1]["y_anchor"], 0)
        self.assertEqual(df_labeled.iloc[2]["y_anchor"], 0)


if __name__ == "__main__":
    unittest.main()
