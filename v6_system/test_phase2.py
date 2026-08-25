"""
v6_system/test_phase2.py
------------------------
Unit Test Suite cho Version 6 Phase 2 Strategy Baseline V0.
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.strategy_v0 import StrategyV0
from v6_system.backtester_v0 import BacktesterV0


class TestV6Phase2StrategyV0(unittest.TestCase):
    """Test suite kiểm thử logic Strategy V0 và Backtester V0."""

    def test_sell_dca_trigger_and_tp(self):
        """Kiểm tra giá tăng chạm Anchor + Step -> Mở SELL và đóng giỏ khi TP đạt."""
        # Anchor = 2000. Step = 2.0. TP = $2.0.
        # Nến 1: Open 2000.0, High 2002.5 -> Touch 2002.0 -> SELL level 1 at 2002.0
        # Nến 2: Close 1999.5 -> PnL SELL 2002.0 at 1999.5 (spread 0.20) = (2002.0 - (1999.5 + 0.20)) * 1 = $2.30 >= $2.0 -> TP Hit!
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 03:01:00"], utc=True),
            "date_vn": ["2023-05-10", "2023-05-10"],
            "open": [2000.0, 2002.0],
            "high": [2002.5, 2002.0],
            "low": [1999.5, 1999.0],
            "close": [2002.0, 1999.5],
            "volume": [10.0, 15.0]
        })

        strat = StrategyV0(step=2.0, max_dca=5, tp_dollars=2.0, lot_size=0.01, spread=0.20)
        res = strat.run_daily_session(df_session)

        self.assertTrue(res["traded"])
        self.assertEqual(res["direction"], "SELL")
        self.assertEqual(res["max_level"], 1)
        self.assertEqual(res["exit_reason"], "TP_HIT")
        self.assertEqual(res["pnl"], 2.0)

    def test_buy_dca_trigger_and_force_close(self):
        """Kiểm tra giá giảm chạm Anchor - Step -> Mở BUY và Force Close tại nến cuối."""
        # Anchor = 2000. Step = 2.0.
        # Nến 1 (10:00): Open 2000.0, Low 1997.5 -> Touch 1998.0 -> BUY level 1 at 1998.0
        # Nến 2 (12:00): Close 1997.0 -> Force close at 1997.0
        # PnL BUY 1998.0 (spread 0.20) at 1997.0 = 1997.0 - (1998.0 + 0.20) = -1.20 USD
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 05:00:00"], utc=True),
            "date_vn": ["2023-05-10", "2023-05-10"],
            "open": [2000.0, 1997.5],
            "high": [2000.0, 1998.0],
            "low": [1997.5, 1996.5],
            "close": [1998.0, 1997.0],
            "volume": [10.0, 15.0]
        })

        strat = StrategyV0(step=2.0, max_dca=5, tp_dollars=2.0, lot_size=0.01, spread=0.20)
        res = strat.run_daily_session(df_session)

        self.assertTrue(res["traded"])
        self.assertEqual(res["direction"], "BUY")
        self.assertEqual(res["max_level"], 1)
        self.assertEqual(res["exit_reason"], "FORCE_CLOSE_1200")
        self.assertAlmostEqual(res["pnl"], -1.20, places=2)

    def test_multi_level_dca_scaling(self):
        """Kiểm tra nhồi thêm tầng DCA khi giá đi xa (Level 1, Level 2)."""
        # Anchor = 2000. Step = 2.0.
        # Touch 2002.0 -> SELL 1
        # Touch 2004.0 -> SELL 2
        df_session = pd.DataFrame({
            "dt_utc": pd.to_datetime(["2023-05-10 03:00:00", "2023-05-10 03:01:00", "2023-05-10 05:00:00"], utc=True),
            "date_vn": ["2023-05-10", "2023-05-10", "2023-05-10"],
            "open": [2000.0, 2002.0, 2004.0],
            "high": [2002.5, 2004.5, 2004.5],
            "low": [1999.5, 2001.5, 2003.0],
            "close": [2002.0, 2004.0, 2003.5],
            "volume": [10.0, 15.0, 20.0]
        })

        strat = StrategyV0(step=2.0, max_dca=5, tp_dollars=5.0, lot_size=0.01, spread=0.20)
        res = strat.run_daily_session(df_session)

        self.assertTrue(res["traded"])
        self.assertEqual(res["direction"], "SELL")
        self.assertEqual(res["max_level"], 2)
        self.assertEqual(res["trades_count"], 2)


if __name__ == "__main__":
    unittest.main()
