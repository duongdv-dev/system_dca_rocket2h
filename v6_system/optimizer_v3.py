"""
v6_system/optimizer_v3.py
-------------------------
Engine Optimizer Grid Search & Đánh Giá Đa Chiều (Phase 3).
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from v6_system.strategy_v1 import StrategyV1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("OptimizerV3")


class OptimizerV3:
    """Class tối ưu hóa không gian tham số và tính toán Scorecard 10 tiêu chí."""

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital

    def evaluate_combination(self, df_m1: pd.DataFrame, step: float, max_dca: int, multiplier: float) -> Dict[str, Any]:
        """
        Đánh giá 1 bộ tham số đơn lẻ trên dữ liệu 2020-2025.
        """
        strategy = StrategyV1(
            step=step,
            max_dca=max_dca,
            multiplier=multiplier,
            tp_dollars=max(2.0, step * 0.5),
            lot_base=0.01,
            spread=0.20
        )

        start_t = pd.to_datetime("10:00:00").time()
        end_t = pd.to_datetime("12:00:00").time()
        session_mask = (df_m1["time_vn"] >= start_t) & (df_m1["time_vn"] <= end_t)
        df_session_all = df_m1[session_mask].copy()

        daily_records = []
        for date_val, group in df_session_all.groupby("date_vn"):
            res = strategy.run_daily_session(group)
            daily_records.append(res)

        df_res = pd.DataFrame(daily_records)
        if df_res.empty or not df_res["traded"].any():
            return {}

        traded_df = df_res[df_res["traded"] == True].copy()
        traded_days = len(traded_df)
        total_days = len(df_res)

        total_pnl = float(traded_df["pnl"].sum())
        wins = traded_df[traded_df["pnl"] > 0]
        losses = traded_df[traded_df["pnl"] < 0]

        win_count = len(wins)
        win_rate = (win_count / traded_days) * 100.0 if traded_days > 0 else 0.0

        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(abs(losses["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Equity Curve & Drawdown
        traded_df["equity"] = self.initial_capital + traded_df["pnl"].cumsum()
        traded_df["peak"] = traded_df["equity"].cummax()
        traded_df["drawdown"] = traded_df["equity"] - traded_df["peak"]
        traded_df["drawdown_pct"] = (traded_df["drawdown"] / traded_df["peak"]) * 100.0

        max_dd_dollars = float(abs(traded_df["drawdown"].min()))
        max_dd_pct = float(abs(traded_df["drawdown_pct"].min()))

        recovery_factor = (total_pnl / max_dd_dollars) if max_dd_dollars > 0 else (total_pnl if total_pnl > 0 else 0.0)
        avg_trade_pnl = float(total_pnl / traded_days) if traded_days > 0 else 0.0

        # Worst Day PnL
        worst_day_pnl = float(traded_df["pnl"].min())

        # Monthly PnL & Worst Month
        traded_df["month"] = pd.to_datetime(traded_df["date"]).dt.to_period("M")
        monthly_pnl = traded_df.groupby("month")["pnl"].sum()
        worst_month_pnl = float(monthly_pnl.min()) if not monthly_pnl.empty else 0.0

        # Peak Level & Exposure
        max_dca_reached = int(traded_df["max_level"].max()) if not traded_df.empty else 0
        max_exposure_lots = float(traded_df["max_exposure_lots"].max()) if not traded_df.empty else 0.0

        # Force Close %
        fc_count = int((traded_df["exit_reason"] == "FORCE_CLOSE_1200").sum())
        force_close_pct = (fc_count / traded_days) * 100.0 if traded_days > 0 else 0.0

        # Composite Score (Cân bằng giữa RF, PF, WinRate, Low DD, Low FC%)
        pf_score = min(profit_factor, 10.0)
        rf_score = max(recovery_factor, -10.0)
        composite_score = (
            rf_score * 0.35 +
            pf_score * 0.25 +
            (win_rate / 100.0) * 15.0 -
            (force_close_pct / 100.0) * 10.0 -
            (max_dd_pct / 100.0) * 15.0
        )

        return {
            "step": step,
            "max_dca": max_dca,
            "multiplier": multiplier,
            "total_days": total_days,
            "traded_days": traded_days,
            "net_profit": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_dd_dollars": round(max_dd_dollars, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "recovery_factor": round(recovery_factor, 2),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "worst_day_pnl": round(worst_day_pnl, 2),
            "worst_month_pnl": round(worst_month_pnl, 2),
            "max_dca_reached": max_dca_reached,
            "max_exposure_lots": round(max_exposure_lots, 4),
            "force_close_pct": round(force_close_pct, 2),
            "composite_score": round(composite_score, 2)
        }

    def run_grid_search(
        self,
        df_m1: pd.DataFrame,
        steps: List[float],
        max_dcas: List[int],
        multipliers: List[float]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Thực thi Grid Search trên toàn bộ không gian tham số.
        """
        total_combos = len(steps) * len(max_dcas) * len(multipliers)
        logger.info(f"Bắt đầu Grid Search {total_combos} tổ hợp tham số...")

        results = []
        count = 0

        for st in steps:
            for dca in max_dcas:
                for mult in multipliers:
                    count += 1
                    if count % 50 == 0 or count == total_combos:
                        logger.info(f"Đã xử lý {count}/{total_combos} tổ hợp...")

                    res = self.evaluate_combination(df_m1, st, dca, mult)
                    if res:
                        results.append(res)

        df_grid = pd.DataFrame(results)
        df_grid = df_grid.sort_values("composite_score", ascending=False).reset_index(drop=True)
        
        top_combos = df_grid.head(10).to_dict(orient="records")
        return df_grid, top_combos
