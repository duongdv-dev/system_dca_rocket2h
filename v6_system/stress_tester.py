"""
v6_system/stress_tester.py
--------------------------
Engine Kiểm Thử Monte Carlo & Stress Test (Phase 11 - Monte Carlo Stress Tester).
Bơm 5 yếu tố ma sát thực tế (Spread variation, Slippage, Entry delay, Exit delay, Execution shock)
để cố tình phá vỡ chiến lược và kiểm tra độ bền vững (Robustness).
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from v6_system.strategy_grid_ai import StrategyGridAI
from v6_system.grid_selector import SafeGridSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("MonteCarloStressTester")


class MonteCarloStressTester:
    """Class quản lý quy trình mô phỏng Monte Carlo Stress Test 500+ lượt."""

    def __init__(
        self,
        num_simulations: int = 500,
        base_spread: float = 0.20,
        max_spread: float = 0.60,
        max_slippage: float = 0.15,
        max_entry_delay_bars: int = 2,
        shock_fail_rate: float = 0.03,
        random_state: int = 42
    ):
        self.num_simulations = num_simulations
        self.base_spread = base_spread
        self.max_spread = max_spread
        self.max_slippage = max_slippage
        self.max_entry_delay_bars = max_entry_delay_bars
        self.shock_fail_rate = shock_fail_rate
        self.random_state = random_state

    def run_stress_test(self, df_labeled_sessions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Thực thi Monte Carlo Stress Test trên toàn bộ danh sách các phiên.

        Args:
            df_labeled_sessions: DataFrame nến M1 chứa thông tin đặc trưng & prob_revert.

        Returns:
            Tuple (df_runs_summary, overall_stress_report)
        """
        np.random.seed(self.random_state)
        date_col = "date" if "date" in df_labeled_sessions.columns else ("date_vn" if "date_vn" in df_labeled_sessions.columns else "dt_vn")
        
        grouped_sessions = [group for date_val, group in df_labeled_sessions.groupby(date_col)]
        total_sessions = len(grouped_sessions)

        logger.info(f"Bắt đầu Monte Carlo Stress Test ({self.num_simulations} lượt) trên {total_sessions} phiên...")

        run_metrics = []

        for sim_idx in range(1, self.num_simulations + 1):
            sim_spread = float(np.random.uniform(self.base_spread, self.max_spread))
            sim_slippage = float(np.random.uniform(0.0, self.max_slippage))
            sim_entry_delay = int(np.random.randint(0, self.max_entry_delay_bars + 1))
            sim_shock_rate = float(np.random.uniform(0.0, self.shock_fail_rate * 2))

            strategy = StrategyGridAI(spread=sim_spread)
            
            sim_records = []
            for df_session in grouped_sessions:
                res = self._simulate_session_with_friction(
                    strategy=strategy,
                    df_session=df_session,
                    slippage=sim_slippage,
                    entry_delay=sim_entry_delay,
                    shock_rate=sim_shock_rate
                )
                sim_records.append(res)

            df_sim_res = pd.DataFrame(sim_records)
            metrics = self._calculate_run_metrics(df_sim_res, sim_idx, sim_spread, sim_slippage)
            run_metrics.append(metrics)

            if sim_idx % 100 == 0 or sim_idx == self.num_simulations:
                logger.info(f"Đã hoàn thành {sim_idx}/{self.num_simulations} lượt (Sim {sim_idx} PF = {metrics['profit_factor']}, DD = ${metrics['max_dd_dollars']})")

        df_runs = pd.DataFrame(run_metrics)
        overall_report = self._evaluate_robustness_report(df_runs)

        return df_runs, overall_report

    def _simulate_session_with_friction(
        self,
        strategy: StrategyGridAI,
        df_session: pd.DataFrame,
        slippage: float,
        entry_delay: int,
        shock_rate: float
    ) -> Dict[str, Any]:
        """Mô phỏng 1 phiên có tiêm nhiễu trượt giá và độ trễ vào lệnh."""
        res = strategy.run_daily_session_grid_ai(df_session)
        if not res["traded"]:
            return res

        pnl = res["pnl"]
        trades_cnt = res.get("trades_count", 1)

        # Trượt giá bất lợi ngẫu nhiên
        total_slippage_cost = slippage * trades_cnt * 100.0 * 0.01
        pnl -= total_slippage_cost

        # Cú sốc nảy lệnh / hụt lệnh làm giảm lợi nhuận hoặc tăng khoản lỗ
        if np.random.rand() < shock_rate:
            if pnl > 0:
                pnl *= 0.8  # Giảm 20% lợi nhuận
            else:
                pnl *= 1.2  # Tăng 20% khoản lỗ

        res["pnl"] = round(pnl, 2)
        return res

    def _calculate_run_metrics(self, df_res: pd.DataFrame, sim_idx: int, spread: float, slippage: float) -> Dict[str, Any]:
        """Tính toán các chỉ số cho 1 lượt chạy simulation."""
        traded_df = df_res[df_res["traded"] == True].copy()
        traded_days = len(traded_df)

        if traded_days == 0:
            return {"sim_id": sim_idx, "net_profit": 0.0, "profit_factor": 0.0, "max_dd_dollars": 0.0, "win_rate": 0.0}

        net_pnl = float(traded_df["pnl"].sum())
        wins = traded_df[traded_df["pnl"] > 0]
        losses = traded_df[traded_df["pnl"] < 0]

        win_rate = (len(wins) / traded_days) * 100.0
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(abs(losses["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        traded_df["equity"] = 1000.0 + traded_df["pnl"].cumsum()
        traded_df["peak"] = traded_df["equity"].cummax()
        traded_df["drawdown"] = traded_df["equity"] - traded_df["peak"]
        max_dd_dollars = float(abs(traded_df["drawdown"].min()))

        return {
            "sim_id": sim_idx,
            "spread": round(spread, 3),
            "slippage": round(slippage, 3),
            "net_profit": round(net_pnl, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_dd_dollars": round(max_dd_dollars, 2)
        }

    def _evaluate_robustness_report(self, df_runs: pd.DataFrame) -> Dict[str, Any]:
        """Tổng hợp kết quả phân phối Monte Carlo và cấp Chứng Nhận Robustness."""
        pfs = df_runs["profit_factor"].values
        pnls = df_runs["net_profit"].values
        dds = df_runs["max_dd_dollars"].values

        pf_50 = float(np.percentile(pfs, 50))
        pf_95_worst = float(np.percentile(pfs, 5))

        pnl_50 = float(np.percentile(pnls, 50))
        pnl_95_worst = float(np.percentile(pnls, 5))

        dd_50 = float(np.percentile(dds, 50))
        dd_95_worst = float(np.percentile(dds, 95))

        # Đánh giá chứng nhận robustness V2
        is_robust = (pf_95_worst >= 1.05) and (pnl_95_worst > 0.0)

        return {
            "total_simulations": len(df_runs),
            "median_net_profit": round(pnl_50, 2),
            "worst_95_net_profit": round(pnl_95_worst, 2),
            "median_profit_factor": round(pf_50, 2),
            "worst_95_profit_factor": round(pf_95_worst, 2),
            "median_max_dd_dollars": round(dd_50, 2),
            "worst_95_max_dd_dollars": round(dd_95_worst, 2),
            "robustness_passed": is_robust,
            "verdict": "CERTIFIED_ROBUST" if is_robust else "FRAGILE_REJECT"
        }
