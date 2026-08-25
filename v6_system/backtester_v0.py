"""
v6_system/backtester_v0.py
--------------------------
Engine Backtest cho Phase 2 Strategy Baseline V0.
Chịu trách nhiệm giả lập chiến lược qua toàn bộ các ngày giao dịch 2020-2025
và thống kê hiệu năng đầy đủ (PnL, Win Rate, Profit Factor, Max Drawdown, DCA Levels).
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

from v6_system.strategy_v0 import StrategyV0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("BacktesterV0")


class BacktesterV0:
    """Class thực thi Backtest cho Strategy V0."""

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital

    def run_backtest(self, df_m1: pd.DataFrame, strategy: StrategyV0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Chạy backtest từng ngày trên dữ liệu M1 đã chuẩn hóa.

        Args:
            df_m1: DataFrame M1 đã sạch từ Phase 1 (chứa cột dt_vn, date_vn, time_vn, open, high, low, close).
            strategy: Đối tượng StrategyV0 đã khởi tạo tham số.

        Returns:
            Tuple (df_results, summary_metrics)
        """
        logger.info(f"Bắt đầu Backtest V0 (Step=${strategy.step}, MaxDCA={strategy.max_dca}, TP=${strategy.tp_dollars})...")

        # Lọc chỉ nến thuộc khung giờ phiên [10:00:00, 12:00:00] VN
        start_t = pd.to_datetime("10:00:00").time()
        end_t = pd.to_datetime("12:00:00").time()
        
        session_mask = (df_m1["time_vn"] >= start_t) & (df_m1["time_vn"] <= end_t)
        df_session_all = df_m1[session_mask].copy()

        daily_results = []
        grouped = df_session_all.groupby("date_vn")

        for date_val, group in grouped:
            res = strategy.run_daily_session(group)
            daily_results.append(res)

        df_res = pd.DataFrame(daily_records_cleaned(daily_results))

        # Tính toán chỉ số hiệu năng
        summary = self.calculate_performance_metrics(df_res)
        return df_res, summary

    def calculate_performance_metrics(self, df_res: pd.DataFrame) -> Dict[str, Any]:
        """Tính toán tổng hợp các chỉ số hiệu năng chuyên sâu."""
        if df_res.empty:
            return {"error": "Không có dữ liệu kết quả."}

        traded_df = df_res[df_res["traded"] == True].copy()
        
        total_days = len(df_res)
        traded_days = len(traded_df)
        untraded_days = total_days - traded_days

        if traded_days == 0:
            return {
                "total_days": total_days,
                "traded_days": 0,
                "net_profit": 0.0,
                "message": "Không có giao dịch nào được kích hoạt."
            }

        total_pnl = traded_df["pnl"].sum()
        wins = traded_df[traded_df["pnl"] > 0]
        losses = traded_df[traded_df["pnl"] < 0]
        breakevens = traded_df[traded_df["pnl"] == 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / traded_days) * 100.0

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = wins["pnl"].mean() if win_count > 0 else 0.0
        avg_loss = losses["pnl"].mean() if loss_count > 0 else 0.0
        expectancy = (total_pnl / traded_days)

        # Tính Equity Curve & Max Drawdown
        traded_df["equity"] = self.initial_capital + traded_df["pnl"].cumsum()
        traded_df["peak"] = traded_df["equity"].cummax()
        traded_df["drawdown"] = traded_df["equity"] - traded_df["peak"]
        traded_df["drawdown_pct"] = (traded_df["drawdown"] / traded_df["peak"]) * 100.0

        max_dd_dollars = abs(traded_df["drawdown"].min())
        max_dd_pct = abs(traded_df["drawdown_pct"].min())
        final_equity = self.initial_capital + total_pnl
        return_pct = (total_pnl / self.initial_capital) * 100.0

        # Thống kê phân phối tầng DCA
        dca_distribution = traded_df["max_level"].value_counts().to_dict()
        # Thống kê lý do đóng lệnh
        exit_reasons = traded_df["exit_reason"].value_counts().to_dict()

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_days": total_days,
            "traded_days": traded_days,
            "untraded_days": untraded_days,
            "net_profit": round(total_pnl, 2),
            "return_pct": round(return_pct, 2),
            "win_count": win_count,
            "loss_count": loss_count,
            "breakeven_count": len(breakevens),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown_dollars": round(max_dd_dollars, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "tp_hit_count": exit_reasons.get("TP_HIT", 0),
            "force_close_count": exit_reasons.get("FORCE_CLOSE_1200", 0),
            "dca_level_distribution": {int(k): int(v) for k, v in sorted(dca_distribution.items())}
        }


def daily_records_cleaned(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for r in records:
        cleaned.append({
            "date": r.get("date"),
            "traded": r.get("traded", False),
            "direction": r.get("direction"),
            "pnl": r.get("pnl", 0.0),
            "exit_reason": r.get("exit_reason"),
            "max_level": r.get("max_level", 0),
            "trades_count": r.get("trades_count", 0),
            "anchor_price": r.get("anchor_price", 0.0)
        })
    return cleaned
