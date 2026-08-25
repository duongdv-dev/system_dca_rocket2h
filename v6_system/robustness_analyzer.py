"""
v6_system/robustness_analyzer.py
--------------------------------
Engine Phân Tích Tính Ổn Định (Robustness Testing) & Phát Hiện Parameter Plateau (Phase 4).
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("RobustnessAnalyzer")


class RobustnessAnalyzer:
    """Class phân tích độ nhạy tham số 2D và phát hiện vùng cao nguyên ổn định (Plateau)."""

    def __init__(self, df_grid: pd.DataFrame):
        """
        Args:
            df_grid: DataFrame ma trận kết quả từ Grid Search Phase 3
                     (chứa các cột: step, max_dca, multiplier, net_profit, profit_factor, recovery_factor, v.v.).
        """
        self.df_grid = df_grid.copy()

    def build_2d_heatmap(self, max_dca_filter: int = 5, metric: str = "recovery_factor") -> pd.DataFrame:
        """Tạo bảng ma trận 2D giữa DCA Step (dòng) và Multiplier (cột)."""
        df_sub = self.df_grid[self.df_grid["max_dca"] == max_dca_filter]
        if df_sub.empty:
            df_sub = self.df_grid.copy()

        pivot = df_sub.pivot_table(index="step", columns="multiplier", values=metric, aggfunc="mean")
        return pivot

    def analyze_plateaus(self, max_dca_filter: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Xác định các vùng Plateau dựa trên lân cận 3x3 cho từng điểm tham số.

        Returns:
            Tuple (df_plateau_metrics, pivot_symbols)
        """
        df_sub = self.df_grid[self.df_grid["max_dca"] == max_dca_filter].copy()
        if df_sub.empty:
            df_sub = self.df_grid.copy()

        steps = sorted(df_sub["step"].unique())
        multipliers = sorted(df_sub["multiplier"].unique())

        pivot_rf = df_sub.pivot_table(index="step", columns="multiplier", values="recovery_factor", aggfunc="mean")
        pivot_pf = df_sub.pivot_table(index="step", columns="multiplier", values="profit_factor", aggfunc="mean")
        pivot_pnl = df_sub.pivot_table(index="step", columns="multiplier", values="net_profit", aggfunc="mean")

        results = []

        for i, st in enumerate(steps):
            for j, mult in enumerate(multipliers):
                # Xác định tọa độ lân cận 3x3
                st_neighbors = steps[max(0, i - 1): min(len(steps), i + 2)]
                mult_neighbors = multipliers[max(0, j - 1): min(len(multipliers), j + 2)]

                # Trích xuất giá trị các ô trong lân cận 3x3
                rf_vals = pivot_rf.loc[st_neighbors, mult_neighbors].values.flatten()
                pf_vals = pivot_pf.loc[st_neighbors, mult_neighbors].values.flatten()
                pnl_vals = pivot_pnl.loc[st_neighbors, mult_neighbors].values.flatten()

                rf_vals = rf_vals[~np.isnan(rf_vals)]
                pf_vals = pf_vals[~np.isnan(pf_vals)]
                pnl_vals = pnl_vals[~np.isnan(pnl_vals)]

                if len(rf_vals) == 0:
                    continue

                mean_rf = float(np.mean(rf_vals))
                std_rf = float(np.std(rf_vals))
                mean_pf = float(np.mean(pf_vals))
                mean_pnl = float(np.mean(pnl_vals))

                # Thuật toán Plateau Score
                plateau_score = (mean_rf * 0.6 + mean_pf * 0.4) / (1.0 + std_rf)

                # Phân loại ký hiệu Plateau trực quan (+++, ++, +, -)
                if mean_rf >= 2.0 and std_rf <= 0.8 and mean_pnl > 0:
                    symbol = "+++"   # Vùng cao nguyên vàng (Gold Plateau)
                    classification = "Gold Plateau"
                elif mean_rf >= 1.5 and std_rf <= 1.2 and mean_pnl > 0:
                    symbol = "++"    # Vùng ổn định (Stable Region)
                    classification = "Stable Region"
                elif mean_rf >= 1.0 and mean_pnl > 0:
                    symbol = "+"     # Vùng chấp nhận được (Acceptable)
                    classification = "Acceptable"
                else:
                    symbol = "-"     # Vùng nguy hiểm / vực thẫm (Danger Zone / Cliff)
                    classification = "Danger Zone"

                results.append({
                    "max_dca": max_dca_filter,
                    "step": st,
                    "multiplier": mult,
                    "self_rf": float(pivot_rf.loc[st, mult]),
                    "self_pnl": float(pivot_pnl.loc[st, mult]),
                    "neighborhood_mean_rf": round(mean_rf, 2),
                    "neighborhood_std_rf": round(std_rf, 2),
                    "neighborhood_mean_pnl": round(mean_pnl, 2),
                    "plateau_score": round(plateau_score, 2),
                    "symbol": symbol,
                    "classification": classification
                })

        df_plateau = pd.DataFrame(results)
        pivot_symbols = df_plateau.pivot_table(index="step", columns="multiplier", values="symbol", aggfunc="first")
        return df_plateau, pivot_symbols

    def generate_ascii_heatmap(self, pivot_symbols: pd.DataFrame, title: str = "PARAMETER PLATEAU HEATMAP") -> str:
        """Tạo chuỗi Heatmap ASCII hiển thị rõ ràng vùng cao nguyên."""
        lines = []
        lines.append("=" * 75)
        lines.append(f"          {title}")
        lines.append("=" * 75)
        lines.append("Ký hiệu: [+++] Vùng cao nguyên vàng (Rất ổn định) | [++] Vùng ổn định")
        lines.append("         [+] Vùng chấp nhận được                 | [-] Vùng nguy hiểm / Vực thẫm")
        lines.append("-" * 75)

        # Header dòng Multiplier
        mult_cols = list(pivot_symbols.columns)
        header_str = f"{'Step \\ Mult':<12} | " + " | ".join([f"{m:>5.2f}" for m in mult_cols])
        lines.append(header_str)
        lines.append("-" * 75)

        for step_val, row in pivot_symbols.iterrows():
            row_str = f"Step {step_val:<7} | " + " | ".join([f"{str(val):^5}" for val in row.values])
            lines.append(row_str)

        lines.append("=" * 75)
        return "\n".join(lines)
