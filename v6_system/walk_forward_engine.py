"""
v6_system/walk_forward_engine.py
--------------------------------
Engine Kiểm Thử Cuộn Thời Gian Walk-Forward (Phase 10 - Walk-Forward Testing).
Đảm bảo đánh giá khả năng khái quát hóa (Generalization) của AI trên dữ liệu Out-of-Sample chưa từng thấy.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from v6_system.ml_gatekeeper import MLGatekeeper
from v6_system.strategy_grid_ai import StrategyGridAI
from v6_system.grid_selector import SafeGridSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("WalkForwardEngine")


class WalkForwardEngine:
    """Class quản lý quy trình kiểm thử cuộn thời gian 4 Folds Walk-Forward."""

    FOLDS_CONFIG = [
        {"fold": 1, "train_years": [2020, 2021], "test_year": 2022},
        {"fold": 2, "train_years": [2020, 2021, 2022], "test_year": 2023},
        {"fold": 3, "train_years": [2020, 2021, 2022, 2023], "test_year": 2024},
        {"fold": 4, "train_years": [2020, 2021, 2022, 2023, 2024], "test_year": 2025}
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_walk_forward(self, df_labeled_features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Thực thi kiểm thử Walk-Forward 4 Folds trên dữ liệu 2020-2025.

        Args:
            df_labeled_features: DataFrame chứa 25 đặc trưng và nhãn target ('y_anchor' hoặc 'target_revert').

        Returns:
            Tuple (df_combined_oos_trades, summary_report)
        """
        df = df_labeled_features.copy()
        
        date_col = "date" if "date" in df.columns else ("date_vn" if "date_vn" in df.columns else "dt_vn")
        df["year"] = pd.to_datetime(df[date_col]).dt.year

        fold_reports = []
        all_oos_records = []

        logger.info("Bắt đầu thực thi Walk-Forward Testing 4 Folds (Expanding Windows)...")

        for cfg in self.FOLDS_CONFIG:
            f_num = cfg["fold"]
            t_years = cfg["train_years"]
            test_yr = cfg["test_year"]

            logger.info(f"--- FOLD {f_num}: Train {t_years} | Out-of-Sample Test {test_yr} ---")

            # 1. Phân chia dữ liệu Train / Test cho Fold hiện tại
            train_mask = df["year"].isin(t_years)
            test_mask = df["year"] == test_yr

            train_df = df[train_mask].dropna(subset=MLGatekeeper.FEATURE_COLS).copy()
            test_df = df[test_mask].dropna(subset=MLGatekeeper.FEATURE_COLS).copy()

            if train_df.empty or test_df.empty:
                logger.warning(f"Fold {f_num} thiếu dữ liệu Train/Test. Bỏ qua fold này.")
                continue

            # 2. Huấn luyện lại mô hình AI Gatekeeper trên dữ liệu Train của Fold
            gatekeeper = MLGatekeeper(random_state=self.random_state)
            train_df["target_revert"] = train_df["y_anchor"] if "y_anchor" in train_df.columns else train_df["target_revert"]
            test_df["target_revert"] = test_df["y_anchor"] if "y_anchor" in test_df.columns else test_df["target_revert"]

            X_train = train_df[MLGatekeeper.FEATURE_COLS].astype(np.float32)
            y_train = train_df["target_revert"].values.astype(np.int32)
            X_test = test_df[MLGatekeeper.FEATURE_COLS].astype(np.float32)
            y_test = test_df["target_revert"].values.astype(np.int32)

            gatekeeper.model.fit(X_train, y_train)

            # 3. Dự báo xác suất Out-of-Sample trên năm Test
            test_df["prob_revert"] = gatekeeper.model.predict_proba(X_test)[:, 1]

            # 4. Giả lập giao dịch Out-of-Sample năm test_yr bằng Safe Grid AI Selector
            strategy = StrategyGridAI()
            oos_fold_records = []

            for date_val, group in test_df.groupby(date_col):
                res = strategy.run_daily_session_grid_ai(group)
                res["fold"] = f_num
                res["test_year"] = test_yr
                oos_fold_records.append(res)
                all_oos_records.append(res)

            df_fold_oos = pd.DataFrame(oos_fold_records)

            # Tính toán hiệu năng OOS của Fold
            fold_perf = self._calculate_metrics(df_fold_oos, f"Fold {f_num} (Test {test_yr})")
            fold_reports.append(fold_perf)

            logger.info(f"Fold {f_num} (OOS {test_yr}): Net PnL = ${fold_perf['net_profit']}, WinRate = {fold_perf['win_rate']}%, PF = {fold_perf['profit_factor']}")

        # 5. Ghép nối kết quả Out-of-Sample từ cả 4 Folds (2022, 2023, 2024, 2025)
        df_combined_oos = pd.DataFrame(all_oos_records)
        combined_summary = self._calculate_metrics(df_combined_oos, "Combined Out-of-Sample (2022-2025)")

        # Tính chỉ số Walk-Forward Efficiency (WFE)
        avg_is_pf = np.mean([f["profit_factor"] for f in fold_reports]) if fold_reports else 1.0
        wfe = (combined_summary["profit_factor"] / avg_is_pf) if avg_is_pf > 0 else 0.0
        combined_summary["walk_forward_efficiency"] = round(float(wfe), 2)
        combined_summary["fold_details"] = fold_reports

        logger.info(f"Walk-Forward Testing Hoàn tất! Combined OOS Net PnL: ${combined_summary['net_profit']}, WFE Ratio: {combined_summary['walk_forward_efficiency']}")
        return df_combined_oos, combined_summary

    def _calculate_metrics(self, df_res: pd.DataFrame, title: str) -> Dict[str, Any]:
        """Hàm trợ giúp tính toán chỉ số hiệu năng."""
        traded_df = df_res[df_res["traded"] == True].copy()
        total_days = len(df_res)
        traded_days = len(traded_df)

        if traded_days == 0:
            return {"title": title, "net_profit": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "traded_days": 0}

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
            "title": title,
            "total_days": total_days,
            "traded_days": traded_days,
            "net_profit": round(net_pnl, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_dd_dollars": round(max_dd_dollars, 2),
            "losing_days": len(losses)
        }
