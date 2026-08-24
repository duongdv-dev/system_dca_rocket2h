"""
Out-of-Sample (OOS) Backtest Validator Module
---------------------------------------------
Kiểm tra tính xác thực định lượng: Huấn luyện mô hình trên tập quá khứ (Ví dụ: 2020-2023),
sau đó Backtest độc lập hoàn toàn trên dữ liệu tương lai chưa từng thấy Out-of-Sample (Ví dụ: 2024).
So sánh hiệu quả giữa chiến lược Gốc (Không lọc) và chiến lược Lọc qua ML Gatekeeper.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OOSBacktestValidator:
    """Kiểm tra & Mô phỏng Backtest Out-of-Sample độc lập."""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def run_oos_validation(
        self,
        daily_df: pd.DataFrame,
        train_years: List[int],
        test_years: List[int],
        thresholds: List[float] = [0.50, 0.60, 0.70, 0.75, 0.80]
    ) -> Dict[str, Any]:
        """
        Thực thi kiểm tra Out-of-Sample: Train trên train_years, Test trên test_years.

        Args:
            daily_df: DataFrame chứa đặc trưng hàng ngày và nhãn target.
            train_years: Danh sách các năm dùng để Huấn luyện (Train).
            test_years: Danh sách các năm dùng để Kiểm thử độc lập (Test OOS).
            thresholds: Danh sách ngưỡng xác suất P(Revert) để kiểm tra.

        Returns:
            Dictionary kết quả thống kê OOS.
        """
        logger.info(f"Đang thực hiện Out-of-Sample Validation: Train = {train_years} | Test OOS = {test_years}")

        daily_df["year"] = pd.to_datetime(daily_df["date"]).dt.year

        df_train = daily_df[daily_df["year"].isin(train_years)].copy()
        df_test = daily_df[daily_df["year"].isin(test_years)].copy()

        if len(df_train) == 0:
            raise ValueError(f"Không có dữ liệu cho các năm train: {train_years}")
        if len(df_test) == 0:
            raise ValueError(f"Không có dữ liệu cho các năm test: {test_years}")

        logger.info(f"Tập Train ({train_years}): {len(df_train)} ngày | Tập Test OOS ({test_years}): {len(df_test)} ngày")

        X_train, y_train = df_train[self.feature_cols], df_train["reverted"].values
        X_test, y_test = df_test[self.feature_cols], df_test["reverted"].values

        # 1. Huấn luyện mô hình LightGBM CHỈ trên tập Train
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train, y_train)

        # 2. Dự báo xác suất P(Revert) trên tập Test OOS
        test_probs = clf.predict_proba(X_test)[:, 1]
        df_test["p_revert"] = test_probs

        # 3. Mô phỏng PnL Backtest chi tiết trên tập Test OOS
        # Giả định giao dịch: Khi lệch giá >= 1.0$, vào lệnh đảo chiều hướng về P0.
        # Nếu reverted == 1: Thắng +1.0$ (hoặc mức lệch thực tế).
        # Nếu reverted == 0: Thua tại mốc 12:00 PM (Lỗ = max_mae hoặc giá đóng cửa 12:00 so với P0).
        df_test["sim_pnl_usd"] = np.where(
            df_test["has_deviated"] == 1,
            np.where(df_test["reverted"] == 1, 1.0, -df_test["max_mae"]),
            0.0
        )

        # Kết quả Baseline (Không lọc ML) trên tập Test
        baseline_deviated = df_test[df_test["has_deviated"] == 1]
        n_baseline_trades = len(baseline_deviated)
        n_baseline_wins = len(baseline_deviated[baseline_deviated["reverted"] == 1])
        baseline_winrate = (n_baseline_wins / n_baseline_trades * 100.0) if n_baseline_trades > 0 else 0.0
        baseline_pnl = baseline_deviated["sim_pnl_usd"].sum()
        
        gross_profit_base = baseline_deviated.loc[baseline_deviated["sim_pnl_usd"] > 0, "sim_pnl_usd"].sum()
        gross_loss_base = abs(baseline_deviated.loc[baseline_deviated["sim_pnl_usd"] < 0, "sim_pnl_usd"].sum())
        baseline_pf = (gross_profit_base / gross_loss_base) if gross_loss_base > 0 else np.nan

        # Đánh giá hiệu quả lọc theo từng Ngưỡng ML trên tập Test OOS
        threshold_results = []
        for th in thresholds:
            passed_df = df_test[(df_test["has_deviated"] == 1) & (df_test["p_revert"] >= th)]
            n_trades = len(passed_df)
            if n_trades > 0:
                n_wins = len(passed_df[passed_df["reverted"] == 1])
                win_rate = n_wins / n_trades * 100.0
                total_pnl = passed_df["sim_pnl_usd"].sum()
                gross_win = passed_df.loc[passed_df["sim_pnl_usd"] > 0, "sim_pnl_usd"].sum()
                gross_loss = abs(passed_df.loc[passed_df["sim_pnl_usd"] < 0, "sim_pnl_usd"].sum())
                profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.0
                filtered_loss_days = len(df_test[(df_test["has_deviated"] == 1) & (df_test["reverted"] == 0) & (df_test["p_revert"] < th)])
            else:
                win_rate = 0.0
                total_pnl = 0.0
                profit_factor = 0.0
                filtered_loss_days = 0

            threshold_results.append({
                "Ngưỡng_Lọc_P": th,
                "Số_Lệnh_Vào": n_trades,
                "WinRate_OOS(%)": round(win_rate, 2),
                "Mức_Tăng_WinRate(%)": round(win_rate - baseline_winrate, 2),
                "Tổng_PnL_Ước_Tính($)": round(total_pnl, 2),
                "Profit_Factor": round(profit_factor, 2),
                "Số_Ngày_Lỗ_Đã_Tránh": filtered_loss_days
            })

        results_df = pd.DataFrame(threshold_results)

        # In báo cáo OOS đẹp mắt
        print("\n" + "=" * 75)
        print(f"KẾT QUẢ KIỂM TRA OUT-OF-SAMPLE (OOS) XÁC THỰC MÔ HÌNH")
        print("=" * 75)
        print(f"  - Dữ liệu Huấn luyện (Train) : Năm {train_years} ({len(df_train)} ngày)")
        print(f"  - Dữ liệu Kiểm thử (Test OOS): Năm {test_years} ({len(df_test)} ngày)")
        print("-" * 75)
        print("KẾT QUẢ GỐC (BASELINE - KHÔNG DÙNG ML FILTERING) NĂM TEST:")
        print(f"  - Tổng số lệnh lệch giá >= 1.0$ : {n_baseline_trades} lệnh")
        print(f"  - Số lệnh thắng                 : {n_baseline_wins} lệnh")
        print(f"  - Win Rate Gốc (OOS)            : {baseline_winrate:.2f}%")
        print(f"  - Tổng PnL Gốc Ước tính         : {baseline_pnl:.2f} USD")
        print(f"  - Profit Factor Gốc             : {baseline_pf:.2f}" if not np.isnan(baseline_pf) else "  - Profit Factor Gốc             : N/A")
        print("-" * 75)
        print("KẾT QUẢ LỌC QUA MÔ HÌNH ML GATEKEEPER TRÊN DỮ LIỆU CHƯA TỪNG THẤY (OOS):")
        print(results_df.to_string(index=False))
        print("=" * 75 + "\n")

        return {
            "df_test": df_test,
            "baseline_metrics": {
                "n_trades": n_baseline_trades,
                "winrate": baseline_winrate,
                "pnl": baseline_pnl,
                "pf": baseline_pf
            },
            "oos_threshold_results": results_df
        }
