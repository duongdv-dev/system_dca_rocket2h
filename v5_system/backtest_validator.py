"""
Out-of-Sample (OOS) Backtest Validator Module
---------------------------------------------
Kiểm tra tính xác thực định lượng: Huấn luyện mô hình trên tập quá khứ (Ví dụ: 2020-2023),
sau đó Backtest độc lập hoàn toàn trên dữ liệu tương lai chưa từng thấy Out-of-Sample (Ví dụ: 2024).
Áp dụng cơ chế Quản lý Rủi ro Thực chiến: Cắt lỗ cứng (Hard Stop Loss - SL) và Thoát lệnh theo Giờ (Time-Stop 12:00 PM).
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
    """Kiểm tra & Mô phỏng Backtest Out-of-Sample thực chiến với Quản lý Rủi ro."""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def calculate_realistic_pnl(
        self,
        row: pd.Series,
        entry_dev: float = 1.0,
        sl_usd: float = 3.0
    ) -> float:
        """
        Tính toán PnL thực chiến cho 1 lệnh giao dịch theo nguyên tắc Quản lý Rủi ro:
        - Điểm vào lệnh (Entry): Khi giá lệch entry_dev (1.0$) từ P0.
        - Chốt lời (Take Profit): Về lại P0 (+1.0$ lợi nhuận).
        - Cắt lỗ cứng (Stop Loss - SL): Nếu lệch thêm sl_usd (Mức lỗ tối đa = -sl_usd).
        - Thoát lệnh thời gian (Time-Stop exit 12:00 PM): Nếu chưa chạm P0 và chưa chạm SL, đóng lệnh tại giá 12:00 PM.
        """
        if row["has_deviated"] != 1:
            return 0.0

        max_mae = row["max_mae"]
        reverted = row["reverted"]
        close_dev_1200 = row.get("close_dev_1200", max_mae)

        # 1. Kiểm tra nếu chạm Cắt lỗ cứng (Hard Stop Loss)
        if max_mae >= (entry_dev + sl_usd):
            return -sl_usd

        # 2. Kiểm tra nếu đảo chiều thành công về P0 trước 12:00 PM
        if reverted == 1:
            return entry_dev

        # 3. Thoát lệnh tại mốc 12:00 PM (Time-Stop Exit)
        # Mức lỗ là độ lệch của giá lúc 12:00 PM so với P0
        loss_at_1200 = max(0.0, close_dev_1200 - entry_dev)
        return -loss_at_1200

    def run_oos_validation(
        self,
        daily_df: pd.DataFrame,
        train_years: List[int],
        test_years: List[int],
        sl_usd: float = 3.0,
        thresholds: List[float] = [0.50, 0.60, 0.70, 0.75, 0.80]
    ) -> Dict[str, Any]:
        """
        Thực thi kiểm tra Out-of-Sample với Quản lý Rủi ro Thực chiến.

        Args:
            daily_df: DataFrame chứa đặc trưng hàng ngày.
            train_years: Các năm dùng để Huấn luyện.
            test_years: Các năm kiểm thử OOS độc lập.
            sl_usd: Mức cắt lỗ cứng (Stop Loss) tính theo USD (mặc định 3.0$).
            thresholds: Các ngưỡng xác suất ML cần kiểm tra.

        Returns:
            Dictionary kết quả thống kê OOS.
        """
        logger.info(f"Đang thực hiện Out-of-Sample Validation (SL = {sl_usd}$): Train = {train_years} | Test OOS = {test_years}")

        daily_df["year"] = pd.to_datetime(daily_df["date"]).dt.year

        df_train = daily_df[daily_df["year"].isin(train_years)].copy()
        df_test = daily_df[daily_df["year"].isin(test_years)].copy()

        if len(df_train) == 0:
            raise ValueError(f"Không có dữ liệu cho các năm train: {train_years}")
        if len(df_test) == 0:
            raise ValueError(f"Không có dữ liệu cho các năm test: {test_years}")

        X_train, y_train = df_train[self.feature_cols], df_train["reverted"].values
        X_test, y_test = df_test[self.feature_cols], df_test["reverted"].values

        # Huấn luyện mô hình LightGBM CHỈ trên tập Train
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

        # Dự báo xác suất P(Revert) trên tập Test OOS
        test_probs = clf.predict_proba(X_test)[:, 1]
        df_test["p_revert"] = test_probs

        # Tính toán PnL Thực chiến cho từng ngày trong tập Test
        df_test["sim_pnl_usd"] = df_test.apply(
            lambda r: self.calculate_realistic_pnl(r, entry_dev=1.0, sl_usd=sl_usd),
            axis=1
        )

        # Thống kê Baseline (Không lọc ML) trên tập Test OOS
        baseline_deviated = df_test[df_test["has_deviated"] == 1]
        n_baseline_trades = len(baseline_deviated)
        n_baseline_wins = len(baseline_deviated[baseline_deviated["sim_pnl_usd"] > 0])
        baseline_winrate = (n_baseline_wins / n_baseline_trades * 100.0) if n_baseline_trades > 0 else 0.0
        baseline_pnl = baseline_deviated["sim_pnl_usd"].sum()
        
        gross_profit_base = baseline_deviated.loc[baseline_deviated["sim_pnl_usd"] > 0, "sim_pnl_usd"].sum()
        gross_loss_base = abs(baseline_deviated.loc[baseline_deviated["sim_pnl_usd"] < 0, "sim_pnl_usd"].sum())
        baseline_pf = (gross_profit_base / gross_loss_base) if gross_loss_base > 0 else np.nan

        # Đánh giá hiệu quả khi lọc qua ML Gatekeeper
        threshold_results = []
        for th in thresholds:
            passed_df = df_test[(df_test["has_deviated"] == 1) & (df_test["p_revert"] >= th)]
            n_trades = len(passed_df)
            if n_trades > 0:
                n_wins = len(passed_df[passed_df["sim_pnl_usd"] > 0])
                win_rate = n_wins / n_trades * 100.0
                total_pnl = passed_df["sim_pnl_usd"].sum()
                gross_win = passed_df.loc[passed_df["sim_pnl_usd"] > 0, "sim_pnl_usd"].sum()
                gross_loss = abs(passed_df.loc[passed_df["sim_pnl_usd"] < 0, "sim_pnl_usd"].sum())
                profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.0
                filtered_loss_days = len(df_test[(df_test["has_deviated"] == 1) & (df_test["sim_pnl_usd"] < 0) & (df_test["p_revert"] < th)])
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
                "Tổng_PnL_Thực_Chiến($)": round(total_pnl, 2),
                "Profit_Factor": round(profit_factor, 2),
                "Số_Ngày_Lỗ_Đã_Tránh": filtered_loss_days
            })

        results_df = pd.DataFrame(threshold_results)

        # In giải thích và báo cáo tiếng Việt chi tiết
        print("\n" + "=" * 75)
        print(f"KẾT QUẢ KIỂM TRA OUT-OF-SAMPLE (OOS) VỚI QUẢN LÝ RỦI RO THỰC CHIẾN")
        print("=" * 75)
        print(f"  - Tập Huấn luyện (Train) : Năm {train_years} ({len(df_train)} ngày)")
        print(f"  - Tập Kiểm thử (Test OOS): Năm {test_years} ({len(df_test)} ngày)")
        print(f"  - Quy tắc Cắt lỗ cứng (Hard Stop-Loss) : {sl_usd} USD")
        print(f"  - Quy tắc Thoát giờ (Time-Stop Hard Exit): 12:00 PM VN Time")
        print("-" * 75)
        print("KẾT QUẢ GỐC (BASELINE - KHÔNG DÙNG LỌC ML):")
        print(f"  - Tổng số lệnh lệch giá >= 1.0$ : {n_baseline_trades} lệnh")
        print(f"  - Số lệnh thắng (+1.0$)         : {n_baseline_wins} lệnh")
        print(f"  - Win Rate Gốc (OOS)            : {baseline_winrate:.2f}%")
        print(f"  - Tổng PnL Gốc Thực chiến       : {baseline_pnl:.2f} USD")
        print(f"  - Profit Factor Gốc             : {baseline_pf:.2f}" if not np.isnan(baseline_pf) else "  - Profit Factor Gốc             : N/A")
        print("-" * 75)
        print("KẾT QUẢ HIỆU QUẢ KHI QUA MÔ HÌNH ML GATEKEEPER (TRÊN DỮ LIỆU OOS CHƯA HỌC):")
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
