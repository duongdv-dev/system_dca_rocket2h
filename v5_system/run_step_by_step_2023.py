"""
Step-by-Step Pipeline Execution & Data Extraction Script for Year 2023
----------------------------------------------------------------------
Chạy từng bước hệ thống v5 cho riêng năm 2023, trích xuất toàn bộ dữ liệu,
chỉ số thống kê EDA, tầm quan trọng đặc trưng, và các file artifact cho bước tiếp theo.
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from v5_system.data_loader import M1DataLoader
from v5_system.feature_engineering import IntradayFeatureExtractor, EDAReporter
from v5_system.ml_gatekeeper import LightGBMGatekeeper, ONNXExporter
from v5_system.backtest_validator import OOSBacktestValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_step_by_step_2023")


def execute_2023_step_by_step():
    """
    Thực thi từng bước 1 cho riêng năm 2023 và xuất toàn bộ dữ liệu.
    """
    print("\n" + "=" * 80)
    print("      THỰC THI TỪNG BƯỚC HỆ THỐNG v5 - CHUYÊN SÂU NĂM 2023      ")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU NĂM 2023
    # -------------------------------------------------------------------------
    print(">>> BƯỚC 1: Tiền xử lý dữ liệu nến M1 XAUUSD năm 2023...")
    loader = M1DataLoader(target_tz="Asia/Ho_Chi_Minh")
    m1_df = loader.load_directory(directory_path=BASE_DIR, pattern="XAUUSD_*_m1.csv", years=[2023])

    print(f"  [✓] Tổng số nến M1 nạp được: {len(m1_df):,} nến")
    print(f"  [✓] Khoảng thời gian: {m1_df['dt_vn'].min()} -> {m1_df['dt_vn'].max()}")
    print(f"  [✓] Số cột dữ liệu: {list(m1_df.columns)}")

    # -------------------------------------------------------------------------
    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG HÀNH VI & THỐNG KÊ EDA 2023
    # -------------------------------------------------------------------------
    print("\n>>> BƯỚC 2: Trích xuất đặc trưng trước 10:00 & Phân tích Thống kê EDA 2023...")
    extractor = IntradayFeatureExtractor(min_dev_usd=1.0)
    daily_df_2023 = extractor.process_dataset(m1_df)

    csv_2023_path = os.path.join(BASE_DIR, "v5_daily_features_2023.csv")
    daily_df_2023.to_csv(csv_2023_path, index=False)
    print(f"  [✓] Đã xuất file dữ liệu đặc trưng 2023: {csv_2023_path} ({len(daily_df_2023)} ngày giao dịch)")

    eda_report = EDAReporter.generate_eda_report(daily_df_2023)

    # -------------------------------------------------------------------------
    # BƯỚC 3: HUẤN LUYỆN MÔ HÌNH LIGHTGBM GATEKEEPER NĂM 2023
    # -------------------------------------------------------------------------
    print("\n>>> BƯỚC 3: Huấn luyện mô hình LightGBM Gatekeeper & Phân tích Ngưỡng 2023...")
    gatekeeper = LightGBMGatekeeper()
    cv_results = gatekeeper.train_and_evaluate_cv(daily_df_2023, n_splits=5)
    gatekeeper.print_ml_report(cv_results)

    # Export artifacts
    pkl_2023_path = os.path.join(BASE_DIR, "v5_gatekeeper_2023.pkl")
    onnx_2023_path = os.path.join(BASE_DIR, "v5_gatekeeper_2023.onnx")

    joblib.dump(gatekeeper.model, pkl_2023_path)
    print(f"  [✓] Đã lưu mô hình Binary PKL năm 2023: {pkl_2023_path}")

    ONNXExporter.export_to_onnx(
        model=gatekeeper.model,
        feature_cols=gatekeeper.FEATURE_COLS,
        output_path=onnx_2023_path
    )

    # -------------------------------------------------------------------------
    # BƯỚC 4: TỔNG HỢP & XUẤT BÁO CÁO THỐNG KÊ CHI TIẾT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TÓM TẮT KẾT QUẢ DỮ LIỆU NĂM 2023 CHO BƯỚC TIẾP THEO:")
    print("=" * 80)
    print(f"1. Tổng số ngày giao dịch phân tích : {len(daily_df_2023)} ngày")
    print(f"2. Tỷ lệ thắng Gốc (Win Rate)      : {eda_report['overall_win_rate_pct']:.2f}%")
    print(f"3. Mức lệch tối đa trung vị (MAE)   : {eda_report['mae_percentiles']['Phân vị 50th (Trung vị)']:.2f} USD")
    print(f"4. Mức lệch tối đa Phân vị 90th    : {eda_report['mae_percentiles']['Phân vị 90th']:.2f} USD")
    print(f"5. Thời gian trung vị đảo chiều    : {eda_report['median_revert_time_min']:.1f} phút")
    print(f"6. File dataset đã sẵn sàng        : {csv_2023_path}")
    print(f"7. Mô hình ONNX đã sẵn sàng         : {onnx_2023_path}")
    print("=" * 80 + "\n")

    return daily_df_2023, eda_report, cv_results


if __name__ == "__main__":
    execute_2023_step_by_step()
