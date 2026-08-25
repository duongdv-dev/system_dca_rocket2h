"""
Parametric Step-by-Step Pipeline Runner & Artifact Exporter (v5_system)
-----------------------------------------------------------------------
Cho phép người dùng truyền tham số năm giao dịch (--years), mức lệch tối thiểu (--min-dev),
và cắt lỗ cứng (--sl-usd) để thực thi từng bước 1 và xuất toàn bộ dữ liệu linh hoạt.
"""

import os
import sys
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from v5_system.data_loader import M1DataLoader
from v5_system.feature_engineering import IntradayFeatureExtractor, EDAReporter
from v5_system.ml_gatekeeper import LightGBMGatekeeper, ONNXExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_step_by_step")


def run_parametric_step_by_step(
    years: Optional[List[int]] = None,
    min_dev_usd: float = 1.0,
    sl_usd: float = 3.0,
    data_dir: str = BASE_DIR
) -> None:
    """
    Thực thi từng bước và trích xuất dữ liệu linh hoạt theo năm người dùng chọn.
    """
    years_tag = "_".join(map(str, sorted(years))) if years else "all_years"
    years_desc = f"Các năm: {years}" if years else "Tất cả các năm (2020-2025)"

    print("\n" + "=" * 80)
    print(f"      THỰC THI TỪNG BƯỚC HỆ THỐNG v5 - {years_desc.upper()}      ")
    print(f"          Tham số: Lệch min = {min_dev_usd}$ | Cắt lỗ SL = {sl_usd}$            ")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU NẾN M1
    # -------------------------------------------------------------------------
    logger.info(f">>> BƯỚC 1: Tiền xử lý dữ liệu nến M1 XAUUSD ({years_desc})...")
    loader = M1DataLoader(target_tz="Asia/Ho_Chi_Minh")
    
    try:
        m1_df = loader.load_directory(directory_path=data_dir, pattern="XAUUSD_*_m1.csv", years=years)
    except Exception as err:
        logger.error(f"Giai đoạn 1 thất bại: {err}")
        return

    logger.info(f" [✓] Nạp thành công {len(m1_df):,} nến M1 từ {m1_df['dt_vn'].min()} -> {m1_df['dt_vn'].max()}")

    # -------------------------------------------------------------------------
    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG HÀNH VI & THỐNG KÊ EDA
    # -------------------------------------------------------------------------
    logger.info(f">>> BƯỚC 2: Trích xuất đặc trưng trước 10:00 & Phân tích Thống kê EDA ({years_desc})...")
    extractor = IntradayFeatureExtractor(min_dev_usd=min_dev_usd)
    daily_df = extractor.process_dataset(m1_df)

    if len(daily_df) == 0:
        logger.error("Không trích xuất được ngày giao dịch hợp lệ nào. Dừng thực thi.")
        return

    # Đường dẫn xuất file đặc trưng theo tham số năm
    csv_output_path = os.path.join(data_dir, f"v5_daily_features_{years_tag}.csv")
    daily_df.to_csv(csv_output_path, index=False)
    logger.info(f" [✓] Đã xuất dataset đặc trưng: {csv_output_path} ({len(daily_df)} ngày giao dịch)")

    # Báo cáo EDA
    eda_report = EDAReporter.generate_eda_report(daily_df)

    # -------------------------------------------------------------------------
    # BƯỚC 3: HUẤN LUYỆN MÔ HÌNH LIGHTGBM GATEKEEPER & CROSS-VALIDATION
    # -------------------------------------------------------------------------
    logger.info(f">>> BƯỚC 3: Huấn luyện mô hình LightGBM Gatekeeper & Phân tích Ngưỡng ({years_desc})...")
    gatekeeper = LightGBMGatekeeper()
    
    try:
        cv_results = gatekeeper.train_and_evaluate_cv(daily_df, n_splits=5)
        gatekeeper.print_ml_report(cv_results)

        # Xuất artifacts định danh theo năm
        pkl_output_path = os.path.join(data_dir, f"v5_gatekeeper_{years_tag}.pkl")
        onnx_output_path = os.path.join(data_dir, f"v5_gatekeeper_{years_tag}.onnx")

        joblib.dump(gatekeeper.model, pkl_output_path)
        logger.info(f" [✓] Đã lưu file mô hình Python Binary: {pkl_output_path}")

        ONNXExporter.export_to_onnx(
            model=gatekeeper.model,
            feature_cols=gatekeeper.FEATURE_COLS,
            output_path=onnx_output_path
        )
        logger.info(f" [✓] Đã lưu file mô hình ONNX cho MT5 EA: {onnx_output_path}")

    except Exception as err:
        logger.error(f"Giai đoạn 3 gặp lỗi: {err}", exc_info=True)

    # -------------------------------------------------------------------------
    # BƯỚC 4: TỔNG HỢP & HOÀN THÀNH
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"HOÀN THÀNH TRÍCH XUẤT DỮ LIỆU & HUẤN LUYỆN DÀNH CHO: {years_desc.upper()}")
    print("=" * 80)
    print(f" 1. Dataset đặc trưng CSV : {csv_output_path}")
    print(f" 2. Mô hình Binary PKL    : {pkl_output_path}")
    print(f" 3. Mô hình ONNX MT5 EA   : {onnx_output_path}")
    print(f" 4. Win Rate Gốc (EDA)    : {eda_report['overall_win_rate_pct']:.2f}%")
    print(f" 5. MAE Trung vị (Median) : {eda_report['mae_percentiles']['Phân vị 50th (Trung vị)']:.2f} USD")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline v5 từng bước với tham số linh hoạt")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Danh sách năm cần chạy (Ví dụ: --years 2023 hoặc --years 2020 2021 2022 2023)")
    parser.add_argument("--min-dev", type=float, default=1.0, help="Mức lệch giá USD tối thiểu để tính nhãn đảo chiều (Mặc định: 1.0$)")
    parser.add_argument("--sl-usd", type=float, default=3.0, help="Mức Cắt lỗ cứng Stop-Loss tính theo USD (Mặc định: 3.0$)")
    parser.add_argument("--data-dir", type=str, default=BASE_DIR, help="Thư mục dữ liệu gốc")

    args = parser.parse_args()

    run_parametric_step_by_step(
        years=args.years,
        min_dev_usd=args.min_dev,
        sl_usd=args.sl_usd,
        data_dir=args.data_dir
    )
