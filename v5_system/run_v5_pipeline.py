"""
Master Orchestrator Script for v5 Intraday Anchor Mean Reversion System (Phiên bản Tiếng Việt)
-----------------------------------------------------------------------------------------
Thực thi Giai đoạn 1 (Tiền xử lý dữ liệu & Đồng bộ múi giờ UTC+7), 
Giai đoạn 2 (Trích xuất đặc trưng hành vi & Báo cáo EDA định lượng), 
và Giai đoạn 3 (Huấn luyện LightGBM Gatekeeper & Xuất mô hình ONNX cho MT5 EA).
"""

import os
import sys
import logging
import argparse
import joblib
import pandas as pd

# Thêm thư mục gốc vào sys.path
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
logger = logging.getLogger("run_v5_pipeline")


def run_pipeline(
    data_dir: str = BASE_DIR,
    file_pattern: str = "XAUUSD_*_m1.csv",
    min_dev_usd: float = 1.0,
    output_onnx_path: str = os.path.join(BASE_DIR, "v5_gatekeeper_model.onnx"),
    output_pkl_path: str = os.path.join(BASE_DIR, "v5_gatekeeper_model.pkl"),
    output_csv_path: str = os.path.join(BASE_DIR, "v5_daily_features.csv"),
) -> None:
    """
    Chạy toàn bộ pipeline nghiên cứu định lượng và học máy 3 giai đoạn end-to-end.
    """
    print("\n" + "#" * 75)
    print("      HỆ THỐNG GIAO DỊCH ĐỊNH LƯỢNG INTRADAY ANCHOR MEAN REVERSION (v5_system)      ")
    print("          XAUUSD 10:00 - 12:00 Giờ Việt Nam (UTC+7 / Asia/Ho_Chi_Minh)            ")
    print("#" * 75 + "\n")

    # =========================================================================
    # GIAI ĐOẠN 1: Tiền xử lý dữ liệu & Chuẩn hóa múi giờ Việt Nam (UTC+7)
    # =========================================================================
    logger.info("=== GIAI ĐOẠN 1: Tiền xử lý dữ liệu & Đồng bộ Múi giờ UTC+7 ===")
    loader = M1DataLoader(target_tz="Asia/Ho_Chi_Minh")
    
    try:
        m1_df = loader.load_directory(directory_path=data_dir, pattern=file_pattern)
    except Exception as err:
        logger.error(f"Giai đoạn 1 thất bại khi tải dữ liệu: {err}")
        return

    # =========================================================================
    # GIAI ĐOẠN 2: Phân tích EDA định lượng & Trích xuất Đặc trưng Hành vi
    # =========================================================================
    logger.info("=== GIAI ĐOẠN 2: Trích xuất Đặc trưng Hành vi trước 10:00 & Báo cáo EDA ===")
    extractor = IntradayFeatureExtractor(min_dev_usd=min_dev_usd)
    daily_df = extractor.process_dataset(m1_df)

    if len(daily_df) == 0:
        logger.error("Không trích xuất được ngày giao dịch hợp lệ nào. Dừng thực thi.")
        return

    # Xuất dataset đặc trưng hàng ngày ra CSV để kiểm tra
    daily_df.to_csv(output_csv_path, index=False)
    logger.info(f"Đã lưu dataset đặc trưng hàng ngày đã xử lý vào file: {output_csv_path}")

    # Xuất Báo cáo Thống kê EDA
    eda_report = EDAReporter.generate_eda_report(daily_df)

    # =========================================================================
    # GIAI ĐOẠN 3: Mô hình Học máy LightGBM Gatekeeper & Xuất mô hình ONNX
    # =========================================================================
    logger.info("=== GIAI ĐOẠN 3: Huấn luyện LightGBM Gatekeeper & Purged Walk-Forward CV ===")
    gatekeeper = LightGBMGatekeeper()
    
    try:
        cv_results = gatekeeper.train_and_evaluate_cv(daily_df, n_splits=5)
        gatekeeper.print_ml_report(cv_results)

        # Lưu file mô hình binary pkl
        joblib.dump(gatekeeper.model, output_pkl_path)
        logger.info(f"Đã lưu binary mô hình LightGBM tại: {output_pkl_path}")

        # Xuất file ONNX cho Robot EA MT5
        ONNXExporter.export_to_onnx(
            model=gatekeeper.model,
            feature_cols=gatekeeper.FEATURE_COLS,
            output_path=output_onnx_path
        )

    except Exception as err:
        logger.error(f"Giai đoạn 3 huấn luyện mô hình Gatekeeper gặp lỗi: {err}", exc_info=True)

    print("\n" + "=" * 75)
    print("HOÀN THÀNH TOÀN BỘ PIPELINE (v5_system)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline Intraday Anchor Mean Reversion v5")
    parser.add_argument("--data-dir", type=str, default=BASE_DIR, help="Thư mục chứa các file M1 CSV")
    parser.add_argument("--min-dev", type=float, default=1.0, help="Mức lệch giá USD tối thiểu để tính nhãn đảo chiều")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        min_dev_usd=args.min_dev
    )
