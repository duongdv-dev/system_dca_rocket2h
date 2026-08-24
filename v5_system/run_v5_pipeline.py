"""
Master Orchestrator Script for v5 Intraday Anchor Mean Reversion System
----------------------------------------------------------------------
Executes Stage 1 (Data Preprocessing), Stage 2 (Quantitative EDA & Feature Extraction),
and Stage 3 (LightGBM ML Gatekeeper & ONNX Export) end-to-end.
"""

import os
import sys
import logging
import argparse
import joblib
import pandas as pd

# Add root directory to sys.path
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
    Runs the complete 3-stage quantitative research and ML pipeline.
    """
    print("\n" + "#" * 70)
    print("      INTRADAY ANCHOR MEAN REVERSION PIPELINE (v5_system)      ")
    print("          XAUUSD 10:00 - 12:00 VN Time (UTC+7 / Asia/Ho_Chi_Minh)   ")
    print("#" * 70 + "\n")

    # =========================================================================
    # STAGE 1: Data Preprocessing & Timezone Alignment
    # =========================================================================
    logger.info("=== STAGE 1: Data Preprocessing & Timezone Alignment ===")
    loader = M1DataLoader(target_tz="Asia/Ho_Chi_Minh")
    
    try:
        m1_df = loader.load_directory(directory_path=data_dir, pattern=file_pattern)
    except Exception as err:
        logger.error(f"Stage 1 failed to load data: {err}")
        return

    # =========================================================================
    # STAGE 2: Quantitative EDA & Behavioral Feature Engineering
    # =========================================================================
    logger.info("=== STAGE 2: Behavioral Feature Engineering & Quantitative EDA ===")
    extractor = IntradayFeatureExtractor(min_dev_usd=min_dev_usd)
    daily_df = extractor.process_dataset(m1_df)

    if len(daily_df) == 0:
        logger.error("No valid trading day records extracted. Aborting.")
        return

    # Export daily features to CSV for transparency
    daily_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved processed daily features dataset to: {output_csv_path}")

    # Generate EDA Report
    eda_report = EDAReporter.generate_eda_report(daily_df)

    # =========================================================================
    # STAGE 3: Machine Learning Gatekeeper Model (LightGBM) & ONNX Export
    # =========================================================================
    logger.info("=== STAGE 3: LightGBM ML Gatekeeper Training & Purged Walk-Forward CV ===")
    gatekeeper = LightGBMGatekeeper()
    
    try:
        cv_results = gatekeeper.train_and_evaluate_cv(daily_df, n_splits=5)
        gatekeeper.print_ml_report(cv_results)

        # Save PKL model artifact
        joblib.dump(gatekeeper.model, output_pkl_path)
        logger.info(f"Saved LightGBM model binary to: {output_pkl_path}")

        # Export to ONNX for MT5 EA deployment
        ONNXExporter.export_to_onnx(
            model=gatekeeper.model,
            feature_cols=gatekeeper.FEATURE_COLS,
            output_path=output_onnx_path
        )

    except Exception as err:
        logger.error(f"Stage 3 ML Gatekeeper training encountered an error: {err}", exc_info=True)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE (v5_system)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v5 Intraday Anchor Mean Reversion Pipeline")
    parser.add_argument("--data-dir", type=str, default=BASE_DIR, help="Directory containing M1 CSV files")
    parser.add_argument("--min-dev", type=float, default=1.0, help="Minimum USD deviation trigger for reversion target")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        min_dev_usd=args.min_dev
    )
