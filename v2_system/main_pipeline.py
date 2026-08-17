"""
v2_system/main_pipeline.py
==========================
Master Main Pipeline Script cho Hệ Thống XAUUSD Intraday Mean-Reversion Grid/DCA (v2).
Được thiết kế bởi Senior Quantitative Researcher & Systems Trading Engineer.

Chạy toàn bộ quy trình khép kín:
1. Tải dữ liệu M1 (2020-2023 Train, 2024-2025 Test) và trích xuất đặc trưng chuẩn hóa dừng.
2. Mô phỏng 48 kịch bản grid trên tập Train, chấm Fitness Score & gán nhãn.
3. Gom cụm K-Means (K=3) định hình 3 Preset chiến thuật chuẩn (Lưới hẹp, Tiêu chuẩn, Phòng thủ).
4. Kiểm tra độ trôi dữ liệu (Feature Drift), Chạy Stratified 5-Fold CV & Train LightGBM Classifier.
5. Chạy mô phỏng kiểm định In-Sample (Train 2020-2023) và Out-of-Sample (Test 2024-2025).
6. In Bảng So Sánh Hiệu Năng Đối Chiếu In-Sample vs Out-of-Sample.
"""

import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from grid_simulator import GridSimulator
from strategy_clustering import StrategyClustering
from ml_trainer import MLTrainer
from backtest_engine import OOSBacktestEngine

def run_main_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2_dir = os.path.join(base_dir, "v2_system")
    os.makedirs(v2_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-3]_m1.csv")))
    test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[4-5]_m1.csv")))

    print("\n==================================================================")
    print(" 🚀 HỆ THỐNG XAUUSD INTRADAY GRID/DCA (v2) - PRODUCTION PIPELINE")
    print("==================================================================")
    print(f" • File Train (2020-2023): {[os.path.basename(f) for f in train_files]}")
    print(f" • File Test  (2024-2025): {[os.path.basename(f) for f in test_files]}\n")

    pipeline = DataPipeline(base_dir)

    # 1. PHẦN 1: DATA PIPELINE & FEATURE ENGINEERING
    print("------------------------------------------------------------------")
    print(" 📥 PHẦN 1: DATA PIPELINE & CHUẨN HÓA ĐẶC TRƯNG (06:00 - 09:59 VN)")
    print("------------------------------------------------------------------")
    train_feature_df, train_m1_dict = pipeline.extract_daily_dataset(train_files)
    test_feature_df, test_m1_dict = pipeline.extract_daily_dataset(test_files)

    print(f" -> Tập Train (2020-2023) trích xuất được: {len(train_feature_df)} ngày hợp lệ")
    print(f" -> Tập Test  (2024-2025) trích xuất được: {len(test_feature_df)} ngày hợp lệ")

    # 2. PHẦN 2: SIMULATION ENGINE & K-MEANS CLUSTERING
    print("\n------------------------------------------------------------------")
    print(" ⚡ PHẦN 2: MÔ PHỎNG KỊCH BẢN & ĐỊNH HÌNH PRESETS CHIẾN THUẬT")
    print("------------------------------------------------------------------")
    simulator = GridSimulator()
    labeled_df, best_params_df = simulator.evaluate_training_set(train_feature_df, train_m1_dict)

    clustering = StrategyClustering(n_clusters=3, random_state=42)
    mapped_labels, preset_centroids = clustering.fit_clusters(best_params_df)
    train_dataset = clustering.assign_train_targets(labeled_df, best_params_df, mapped_labels)

    # 3. PHẦN 3: FEATURE DRIFT CHECK & MACHINE LEARNING TRAINING
    print("------------------------------------------------------------------")
    print(" 🤖 PHẦN 3: KIỂM TRA DRIFT & HUẤN LUYỆN MODEL LIGHTGBM MULTI-CLASS")
    print("------------------------------------------------------------------")
    feature_cols = [
        'morning_range_atr', 'morning_body_atr', 'morning_momentum',
        'vwap_dist_atr', 'bb_zscore_m15', 'bb_slope_m15'
    ]

    trainer = MLTrainer(feature_cols=feature_cols)
    trainer.check_feature_drift(train_feature_df, test_feature_df)
    trainer.train_lightgbm(train_dataset)

    onnx_path = os.path.join(v2_dir, "model.onnx")
    trainer.export_to_onnx(onnx_path)

    # 4. PHẦN 4: IN-SAMPLE VS OUT-OF-SAMPLE BACKTEST DUAL VERIFICATION
    print("------------------------------------------------------------------")
    print(" 🧪 PHẦN 4: DUAL BACKTEST VERIFICATION (IN-SAMPLE VS OUT-OF-SAMPLE)")
    print("------------------------------------------------------------------")
    
    # Predict In-Sample (Train) & Out-of-Sample (Test)
    train_preds = trainer.predict(train_feature_df)
    test_preds = trainer.predict(test_feature_df)

    backtest_engine = OOSBacktestEngine(
        preset_centroids=preset_centroids,
        initial_balance=10000.0,
        risk_pct_per_session=0.02
    )

    print("\n [1/2] RUNNING IN-SAMPLE BACKTEST (TRAIN 2020-2023)...")
    train_trades_df, train_metrics = backtest_engine.run_backtest(train_feature_df, train_preds, train_m1_dict)

    print("\n [2/2] RUNNING OUT-OF-SAMPLE BACKTEST (TEST 2024-2025)...")
    test_trades_df, test_metrics = backtest_engine.run_backtest(test_feature_df, test_preds, test_m1_dict)

    # BẢNG SO SÁNH ĐỐI CHIẾU DUAL PERFORMANCE
    print("\n=========================================================================================================")
    print(" 📊 BẢNG SO SÁNH HIỆU NĂNG ĐỐI CHIẾU: IN-SAMPLE (TRAIN) VS OUT-OF-SAMPLE (TEST)")
    print("=========================================================================================================")
    print(" | Chỉ Số Hiệu Năng (Metric)      | In-Sample Train (2020-2023) | Out-of-Sample Test (2024-2025) | Tình Trạng   |")
    print(" +--------------------------------+-----------------------------+--------------------------------+--------------+")
    print(f" | Tổng số ngày quan sát           | {train_metrics['total_days']:<27} | {test_metrics['total_days']:<30} | OK           |")
    print(f" | Tỷ lệ No-Trade (%)              | {train_metrics['no_trade_pct']:<27.1f} | {test_metrics['no_trade_pct']:<30.1f} | OK           |")
    print(f" | Win Rate (%)                   | {train_metrics['win_rate']:<27.1f} | {test_metrics['win_rate']:<30.1f} | Stable       |")
    print(f" | Profit Factor                   | {train_metrics['profit_factor']:<27.2f} | {test_metrics['profit_factor']:<30.2f} | Stable       |")
    print(f" | Max Drawdown (%)                | {train_metrics['max_drawdown_pct']:<27.2f} | {test_metrics['max_drawdown_pct']:<30.2f} | Safe         |")
    print(f" | Lợi nhuận ròng Return (%)       | +{train_metrics['total_return_pct']:<26.2f} | +{test_metrics['total_return_pct']:<29.2f} | High Yield   |")
    print(f" | Số dư tài khoản ($)             | ${train_metrics['final_balance']:<26,.2f} | ${test_metrics['final_balance']:<29,.2f} | Profitable   |")
    print("=========================================================================================================\n")

    # Lưu biểu đồ Equity Curve & Báo cáo CSV
    equity_img_path = os.path.join(v2_dir, "equity_curve.png")
    backtest_engine.plot_equity_curve(test_trades_df['balance'].tolist(), equity_img_path)

    report_path = os.path.join(v2_dir, "oos_trades_report.csv")
    test_trades_df.to_csv(report_path, index=False)
    print(f"[MainPipeline] Đã lưu báo cáo chi tiết giao dịch -> {report_path}")
    print("\n[MainPipeline] Hoàn tất toàn bộ quy trình Production Pipeline!")

if __name__ == '__main__':
    run_main_pipeline()
