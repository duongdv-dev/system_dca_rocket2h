"""
v2_system/main_pipeline.py
==========================
Master Main Pipeline Script cho Hệ Thống XAUUSD Intraday Mean-Reversion Grid/DCA (v2).
Được thiết kế bởi Senior Quantitative Researcher & Systems Trading Engineer.

Chạy toàn bộ quy trình khép kín và stream log trực tiếp ra console stdout (Unbuffered):
1. Tải dữ liệu M1 (2020-2023 Train, 2024-2025 Test) và trích xuất đặc trưng lúc 09:59.
2. Mô phỏng 72 kịch bản grid trên tập Train, chấm Fitness Score & gán nhãn.
3. Gom cụm K-Means (K=3) định hình 3 Preset chiến thuật chuẩn (Lưới hẹp, Tiêu chuẩn, Phòng thủ).
4. Huấn luyện LightGBM Classifier đa lớp & Xuất mô hình sang file `model.onnx`.
5. Run Out-of-Sample Backtest (2024-2025) tích hợp Risk Engine (2%), Time-Decay TP & Hard Exit 12:00.
6. In báo cáo chi tiết Kế hoạch, Chiến thuật & Kết quả Backtest ra stdout.
"""

import os
import sys
import glob
import pandas as pd
from data_pipeline import DataPipeline
from grid_simulator import GridSimulator
from strategy_clustering import StrategyClustering
from ml_trainer import MLTrainer
from backtest_engine import OOSBacktestEngine

def run_main_pipeline():
    # 1. Xác định đường dẫn dữ liệu
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2_dir = os.path.join(base_dir, "v2_system")
    os.makedirs(v2_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-3]_m1.csv")))
    test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[4-5]_m1.csv")))

    print("\n==================================================================")
    print(" 🚀 HỆ THỐNG XAUUSD INTRADAY GRID/DCA (v2) - DOCKER PRODUCTION PIPELINE")
    print("==================================================================")
    print(f" • File Train (2020-2023): {[os.path.basename(f) for f in train_files]}")
    print(f" • File Test  (2024-2025): {[os.path.basename(f) for f in test_files]}\n")

    pipeline = DataPipeline(base_dir)

    # 2. PHẦN 1: DATA PIPELINE & FEATURE ENGINEERING
    print("------------------------------------------------------------------")
    print(" 📥 PHẦN 1: DATA PIPELINE & FEATURE ENGINEERING (06:00 - 09:59 VN)")
    print("------------------------------------------------------------------")
    train_feature_df, train_m1_dict = pipeline.extract_daily_dataset(train_files)
    test_feature_df, test_m1_dict = pipeline.extract_daily_dataset(test_files)

    print(f" -> Tập Train (2020-2023) trích xuất được: {len(train_feature_df)} ngày hợp lệ")
    print(f" -> Tập Test  (2024-2025) trích xuất được: {len(test_feature_df)} ngày hợp lệ")

    # 3. PHẦN 2: SIMULATION ENGINE & K-MEANS CLUSTERING
    print("\n------------------------------------------------------------------")
    print(" ⚡ PHẦN 2: MÔ PHỎNG 72 KỊCH BẢN & ĐỊNH HÌNH KẾ HOẠCH CHIẾN THUẬT")
    print("------------------------------------------------------------------")
    simulator = GridSimulator()
    labeled_df, best_params_df = simulator.evaluate_training_set(train_feature_df, train_m1_dict)

    clustering = StrategyClustering(n_clusters=3, random_state=42)
    mapped_labels, preset_centroids = clustering.fit_clusters(best_params_df)
    train_dataset = clustering.assign_train_targets(labeled_df, best_params_df, mapped_labels)

    print("\n 🎯 KẾ HOẠCH BẢNG CHIẾN THUẬT ĐƯỢC THIẾT LẬP (CENTROIDS):")
    print(" +---------+-------------------------+-------------+----------+------------+------------+--------------+")
    print(" | Class   | Tên Chiến Thuật         | Step_0      | Step_Exp | Max_Orders | Multiplier | TP_BE Ratio  |")
    print(" +---------+-------------------------+-------------+----------+------------+------------+--------------+")
    print(" | Class 0 | No-Trade (Bảo vệ vốn)   |     N/A     |   N/A    |     0      |    N/A     |     N/A      |")
    for p_id in [1, 2, 3]:
        p = preset_centroids[p_id]
        print(f" | Class {p_id} | {p['name']:<23} | {p['step_0_ratio']:<11} | {p['step_exp']:<8} | {p['max_orders']:<10} | {p['multiplier']:<10} | {p['tp_be_ratio']:<12} |")
    print(" +---------+-------------------------+-------------+----------+------------+------------+--------------+\n")

    # 4. PHẦN 3: MACHINE LEARNING TRAINING & ONNX EXPORT
    print("------------------------------------------------------------------")
    print(" 🤖 PHẦN 3: HUẤN LUYỆN MODEL LIGHTGBM & XUẤT NATIVE ONNX MODEL")
    print("------------------------------------------------------------------")
    feature_cols = [
        'atr_14_m15', 'morning_range', 'morning_body',
        'morning_momentum', 'vwap_dist_atr', 'bb_zscore_m15', 'bb_slope_m15'
    ]

    trainer = MLTrainer(feature_cols=feature_cols)
    trainer.train_lightgbm(train_dataset)

    onnx_path = os.path.join(v2_dir, "model.onnx")
    trainer.export_to_onnx(onnx_path)

    # 5. PHẦN 4: OUT-OF-SAMPLE BACKTEST (2024-2025)
    print("\n------------------------------------------------------------------")
    print(" 🧪 PHẦN 4: OUT-OF-SAMPLE BACKTEST CHI TIẾT (2 NĂM 2024 - 2025)")
    print("------------------------------------------------------------------")
    predictions = trainer.predict(test_feature_df)

    oos_engine = OOSBacktestEngine(
        preset_centroids=preset_centroids,
        initial_balance=10000.0,
        risk_pct_per_session=0.02
    )

    trades_df, metrics = oos_engine.run_backtest(test_feature_df, predictions, test_m1_dict)

    # Save Equity Curve Plot & Performance Report CSV (Lưu trữ nhẹ)
    equity_img_path = os.path.join(v2_dir, "equity_curve.png")
    oos_engine.plot_equity_curve(trades_df['balance'].tolist(), equity_img_path)

    report_path = os.path.join(v2_dir, "oos_trades_report.csv")
    trades_df.to_csv(report_path, index=False)
    
    print("✨ DOCKER RUN COMPLETE! Toàn bộ log và kết quả chi tiết đã được in thành công ra console stdout.")

if __name__ == '__main__':
    run_main_pipeline()
