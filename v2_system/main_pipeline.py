"""
v2_system/main_pipeline.py
==========================
Master Main Pipeline Script cho Training & Backtest Kiểm Định (v2).
Được thiết kế bởi Senior Quantitative Researcher & Systems Trading Engineer.

Tập trung 100% vào kiểm tra Training & Backtest (Bỏ qua đóng gói ONNX):
1. Tải dữ liệu M1 (2020-2023 Train, 2024-2025 Test) và trích xuất đặc trưng chuẩn hóa dừng.
2. CHẠY ROLLING WALK-FORWARD VALIDATION (2020-2023):
   - Train 6 tháng -> Test 6 tháng tiếp theo (Cuộn 7 cửa sổ bán niên).
3. HUẤN LUYỆN MASTER MODEL (Full Train 2020-2023).
4. CHẠY TEST CUỐI KỲ 2 NĂM OUT-OF-SAMPLE (2024-2025).
5. Xuất báo cáo CSV & Biểu đồ Equity Curve kiểm định.
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
from walk_forward_engine import WalkForwardEngine

def run_main_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2_dir = os.path.join(base_dir, "v2_system")
    os.makedirs(v2_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-3]_m1.csv")))
    test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[4-5]_m1.csv")))

    print("\n==================================================================")
    print(" 🚀 HỆ THỐNG XAUUSD INTRADAY GRID/DCA (v2) - TRAINING & BACKTEST EVALUATION")
    print("==================================================================")
    print(f" • File Train (2020-2023): {[os.path.basename(f) for f in train_files]}")
    print(f" • File Test  (2024-2025): {[os.path.basename(f) for f in test_files]}\n")

    pipeline = DataPipeline(base_dir)

    # 1. PHẦN 1: DATA PIPELINE & CHUẨN HÓA ĐẶC TRƯNG
    print("------------------------------------------------------------------")
    print(" 📥 PHẦN 1: DATA PIPELINE & CHUẨN HÓA ĐẶC TRƯNG (06:00 - 09:59 VN)")
    print("------------------------------------------------------------------")
    train_feature_df, train_m1_dict = pipeline.extract_daily_dataset(train_files)
    test_feature_df, test_m1_dict = pipeline.extract_daily_dataset(test_files)

    print(f" -> Tập Train (2020-2023) trích xuất được: {len(train_feature_df)} ngày hợp lệ")
    print(f" -> Tập Test  (2024-2025) trích xuất được: {len(test_feature_df)} ngày hợp lệ")

    feature_cols = [
        'morning_range_atr', 'morning_body_atr', 'morning_momentum',
        'vwap_dist_atr', 'bb_zscore_m15', 'bb_slope_m15'
    ]

    # 2. PHẦN 2: ROLLING WALK-FORWARD VALIDATION (2020 - 2023)
    print("\n------------------------------------------------------------------")
    print(" 🔄 PHẦN 2: ROLLING WALK-FORWARD (TRAIN 6M -> TEST 6M TIẾP THEO)")
    print("------------------------------------------------------------------")
    wf_engine = WalkForwardEngine(
        feature_cols=feature_cols,
        initial_balance=10000.0,
        risk_pct_per_session=0.02
    )
    wf_trades_df, wf_metrics = wf_engine.run_walk_forward_process(train_feature_df, train_m1_dict)

    # 3. PHẦN 3: MASTER MODEL TRAINING (FULL 2020-2023)
    print("------------------------------------------------------------------")
    print(" 🤖 PHẦN 3: HUẤN LUYỆN MASTER MODEL (FULL 2020-2023)")
    print("------------------------------------------------------------------")
    simulator = GridSimulator()
    labeled_df, best_params_df = simulator.evaluate_training_set(train_feature_df, train_m1_dict)

    clustering = StrategyClustering(n_clusters=3, random_state=42)
    mapped_labels, master_preset_centroids = clustering.fit_clusters(best_params_df)
    train_dataset = clustering.assign_train_targets(labeled_df, best_params_df, mapped_labels)

    trainer = MLTrainer(feature_cols=feature_cols)
    trainer.check_feature_drift(train_feature_df, test_feature_df)
    trainer.train_lightgbm(train_dataset)

    # 4. PHẦN 4: FINAL OUT-OF-SAMPLE TEST (2 NĂM CUỐI 2024-2025)
    print("------------------------------------------------------------------")
    print(" 🧪 PHẦN 4: FINAL OUT-OF-SAMPLE TEST (2 NĂM CUỐI 2024 - 2025)")
    print("------------------------------------------------------------------")
    test_preds = trainer.predict(test_feature_df)

    final_backtest_engine = OOSBacktestEngine(
        preset_centroids=master_preset_centroids,
        initial_balance=10000.0,
        risk_pct_per_session=0.02
    )

    test_trades_df, test_metrics = final_backtest_engine.run_backtest(test_feature_df, test_preds, test_m1_dict)

    # BẢNG TỔNG HỢP CUỐI CÙNG
    print("\n=========================================================================================================")
    print(" 📊 BẢNG TỔNG HỢP HIỆU NĂNG TOÀN BỘ QUY TRÌNH WALK-FORWARD & FINAL TEST 2024-2025")
    print("=========================================================================================================")
    print(f" • Lợi Nhuận Walk-Forward (2020-2023):  +{wf_metrics['overall_return_pct']:.2f}% (Số dư 2023: ${wf_metrics['final_balance']:,.2f})")
    print(f" • Win Rate Walk-Forward Trung Bình 6M: {wf_metrics['avg_win_rate']:.1f}%")
    print(f" • Profit Factor Walk-Forward 6M:       {wf_metrics['avg_pf']:.2f}")
    print(" ---------------------------------------------------------------------------------------------------------")
    print(f" • Lợi Nhuận Final Test (2024-2025):     +{test_metrics['total_return_pct']:.2f}% (Số dư 2025: ${test_metrics['final_balance']:,.2f})")
    print(f" • Win Rate Final Test (2024-2025):      {test_metrics['win_rate']:.1f}%")
    print(f" • Profit Factor Final Test (2024-2025): {test_metrics['profit_factor']:.2f}")
    print(f" • Max Drawdown Final Test (2024-2025):  {test_metrics['max_drawdown_pct']:.2f}%")
    print("=========================================================================================================\n")

    # Lưu biểu đồ Equity Curve & Báo cáo CSV
    equity_img_path = os.path.join(v2_dir, "equity_curve.png")
    final_backtest_engine.plot_equity_curve(test_trades_df['balance'].tolist(), equity_img_path)

    report_path = os.path.join(v2_dir, "oos_trades_report.csv")
    test_trades_df.to_csv(report_path, index=False)
    print(f"[MainPipeline] Đã lưu báo cáo chi tiết giao dịch -> {report_path}")
    print("\n[MainPipeline] Hoàn tất toàn bộ quy trình Training & Backtest Evaluation!")

if __name__ == '__main__':
    run_main_pipeline()
