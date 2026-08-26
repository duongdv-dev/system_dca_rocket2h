"""
v6_system/run_phase11.py
------------------------
Runner script thực thi Phase 11 - Monte Carlo / Stress Test.
Chạy 500 lượt mô phỏng tiêm nhiễu ma sát (Spread expansion, Slippage, Latency, Shock)
và cấp Chứng Nhận Robustness Certificate cho hệ thống Version 6.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.feature_engineering import V6FeatureEngineer
from v6_system.label_builder import V6LabelBuilder
from v6_system.ml_gatekeeper import MLGatekeeper
from v6_system.stress_tester import MonteCarloStressTester
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 90)
    print("       VERSION 6 - PHASE 11: MONTE CARLO / STRESS TESTING & ROBUSTNESS CERTIFICATION")
    print("=" * 90)

    # 1. Nạp dữ liệu & Trích xuất đặc trưng
    print("\n[Bước 1/4] Nạp dữ liệu M1 2020-2025 & Trích xuất Đặc trưng & AI Probabilities...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    fe = V6FeatureEngineer()
    df_features = fe.extract_session_features_and_targets(df_clean)

    label_builder = V6LabelBuilder()
    df_labeled = label_builder.build_all_labels(df_features)

    gatekeeper = MLGatekeeper(random_state=42)
    df_labeled["target_revert"] = df_labeled["y_anchor"]
    gatekeeper.train_and_evaluate(df_labeled, split_year=2024)

    X_all = df_labeled[MLGatekeeper.FEATURE_COLS].astype(float)
    df_labeled["prob_revert"] = gatekeeper.model.predict_proba(X_all)[:, 1]

    # 2. Thực thi Monte Carlo Stress Test 500 Lượt
    print("\n[Bước 2/4] Chạy Monte Carlo Stress Test 500 Lượt Giả Lập Ma Sát Thực Tế...")
    tester = MonteCarloStressTester(num_simulations=500, random_state=42)
    df_runs, overall_report = tester.run_stress_test(df_labeled)

    # 3. Xuất kết quả báo cáo
    print("\n[Bước 3/4] Xuất báo cáo kết quả Stress Test...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_path = os.path.join(OUTPUT_DIR, "phase11_stress_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "phase11_monte_carlo_distribution.csv")
    df_runs.to_csv(csv_path, index=False)

    # 4. Hiển thị bảng so sánh Ideal vs Median vs 95th Percentile Worst Case
    print("\n" + "=" * 90)
    print("             BẢNG TỔNG HỢP KẾT QUẢ MONTE CARLO STRESS TEST (500 SIMULATIONS)")
    print("=" * 90)
    hdr = f"{'Metric / Scenario':<35} | {'Median (50th Pct)':<20} | {'Worst Case (95th Pct)':<20}"
    print(hdr)
    print("-" * 90)
    print(f"{'Net Profit ($)':<35} | ${overall_report['median_net_profit']:<19} | ${overall_report['worst_95_net_profit']:<19}")
    print(f"{'Profit Factor (PF)':<35} | {overall_report['median_profit_factor']:<20} | {overall_report['worst_95_profit_factor']:<20}")
    print(f"{'Max Drawdown ($)':<35} | ${overall_report['median_max_dd_dollars']:<19} | ${overall_report['worst_95_max_dd_dollars']:<19}")
    print("=" * 90)

    # Kết luận Robustness Certificate
    print("\n" + "#" * 85)
    print("                  PHÁN QUYẾT CHỨNG NHẬN ROBUSTNESS (OFFICIAL CERTIFICATE)")
    print("#" * 85)
    if overall_report["robustness_passed"]:
        print("-> ĐẠT CHỨNG NHẬN: CERTIFIED ROBUST!")
        print("   Chiến lược giữ vững Profit Factor > 1.10 ngay cả trong 95% trường hợp xấu nhất.")
        print("   Hệ thống sẵn sàng vận hành live mượt mà chịu được ma sát spread/slippage thực tế.")
    else:
        print("-> CẢNH BÁO LOẠI: FRAGILE REJECT!")
        print("   Chiến lược bị suy giảm hiệu năng khi gặp ma sát thực tế cao.")
    print("#" * 85 + "\n")


if __name__ == "__main__":
    main()
