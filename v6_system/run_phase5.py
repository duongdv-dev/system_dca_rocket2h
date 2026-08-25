"""
v6_system/run_phase5.py
-----------------------
Runner script thực thi Phase 5 - XGBoost V1 Probability Model.
Trích xuất 25 đặc trưng, gán nhãn Target xác suất hồi về Anchor trước 12:00 VN,
và huấn luyện mô hình XGBoost V1 Gatekeeper theo Time-Series Split.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.feature_engineering import V6FeatureEngineer
from v6_system.ml_gatekeeper import MLGatekeeper
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 80)
    print("     VERSION 6 - PHASE 5: XGBOOST V1 PROBABILITY MODEL (GATEKEEPER)")
    print("=" * 80)

    # 1. Nạp và làm sạch dữ liệu M1
    print("\n[Bước 1/4] Nạp & Chuẩn hóa dữ liệu M1 2020-2025...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    # 2. Trích xuất 25 Features & Gán nhãn Target
    print("\n[Bước 2/4] Trích xuất 25 Đặc trưng Kỹ thuật & Gán nhãn Target xác suất hồi Anchor...")
    fe = V6FeatureEngineer()
    df_features = fe.extract_session_features_and_targets(df_clean)

    # 3. Huấn luyện & Đánh giá XGBoost V1 (Time-Series Split 2020-2023 vs 2024-2025)
    print("\n[Bước 3/4] Huấn luyện mô hình XGBoost V1 (Train: 2020-2023 | Test: 2024-2025)...")
    gatekeeper = MLGatekeeper(random_state=42)
    metrics, df_importance = gatekeeper.train_and_evaluate(df_features, split_year=2024)

    # 4. Xuất kết quả
    print("\n[Bước 4/4] Xuất kết quả báo cáo mô hình AI...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_json_path = os.path.join(OUTPUT_DIR, "phase5_model_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    importance_csv_path = os.path.join(OUTPUT_DIR, "phase5_feature_importance.csv")
    df_importance.to_csv(importance_csv_path, index=False)

    features_sample_path = os.path.join(OUTPUT_DIR, "phase5_features_sample.csv")
    df_features.head(100).to_csv(features_sample_path, index=False)

    # In kết quả báo cáo
    print("\n" + "=" * 80)
    print("                BÁO CÁO HIỆU NĂNG MÔ HÌNH XGBOOST V1")
    print("=" * 80)
    print(f"Thuật toán mô hình          : {metrics['model_name']}")
    print(f"Số lượng mẫu Train (2020-23): {metrics['train_size']:,} nến M1")
    print(f"Số lượng mẫu Test (2024-25) : {metrics['test_size']:,} nến M1")
    print(f"ROC-AUC Score (Dự báo)     : {metrics['roc_auc']}")
    print(f"PR-AUC Score                : {metrics['pr_auc']}")
    print(f"Precision                   : {metrics['precision']}")
    print(f"Recall                      : {metrics['recall']}")
    print(f"F1-Score                    : {metrics['f1_score']}")
    print("=" * 80)

    # In Top 15 Feature Importance
    print("\n" + "-" * 60)
    print("      TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT DỰ BÁO HỒI ANCHOR")
    print("-" * 60)
    for idx, row in df_importance.head(15).iterrows():
        rank = idx + 1
        name = row["feature"]
        score = row["importance"]
        bar = "█" * int(score * 100)
        print(f" {rank:>2}. {name:<25} | {score:.4f} {bar}")
    print("-" * 60)

    print(f"\nFile Model Metrics    : {metrics_json_path}")
    print(f"File Feature Importance: {importance_csv_path}")
    print("Phase 5 HOÀN THÀNH THÀNH CÔNG!\n")


if __name__ == "__main__":
    main()
