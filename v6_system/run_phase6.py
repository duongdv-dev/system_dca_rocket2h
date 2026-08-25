"""
v6_system/run_phase6.py
-----------------------
Runner script thực thi Phase 6 - Advanced AI Labeling System (Xây Label cho AI).
Trích xuất 25 đặc trưng kỹ thuật, gán nhãn 3 loại Target chuyên sâu (y_anchor, y_basket_tp, y_safe_revert),
và huấn luyện so sánh hiệu năng mô hình AI Gatekeeper trên từng loại nhãn.
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
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 80)
    print("     VERSION 6 - PHASE 6: ADVANCED AI LABELING SYSTEM (XÂY LABEL CHO AI)")
    print("=" * 80)

    # 1. Nạp và làm sạch dữ liệu M1
    print("\n[Bước 1/4] Nạp & Chuẩn hóa dữ liệu M1 2020-2025...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    # 2. Trích xuất 25 Features
    print("\n[Bước 2/4] Trích xuất 25 Đặc trưng Kỹ thuật...")
    fe = V6FeatureEngineer()
    df_features = fe.extract_session_features_and_targets(df_clean)

    # 3. Gán nhãn 3 loại Target chuyên sâu Phase 6
    print("\n[Bước 3/4] Xây dựng 3 loại Nhãn AI (y_anchor, y_basket_tp, y_safe_revert)...")
    label_builder = V6LabelBuilder(default_step=5.0, default_tp=2.0, max_adverse=15.0)
    df_labeled = label_builder.build_all_labels(df_features)

    # 4. Huấn luyện & So sánh mô hình AI trên từng loại nhãn (Time-Series Split 2020-2023 vs 2024-2025)
    print("\n[Bước 4/4] Huấn luyện & So sánh hiệu năng AI trên từng loại nhãn...")
    
    label_types = [
        {"name": "Anchor Reversion (y_anchor)", "col": "y_anchor"},
        {"name": "Basket TP Target (y_basket_tp)", "col": "y_basket_tp"},
        {"name": "Risk-Safe Reversion (y_safe_revert)", "col": "y_safe_revert"}
    ]

    gatekeeper = MLGatekeeper(random_state=42)
    comparison_results = []

    for item in label_types:
        lbl_col = item["col"]
        df_temp = df_labeled.copy()
        df_temp["target_revert"] = df_temp[lbl_col]

        metrics, df_imp = gatekeeper.train_and_evaluate(df_temp, split_year=2024)
        metrics["label_type"] = item["name"]
        comparison_results.append(metrics)

    # Xuất kết quả
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_path = os.path.join(OUTPUT_DIR, "phase6_label_analysis.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)

    sample_csv_path = os.path.join(OUTPUT_DIR, "phase6_features_labeled.csv")
    df_labeled.head(100).to_csv(sample_csv_path, index=False)

    # In bảng so sánh trực quan
    print("\n" + "=" * 95)
    print("                KẾT QUẢ SO SÁNH HIỆU NĂNG MÔ HÌNH AI TRÊN CÁC LOẠI NHÃN")
    print("=" * 95)
    hdr = f"{'Label Type':<35} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}"
    print(hdr)
    print("-" * 95)

    for r in comparison_results:
        l_name = r["label_type"]
        auc = r["roc_auc"]
        prauc = r["pr_auc"]
        prec = r["precision"]
        rec = r["recall"]
        f1 = r["f1_score"]
        print(f"{l_name:<35} | {auc:<8} | {prauc:<8} | {prec:<9} | {rec:<8} | {f1:<8}")

    print("=" * 95)
    print(f"\nFile Báo cáo Label Analysis : {report_path}")
    print(f"File Dataset Labeled Sample : {sample_csv_path}")
    print("Phase 6 HOÀN THÀNH THÀNH CÔNG!\n")


if __name__ == "__main__":
    main()
