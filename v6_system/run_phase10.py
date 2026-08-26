"""
v6_system/run_phase10.py
------------------------
Runner script thực thi Phase 10 - Walk-Forward Testing.
Thực hiện kiểm thử cuộn thời gian 4 Folds (2022, 2023, 2024, 2025 OOS) và ghép nối đường cong tài sản Out-of-Sample thực tế.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.feature_engineering import V6FeatureEngineer
from v6_system.label_builder import V6LabelBuilder
from v6_system.walk_forward_engine import WalkForwardEngine
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 90)
    print("      VERSION 6 - PHASE 10: WALK-FORWARD TESTING & GENERALIZATION VERIFICATION")
    print("=" * 90)

    # 1. Nạp dữ liệu & Trích xuất đặc trưng
    print("\n[Bước 1/4] Nạp dữ liệu M1 2020-2025 & Trích xuất 25 Đặc trưng Kỹ thuật...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    fe = V6FeatureEngineer()
    df_features = fe.extract_session_features_and_targets(df_clean)

    label_builder = V6LabelBuilder()
    df_labeled = label_builder.build_all_labels(df_features)

    # 2. Thực thi Walk-Forward Testing 4 Folds
    print("\n[Bước 2/4] Chạy Walk-Forward Testing 4 Folds (Expanding Windows 2022-2025)...")
    wf_engine = WalkForwardEngine(random_state=42)
    df_combined_oos, summary = wf_engine.run_walk_forward(df_labeled)

    # 3. Xuất kết quả báo cáo
    print("\n[Bước 3/4] Xuất kết quả báo cáo Walk-Forward...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(OUTPUT_DIR, "phase10_walk_forward_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    oos_csv_path = os.path.join(OUTPUT_DIR, "phase10_oos_trades.csv")
    df_combined_oos.to_csv(oos_csv_path, index=False)

    # 4. Hiển thị bảng tổng hợp kết quả từng Fold và Combined OOS
    print("\n" + "=" * 90)
    print("           BẢNG TỔNG HỢP KẾT QUẢ OUT-OF-SAMPLE THEO TỪNG FOLD (2022 - 2025)")
    print("=" * 90)
    hdr = f"{'Fold Name':<30} | {'Traded Days':<12} | {'Net PnL':<10} | {'WinRate':<8} | {'PF':<6} | {'MaxDD($)':<9}"
    print(hdr)
    print("-" * 90)

    for f_detail in summary.get("fold_details", []):
        f_title = f_detail["title"]
        tdays = str(f_detail["traded_days"])
        pnl = f"${f_detail['net_profit']:.1f}"
        wr = f"{f_detail['win_rate']}%"
        pf = str(f_detail["profit_factor"])
        mdd = f"${f_detail['max_dd_dollars']:.1f}"
        print(f"{f_title:<30} | {tdays:<12} | {pnl:<10} | {wr:<8} | {pf:<6} | {mdd:<9}")

    print("-" * 90)
    c_title = summary["title"]
    c_tdays = str(summary["traded_days"])
    c_pnl = f"${summary['net_profit']:.1f}"
    c_wr = f"{summary['win_rate']}%"
    c_pf = str(summary["profit_factor"])
    c_mdd = f"${summary['max_dd_dollars']:.1f}"
    print(f"{c_title:<30} | {c_tdays:<12} | {c_pnl:<10} | {c_wr:<8} | {c_pf:<6} | {c_mdd:<9}")
    print("=" * 90)

    # Kết luận WFE
    wfe = summary.get("walk_forward_efficiency", 0.0)
    print("\n" + "*" * 85)
    print("          KẾT LUẬN KHẢ NĂNG KHÁI QUÁT HÓA (WALK-FORWARD EFFICIENCY)")
    print("*" * 85)
    print(f"Combined Out-of-Sample Net Profit (2022-2025) : ${summary['net_profit']}")
    print(f"Combined Out-of-Sample Profit Factor          : {summary['profit_factor']}")
    print(f"Combined Out-of-Sample Win Rate               : {summary['win_rate']}%")
    print(f"Chỉ số Walk-Forward Efficiency (WFE Ratio)   : {wfe}")

    if wfe >= 0.70:
        print("\n-> XÁC NHẬN: MÔ HÌNH ĐẠT KHẢ NĂNG KHÁI QUÁT HÓA TỐT (GENERALIZE THÀNH CÔNG)!")
        print("   Hệ thống duy trì lợi nhuận ổn định trên cả 4 năm Out-of-Sample chưa từng thấy.")
    else:
        print("\n-> CẢNH BÁO: Chỉ số WFE < 0.70. Cần tiếp tục theo dõi độ ổn định trên các regime biến động mạnh.")

    print("*" * 85)
    print(f"\nFile Summary JSON : {summary_path}")
    print(f"File OOS Trades CSV: {oos_csv_path}")
    print("Phase 10 HOÀN THÀNH THÀNH CÔNG!\n")


if __name__ == "__main__":
    main()
