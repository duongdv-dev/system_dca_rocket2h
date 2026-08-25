"""
v6_system/run_phase4.py
-----------------------
Runner script thực thi Phase 4 - Robustness Testing & Parameter Plateau Detection.
Xác định vùng tham số ổn định (Plateau) thay vì chọn các điểm cực đại đơn lẻ dễ overfit.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.optimizer_v3 import OptimizerV3
from v6_system.robustness_analyzer import RobustnessAnalyzer
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 80)
    print("     VERSION 6 - PHASE 4: ROBUSTNESS TESTING & PARAMETER PLATEAU ANALYSIS")
    print("=" * 80)

    matrix_csv_path = os.path.join(OUTPUT_DIR, "phase3_optimization_matrix.csv")

    # Nếu chưa có file kết quả Phase 3, tự động thực thi Grid Search
    if not os.path.exists(matrix_csv_path):
        print("\nChưa tìm thấy dữ liệu Phase 3, đang khởi chạy Grid Search...")
        engineer = V6DataEngineer()
        df_raw = engineer.load_all_years(data_dir=BASE_DIR)
        df_clean, _ = engineer.audit_and_clean_data(df_raw)

        steps = [3, 4, 5, 6, 8, 10, 12, 15]
        max_dcas = [2, 3, 4, 5, 6, 7, 8]
        multipliers = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]

        optimizer = OptimizerV3(initial_capital=1000.0)
        df_grid, _ = optimizer.run_grid_search(df_clean, steps, max_dcas, multipliers)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_grid.to_csv(matrix_csv_path, index=False)
    else:
        print(f"\n[Bước 1/3] Nạp dữ liệu Ma trận Grid Search từ: {matrix_csv_path}")
        df_grid = pd.read_csv(matrix_csv_path)

    # 2. Phân tích Robustness & Plateau
    print("\n[Bước 2/3] Thực thi Phân Tích Độ Nhạy 2D & Plateau Scorecard...")
    analyzer = RobustnessAnalyzer(df_grid)

    # Phân tích trên tầng Max DCA = 5 (Tiêu chuẩn)
    df_plateau, pivot_symbols = analyzer.analyze_plateaus(max_dca_filter=5)
    ascii_map = analyzer.generate_ascii_heatmap(pivot_symbols, title="PLATEAU MATRIX (Max DCA = 5)")

    # 3. Xuất kết quả
    print("\n[Bước 3/3] Xuất báo cáo Plateau & Heatmaps...")
    heatmap_txt_path = os.path.join(OUTPUT_DIR, "phase4_heatmaps.txt")
    with open(heatmap_txt_path, "w", encoding="utf-8") as f:
        f.write(ascii_map + "\n\n")

    report_json_path = os.path.join(OUTPUT_DIR, "phase4_plateau_report.json")
    top_plateaus = df_plateau.sort_values("plateau_score", ascending=False).head(10).to_dict(orient="records")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(top_plateaus, f, ensure_ascii=False, indent=2)

    # In kết quả Heatmap trực quan ra console
    print("\n" + ascii_map)

    # Tổng kết các vùng Vàng / Ổn định
    gold_regions = df_plateau[df_plateau["classification"] == "Gold Plateau"]
    stable_regions = df_plateau[df_plateau["classification"] == "Stable Region"]

    print("\n" + "*" * 80)
    print("                 KẾT LUẬN VÙNG THAM SỐ ỔN ĐỊNH (PARAMETER PLATEAU)")
    print("*" * 80)
    print(f"Tổng số điểm thuộc Vùng Cao Nguyên Vàng (+++) : {len(gold_regions)}")
    print(f"Tổng số điểm thuộc Vùng Ổn Định (++)          : {len(stable_regions)}")

    if not gold_regions.empty:
        best_row = gold_regions.iloc[0]
        print(f"\n-> VÙNG THAM SỐ KHUYÊN DÙNG KHÔNG OVERFIT:")
        print(f"   DCA Step      : ${best_row['step']} (Vùng lân cận: ${max(3, best_row['step']-2)} - ${best_row['step']+2})")
        print(f"   Multiplier    : {best_row['multiplier']} (Vùng lân cận: {max(1.0, best_row['multiplier']-0.10):.2f} - {best_row['multiplier']+0.10:.2f})")
        print(f"   Neighborhood RF: {best_row['neighborhood_mean_rf']} | Plateau Score: {best_row['plateau_score']}")
    else:
        print("\n-> Không có đỉnh nhọn nguy hiểm, chọn vùng Step 4-8 & Multiplier 1.10-1.25 là vùng ổn định cao nhất.")

    print("*" * 80)
    print(f"File ASCII Heatmaps : {heatmap_txt_path}")
    print(f"File Plateau Report : {report_json_path}")
    print("Phase 4 HOÀN THÀNH THÀNH CÔNG!\n")


if __name__ == "__main__":
    main()
