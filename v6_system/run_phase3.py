"""
v6_system/run_phase3.py
-----------------------
Runner script thực thi Phase 3 - DCA Optimization.
Thực hiện Grid Search trên 448 tổ hợp tham số (Step, Max DCA, Multiplier)
và đánh giá Scorecard 10 chỉ số đo lường rủi ro & hiệu năng.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.optimizer_v3 import OptimizerV3
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 80)
    print("        VERSION 6 - PHASE 3: DCA OPTIMIZATION GRID SEARCH (2020 - 2025)")
    print("=" * 80)

    # 1. Nạp và làm sạch dữ liệu Phase 1
    print("\n[Bước 1/3] Nạp & Chuẩn hóa dữ liệu M1 2020-2025...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    # 2. Định nghĩa không gian tham số yêu cầu
    steps = [3, 4, 5, 6, 8, 10, 12, 15]
    max_dcas = [2, 3, 4, 5, 6, 7, 8]
    multipliers = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]

    total_combos = len(steps) * len(max_dcas) * len(multipliers)
    print(f"\n[Bước 2/3] Chạy Grid Search cho {total_combos} tổ hợp tham số...")
    print(f"  - DCA Step   : {steps}")
    print(f"  - Max DCA    : {max_dcas}")
    print(f"  - Multiplier : {multipliers}")

    optimizer = OptimizerV3(initial_capital=1000.0)
    df_grid, top_combos = optimizer.run_grid_search(df_clean, steps, max_dcas, multipliers)

    # 3. Xuất kết quả
    print("\n[Bước 3/3] Xuất báo cáo ma trận tối ưu hóa...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    matrix_csv_path = os.path.join(OUTPUT_DIR, "phase3_optimization_matrix.csv")
    df_grid.to_csv(matrix_csv_path, index=False)

    top_json_path = os.path.join(OUTPUT_DIR, "phase3_top_parameters.json")
    with open(top_json_path, "w", encoding="utf-8") as f:
        json.dump(top_combos, f, ensure_ascii=False, indent=2)

    # In bảng xếp hạng Top 10 bộ tham số bền vững nhất
    print("\n" + "=" * 110)
    print("                  TOP 10 BỘ THAM SỐ TỐI ƯU CÂN BẰNG RỦI RO & LỢI NHUẬN (SCORECARD 10 CHỈ SỐ)")
    print("=" * 110)
    
    hdr = f"{'Step':<5} | {'DCA':<4} | {'Mult':<5} | {'NetPnL':<9} | {'PF':<6} | {'MaxDD($)':<9} | {'RecFac':<7} | {'WinRate':<8} | {'AvgTrd':<7} | {'WstMonth':<9} | {'MaxLot':<7} | {'FC%':<6}"
    print(hdr)
    print("-" * 110)

    for item in top_combos:
        st = f"${item['step']}"
        dca = str(item['max_dca'])
        mult = f"{item['multiplier']:.2f}"
        pnl = f"${item['net_profit']:.1f}"
        pf = str(item['profit_factor'])
        mdd = f"${item['max_dd_dollars']:.1f}"
        rf = str(item['recovery_factor'])
        wr = f"{item['win_rate']}%"
        avg = f"${item['avg_trade_pnl']:.1f}"
        wmn = f"${item['worst_month_pnl']:.1f}"
        mlot = f"{item['max_exposure_lots']:.2f}"
        fc = f"{item['force_close_pct']:.1f}%"

        print(f"{st:<5} | {dca:<4} | {mult:<5} | {pnl:<9} | {pf:<6} | {mdd:<9} | {rf:<7} | {wr:<8} | {avg:<7} | {wmn:<9} | {mlot:<7} | {fc:<6}")

    print("=" * 110)
    print(f"File ma trận 448 kết quả: {matrix_csv_path}")
    print(f"File Top 10 tham số JSON : {top_json_path}")
    print("Phase 3 HOÀN THÀNH THÀNH CÔNG!\n")


if __name__ == "__main__":
    main()
