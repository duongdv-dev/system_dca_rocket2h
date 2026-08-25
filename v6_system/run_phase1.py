"""
v6_system/run_phase1.py
-----------------------
Runner script cho Version 6 Phase 1 Data Engineering.
Thực thi toàn bộ pipeline: Nạp -> Kiểm định -> Làm sạch -> Tạo Session 10:00-12:00 -> Export.
"""

import sys
import os
import json

# Đảm bảo import được v6_system từ root workspace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 70)
    print("      VERSION 6 - PHASE 1: DATA ENGINEERING PIPELINE (2020 - 2025)")
    print("=" * 70)

    engineer = V6DataEngineer()

    # 1. Nạp dữ liệu thô 2020-2025
    print("\n[Bước 1/4] Nạp dữ liệu Dukascopy M1 (2020 - 2025)...")
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)

    # 2. Kiểm định & Làm sạch
    print("\n[Bước 2/4] Kiểm định Data Engineering (Timeframe, Timezone/DST, OHLC, Volume, Duplicates, Gaps)...")
    df_clean, report = engineer.audit_and_clean_data(df_raw)

    # 3. Tạo phiên 10:00 -> 12:00 VN
    print("\n[Bước 3/4] Trích xuất & Tạo Dataset Session (10:00 -> 12:00 VN)...")
    df_daily = engineer.extract_daily_sessions(df_clean)

    # 4. Xuất kết quả
    print("\n[Bước 4/4] Xuất kết quả và báo cáo chất lượng...")
    engineer.export_results(df_clean, df_daily, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("                  BÁO CÁO TÓM TẮT PHÁP LÝ KỸ THUẬT (PHASE 1)")
    print("=" * 70)
    print(f"Tổng số nến M1 thô ban đầu       : {report['total_raw_rows']:,}")
    print(f"Bản ghi trùng lặp đã xóa (Dup)   : {report['duplicates_removed']:,}")
    print(f"Bản ghi lỗi OHLC/giá đã xóa      : {report['invalid_ohlc_removed']:,}")
    print(f"Bản ghi Volume âm / bằng 0       : {report['invalid_volume_count']} âm / {report['zero_volume_count']:,} bằng 0")
    print(f"Khoảng mất nến Intraday          : {report['missing_candles_intervals']} khoảng ({report['missing_candles_count']:,} nến M1)")
    print(f"Gap giá bất thường (> $5.0)      : {report['abnormal_gaps_count']}")
    print(f"Monotonic Timestamp Order        : {report['timestamp_is_monotonic']}")
    print(f"Múi giờ chuẩn hóa                : {report['target_timezone']}")
    print(f"Tổng nến M1 sạch                 : {report['final_clean_rows']:,}")
    print(f"Tổng số ngày phiên (10:00-12:00) : {len(df_daily):,} ngày")
    print("=" * 70)
    print(f"File session output: {os.path.join(OUTPUT_DIR, 'daily_sessions_10_12.csv')}")
    print(f"File quality report: {os.path.join(OUTPUT_DIR, 'data_quality_report.json')}")
    print("Phase 1 HOÀN THÀNH THÀNH CÔNG!\n")


if __name__ == "__main__":
    main()
