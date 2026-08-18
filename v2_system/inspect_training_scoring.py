"""
v2_system/inspect_training_scoring.py
======================================
Script In FULL 100% Tất Cả Các Ngày Train & Tất Cả 72 Kịch Bản Chiến Thuật (Độc Lập Từng Ngày).
Được thiết kế bởi Senior Quantitative Researcher.

Mục đích:
In full toàn bộ kết quả thử nghiệm 72 kịch bản grid trên nến M1 (10:00 - 12:00) cho TẤT CẢ các ngày tập Train (2020-2023).
Tự động lưu báo cáo đầy đủ ra file txt: `v2_system/full_training_scan_report.txt`.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from grid_simulator import GridSimulator

def inspect_full_scoring():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2_dir = os.path.join(base_dir, "v2_system")
    train_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-3]_m1.csv")))

    print("\n==================================================================")
    print(" 🔍 IN FULL TẤT CẢ CÁC NGÀY TRAIN & TẤT CẢ 72 KỊCH BẢN CHIẾN THUẬT (ĐỘC LẬP TỪNG NGÀY)")
    print("==================================================================")
    
    pipeline = DataPipeline(base_dir)
    train_feature_df, train_m1_dict = pipeline.extract_daily_dataset(train_files)

    simulator = GridSimulator()
    param_grid = GridSimulator.generate_parameter_grid()

    total_days = len(train_feature_df)
    print(f"\n[Info] Đã nạp FULL {total_days} ngày train. Số kịch bản quét mỗi ngày: {len(param_grid)} kịch bản.\n")

    report_path = os.path.join(v2_dir, "full_training_scan_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as out_f:
        out_f.write("=========================================================================================================\n")
        out_f.write(" 📊 BÁO CÁO FULL ALL DAYS & ALL 72 SCENARIOS - STRICT DAILY ISOLATION & ATR NORMALIZATION (2020-2023)\n")
        out_f.write("=========================================================================================================\n\n")

        for day_idx, date_str in enumerate(train_feature_df['date'], start=1):
            row = train_feature_df[train_feature_df['date'] == date_str].iloc[0]
            obs_df, exec_df = train_m1_dict[date_str]

            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']
            vwap_dist = row['vwap_dist_atr']
            bb_z = row['bb_zscore_m15']

            header_str = (
                f"\n---------------------------------------------------------------------------------------------------------\n"
                f" 📅 NGÀY {day_idx}/{total_days}: {date_str} (Khởi tạo Độc Lập 100%)\n"
                f"---------------------------------------------------------------------------------------------------------\n"
                f" • 09:59 AM Snapshot: Close=${close_0959:.2f} | VWAP=${daily_vwap:.2f} | ATR=${atr_14:.2f} | VWAP_Dist/ATR={vwap_dist:.2f} | BB_Z={bb_z:.2f}\n"
                f" • Hướng Giao Dịch  : {'SELL (Close >= VWAP)' if close_0959 >= daily_vwap else 'BUY (Close < VWAP)'}\n"
                f"---------------------------------------------------------------------------------------------------------\n"
                f" | Kịch Bản | Step_0 | Step_Exp | Max_Ord | Multiplier | TP_BE | Lệnh Khớp | Kết Quả 10:00-12:00 | PnL ($)     | DD ($)     | PnL (ATR) | Fitness Score |\n"
                f" +----------+--------+----------+---------+------------+-------+-----------+---------------------+-------------+------------+-----------+---------------+\n"
            )

            print(header_str, end="")
            out_f.write(header_str)

            day_scenarios = []
            for p_idx, p in enumerate(param_grid):
                res = simulator.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
                res['preset_idx'] = p_idx + 1
                res['params'] = p
                day_scenarios.append(res)

            # In FULL tất cả 72 kịch bản của ngày này
            for s in day_scenarios:
                p = s['params']
                outcome = f"Hit TP (m{s['hit_minute']})" if s['hit_tp'] else "Kẹt 12:00"
                row_str = f" | P_{s['preset_idx']:<6} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<7} | {p['multiplier']:<10.1f} | {p['tp_be_ratio']:<5.1f} | {s['num_orders']:<9} | {outcome:<19} | {s['net_profit']:<11,.2f} | {s['max_drawdown']:<10,.2f} | {s['pnl_atr']:<9.2f} | {s['fitness_score']:<13,.1f} |\n"
                print(row_str, end="")
                out_f.write(row_str)

            # Tìm kịch bản thắng của ngày
            day_scenarios.sort(key=lambda x: x['fitness_score'], reverse=True)
            winner = day_scenarios[0]
            wp = winner['params']

            footer_str = (
                f" +----------+--------+----------+---------+------------+-------+-----------+---------------------+-------------+------------+-----------+---------------+\n"
                f" 🏆 WINNER NGÀY {date_str}: P_{winner['preset_idx']} (Score = {winner['fitness_score']:.1f}) | Step_0={wp['step_0_ratio']}x | Exp={wp['step_exp']} | MaxOrd={int(wp['max_orders'])} | Mult={wp['multiplier']} | TP_BE={wp['tp_be_ratio']}x | PnL=+${winner['net_profit']:.2f} | PnL_ATR={winner['pnl_atr']:.2f} | DD=${winner['max_drawdown']:.2f}\n\n"
            )
            print(footer_str, end="")
            out_f.write(footer_str)
            sys.stdout.flush()

    print("\n==================================================================")
    print(f" ✨ CẢM ƠN BẠN! ĐÃ IN FULL TẤT CẢ CÁC NGÀY VÀ KỊCH BẢN (ĐỘC LẬP TỪNG NGÀY)!")
    print(f" 📂 File báo cáo đầy đủ đã được lưu tại: {report_path}")
    print("==================================================================")

if __name__ == '__main__':
    inspect_full_scoring()
