"""
v2_system/inspect_training_scoring.py
======================================
Script In FULL 100% Tất Cả Các Ngày Train & Các Kịch Bản (Order 1 Chờ Giãn Step_0, TP Cố Định Open 10:00).
Được thiết kế bởi Senior Quantitative Researcher.
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
    print(" 🔍 IN FULL ALL DAYS (ORDER 1 CHỜ GIÃN STEP_0, TP CỐ ĐỊNH OPEN 10:00)")
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
        out_f.write(" 📊 BÁO CÁO FULL ALL DAYS & ALL SCENARIOS - ORDER 1 STRETCH STEP_0 & FIXED 10:00 OPEN TP\n")
        out_f.write("=========================================================================================================\n\n")

        for day_idx, date_str in enumerate(train_feature_df['date'], start=1):
            row = train_feature_df[train_feature_df['date'] == date_str].iloc[0]
            obs_df, exec_df = train_m1_dict[date_str]

            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']
            vwap_dist = row['vwap_dist_atr']
            bb_z = row['bb_zscore_m15']
            price_1000 = exec_df['open'].iloc[0]

            header_str = (
                f"\n---------------------------------------------------------------------------------------------------------\n"
                f" 📅 NGÀY {day_idx}/{total_days}: {date_str} (Open 10:00 = ${price_1000:.2f})\n"
                f"---------------------------------------------------------------------------------------------------------\n"
                f" • 09:59 AM Snapshot: Close=${close_0959:.2f} | VWAP=${daily_vwap:.2f} | ATR=${atr_14:.2f} | VWAP_Dist/ATR={vwap_dist:.2f} | BB_Z={bb_z:.2f}\n"
                f" • Hướng Giao Dịch  : {'SELL (Close >= VWAP)' if close_0959 >= daily_vwap else 'BUY (Close < VWAP)'}\n"
                f"---------------------------------------------------------------------------------------------------------\n"
                f" | Kịch Bản | Step_0 | Step_Exp | Max_Ord | Multiplier | TP Target | Lệnh Khớp | Kết Quả 10:00-12:00 | Net PnL ($) | Max DD ($) | Fitness Score |\n"
                f" +----------+--------+----------+---------+------------+-----------+-----------+---------------------+-------------+------------+---------------+\n"
            )

            print(header_str, end="")
            out_f.write(header_str)

            day_scenarios = []
            for p_idx, p in enumerate(param_grid):
                res = simulator.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
                res['preset_idx'] = p_idx + 1
                res['params'] = p
                day_scenarios.append(res)

            for s in day_scenarios:
                p = s['params']
                outcome = f"Hit TP (m{s['hit_minute']})" if s['hit_tp'] else ("Chưa khớp Order 1" if s['num_orders']==0 else "Kẹt 12:00")
                tp_target_str = f"${price_1000:.1f}"
                row_str = f" | P_{s['preset_idx']:<6} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<7} | {p['multiplier']:<10.1f} | {tp_target_str:<9} | {s['num_orders']:<9} | {outcome:<19} | {s['net_profit']:<11,.2f} | {s['max_drawdown']:<10,.2f} | {s['fitness_score']:<13,.1f} |\n"
                print(row_str, end="")
                out_f.write(row_str)

            day_scenarios.sort(key=lambda x: x['fitness_score'], reverse=True)
            winner = day_scenarios[0]
            wp = winner['params']

            footer_str = (
                f" +----------+--------+----------+---------+------------+-----------+-----------+---------------------+-------------+------------+---------------+\n"
                f" 🏆 WINNER NGÀY {date_str}: P_{winner['preset_idx']} (Score = {winner['fitness_score']:.1f}) | Step_0={wp['step_0_ratio']}x | Exp={wp['step_exp']} | MaxOrd={int(wp['max_orders'])} | Mult={wp['multiplier']} | TP Target=${price_1000:.2f} | PnL=+${winner['net_profit']:.2f} | DD=${winner['max_drawdown']:.2f}\n\n"
            )
            print(footer_str, end="")
            out_f.write(footer_str)
            sys.stdout.flush()

    print("\n==================================================================")
    print(f" ✨ HOÀN TẤT IN FULL BÁO CÁO (ORDER 1 CHỜ GIÃN STEP_0, TP CỐ ĐỊNH OPEN 10:00)!")
    print(f" 📂 File báo cáo đầy đủ đã được lưu tại: {report_path}")
    print("==================================================================")

if __name__ == '__main__':
    inspect_full_scoring()
