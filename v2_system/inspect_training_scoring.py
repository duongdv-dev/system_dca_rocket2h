"""
v2_system/inspect_training_scoring.py
======================================
Script Chuyên Biệt Kiểm Tra & Minh Họa Quy Trình Quét Kịch Bản & Chấm Điểm Training.
Được thiết kế bởi Senior Quantitative Researcher.

Mục đích:
Giúp người dùng nhìn rõ từng bước:
1. Đọc dữ liệu nến M1 và trích xuất chỉ số 09:59.
2. Thử nghiệm từng kịch bản tham số trên nến M1 (10:00 - 12:00).
3. Công thức tính Fitness Score cho từng kịch bản.
4. Lựa chọn bộ tham số xuất sắc nhất (Winning Preset) cho ngày hôm đó.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from grid_simulator import GridSimulator

def inspect_scoring():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-3]_m1.csv")))

    print("\n==================================================================")
    print(" 🔍 BÁO CÁO CHI TIẾT QUY TRÌNH QUÉT KỊCH BẢN & CHẤM ĐIỂM TRAINING")
    print("==================================================================")
    
    pipeline = DataPipeline(base_dir)
    train_feature_df, train_m1_dict = pipeline.extract_daily_dataset(train_files)

    simulator = GridSimulator()
    param_grid = GridSimulator.generate_parameter_grid()

    print(f"\n[Info] Đã nạp {len(train_feature_df)} ngày train. Tổng số kịch bản quét mỗi ngày: {len(param_grid)} kịch bản.\n")

    # Chọn 3 ngày tiêu biểu để in log chi tiết
    sample_dates = train_feature_df['date'].iloc[:3].tolist()

    for day_idx, date_str in enumerate(sample_dates, start=1):
        row = train_feature_df[train_feature_df['date'] == date_str].iloc[0]
        obs_df, exec_df = train_m1_dict[date_str]

        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        m_range = row['morning_range_atr'] * atr_14
        m_momentum = row['morning_momentum']

        print("---------------------------------------------------------------------------------------------------------")
        print(f" 📅 NGÀY TRAIN VÍ DỤ {day_idx}: {date_str}")
        print("---------------------------------------------------------------------------------------------------------")
        print(f" • Chỉ số 09:59 AM : Close_0959 = ${close_0959:.2f} | Daily VWAP = ${daily_vwap:.2f} | ATR M15 = ${atr_14:.2f}")
        print(f" • Biên độ sáng    : Range = ${m_range:.2f} | Momentum = {m_momentum:.2f} | VwAP Dist/ATR = {row['vwap_dist_atr']:.2f}")
        print(f" • Hướng Giao Dịch : {'SELL (Giá 09:59 > VWAP)' if close_0959 >= daily_vwap else 'BUY (Giá 09:59 < VWAP)'}")
        print("---------------------------------------------------------------------------------------------------------")
        print(" | Kịch Bản | Step_0 | Step_Exp | Max_Ord | Multiplier | TP_BE | Lệnh Khớp | Kết Quả 10:00-12:00 | Net PnL ($) | Max DD ($) | Fitness Score |")
        print(" +----------+--------+----------+---------+------------+-------+-----------+---------------------+-------------+------------+---------------+")

        day_scenarios = []

        for p_idx, p in enumerate(param_grid):
            res = simulator.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
            res['preset_idx'] = p_idx + 1
            res['params'] = p
            day_scenarios.append(res)

        # In 8 kịch bản tiêu biểu của ngày này
        for s in day_scenarios[:8]:
            p = s['params']
            outcome = f"Hit TP (Phút {s['hit_minute']})" if s['hit_tp'] else "Kẹt Lệnh 12:00"
            print(f" | P_{s['preset_idx']:<6} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<7} | {p['multiplier']:<10.1f} | {p['tp_be_ratio']:<5.1f} | {s['num_orders']:<9} | {outcome:<19} | {s['net_profit']:<11,.2f} | {s['max_drawdown']:<10,.2f} | {s['fitness_score']:<13,.1f} |")

        # Tìm kịch bản chiến thắng của ngày
        day_scenarios.sort(key=lambda x: x['fitness_score'], reverse=True)
        winner = day_scenarios[0]
        wp = winner['params']

        print(" +----------+--------+----------+---------+------------+-------+-----------+---------------------+-------------+------------+---------------+")
        print(f" 🏆 KỊCH BẢN TỐI ƯU NHẤT NGÀY {date_str}: Kịch Bản P_{winner['preset_idx']} (Score = {winner['fitness_score']:.1f})")
        print(f"     Tham số: Step_0 = {wp['step_0_ratio']}x ATR | Step_Exp = {wp['step_exp']} | Max_Orders = {int(wp['max_orders'])} | Multiplier = {wp['multiplier']} | TP_BE = {wp['tp_be_ratio']}x ATR")
        print(f"     Kết quả: PnL = +${winner['net_profit']:.2f} | Max DD = ${winner['max_drawdown']:.2f} | Cắn TP: {winner['hit_tp']}\n")

    print("==================================================================")
    print(" ✨ HOÀN TẤT MINH HỌA QUY TRÌNH QUÉT & CHẤM ĐIỂM TRAINING!")
    print("==================================================================")

if __name__ == '__main__':
    inspect_scoring()
