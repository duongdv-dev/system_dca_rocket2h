"""
v2_system/run_one_day_diagnostic.py
===================================
Script Mô Phỏng & In FULL Chấm Điểm 288 Chiến Thuật Tham Số (Full Strategy Scoring & Ranking).
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

def run_full_strategy_diagnostic(target_date: str = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_2024_m1.csv"))) + sorted(glob.glob(os.path.join(base_dir, "XAUUSD_2025_m1.csv")))

    if not test_files:
        test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_2023_m1.csv")))

    pipeline = DataPipeline(base_dir)
    feature_df, daily_m1_dict = pipeline.extract_daily_dataset(test_files)

    if feature_df.empty:
        print("❌ Không tìm thấy dữ liệu hợp lệ!")
        return

    # Nếu không chỉ định ngày, tự động tìm ngày ĐẦU TIÊN CÓ LỆNH KHỚP THỰC TẾ
    if target_date is None or target_date not in daily_m1_dict:
        simulator_temp = GridSimulator()
        found_date = None
        for d_str in feature_df['date']:
            obs_t, exec_t = daily_m1_dict[d_str]
            r_t = feature_df[feature_df['date'] == d_str].iloc[0]
            test_res = simulator_temp.simulate_day_scenario(exec_t, r_t['atr_14_m15'], r_t['close_0959'], r_t['daily_vwap'], {'step_0_ratio': 0.6, 'step_exp': 1.1, 'max_orders': 4, 'multiplier': 1.2})
            if test_res['num_orders'] > 0:
                found_date = d_str
                break
        target_date = found_date if found_date else feature_df['date'].iloc[0]

    row = feature_df[feature_df['date'] == target_date].iloc[0]
    obs_df, exec_df = daily_m1_dict[target_date]

    atr_14 = row['atr_14_m15']
    close_0959 = row['close_0959']
    daily_vwap = row['daily_vwap']
    vwap_dist_atr = row['vwap_dist_atr']
    bb_zscore = row['bb_zscore_m15']
    bb_slope = row['bb_slope_m15']
    morning_momentum = row['morning_momentum']
    price_1000 = exec_df['open'].iloc[0]

    direction = -1 if close_0959 >= daily_vwap else 1
    dir_str = "SELL (Kỳ vọng giảm về Open 10:00)" if direction == -1 else "BUY (Kỳ vọng tăng về Open 10:00)"

    min_low_10_12 = exec_df['low'].min()
    max_high_10_12 = exec_df['high'].max()

    print("\n=========================================================================================================")
    print(f" 🔍 QUÉT FULL TẤT CẢ 288 CHIẾN THUẬT & CHẤM ĐIỂM NGÀY: {target_date}")
    print("=========================================================================================================")
    print(f" 1. TỔNG QUAN CHỈ SỐ THỊ TRƯỜNG LÚC 09:59:59 AM:")
    print(f"    • Giá Open 10:00 AM (price_1000): ${price_1000:,.2f}")
    print(f"    • Biến động M1 (10:00 - 12:00):  Min Low = ${min_low_10_12:,.2f} | Max High = ${max_high_10_12:,.2f}")
    print(f"    • Giá Close 09:59 AM:            ${close_0959:,.2f}")
    print(f"    • Daily VWAP:                    ${daily_vwap:,.2f} (Lệch VWAP: {vwap_dist_atr:+.2f}x ATR)")
    print(f"    • ATR(14) M15:                   ${atr_14:,.2f}")
    print(f"    • BB Z-Score / Slope M15:        Z={bb_zscore:.2f} | Slope={bb_slope:.2f}")
    print(f"    • Động lượng phiên sáng:         Momentum = {morning_momentum:.2f}")
    print(f"    👉 Hướng Giao Dịch Quyết Định:   {dir_str}\n")

    simulator = GridSimulator()
    param_grid = GridSimulator.generate_parameter_grid()

    print(f" 2. ĐANG THỬ NGHIỆM TẤT CẢ {len(param_grid)} CHIẾN THUẬT THAM SỐ TRÊN NẾN M1 (10:00 - 12:00)...")

    strategy_results = []
    for p_idx, p in enumerate(param_grid, start=1):
        res = simulator.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
        res['preset_idx'] = p_idx
        res['params'] = p
        strategy_results.append(res)

    # Ưu tiên xếp hạng những chiến thuật CÓ KHỚP LỆNH và CÓ ĐIỂM CAO
    strategy_results.sort(key=lambda x: (x['num_orders'] > 0, x['fitness_score']), reverse=True)

    traded_strategies = [s for s in strategy_results if s['num_orders'] > 0]
    if not traded_strategies:
        min_step0_needed = abs(min_low_10_12 - price_1000) / atr_14 if direction == 1 else abs(max_high_10_12 - price_1000) / atr_14
        print(f"    ⚠️ GHI CHÚ ĐỊNH LƯỢNG: Ngày {target_date} Vàng chỉ biến động tối đa {min_step0_needed:.2f}x ATR từ Open 10:00.")
        print(f"    👉 Do Vàng không nảy giãn đủ khoảng cách Step_0 tối thiểu (0.6x ATR = ${0.6*atr_14:.2f}), hệ thống ĐỨNG NGOÀI (NO-TRADE) bảo vệ vốn 100%!\n")

    print("\n 3. BẢNG XẾP HẠNG TOP 20 CHIẾN THUẬT ĐẠT ĐIỂM FITNESS SCORE CAO NHẤT:")
    print(" -----------------------------------------------------------------------------------------------------------------")
    print(" | Hạng | Set ID | Step_0 | Step_Exp | Max_Ord | Multiplier | Lệnh Khớp | Kết Quả M1 | Net PnL ($) | Max DD ($) | Score  |")
    print(" +------+--------+--------+----------+---------+------------+-----------+------------+-------------+------------+--------+")

    for rank, s in enumerate(strategy_results[:20], start=1):
        p = s['params']
        outcome = f"Hit TP (m{s['hit_minute']})" if s['hit_tp'] else ("Chưa khớp Order 1" if s['num_orders']==0 else "Kẹt 12:00")
        print(f" | {rank:<4} | P_{s['preset_idx']:<4} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<7} | {p['multiplier']:<10.1f} | {s['num_orders']:<9} | {outcome:<10} | {s['net_profit']:<11,.2f} | {s['max_drawdown']:<10,.2f} | {s['fitness_score']:<6.1f} |")
    print(" -----------------------------------------------------------------------------------------------------------------")

    winner = strategy_results[0]
    wp = winner['params']
    print(f"\n 🏆 CHIẾN THUẬT CHIẾN THẮNG NGÀY {target_date}: Set P_{winner['preset_idx']}")
    print(f"    • Step_0: {wp['step_0_ratio']}x ATR | Step_Exp: {wp['step_exp']} | Max_Orders: {int(wp['max_orders'])} | Multiplier: {wp['multiplier']}x")
    print(f"    • Điểm Fitness Score: {winner['fitness_score']:.1f}")
    print(f"    • Lợi Nhuận Net PnL : +${winner['net_profit']:,.2f}")
    print(f"    • Max Drawdown      : ${winner['max_drawdown']:,.2f}")
    print(f"    • Số lệnh đã khớp   : {winner['num_orders']} lệnh\n")

    report_file = os.path.join(base_dir, "v2_system", f"full_strategy_scoring_{target_date}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=========================================================================================================\n")
        f.write(f" 📊 BÁO CÁO FULL CHẤM ĐIỂM TẤT CẢ {len(param_grid)} CHIẾN THUẬT - NGÀY {target_date}\n")
        f.write(f"=========================================================================================================\n\n")
        f.write(" | Hạng | Set ID | Step_0 | Step_Exp | Max_Ord | Multiplier | Lệnh Khớp | Kết Quả M1 | Net PnL ($) | Max DD ($) | Score  |\n")
        f.write(" +------+--------+--------+----------+---------+------------+-----------+------------+-------------+------------+--------+\n")
        for rank, s in enumerate(strategy_results, start=1):
            p = s['params']
            outcome = f"Hit TP (m{s['hit_minute']})" if s['hit_tp'] else ("Chưa khớp Order 1" if s['num_orders']==0 else "Kẹt 12:00")
            f.write(f" | {rank:<4} | P_{s['preset_idx']:<4} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<7} | {p['multiplier']:<10.1f} | {s['num_orders']:<9} | {outcome:<10} | {s['net_profit']:<11,.2f} | {s['max_drawdown']:<10,.2f} | {s['fitness_score']:<6.1f} |\n")

    print(f" 📂 Báo cáo FULL 288 chiến thuật đã được lưu tại: {report_file}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_strategy_diagnostic(target)
