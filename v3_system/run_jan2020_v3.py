"""
v3_system/run_jan2020_v3.py
============================
Script In Trực Tiếp Bảng Ma Trận Diễn Biến PnL 22 Ngày Của Các Set Tham Số Cho User Xem Trên Console.
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
In trực tiếp ra màn hình Console diễn biến PnL của từng ngày (Jan 02 -> Jan 31) cho các Set tham số
để người dùng đối chiếu sự biến động của cùng 1 Set qua cả tháng mà không cần mở file CSV.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v3_preset_generator import V3PresetGenerator
from v3_regime_classifier import V3RegimeClassifier

def run_preset_matrix_jan2020():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")

    print("\n=========================================================================================================")
    print(" 🚀 IN TRỰC TIẾP MA TRẬN PNL 22 NGÀY CỦA CÁC SET THAM SỐ THÁNG 1/2020 LÊN CONSOLE")
    print("=========================================================================================================")

    pipeline = V3DataPipeline(base_dir)
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    feature_df, daily_m1_dict = pipeline.compute_daily_features(df_raw, target_month_str="2020-01")

    total_days = len(feature_df)
    trading_dates = feature_df['date'].tolist()

    classifier = V3RegimeClassifier()
    feature_df = classifier.label_dataset_regimes(feature_df)

    generator = V3PresetGenerator()
    presets_list = V3PresetGenerator.generate_540_candidate_presets()

    preset_matrix_results = []

    for p_idx, p in enumerate(presets_list, start=1):
        daily_pnl_dict = {}
        daily_orders_dict = {}
        
        initial_balance = 10000.0
        curr_balance = initial_balance
        equity_curve = [curr_balance]
        
        win_days = 0
        loss_days = 0
        no_trade_days = 0

        for date_str in trading_dates:
            obs_df, exec_df = daily_m1_dict[date_str]
            row = feature_df[feature_df['date'] == date_str].iloc[0]

            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']

            res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, p)
            day_pnl = res['net_profit']
            num_orders = res['num_orders']

            daily_pnl_dict[date_str] = day_pnl
            daily_orders_dict[date_str] = num_orders

            curr_balance += day_pnl
            equity_curve.append(curr_balance)

            if num_orders == 0:
                no_trade_days += 1
            elif day_pnl > 0:
                win_days += 1
            else:
                loss_days += 1

        traded_days = win_days + loss_days
        win_rate = (win_days / traded_days * 100.0) if traded_days > 0 else 0.0
        
        gross_profit = sum(v for v in daily_pnl_dict.values() if v > 0)
        gross_loss = abs(sum(v for v in daily_pnl_dict.values() if v < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd_arr = (eq_arr - peak) / peak
        max_dd_pct = abs(np.min(dd_arr)) * 100.0

        net_return_pct = ((curr_balance - initial_balance) / initial_balance) * 100.0

        rec = {
            'preset_id': p_idx,
            'step_0_ratio': p['step_0_ratio'],
            'step_exp': p['step_exp'],
            'max_orders': int(p['max_orders']),
            'multiplier': p['multiplier'],
            'total_pnl': curr_balance - initial_balance,
            'net_return_pct': net_return_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_dd_pct': max_dd_pct,
            'win_days': win_days,
            'loss_days': loss_days,
            'no_trade_days': no_trade_days,
            'daily_pnl': daily_pnl_dict
        }

        preset_matrix_results.append(rec)

    # Sắp xếp các Presets theo Lợi Nhuận Ròng Tháng 1/2020 giảm dần
    preset_matrix_results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # 1. BẢNG TỔNG HỢP XẾP HẠNG TOP 10 PRESETS THÁNG 1/2020
    print("\n=========================================================================================================")
    print(" 🏆 TOP 10 SETS THAM SỐ CÓ TỔNG LỢI NHUẬN CAO NHẤT THÁNG 1/2020")
    print("=========================================================================================================")
    print(" | Hạng | Set ID | Step_0 | Exp  | MaxOrd | Mult  | PnL Tháng 1 ($) | Return (%) | Win Rate (%) | Max DD (%) | PF   | Win/Loss |")
    print(" +------+--------+--------+------+--------+-------+-----------------+------------+--------------+------------+------+----------+")

    for rank, r in enumerate(preset_matrix_results[:10], start=1):
        print(f" | {rank:<4} | P_{r['preset_id']:<4} | {r['step_0_ratio']:<6.1f} | {r['step_exp']:<4.2f} | {r['max_orders']:<6} | {r['multiplier']:<5.1f} | {r['total_pnl']:<15,.2f} | +{r['net_return_pct']:<10.2f} | {r['win_rate']:<12.1f} | {r['max_dd_pct']:<10.2f} | {r['profit_factor']:<4.2f} | {r['win_days']}W / {r['loss_days']}L |")
    print("=========================================================================================================\n")

    # 2. IN TRỰC TIẾP MA TRẬN PNL TỪNG NGÀY CHO TOP 10 PRESETS LÊN CONSOLE
    # Chia 22 ngày thành 2 nửa (11 ngày đầu & 11 ngày sau) để hiển thị bảng vuông vức đẹp mắt trên terminal
    dates_part1 = trading_dates[:11]
    dates_part2 = trading_dates[11:]

    print("=========================================================================================================")
    print(" 📊 MA TRẬN PNL P1 (11 NGÀY ĐẦU THÁNG 1/2020: 02/01 -> 16/01)")
    print("=========================================================================================================")
    
    header_p1 = " | Set ID | " + " | ".join([d[5:] for d in dates_part1]) + " |"
    sep_p1 = " +--------+" + "+".join(["--------" for _ in dates_part1]) + "+"
    print(header_p1)
    print(sep_p1)

    for r in preset_matrix_results[:10]:
        row_str = f" | P_{r['preset_id']:<4} | "
        pnl_vals = []
        for d in dates_part1:
            val = r['daily_pnl'][d]
            if val > 0:
                pnl_vals.append(f"+${val:<5.0f}")
            elif val < 0:
                pnl_vals.append(f"-${abs(val):<5.0f}")
            else:
                pnl_vals.append(f"  $0   ")
        row_str += " | ".join(pnl_vals) + " |"
        print(row_str)
    print(sep_p1 + "\n")

    print("=========================================================================================================")
    print(" 📊 MA TRẬN PNL P2 (11 NGÀY CUỐI THÁNG 1/2020: 17/01 -> 31/01)")
    print("=========================================================================================================")
    
    header_p2 = " | Set ID | " + " | ".join([d[5:] for d in dates_part2]) + " |"
    sep_p2 = " +--------+" + "+".join(["--------" for _ in dates_part2]) + "+"
    print(header_p2)
    print(sep_p2)

    for r in preset_matrix_results[:10]:
        row_str = f" | P_{r['preset_id']:<4} | "
        pnl_vals = []
        for d in dates_part2:
            val = r['daily_pnl'][d]
            if val > 0:
                pnl_vals.append(f"+${val:<5.0f}")
            elif val < 0:
                pnl_vals.append(f"-${abs(val):<5.0f}")
            else:
                pnl_vals.append(f"  $0   ")
        row_str += " | ".join(pnl_vals) + " |"
        print(row_str)
    print(sep_p2 + "\n")

    # 3. IN CHI TIẾT THEO NGÀY CHO SET TOP 1, TOP 2 VÀ TOP 3
    print("=========================================================================================================")
    print(" 🔍 SOI ĐỐI CHIẾU TIẾN TRÌNH TỪNG NGÀY CỦA TOP 3 SETS THÁNG 1/2020")
    print("=========================================================================================================")
    print(" | STT | Ngày VN    | Nhóm Xu Hướng (Regime)    | PnL Top 1 (P_" + str(preset_matrix_results[0]['preset_id']) + ")  | PnL Top 2 (P_" + str(preset_matrix_results[1]['preset_id']) + ")  | PnL Top 3 (P_" + str(preset_matrix_results[2]['preset_id']) + ")  |")
    print(" +-----+------------+---------------------------+----------------+----------------+----------------+")

    for idx, date_str in enumerate(trading_dates, start=1):
        row = feature_df[feature_df['date'] == date_str].iloc[0]
        regime_name = row['regime_name']

        v1 = preset_matrix_results[0]['daily_pnl'][date_str]
        v2 = preset_matrix_results[1]['daily_pnl'][date_str]
        v3 = preset_matrix_results[2]['daily_pnl'][date_str]

        s1 = f"+${v1:,.2f}" if v1 > 0 else (f"-${abs(v1):,.2f}" if v1 < 0 else "$0.00")
        s2 = f"+${v2:,.2f}" if v2 > 0 else (f"-${abs(v2):,.2f}" if v2 < 0 else "$0.00")
        s3 = f"+${v3:,.2f}" if v3 > 0 else (f"-${abs(v3):,.2f}" if v3 < 0 else "$0.00")

        print(f" | {idx:<3} | {date_str:<10} | {regime_name:<25} | {s1:<14} | {s2:<14} | {s3:<14} |")

    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_preset_matrix_jan2020()
