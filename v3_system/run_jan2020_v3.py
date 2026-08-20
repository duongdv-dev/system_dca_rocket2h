"""
v3_system/run_jan2020_v3.py
============================
Script Độc Lập Chạy So Sánh & Trích Xuất Set Tối Ưu Nhất Dựa Trên Tín Hiệu R Trực Tiếp (Tháng 1/2020).
Được thiết kế bởi Senior Quantitative Researcher.

Quy trình V3 Mới Theo Yêu Cầu Của User:
1. KHÔNG PHÂN LOẠI NHÓM (Bỏ hoàn toàn việc chia nhóm).
2. Trích xuất Tín Hiệu Thị Trường Liên Tục Chuẩn Hóa Theo R lúc 09:59 AM:
   - delta_open_0600_1000_r : Độ lệch Open 06:00 -> Open 10:00 theo R (ATR units).
   - delta_vwap_r           : Độ lệch Close 09:59 -> Daily VWAP theo R (ATR units).
   - range_morning_r        : Biên độ sóng sáng (06:00 - 09:59) theo R.
   - bb_zscore_m15 & bb_slope_m15.
3. Chạy Backtest so sánh trực tiếp các Candidate Presets trên nến M1 từng ngày.
4. Trích xuất Set có Điểm Fitness Score cao nhất & phù hợp nhất cho từng ngày.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v3_preset_generator import V3PresetGenerator

def run_january_2020_continuous_r_signals():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")

    print("\n=========================================================================================================")
    print(" 🚀 HỆ THỐNG V3: SO SÁNH & TRÍCH XUẤT SET TỐI ƯU DỰA TRÊN TÍN HIỆU R TRỰC TIẾP (THÁNG 1/2020)")
    print("=========================================================================================================")
    print(f" • Dữ liệu: File {os.path.basename(csv_2020)}")
    print(f" • Phương Pháp: Tín Hiệu Liên Tục R-Ratio (KHÔNG CHIA NHÓM)\n")

    pipeline = V3DataPipeline(base_dir)
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    feature_df, daily_m1_dict = pipeline.compute_daily_features(df_raw, target_month_str="2020-01")

    total_days = len(feature_df)
    trading_dates = feature_df['date'].tolist()
    print(f" -> Đã trích xuất {total_days} ngày giao dịch trong Tháng 1/2020.\n")

    generator = V3PresetGenerator()
    presets_list = V3PresetGenerator.generate_540_candidate_presets()

    daily_winners = []

    print("=========================================================================================================")
    print(" 🔍 BẢNG SO SÁNH & TRÍCH XUẤT SET ĐẠT ĐIỂM R CAO NHẤT TỪNG NGÀY THÁNG 1/2020")
    print("=========================================================================================================")
    print(" | STT | Ngày VN    | ΔOpen (R) | ΔVWAP (R) | Range (R) | Hướng | Best Set ID | Step_0 | Exp  | MaxOrd | Mult  | Net PnL ($) | Score (R) |")
    print(" +-----+------------+-----------+-----------+-----------+-------+-------------+--------+------+--------+-------+-------------+-----------+")

    for idx, date_str in enumerate(trading_dates, start=1):
        row = feature_df[feature_df['date'] == date_str].iloc[0]
        obs_df, exec_df = daily_m1_dict[date_str]

        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        price_1000 = exec_df['open'].iloc[0]

        delta_open_r = row['delta_open_0600_1000_r']
        delta_vwap_r = row['delta_vwap_r']
        range_morning_r = row['range_morning_r']

        direction_str = "SELL" if close_0959 >= daily_vwap else "BUY"

        best_score = -float('inf')
        best_p_idx = -1
        best_param = None
        best_res = None

        # So sánh trực tiếp tất cả các kịch bản tham số cho nến M1 ngày hôm nay
        for p_idx, p in enumerate(presets_list, start=1):
            res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, p)
            if res['fitness_score'] > best_score:
                best_score = res['fitness_score']
                best_p_idx = p_idx
                best_param = p
                best_res = res

        has_traded = (best_res is not None and best_res['num_orders'] > 0 and best_score > 0.0)

        if has_traded:
            p_str = f"P_{best_p_idx}"
            s0_str = f"{best_param['step_0_ratio']:.1f}x"
            exp_str = f"{best_param['step_exp']:.2f}"
            mo_str = f"{int(best_param['max_orders'])}"
            mult_str = f"{best_param['multiplier']:.1f}x"
            pnl_str = f"+${best_res['net_profit']:,.2f}"
            score_str = f"{best_score:.1f}"
        else:
            p_str = "No-Trade"
            s0_str = "-"
            exp_str = "-"
            mo_str = "-"
            mult_str = "-"
            pnl_str = "$0.00"
            score_str = "0.0"

        print(f" | {idx:<3} | {date_str:<10} | {delta_open_r:<+9.2f} | {delta_vwap_r:<+9.2f} | {range_morning_r:<9.2f} | {direction_str:<5} | {p_str:<11} | {s0_str:<6} | {exp_str:<4} | {mo_str:<6} | {mult_str:<5} | {pnl_str:<11} | {score_str:<9} |")

        daily_winners.append({
            'date': date_str,
            'delta_open_0600_1000_r': delta_open_r,
            'delta_vwap_r': delta_vwap_r,
            'range_morning_r': range_morning_r,
            'morning_momentum': row['morning_momentum'],
            'bb_zscore_m15': row['bb_zscore_m15'],
            'bb_slope_m15': row['bb_slope_m15'],
            'direction': direction_str,
            'best_preset_id': best_p_idx if has_traded else 0,
            'step_0_ratio': best_param['step_0_ratio'] if has_traded else np.nan,
            'step_exp': best_param['step_exp'] if has_traded else np.nan,
            'max_orders': best_param['max_orders'] if has_traded else np.nan,
            'multiplier': best_param['multiplier'] if has_traded else np.nan,
            'net_profit': best_res['net_profit'] if has_traded else 0.0,
            'fitness_score': best_score if has_traded else 0.0
        })

    print("=========================================================================================================\n")

    summary_df = pd.DataFrame(daily_winners)
    traded_days_count = (summary_df['best_preset_id'] > 0).sum()
    total_pnl = summary_df['net_profit'].sum()

    print(" 📈 TỔNG HỢP KẾT QUẢ ĐỐI CHIẾU TRỰC TIẾP THÁNG 1/2020:")
    print(f"   • Số ngày trích xuất được Set thắng (>0 Score): {traded_days_count} / {total_days} ngày")
    print(f"   • Tổng Lợi Nhuận Net PnL Tháng 1/2020:          +${total_pnl:,.2f}\n")

    csv_path = os.path.join(v3_dir, "jan2020_r_signals_daily_winners.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f" 📂 Đã lưu báo cáo CSV Tín Hiệu R & Sets Tối Ưu Từng Ngày tại: {csv_path}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_january_2020_continuous_r_signals()
