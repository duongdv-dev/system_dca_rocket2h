"""
v3_system/run_h1_2020_v3.py
============================
Script Chạy Kiểm Định 6 Tháng Đầu Năm 2020 (2020-01 -> 2020-06) Theo Tín Hiệu R Trực Tiếp.
Được thiết kế bởi Senior Quantitative Researcher.

Quy trình V3 6 Tháng (2020-H1):
1. Nạp dữ liệu 6 tháng đầu năm 2020 từ `XAUUSD_2020_m1.csv`.
2. Trích xuất Tín Hiệu R Liên Tục 09:59 AM cho toàn bộ ~125 ngày giao dịch.
3. So sánh 540 Candidate Presets & Trích xuất Set điểm R cao nhất cho từng ngày.
4. Tổng hợp Báo Cáo Hiệu Năng 6 Tháng, Bảng Thống Kê Theo Tháng & Báo Cáo Tổng Hợp H1 2020.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v3_preset_generator import V3PresetGenerator

def run_h1_2020_v3_evaluation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")
    target_h1_months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]

    print("\n=========================================================================================================")
    print(" 🚀 HỆ THỐNG V3: KIỂM ĐỊNH 6 THÁNG ĐẦU NĂM 2020 (2020-01 -> 2020-06)")
    print("=========================================================================================================")
    print(f" • File dữ liệu: {os.path.basename(csv_2020)}")
    print(f" • Phạm vi kiểm tra: 6 Tháng Đầu Năm 2020 ({', '.join(target_h1_months)})\n")

    pipeline = V3DataPipeline(base_dir)
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    feature_df, daily_m1_dict = pipeline.compute_daily_features(df_raw, target_months=target_h1_months)

    total_days = len(feature_df)
    trading_dates = feature_df['date'].tolist()
    print(f" -> Đã trích xuất thành công: {total_days} ngày giao dịch trong 6 Tháng Đầu 2020.\n")

    generator = V3PresetGenerator()
    presets_list = V3PresetGenerator.generate_540_candidate_presets()

    daily_winners = []

    print("=========================================================================================================")
    print(" 🔍 BẢNG ĐỐI CHIẾU THAM SỐ DYNAMIC STEP & MARTINGALE TỪNG NGÀY 6 THÁNG H1 2020")
    print("=========================================================================================================")
    print(" | STT | Ngày VN    | Hướng | Best Set ID | Dynamic Step_0 | Step Exp | Max Orders | Multiplier | Net Profit ($) | Score (R) |")
    print(" +-----+------------+-------+-------------+----------------+----------+------------+------------+----------------+-----------+")

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
        bb_slope_m15 = row['bb_slope_m15']

        direction_str = "SELL" if close_0959 >= daily_vwap else "BUY"

        best_score = -float('inf')
        best_p_idx = -1
        best_param = None
        best_res = None

        # So sánh 540 Presets trên nến M1 ngày hôm nay
        for p_idx, p in enumerate(presets_list, start=1):
            res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, p, bb_slope_m15=bb_slope_m15)
            if res['fitness_score'] > best_score:
                best_score = res['fitness_score']
                best_p_idx = p_idx
                best_param = p
                best_res = res

        has_traded = (best_res is not None and best_res['num_orders'] > 0 and best_score > 0.0)

        if has_traded:
            p_str = f"Set_{best_p_idx}"
            s0_val = best_param['step_0_ratio'] * atr_14
            s0_str = f"{s0_val:.1f}$ ({best_param['step_0_ratio']:.1f}x)"
            exp_str = f"{best_param['step_exp']:.2f}"
            mo_str = f"{int(best_param['max_orders'])}"
            mult_str = f"{best_param['multiplier']:.2f}x"
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

        print(f" | {idx:<3} | {date_str:<10} | {direction_str:<5} | {p_str:<11} | {s0_str:<14} | {exp_str:<8} | {mo_str:<10} | {mult_str:<10} | {pnl_str:<14} | {score_str:<9} |")

        daily_winners.append({
            'date': date_str,
            'month': row['month'],
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
            'max_drawdown': best_res['max_drawdown'] if has_traded else 0.0,
            'fitness_score': best_score if has_traded else 0.0
        })

    summary_df = pd.DataFrame(daily_winners)

    # 1. BẢNG TỔNG HỢP HIỆU NĂNG TỪNG THÁNG TRONG 6 THÁNG ĐẦU NĂM 2020
    monthly_report = []
    print("\n=========================================================================================================")
    print(" 📊 BẢNG TỔNG HỢP HIỆU NĂNG TỪNG THÁNG TRONG 6 THÁNG ĐẦU NĂM 2020 (2020-01 -> 2020-06)")
    print("=========================================================================================================")
    print(" | Tháng   | Tổng Ngày | Ngày Có Lệnh | Ngày Thắng | Lợi Nhuận Net ($) | Win Rate (%) | Max Drawdown ($) |")
    print(" +---------+-----------+--------------+------------+-------------------+--------------+------------------+")

    for m in target_h1_months:
        sub_m = summary_df[summary_df['month'] == m]
        m_days = len(sub_m)
        traded_sub = sub_m[sub_m['best_preset_id'] > 0]
        traded_cnt = len(traded_sub)
        win_cnt = (traded_sub['net_profit'] > 0).sum()
        m_pnl = sub_m['net_profit'].sum()
        m_winrate = (win_cnt / traded_cnt * 100.0) if traded_cnt > 0 else 0.0
        m_max_dd = sub_m['max_drawdown'].max()

        print(f" | {m:<7} | {m_days:<9} | {traded_cnt:<12} | {win_cnt:<10} | +${m_pnl:<16,.2f} | {m_winrate:<12.1f} | ${m_max_dd:<16,.2f} |")

        monthly_report.append({
            'month': m,
            'total_days': m_days,
            'traded_days': traded_cnt,
            'win_days': win_cnt,
            'net_profit': m_pnl,
            'win_rate': m_winrate,
            'max_drawdown': m_max_dd
        })

    print("=========================================================================================================\n")

    # 2. TỔNG KẾT TOÀN BỘ 6 THÁNG (2020-H1)
    total_traded = (summary_df['best_preset_id'] > 0).sum()
    total_win = (summary_df['net_profit'] > 0).sum()
    overall_pnl = summary_df['net_profit'].sum()
    overall_winrate = (total_win / total_traded * 100.0) if total_traded > 0 else 0.0
    overall_max_dd = summary_df['max_drawdown'].max()

    print(" 🏆 TỔNG KẾT HIỆU NĂNG 6 THÁNG ĐẦU NĂM 2020 (2020-H1):")
    print(f"   • Tổng số ngày giao dịch:             {total_days} ngày")
    print(f"   • Số ngày trích xuất được Set thắng : {total_traded} ngày ({total_traded/total_days*100:.1f}%)")
    print(f"   • Win Rate 6 Tháng H1:               {overall_winrate:.1f}% ({total_win}W / {total_traded-total_win}L)")
    print(f"   • TỔNG LỢI NHUẬN NET 6 THÁNG:        +${overall_pnl:,.2f}")
    print(f"   • Max Drawdown Lớn Nhất:              ${overall_max_dd:,.2f}\n")

    # Xuất file CSV báo cáo
    daily_csv_path = os.path.join(v3_dir, "h1_2020_r_signals_daily_winners.csv")
    monthly_csv_path = os.path.join(v3_dir, "h1_2020_monthly_summary.csv")

    summary_df.to_csv(daily_csv_path, index=False)
    pd.DataFrame(monthly_report).to_csv(monthly_csv_path, index=False)

    print(f" 📂 Đã lưu báo cáo ngày 6 tháng tại: {daily_csv_path}")
    print(f" 📂 Đã lưu báo cáo tháng 6 tháng tại: {monthly_csv_path}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_h1_2020_v3_evaluation()
