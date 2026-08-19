"""
v3_system/run_jan2020_v3.py
============================
Script Chạy Chi Tiết Từng Ngày & In Tổng Hợp Tháng 1/2020 (v3 Architecture - Fixed Typo).
Được thiết kế bởi Senior Quantitative Researcher.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v3_preset_generator import V3PresetGenerator
from v3_regime_classifier import V3RegimeClassifier

def run_january_2020_v3_step1():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")

    print("\n=========================================================================================================")
    print(" 🚀 HỆ THỐNG V3: CHI TIẾT THEO DÕI TỪNG NGÀY & TỔNG HỢP THÁNG 1/2020")
    print("=========================================================================================================")
    print(f" • File dữ liệu nạp: {os.path.basename(csv_2020)}")
    print(f" • Số lượng Presets quét mỗi ngày: 540 Candidate Presets\n")

    pipeline = V3DataPipeline(base_dir)
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    feature_df, daily_m1_dict = pipeline.compute_daily_features(df_raw, target_month_str="2020-01")

    total_days = len(feature_df)
    print(f" -> Đã tìm thấy {total_days} ngày giao dịch trong Tháng 1/2020.\n")

    classifier = V3RegimeClassifier()
    feature_df = classifier.label_dataset_regimes(feature_df)

    generator = V3PresetGenerator()
    presets_540 = V3PresetGenerator.generate_540_candidate_presets()

    report_records = []

    print("=========================================================================================================")
    print(" 🔍 PHẦN 1: QUAN SÁT DIỄN BIẾN CHI TIẾT TỪNG NGÀY TRONG THÁNG 1/2020")
    print("=========================================================================================================\n")

    for idx, row in feature_df.iterrows():
        date_str = row['date']
        obs_df, exec_df = daily_m1_dict[date_str]

        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        vwap_dist_atr = row['vwap_dist_atr']
        bb_zscore = row['bb_zscore_m15']
        bb_slope = row['bb_slope_m15']
        morning_momentum = row['morning_momentum']
        regime_name = row['regime_name']
        regime_desc = row['regime_desc']

        price_1000 = exec_df['open'].iloc[0]
        min_low_10_12 = exec_df['low'].min()
        max_high_10_12 = exec_df['high'].max()

        direction_str = "SELL" if close_0959 >= daily_vwap else "BUY"

        print(f"---------------------------------------------------------------------------------------------------------")
        print(f" 📅 NGÀY {idx+1}/{total_days}: {date_str}")
        print(f"---------------------------------------------------------------------------------------------------------")
        print(f" • 09:59 AM Snapshot: Open 10:00=${price_1000:,.2f} | Close 09:59=${close_0959:,.2f} | VWAP=${daily_vwap:,.2f} (Lệch: {vwap_dist_atr:+.2f}x ATR)")
        print(f" • Chỉ số kỹ thuật : ATR(14)=${atr_14:.2f} | Z-Score={bb_zscore:.2f} | Slope={bb_slope:.2f} | Momentum={morning_momentum:.2f}")
        print(f" • Biến động M1 10-12: Min Low = ${min_low_10_12:,.2f} | Max High = ${max_high_10_12:,.2f}")
        print(f" • Hướng Giao Dịch  : {direction_str}")
        print(f" 🏷️ Nhóm Xu Hướng   : {regime_name} ({regime_desc})")

        # Thử nghiệm 540 Presets
        best_score = -float('inf')
        best_p_idx = -1
        best_param = None
        best_res = None

        for p_idx, p in enumerate(presets_540, start=1):
            res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, p)
            if res['fitness_score'] > best_score:
                best_score = res['fitness_score']
                best_p_idx = p_idx
                best_param = p
                best_res = res

        has_traded = (best_res is not None and best_res['num_orders'] > 0 and best_score > 0.0)

        if has_traded:
            p_info = best_param
            print(f" 🏆 WINNER PRESET   : Set P_{best_p_idx} | Step_0={p_info['step_0_ratio']}x ATR | Exp={p_info['step_exp']} | MaxOrd={int(p_info['max_orders'])} | Mult={p_info['multiplier']}x")
            print(f"    👉 Kết quả lệnh : Khớp {best_res['num_orders']} lệnh | Net PnL = +${best_res['net_profit']:,.2f} | Max DD = ${best_res['max_drawdown']:,.2f} | Fitness Score = {best_score:.1f}\n")
        else:
            print(f" 🛡️ TRẠNG THÁI     : NO-TRADE (Vàng không nảy giãn đủ Step_0 tối thiểu hoặc vi phạm quy tắc an toàn). PnL = $0.00 | Score = 0.0\n")

        sys.stdout.flush()

        report_records.append({
            'date': date_str,
            'regime_id': row['regime_id'],
            'regime_name': regime_name,
            'price_1000': price_1000,
            'direction': direction_str,
            'best_preset_id': best_p_idx if has_traded else 0,
            'step_0_ratio': best_param['step_0_ratio'] if has_traded else np.nan,
            'step_exp': best_param['step_exp'] if has_traded else np.nan,
            'max_orders': best_param['max_orders'] if has_traded else np.nan,
            'multiplier': best_param['multiplier'] if has_traded else np.nan,
            'net_profit': best_res['net_profit'] if has_traded else 0.0,
            'fitness_score': best_score if has_traded else 0.0
        })

    # PHẦN 2: BẢNG TỔNG HỢP CUỐI THÁNG
    summary_df = pd.DataFrame(report_records)

    print("\n=========================================================================================================")
    print(" 📊 PHẦN 2: BẢNG TỔNG HỢP KẾT QUẢ TOÀN BỘ THÁNG 1/2020")
    print("=========================================================================================================")
    print(" | STT | Ngày VN    | Nhóm Xu Hướng (Regime)    | Open 10:00 | Hướng | Best Set ID | Step_0 | Exp  | MaxOrd | Mult  | Net PnL ($) | Score  |")
    print(" +-----+------------+---------------------------+------------+-------+-------------+--------+------+--------+-------+-------------+--------+")

    for idx, r in summary_df.iterrows():
        if r['best_preset_id'] > 0:
            p_str = f"P_{int(r['best_preset_id'])}"
            s0_str = f"{r['step_0_ratio']:.1f}x"
            exp_str = f"{r['step_exp']:.2f}"
            mo_str = f"{int(r['max_orders'])}"
            mult_str = f"{r['multiplier']:.1f}x"
            pnl_str = f"+${r['net_profit']:,.2f}"
            score_str = f"{r['fitness_score']:.1f}"
        else:
            p_str = "No-Trade"
            s0_str = "-"
            exp_str = "-"
            mo_str = "-"
            mult_str = "-"
            pnl_str = "$0.00"
            score_str = "0.0"

        print(f" | {idx+1:<3} | {r['date']:<10} | {r['regime_name']:<25} | {r['price_1000']:<10,.2f} | {r['direction']:<5} | {p_str:<11} | {s0_str:<6} | {exp_str:<4} | {mo_str:<6} | {mult_str:<5} | {pnl_str:<11} | {score_str:<6} |")

    print(" ==========================================================================================================================================\n")

    regime_counts = summary_df['regime_name'].value_counts().to_dict()
    print(" 📈 PHÂN BỔ CÁC NHÓM XU HƯỚNG THÁNG 1/2020:")
    for r_name, count in regime_counts.items():
        pct = (count / len(summary_df)) * 100.0
        print(f"   • {r_name:<25}: {count} ngày ({pct:.1f}%)")

    traded_days_count = (summary_df['best_preset_id'] > 0).sum()
    total_jan_pnl = summary_df['net_profit'].sum()

    print(f"\n 💰 TỔNG PNl THÁNG 1/2020: +${total_jan_pnl:,.2f} ({traded_days_count}/{len(summary_df)} ngày có lệnh thắng)")

    csv_path = os.path.join(v3_dir, "jan2020_v3_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f" 📂 Đã lưu báo cáo CSV Tháng 1/2020 tại: {csv_path}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_january_2020_v3_step1()
