"""
v3_system/run_jan2020_v3.py
============================
Script Chạy Bước 1 V3: Chấm Điểm 540 Presets & Phân Loại Xu Hướng Cho Tháng 1/2020.
Được thiết kế bởi Senior Quantitative Researcher.

Quy trình V3 (Bước 1):
1. Nạp dữ liệu M1 Tháng 1/2020 từ `XAUUSD_2020_m1.csv`.
2. Phân loại từng ngày vào 4 Nhóm Xu Hướng Thị Trường (Range, Uptrend, Downtrend, Outlier).
3. Thử nghiệm 540 Presets tham số cho từng ngày và chấm điểm Fitness Score.
4. Trích xuất Preset thắng nhất (Best Preset) cho từng ngày và in báo cáo chi tiết.
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
    print(" 🚀 BẮT ĐẦU V3 (BƯỚC 1): CHẤM ĐIỂM 540 PRESETS & PHÂN LOẠI XU HƯỚNG THÁNG 1/2020")
    print("=========================================================================================================")
    print(f" • File dữ liệu nạp: {os.path.basename(csv_2020)}")
    print(f" • Phạm vi kiểm tra: Tháng 1 năm 2020 (2020-01-01 -> 2020-01-31)\n")

    pipeline = V3DataPipeline(base_dir)
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    feature_df, daily_m1_dict = pipeline.compute_daily_features(df_raw, target_month_str="2020-01")

    print(f" -> Đã trích xuất thành công: {len(feature_df)} ngày giao dịch trong Tháng 1/2020.")

    # Phân loại nhóm xu hướng cho từng ngày
    classifier = V3RegimeClassifier()
    feature_df = classifier.label_dataset_regimes(feature_df)

    # Sinh 540 Candidate Presets
    generator = V3PresetGenerator()
    presets_540 = V3PresetGenerator.generate_540_candidate_presets()
    print(f" -> Đã sinh thành công {len(presets_540)} Candidate Presets cho hệ thống V3.\n")

    report_records = []

    print(" 📊 BẢNG TỔNG HỢP PHÂN LOẠI XU HƯỚNG & PRESET THẮNG NHẤT THÁNG 1/2020:")
    print(" ==========================================================================================================================================")
    print(" | STT | Ngày VN    | Nhóm Xu Hướng (Regime)    | Open 10:00 | Hướng | Best Set ID | Step_0 | Exp  | MaxOrd | Mult  | Net PnL ($) | Score  |")
    print(" +-----+------------+---------------------------+------------+-------+-------------+--------+------+--------+-------+-------------+--------+")

    for idx, row in feature_df.iterrows():
        date_str = row['date']
        obs_df, exec_df = daily_m1_dict[date_str]

        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        regime_name = row['regime_name']
        price_1000 = exec_df['open'].iloc[0]

        direction_str = "SELL" if close_0959 >= daily_vwap else "BUY"

        best_score = -float('inf')
        best_p_idx = -1
        best_param = None
        best_res = None

        # Thử nghiệm TẤT CẢ 540 Presets cho ngày hôm nay
        for p_idx, p in enumerate(presets_540, start=1):
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

        print(f" | {idx+1:<3} | {date_str:<10} | {regime_name:<25} | {price_1000:<10,.2f} | {direction_str:<5} | {p_str:<11} | {s0_str:<6} | {exp_str:<4} | {mo_str:<6} | {mult_str:<5} | {pnl_str:<11} | {score_str:<6} |")

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

    print(" ==========================================================================================================================================\n")

    # Thống kê phân bố nhóm xu hướng Tháng 1/2020
    summary_df = pd.DataFrame(report_records)
    regime_counts = summary_df['regime_name'].value_counts().to_dict()

    print(" 📈 TỔNG HỢP PHÂN BỔ CÁC NHÓM XU HƯỚNG THÁNG 1/2020:")
    for r_name, count in regime_counts.items():
        pct = (count / len(summary_df)) * 100.0
        print(f"   • {r_name:<25}: {count} ngày ({pct:.1f}%)")

    traded_days_count = (summary_df['best_preset_id'] > 0).sum()
    total_jan_pnl = summary_df['net_profit'].sum()

    print(f"\n 💰 TỔNG PNl THÁNG 1/2020: +${total_jan_pnl:,.2f} ({traded_days_count}/{len(summary_df)} ngày có lệnh thắng)")

    report_txt_path = os.path.join(v3_dir, "jan2020_v3_report.txt")
    summary_df.to_csv(os.path.join(v3_dir, "jan2020_v3_summary.csv"), index=False)
    print(f" 📂 Báo cáo CSV Tháng 1/2020 đã được lưu tại: {os.path.join(v3_dir, 'jan2020_v3_summary.csv')}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_january_2020_v3_step1()
