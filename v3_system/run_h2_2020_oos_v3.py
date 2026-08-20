"""
v3_system/run_h2_2020_oos_v3.py
================================
Script Train 6 Tháng Đầu (2020-H1) -> Backtest Out-of-Sample 6 Tháng Sau (2020-H2).
Được thiết kế bởi Senior Quantitative Researcher.

Quy trình V3 Mới (Dynamic Trigger-Based Direction):
1. Preset CHỈ ĐỊNH NGHĨA CÁC THAM SỐ LƯỚI: [step_0_ratio, step_exp, max_orders, multiplier].
2. Hướng BUY/SELL được quyết định linh hoạt theo giá nảy thực tế sau 10:00 (không ép cứng bởi VWAP).
3. Train H1 (01 -> 06/2020) -> Test OOS H2 (07 -> 12/2020).
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v3_preset_generator import V3PresetGenerator
from v3_ml_trainer import V3MLTrainer

def run_h2_2020_out_of_sample_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")

    h1_months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]
    h2_months = ["2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12"]

    feature_cols = [
        'delta_open_0600_1000_r', 'delta_vwap_r', 'range_morning_r',
        'body_morning_r', 'morning_momentum', 'bb_zscore_m15', 'bb_slope_m15'
    ]

    print("\n=========================================================================================================")
    print(" 🚀 PIPELINE V3: DYNAMIC DIRECTION EXECUTION (TRAIN H1 ──► BACKTEST TEST H2 OOS)")
    print("=========================================================================================================")
    print(f" • File dữ liệu:               {os.path.basename(csv_2020)}")
    print(f" • Tập Train In-Sample  (H1): {', '.join(h1_months)}")
    print(f" • Tập Test Out-of-Sample (H2): {', '.join(h2_months)}\n")

    pipeline = V3DataPipeline(base_dir)
    df_raw = pipeline.load_and_preprocess_file(csv_2020)

    # 1. PHẦN 1: TRAIN IN-SAMPLE (6 THÁNG ĐẦU: 2020-01 -> 2020-06)
    print("---------------------------------------------------------------------------------------------------------")
    print(" 📥 PHẦN 1: HUẤN LUYỆN MODEL TRÊN 6 THÁNG ĐẦU NĂM 2020 (IN-SAMPLE 2020-H1)")
    print("---------------------------------------------------------------------------------------------------------")
    train_feature_df, train_m1_dict = pipeline.compute_daily_features(df_raw, target_months=h1_months)

    print(f" -> Tập Train 2020-H1 có: {len(train_feature_df)} ngày giao dịch.")

    generator = V3PresetGenerator()
    presets_list = V3PresetGenerator.generate_540_candidate_presets()
    preset_params_map = {i: p for i, p in enumerate(presets_list, start=1)}

    train_records = []
    for idx, row in train_feature_df.iterrows():
        date_str = row['date']
        obs_df, exec_df = train_m1_dict[date_str]

        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        bb_slope_m15 = row['bb_slope_m15']

        best_score = -float('inf')
        best_p_idx = -1
        best_param = None
        best_res = None

        for p_idx, p in enumerate(presets_list, start=1):
            res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, p, bb_slope_m15=bb_slope_m15)
            if res['fitness_score'] > best_score:
                best_score = res['fitness_score']
                best_p_idx = p_idx
                best_param = p
                best_res = res

        has_traded = (best_res is not None and best_res['num_orders'] > 0 and best_score > 0.0)

        rec = row.to_dict()
        rec['best_preset_id'] = best_p_idx if has_traded else 0
        rec['net_profit'] = best_res['net_profit'] if has_traded else 0.0
        rec['fitness_score'] = best_score if has_traded else 0.0
        train_records.append(rec)

    train_dataset_df = pd.DataFrame(train_records)
    train_pnl = train_dataset_df['net_profit'].sum()
    train_win_cnt = (train_dataset_df['best_preset_id'] > 0).sum()

    print(f" -> Hoàn tất chấm điểm H1: {train_win_cnt}/{len(train_dataset_df)} ngày đạt Best Preset. Tổng PnL H1 In-Sample: +${train_pnl:,.2f}")

    # Train LightGBM Model
    trainer = V3MLTrainer(feature_cols=feature_cols)
    trainer.train_lightgbm(train_dataset_df)

    # 2. PHẦN 2: BACKTEST OUT-OF-SAMPLE (6 THÁNG SAU: 2020-07 -> 2020-12)
    print("\n---------------------------------------------------------------------------------------------------------")
    print(" 🧪 PHẦN 2: CHẠY BACKTEST TEST OUT-OF-SAMPLE TRÊN 6 THÁNG SAU (2020-H2)")
    print("---------------------------------------------------------------------------------------------------------")
    test_feature_df, test_m1_dict = pipeline.compute_daily_features(df_raw, target_months=h2_months)
    print(f" -> Tập Test Out-of-Sample 2020-H2 có: {len(test_feature_df)} ngày giao dịch.")

    h2_predictions = trainer.predict(test_feature_df)

    initial_balance = 10000.0
    curr_balance = initial_balance
    oos_trades = []

    for idx, row in test_feature_df.iterrows():
        date_str = row['date']
        pred_p_idx = int(h2_predictions[idx])

        obs_df, exec_df = test_m1_dict[date_str]
        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        bb_slope_m15 = row['bb_slope_m15']
        price_1000 = exec_df['open'].iloc[0]

        if pred_p_idx <= 0 or pred_p_idx not in preset_params_map:
            oos_trades.append({
                'date': date_str,
                'month': row['month'],
                'price_1000': price_1000,
                'pred_preset_id': 0,
                'preset_name': 'No-Trade',
                'net_profit': 0.0,
                'max_drawdown': 0.0,
                'balance': curr_balance,
                'hit_tp': False,
                'num_orders': 0
            })
            continue

        pred_params = preset_params_map[pred_p_idx]
        res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, pred_params, bb_slope_m15=bb_slope_m15)

        day_pnl = res['net_profit']
        curr_balance += day_pnl

        oos_trades.append({
            'date': date_str,
            'month': row['month'],
            'price_1000': price_1000,
            'pred_preset_id': pred_p_idx,
            'step_0_ratio': pred_params['step_0_ratio'],
            'step_exp': pred_params['step_exp'],
            'max_orders': pred_params['max_orders'],
            'multiplier': pred_params['multiplier'],
            'net_profit': day_pnl,
            'max_drawdown': res['max_drawdown'],
            'balance': curr_balance,
            'hit_tp': res['hit_tp'],
            'hit_sl': res['hit_sl'],
            'num_orders': res['num_orders']
        })

    oos_df = pd.DataFrame(oos_trades)

    # 3. BẢNG TỔNG HỢP CHI TIẾT TỪNG THÁNG OUT-OF-SAMPLE (H2: THÁNG 7 -> THÁNG 12/2020)
    print("\n=========================================================================================================")
    print(" 📊 BẢNG CHI TIẾT KẾT QUẢ BACKTEST TỪNG THÁNG OUT-OF-SAMPLE (2020-H2)")
    print("=========================================================================================================")
    print(" | Tháng   | Tổng Ngày | Số Ngày Khớp Lệnh | Số Ngày Thắng | Net PnL ($)     | Win Rate (%) | Max Drawdown ($) |")
    print(" +---------+-----------+-------------------+---------------+-----------------+--------------+------------------+")

    monthly_oos_summary = []
    for m in h2_months:
        sub_m = oos_df[oos_df['month'] == m]
        m_days = len(sub_m)
        traded_sub = sub_m[sub_m['num_orders'] > 0]
        traded_cnt = len(traded_sub)
        win_cnt = (traded_sub['net_profit'] > 0).sum()
        m_pnl = sub_m['net_profit'].sum()
        m_winrate = (win_cnt / traded_cnt * 100.0) if traded_cnt > 0 else 0.0
        m_max_dd = sub_m['max_drawdown'].max()

        print(f" | {m:<7} | {m_days:<9} | {traded_cnt:<17} | {win_cnt:<13} | +${m_pnl:<15,.2f} | {m_winrate:<12.1f} | ${m_max_dd:<16,.2f} |")

        monthly_oos_summary.append({
            'month': m,
            'total_days': m_days,
            'traded_days': traded_cnt,
            'win_days': win_cnt,
            'net_profit': m_pnl,
            'win_rate': m_winrate,
            'max_drawdown': m_max_dd
        })

    print("=========================================================================================================\n")

    # 4. BẢNG SO SÁNH TỔNG KẾT IN-SAMPLE (H1) VS OUT-OF-SAMPLE (H2)
    h2_traded_df = oos_df[oos_df['num_orders'] > 0]
    h2_traded_cnt = len(h2_traded_df)
    h2_win_cnt = (h2_traded_df['net_profit'] > 0).sum()
    h2_total_pnl = oos_df['net_profit'].sum()
    h2_winrate = (h2_win_cnt / h2_traded_cnt * 100.0) if h2_traded_cnt > 0 else 0.0
    h2_max_dd = oos_df['max_drawdown'].max()
    h2_return_pct = ((curr_balance - initial_balance) / initial_balance) * 100.0

    print(" 🏆 BẢNG TỔNG KẾT SO SÁNH IN-SAMPLE (6 THÁNG ĐẦU) VS OUT-OF-SAMPLE (6 THÁNG SAU 2020):")
    print(f" ---------------------------------------------------------------------------------------------------------")
    print(f" • TAP TRAIN 6 THÁNG ĐẦU (IN-SAMPLE 2020-H1):")
    print(f"   - Tổng số ngày train:                  {len(train_feature_df)} ngày")
    print(f"   - Tổng Lợi Nhuận In-Sample:             +${train_pnl:,.2f}")
    print(f" ---------------------------------------------------------------------------------------------------------")
    print(f" • TAP TEST 6 THÁNG SAU (OUT-OF-SAMPLE 2020-H2):")
    print(f"   - Số dư vốn ban đầu H2:                ${initial_balance:,.2f}")
    print(f"   - Số dư vốn cuối kỳ H2:               ${curr_balance:,.2f}")
    print(f"   - TỔNG LỢI NHUẬN OUT-OF-SAMPLE NET:    +${h2_total_pnl:,.2f} (+{h2_return_pct:.2f}%)")
    print(f"   - Win Rate Out-of-Sample (H2):          {h2_winrate:.1f}% ({h2_win_cnt}W / {h2_traded_cnt - h2_win_cnt}L)")
    print(f"   - Max Drawdown Lớn Nhất (H2):           ${h2_max_dd:,.2f}")
    print("=========================================================================================================\n")

    # Lưu báo cáo CSV
    oos_trades_csv = os.path.join(v3_dir, "h2_2020_oos_trades.csv")
    oos_summary_csv = os.path.join(v3_dir, "h2_2020_oos_monthly_summary.csv")

    oos_df.to_csv(oos_trades_csv, index=False)
    pd.DataFrame(monthly_oos_summary).to_csv(oos_summary_csv, index=False)

    print(f" 📂 Đã lưu báo cáo chi tiết từng ngày OOS H2 tại: {oos_trades_csv}")
    print(f" 📂 Đã lưu báo cáo tổng hợp từng tháng OOS H2 tại: {oos_summary_csv}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_h2_2020_out_of_sample_pipeline()
