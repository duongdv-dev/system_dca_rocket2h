"""
v3_system/run_v3_advanced_training_pipeline.py
================================================
Script Kiểm Định Đối Chứng Multi-Year Walk-Forward (2020 - 2025):
Chiến Lược Training Cũ (Old Strategy) vs Chiến Lược Training Mới (New Dual-Model Strategy).

Được thiết kế bởi Senior Quantitative Researcher.

Mục tiêu:
1. Nạp toàn bộ dữ liệu 6 năm (2020 -> 2025).
2. Chạy Rolling Walk-Forward (Train 12 tháng -> Test 6 tháng Out-of-Sample cuộn).
3. So sánh trực tiếp hiệu năng thực chiến của 2 chiến lược training:
   • Old Training Strategy : Single-day winner labeling + Standard LightGBM
   • New Training Strategy : Cluster-Based Robust Labeling + Dual-Model (Gatekeeper + Ranker)
4. Xuất Báo Cáo So Sánh Chi Tiết (Win Rate, Profit Factor, Max Drawdown, Net Profit, SL Hits).
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v3_preset_generator import V3PresetGenerator
from v3_preset_clusterer import V3PresetClusterer
from v3_ml_trainer import V3MLTrainer
from v3_robust_labeler import V3RobustLabeler
from v3_dual_model_trainer import V3DualModelTrainer
from v3_regime_classifier import V3RegimeClassifier

def run_comparative_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    # Nạp các file dữ liệu từ 2020 đến 2025
    data_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-5]_m1.csv")))
    
    if not data_files:
        print("❌ Không tìm thấy file dữ liệu XAUUSD_202x_m1.csv trong thư mục workspace!")
        return

    print("\n=========================================================================================================")
    print(" 🚀 KIỂM ĐỊNH ĐỐI CHỨNG MULTI-YEAR WALK-FORWARD (2020 - 2025)")
    print(" 📊 CHIẾN LƯỢC TRAINING CŨ vs CHIẾN LƯỢC TRAINING MỚI (DUAL-MODEL + ROBUST CLUSTERING)")
    print("=========================================================================================================")
    print(f" • Tổng số file dữ liệu năm: {len(data_files)} files ({[os.path.basename(f) for f in data_files]})\n")

    pipeline = V3DataPipeline(base_dir)
    
    all_feature_dfs = []
    all_m1_dicts = {}

    for f in data_files:
        print(f" 📥 Đang xử lý file dữ liệu: {os.path.basename(f)}...")
        df_raw = pipeline.load_and_preprocess_file(f)
        feat_df, m1_dict = pipeline.compute_daily_features(df_raw)
        all_feature_dfs.append(feat_df)
        all_m1_dicts.update(m1_dict)

    full_feature_df = pd.concat(all_feature_dfs, ignore_index=True)
    full_feature_df = full_feature_df.sort_values('date').reset_index(drop=True)
    
    # Gán nhãn Regime Classifier
    regime_classifier = V3RegimeClassifier()
    full_feature_df = regime_classifier.label_dataset_regimes(full_feature_df)

    print(f"\n -> Tổng số ngày giao dịch trích xuất được từ 2020 - 2025: {len(full_feature_df)} ngày.\n")

    # Danh sách các đặc trưng vi cấu trúc
    feature_cols = [
        'delta_open_0600_1000_r', 'delta_vwap_r', 'range_morning_r',
        'body_morning_r', 'morning_momentum', 'bb_zscore_m15', 'bb_slope_m15',
        'atr_ratio', 'asian_range_atr', 'wick_body_ratio_m15', 'vwap_dist_band_ratio',
        'pre_open_momentum_5m', 'asian_breakout_distance_r', 'order_flow_imbalance_proxy'
    ]

    generator = V3PresetGenerator()
    presets_list = V3PresetGenerator.generate_540_candidate_presets()
    robust_labeler = V3RobustLabeler(n_clusters=6, random_state=42)

    # QUY TRÌNH ROLLING WALK-FORWARD (Train 12 tháng -> Test 6 tháng cuộn)
    all_months = sorted(full_feature_df['month'].unique())
    train_window_months = 12
    test_window_months = 6

    old_strategy_oos_results = []
    new_strategy_oos_results = []

    initial_balance = 10000.0
    old_balance = initial_balance
    new_balance = initial_balance

    print("---------------------------------------------------------------------------------------------------------")
    print(" 🔄 BẮT ĐẦU VÒNG LẶP ROLLING WALK-FORWARD (TRAIN 12M ──► TEST 6M OUT-OF-SAMPLE)")
    print("---------------------------------------------------------------------------------------------------------")

    step = 0
    for start_idx in range(0, len(all_months) - train_window_months, test_window_months):
        step += 1
        train_m_list = all_months[start_idx : start_idx + train_window_months]
        test_m_list = all_months[start_idx + train_window_months : start_idx + train_window_months + test_window_months]

        if not test_m_list:
            break

        print(f"\n [Cửa sổ #{step}] Train: {train_m_list[0]} -> {train_m_list[-1]} ({len(train_m_list)}M) | Test OOS: {test_m_list[0]} -> {test_m_list[-1]} ({len(test_m_list)}M)")

        train_sub_df = full_feature_df[full_feature_df['month'].isin(train_m_list)].copy().reset_index(drop=True)
        test_sub_df = full_feature_df[full_feature_df['month'].isin(test_m_list)].copy().reset_index(drop=True)

        if len(train_sub_df) < 20 or len(test_sub_df) < 5:
            continue

        # =========================================================================
        # 1. STRATEGY CŨ (OLD TRAINING STRATEGY: Hindsight Winner + Standard LightGBM)
        # =========================================================================
        # Gán nhãn winner 1 ngày duy nhất
        old_train_records = []
        for idx, row in train_sub_df.iterrows():
            date_str = row['date']
            obs_df, exec_df = all_m1_dicts[date_str]
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
            rec['best_archetype_id'] = V3PresetClusterer.classify_preset(best_param) if has_traded else 0
            rec['fitness_score'] = best_score if has_traded else 0.0
            rec['hit_sl'] = best_res['hit_sl'] if has_traded else False
            rec['dd_atr'] = best_res['dd_atr'] if has_traded else 0.0
            old_train_records.append(rec)

        old_train_df = pd.DataFrame(old_train_records)

        old_trainer = V3MLTrainer(feature_cols=feature_cols[:11]) # Dùng 11 features cũ
        old_trainer.train_lightgbm(old_train_df)
        old_preds = old_trainer.predict(test_sub_df, confidence_threshold=0.30)

        # Backtest Old Strategy on Test OOS
        for idx, row in test_sub_df.iterrows():
            date_str = row['date']
            pred_aid = int(old_preds[idx])
            obs_df, exec_df = all_m1_dicts[date_str]
            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']
            bb_slope_m15 = row['bb_slope_m15']

            if pred_aid <= 0:
                pnl = 0.0
                dd = 0.0
                hit_tp, hit_sl = False, False
                orders = 0
            else:
                params = V3PresetClusterer.get_archetype_representative(pred_aid, presets_list)
                res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, params, bb_slope_m15=bb_slope_m15)
                pnl = res['net_profit']
                dd = res['max_drawdown']
                hit_tp, hit_sl = res['hit_tp'], res['hit_sl']
                orders = res['num_orders']

            old_balance += pnl
            old_strategy_oos_results.append({
                'date': date_str,
                'month': row['month'],
                'pred_archetype_id': pred_aid,
                'net_profit': pnl,
                'max_drawdown': dd,
                'balance': old_balance,
                'hit_tp': hit_tp,
                'hit_sl': hit_sl,
                'num_orders': orders
            })

        # =========================================================================
        # 2. STRATEGY MỚI (NEW TRAINING STRATEGY: Robust Labeling + Dual-Model Trainer)
        # =========================================================================
        # Gán nhãn bền vững theo cụm
        new_train_df = robust_labeler.generate_robust_labels(train_sub_df, all_m1_dicts, feature_cols)

        # Huấn luyện Dual Model
        dual_trainer = V3DualModelTrainer(feature_cols=feature_cols)
        dual_trainer.train_dual_models(new_train_df)
        new_preds = dual_trainer.predict(test_sub_df, safe_threshold=0.55)

        # Backtest New Strategy on Test OOS
        for idx, row in test_sub_df.iterrows():
            date_str = row['date']
            pred_aid = int(new_preds[idx])
            obs_df, exec_df = all_m1_dicts[date_str]
            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']
            bb_slope_m15 = row['bb_slope_m15']

            if pred_aid <= 0:
                pnl = 0.0
                dd = 0.0
                hit_tp, hit_sl = False, False
                orders = 0
            else:
                params = V3PresetClusterer.get_archetype_representative(pred_aid, presets_list)
                res = generator.simulate_day(exec_df, atr_14, close_0959, daily_vwap, params, bb_slope_m15=bb_slope_m15)
                pnl = res['net_profit']
                dd = res['max_drawdown']
                hit_tp, hit_sl = res['hit_tp'], res['hit_sl']
                orders = res['num_orders']

            new_balance += pnl
            new_strategy_oos_results.append({
                'date': date_str,
                'month': row['month'],
                'pred_archetype_id': pred_aid,
                'net_profit': pnl,
                'max_drawdown': dd,
                'balance': new_balance,
                'hit_tp': hit_tp,
                'hit_sl': hit_sl,
                'num_orders': orders
            })

    # =========================================================================
    # TỔNG HỢP BÁO CÁO KẾT QUẢ SO SÁNH OUT-OF-SAMPLE MULTI-YEAR (2020 - 2025)
    # =========================================================================
    df_old_res = pd.DataFrame(old_strategy_oos_results)
    df_new_res = pd.DataFrame(new_strategy_oos_results)

    def calc_metrics(df_res, init_bal):
        traded = df_res[df_res['num_orders'] > 0]
        t_cnt = len(traded)
        w_cnt = (traded['net_profit'] > 0).sum()
        sl_cnt = traded['hit_sl'].sum()
        total_pnl = df_res['net_profit'].sum()
        win_rate = (w_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0
        
        gross_profit = traded[traded['net_profit'] > 0]['net_profit'].sum()
        gross_loss = abs(traded[traded['net_profit'] < 0]['net_profit'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.9

        max_dd = df_res['max_drawdown'].max()
        return {
            'traded_days': t_cnt,
            'win_days': w_cnt,
            'sl_hits': sl_cnt,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'return_pct': (total_pnl / init_bal) * 100.0,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd
        }

    m_old = calc_metrics(df_old_res, initial_balance)
    m_new = calc_metrics(df_new_res, initial_balance)

    print("\n=========================================================================================================")
    print(" 🏆 BẢNG SO SÁNH ĐỐI CHỨNG HIỆU NĂNG OUT-OF-SAMPLE THỰC TẾ (MULTI-YEAR WALK-FORWARD 2020-2025)")
    print("=========================================================================================================")
    print(" | Chỉ Số Kiểm Định Out-of-Sample    | Chiến Lược Training Cũ          | Chiến Lược Training Mới (Dual-Model) |")
    print(" +-----------------------------------+---------------------------------+----------------------------------------+")
    print(f" | Tổng Lợi Nhuận Net Out-of-Sample  | +${m_old['total_pnl']:<16,.2f} | +${m_new['total_pnl']:<22,.2f} |")
    print(f" | Tỷ Suất Sinh Lợi (%)              | +{m_old['return_pct']:<16.2f}% | +{m_new['return_pct']:<22.2f}% |")
    print(f" | Win Rate Out-of-Sample            | {m_old['win_rate']:<16.1f}% | {m_new['win_rate']:<22.1f}% |")
    print(f" | Profit Factor (Hệ số Lợi Nhuận)   | {m_old['profit_factor']:<16.2f}  | {m_new['profit_factor']:<22.2f}  |")
    print(f" | Max Drawdown Lớn Nhất             | ${m_old['max_drawdown']:<16,.2f}  | ${m_new['max_drawdown']:<22,.2f}  |")
    print(f" | Số Ngày Duyệt Khớp Lệnh           | {m_old['traded_days']:<16} ngày | {m_new['traded_days']:<22} ngày |")
    print(f" | Số Lần Dính Hard Stop Loss        | {m_old['sl_hits']:<16} lần  | {m_new['sl_hits']:<22} lần  |")
    print("=========================================================================================================\n")

    # Lưu báo cáo CSV
    old_csv = os.path.join(v3_dir, "multiyear_old_strategy_oos.csv")
    new_csv = os.path.join(v3_dir, "multiyear_new_strategy_oos.csv")
    df_old_res.to_csv(old_csv, index=False)
    df_new_res.to_csv(new_csv, index=False)

    print(f" 📂 Đã lưu báo cáo OOS Chiến lược cũ tại : {old_csv}")
    print(f" 📂 Đã lưu báo cáo OOS Chiến lược MỚI tại: {new_csv}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_comparative_pipeline()
