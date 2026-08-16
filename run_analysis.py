import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

from feature_extractor import FeatureExtractor
from preset_pool import PresetPool
from ai_preset_selector import AIPresetSelector

def main():
    print("==========================================================================")
    print("🚀 HE THONG MULTI-PRESET POOL & AI DAILY META-SELECTOR (ROCKET 2H)")
    print("==========================================================================")
    
    TRAIN_YEARS = ['2020', '2021', '2022', '2023']  # Train Set (4 nam)
    TEST_YEARS = ['2024', '2025']                    # Out-of-Sample Test Set (2 nam)

    csv_files = sorted(glob.glob("XAUUSD_*_m1.csv"))
    if not csv_files:
        print("❌ Khong tim thay file XAUUSD_*_m1.csv nào!")
        return

    train_dfs = []
    test_dfs = []

    for f in csv_files:
        print(f" -> Dang doc {f}...")
        df_temp = pd.read_csv(f)
        df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='ms', utc=True)
        df_temp['year'] = df_temp['datetime'].dt.strftime('%Y')

        train_part = df_temp[df_temp['year'].isin(TRAIN_YEARS)]
        test_part = df_temp[df_temp['year'].isin(TEST_YEARS)]

        if not train_part.empty:
            train_dfs.append(train_part)
        if not test_part.empty:
            test_dfs.append(test_part)

    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_test_all = pd.concat(test_dfs, ignore_index=True)

    # 1. Trich xuat dac trung phien A
    print("\n🔍 1. Trich xuat dac trung phien A (08:00 - 10:00 VN Time)...")
    extractor_train = FeatureExtractor(df_train_all)
    train_features_df, train_days_data = extractor_train.extract_daily_features_and_data()

    extractor_test = FeatureExtractor(df_test_all)
    test_features_df, test_days_data = extractor_test.extract_daily_features_and_data()

    print(f"   • Tap Train (2020-2023): {len(train_features_df)} ngay")
    print(f"   • Tap Test  (2024-2025): {len(test_features_df)} ngay")

    # 2. Khoi tao Preset Pool (10 Presets đa dạng)
    print("\n📚 2. Khoi tao Multi-Preset Pool (10 Presets đa dạng)...")
    pool = PresetPool(account_balance=1000.0)
    for p in pool.pool:
        print(f"   • Preset {p['id']}: {p['name']} ({p['type']})")

    # 3. Mo phong tat ca Presets va tao Ma tran Nhan tren Tap Train (2020-2023)
    print("\n⚙️ 3. Generating Performance Matrix & Optimal Labels (2020-2023)...")
    train_matrix_df, train_best_labels = pool.build_performance_matrix(train_features_df, train_days_data)

    # 4. Huan luyen AI Meta-Selector
    print("\n🤖 4. Training AI Daily Meta-Selector...")
    ai_selector = AIPresetSelector(n_presets=len(pool.pool))
    ai_selector.train_selector(train_features_df, train_best_labels)

    # 5. Out-of-Sample Test (2024-2025): AI tu dong CHON PRESET moi sang
    print("\n🧪 5. Out-of-Sample Backtest: AI Dynamic Preset Selection (2024-2025)...")
    selected_preset_ids = ai_selector.select_preset_for_days(test_features_df)

    # Thuc thi các Presets duoc AI chon va tinh PnL
    test_results = []
    preset_selection_counts = {p['id']: 0 for p in pool.pool}

    for idx, row in test_features_df.iterrows():
        date_str = row['date']
        chosen_p_id = selected_preset_ids[idx]
        preset_selection_counts[chosen_p_id] += 1

        chosen_preset = pool.pool[chosen_p_id]
        if date_str not in test_days_data:
            test_results.append({'profit_usd': 0.0, 'max_drawdown_usd': 0.0, 'status': 'NO_DATA'})
            continue

        trade_df = test_days_data[date_str]
        res = pool.evaluate_preset_on_day(chosen_preset, row, trade_df)
        test_results.append(res)

    # Tinh toan thong ke PnL cho Tap Test được AI điều khiển
    active_results = [r for r in test_results if r['status'] != 'NO_TRADE']
    profits = [r['profit_usd'] for r in active_results]
    wins = [p for p in profits if p > 0]
    losses = [abs(p) for p in profits if p < 0]

    total_profit = sum(profits)
    win_rate = len(wins) / len(profits) * 100.0 if profits else 0.0
    profit_factor = (sum(wins) / (sum(losses) + 1e-8)) if losses else 999.0

    cum_pnl = 0.0
    peak_pnl = 0.0
    max_dd_usd = 0.0
    for p in profits:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd_usd:
            max_dd_usd = dd

    max_dd_pct = (max_dd_usd / pool.account_balance) * 100.0

    print("\n==========================================================================")
    print("📊 BAP CAO HE THONG AI DAILY PRESET SELECTOR TREN TAP TEST (2024-2025)")
    print("==========================================================================")
    print(f" 🚀 TONG LOI NHUAN AI TAO RA (2024-2025): ${total_profit:.2f} USD")
    print(f" • Profit Factor:     {profit_factor:.2f}")
    print(f" • Win Rate:          {win_rate:.2f}%")
    print(f" • Max Drawdown:      {max_dd_pct:.2f}%")
    print(f" • So ngay giao dich: {len(active_results)} ngay (AI da chon NO_TRADE {preset_selection_counts[9]} ngay)")
    
    print("\n📈 DANH SACH PRESETS DUOC AI CHON TRONG 2024-2025:")
    for p in pool.pool:
        count = preset_selection_counts[p['id']]
        pct = (count / len(test_features_df)) * 100.0
        print(f"   • Preset {p['id']} ({p['name']:<22}): Chon {count:<3} ngay ({pct:.1f}%)")

    # Xuat ket qua json
    output_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'architecture': 'Multi-Preset Pool + AI Daily Selector',
        'train_years': TRAIN_YEARS,
        'test_years': TEST_YEARS,
        'test_performance': {
            'total_profit_usd': round(total_profit, 2),
            'profit_factor': round(profit_factor, 2),
            'win_rate': round(win_rate, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'traded_days': len(active_results),
            'no_trade_days': preset_selection_counts[9]
        },
        'preset_counts': preset_selection_counts
    }

    with open('meta_selector_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("\n==========================================================================")
    print("🎉 HOAN THANH! Ket qua da duoc luu vao: meta_selector_results.json")
    print("==========================================================================")

if __name__ == "__main__":
    main()
