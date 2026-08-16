import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

from feature_extractor import FeatureExtractor
from ultimate_strategy_finder import UltimateStrategyFinder

def main():
    print("==========================================================================")
    print("🚀 DCA ROCKET 2H - HIGH-COMPOUNDING YIELD STRATEGY FINDER (+100% TO +300%+)")
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

    print(f"   • Tap Train (2020-2023): {len(train_features_df)} ngay giao dich")
    print(f"   • Tap Test  (2024-2025): {len(test_features_df)} ngay giao dich")

    # 2. Tim kiem Chien luoc DCA Rocket 2h Lợi Nhuận Khung tren Tap Train (2020-2023)
    print("\n🤖 2. Searching for High Compounding Yield DCA Strategy on Train Set (2020-2023)...")
    finder = UltimateStrategyFinder(account_balance=1000.0)
    best_preset, train_metrics = finder.search_high_yield_dca_strategy(train_features_df, train_days_data)

    print("\n✅ DA TIM THAY CHIEN LUOC DCA TOI UU TREN TAP TRAIN:")
    print(f"   • Entry Rule:     {best_preset['entry_rule']}")
    print(f"   • Step ATR Mult:  {best_preset['step_atr_mult']} * ATR | Step Mult: {best_preset['step_multiplier']}")
    print(f"   • TP ATR Mult:    {best_preset['tp_atr_mult']} * ATR | Lot Mult: {best_preset['lot_multiplier']}")
    print(f"   • In-Sample PnL:  Profit = ${train_metrics['total_profit_usd']} USD (+{train_metrics['roi_pct']}%) | PF = {train_metrics['profit_factor']} | WinRate = {train_metrics['win_rate']}% | MaxDD = {train_metrics['max_drawdown_pct']}%")

    # 3. Out-of-Sample Backtest Lãi Kép (Compounding) trên Tập Test (2024-2025)
    print("\n🧪 3. Running Out-of-Sample Compounding Backtest on Test Set (2024-2025)...")
    test_metrics = finder.evaluate_strategy_on_period(test_features_df, test_days_data, best_preset, use_compounding=True)

    print("\n==========================================================================")
    print("📊 BAP CAO LOI NHUAN KHUNG TREN TAP TEST OUT-OF-SAMPLE (2024-2025)")
    print("==========================================================================")
    print(f" 🚀 TAI KHOAN BAN DAU:                $1,000.00 USD")
    print(f" 💰 TAI KHOAN SAU 2 NAM (2024-2025):  ${test_metrics['final_equity_usd']} USD 🚀")
    print(f" 🔥 TONG LOI NHUAN 2 NAM:             +${test_metrics['total_profit_usd']} USD (Tỷ suất +{test_metrics['roi_pct']}%)")
    print(f" • Profit Factor (Hệ số lời/lỗ):     {test_metrics['profit_factor']}")
    print(f" • Win Rate (Tỷ lệ thắng):            {test_metrics['win_rate']}%")
    print(f" • Max Drawdown (Sụt giảm tối đa):   {test_metrics['max_drawdown_pct']}%")
    print(f" • Tổng số ngày giao dịch:            {test_metrics['total_days']} ngày")

    # Ghi ket qua json
    output_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Rocket 2h High-Compounding DCA Strategy',
        'optimal_preset': best_preset,
        'train_years': TRAIN_YEARS,
        'test_years': TEST_YEARS,
        'train_performance': train_metrics,
        'test_out_of_sample_performance': test_metrics
    }

    with open('high_yield_dca_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    with open('preset_candidates.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("\n==========================================================================")
    print("🎉 HOAN THANH! Kết quả lợi nhuận khủng đã được lưu vào: high_yield_dca_results.json")
    print("==========================================================================")

if __name__ == "__main__":
    main()
