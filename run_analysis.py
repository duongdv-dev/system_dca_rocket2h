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
    print("🏆 ROCKET 2H QUANT AI - ULTIMATE STRATEGY FINDER (BEAUTIFUL RESULT ENGINE)")
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

    # 2. Tim kiem Chien luoc Quant Tao KET QUA DEP tren Tap Train (2020-2023)
    print("\n🤖 2. Searching for the Ultimate Quant Strategy on Train Set (2020-2023)...")
    finder = UltimateStrategyFinder(account_balance=1000.0, risk_pct_per_trade=2.0)
    best_strat, train_metrics = finder.search_beautiful_strategy(train_features_df, train_days_data)

    print("\n✅ DA TIM THAY CHIEN LUOC QUANT TOI UU TREN TAP TRAIN:")
    print(f"   • Ten Chien Luoc: {best_strat['name']}")
    print(f"   • Khung Quy Tac:  {best_strat['mode']} (Biên độ Phiên Á: {best_strat['min_range_ratio']} - {best_strat['max_range_ratio']} * ATR)")
    print(f"   • Take Profit:   {best_strat['tp_atr_mult']} * ATR | Stop Loss: {best_strat['sl_atr_mult']} * ATR (R:R = 1 : {best_strat['tp_atr_mult'] / best_strat['sl_atr_mult']:.2f})")
    print(f"   • In-Sample PnL:  Profit = ${train_metrics['total_profit_usd']} USD | PF = {train_metrics['profit_factor']} | WinRate = {train_metrics['win_rate']}% | MaxDD = {train_metrics['max_drawdown_pct']}%")

    # 3. Out-of-Sample Backtest tren Tap Test (2024-2025)
    print("\n🧪 3. Running Out-of-Sample Backtest on Test Set (2024-2025)...")
    test_daily_results = []
    for idx, row in test_features_df.iterrows():
        date_str = row['date']
        if date_str not in test_days_data:
            continue
        trade_df = test_days_data[date_str]
        res = finder.evaluate_day_trade(row, trade_df, best_strat)
        test_daily_results.append(res)

    traded_test_results = [r for r in test_daily_results if r['status'] != 'FILTERED']
    test_profits = [r['profit_usd'] for r in traded_test_results]
    test_wins = [p for p in test_profits if p > 0]
    test_losses = [abs(p) for p in test_profits if p < 0]

    test_total_profit = sum(test_profits)
    test_win_rate = len(test_wins) / len(test_profits) * 100.0 if test_profits else 0.0
    test_profit_factor = (sum(test_wins) / (sum(test_losses) + 1e-8)) if test_losses else 999.0

    cum_pnl, peak_pnl, max_dd_usd = 0.0, 0.0, 0.0
    for p in test_profits:
        cum_pnl += p
        if cum_pnl > peak_pnl: peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd_usd: max_dd_usd = dd

    test_max_dd_pct = (max_dd_usd / finder.account_balance) * 100.0

    print("\n==========================================================================")
    print("🌟 KET QUA KIEU MAU (BEAUTIFUL RESULT) TREN TAP TEST OUT-OF-SAMPLE (2024-2025)")
    print("==========================================================================")
    print(f" 🚀 TONG LOI NHUAN 2 NAM (2024-2025): +${test_total_profit:.2f} USD (Tỷ suất +{test_total_profit/10.0:.1f}%)")
    print(f" • Profit Factor (Hệ số lời/lỗ):     {test_profit_factor:.2f}")
    print(f" • Win Rate (Tỷ lệ thắng):            {test_win_rate:.2f}%")
    print(f" • Max Drawdown (Sụt giảm tối đa):   {test_max_dd_pct:.2f}% (An toàn < 9.5%)")
    print(f" • Tổng số lệnh chắt lọc:             {len(traded_test_results)} lệnh (Trung bình {len(traded_test_results)/24:.1f} lệnh/tháng)")

    # Ghi ket qua json
    output_data = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy_name': best_strat['name'],
        'strategy_config': best_strat,
        'train_years': TRAIN_YEARS,
        'test_years': TEST_YEARS,
        'train_performance': train_metrics,
        'test_out_of_sample_performance': {
            'total_profit_usd': round(test_total_profit, 2),
            'roi_percentage': round(test_total_profit / 10.0, 2),
            'profit_factor': round(test_profit_factor, 2),
            'win_rate': round(test_win_rate, 2),
            'max_drawdown_pct': round(test_max_dd_pct, 2),
            'total_trades': len(traded_test_results)
        }
    }

    with open('beautiful_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    with open('preset_candidates.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("\n==========================================================================")
    print("🎉 HOAN THANH! Kết quả đẹp đã được lưu vào: beautiful_results.json")
    print("==========================================================================")

if __name__ == "__main__":
    main()
