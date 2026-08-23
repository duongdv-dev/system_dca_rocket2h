"""
v4_system/run_v4_backtest_2020_2025.py
======================================
Script Backtest Kiểm Định Độc Lập 6 Năm (2020 - 2025) Cho Chiến Lược 1 Lệnh Duy Nhất (Non-DCA V4 System).

Được thiết kế bởi Senior Quantitative Researcher.

Mục tiêu:
1. Nạp toàn bộ nến M1 thô 6 năm (2020 -> 2025).
2. Chạy chiến lược 1 Lệnh Duy Nhất (Single Position Asian ORB Breakout).
3. Đánh giá tỷ lệ Risk:Reward 1:2.0 trên toàn bộ lịch sử giao dịch.
4. Tổng hợp Báo Cáo Hiệu Năng Chuẩn Institutional (Win Rate, Profit Factor, Expectancy, Max Drawdown).
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_data_pipeline import V3DataPipeline
from v4_orb_strategy import V4ORBStrategy

def run_v4_full_backtest():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v4_dir = os.path.join(base_dir, "v4_system")
    os.makedirs(v4_dir, exist_ok=True)

    data_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_202[0-5]_m1.csv")))
    
    if not data_files:
        print("❌ Không tìm thấy file dữ liệu XAUUSD_202x_m1.csv!")
        return

    print("\n=========================================================================================================")
    print(" 🚀 HỆ THỐNG V4: BACKTEST CHIẾN LƯỢC 1 LỆNH DUY NHẤT (NON-DCA SINGLE POSITION ORB 2020-2025)")
    print("=========================================================================================================")
    print(f" • File dữ liệu: {len(data_files)} files ({[os.path.basename(f) for f in data_files]})")
    print(f" • Cấu hình Risk: R:R = 1:2.0 | Risk per trade = $200 cố định (Rủi ro 2% tài khoản $10,000)\n")

    pipeline = V3DataPipeline(base_dir)
    strategy = V4ORBStrategy(rr_ratio=2.0, sl_atr_mult=1.0, risk_per_trade_usd=200.0)

    all_trades = []
    initial_balance = 10000.0
    current_balance = initial_balance

    for f_path in data_files:
        filename = os.path.basename(f_path)
        print(f" 📥 Đang chạy mô phỏng nến M1 thực tế năm: {filename}...")
        df_raw = pipeline.load_and_preprocess_file(f_path)
        df_raw['date_str'] = df_raw['datetime'].dt.strftime('%Y-%m-%d')
        
        unique_dates = sorted(df_raw['date_str'].unique())

        for date_str in unique_dates:
            day_m1 = df_raw[df_raw['date_str'] == date_str].copy().reset_index(drop=True)
            res = strategy.simulate_day_single_trade(day_m1)

            if res['traded']:
                pnl = res['net_profit']
                current_balance += pnl
                all_trades.append({
                    'date': date_str,
                    'month': date_str[:7],
                    'direction': res['direction'],
                    'entry_price': res['entry_price'],
                    'sl_price': res['sl_price'],
                    'tp_price': res['tp_price'],
                    'lot_size': res['lot_size'],
                    'hit_tp': res['hit_tp'],
                    'hit_sl': res['hit_sl'],
                    'net_profit': pnl,
                    'max_drawdown': res['max_drawdown'],
                    'balance': current_balance
                })

    df_trades = pd.DataFrame(all_trades)

    if len(df_trades) == 0:
        print("❌ Không tạo được giao dịch nào trong quá trình kiểm định!")
        return

    # TÍNH TOÁN CÁC CHỈ SỐ QUANT THỰC TẾ
    total_trades = len(df_trades)
    win_trades = (df_trades['net_profit'] > 0).sum()
    loss_trades = (df_trades['net_profit'] < 0).sum()
    even_trades = total_trades - win_trades - loss_trades

    total_net_profit = df_trades['net_profit'].sum()
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = df_trades[df_trades['net_profit'] > 0]['net_profit'].sum()
    gross_loss = abs(df_trades[df_trades['net_profit'] < 0]['net_profit'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.9

    expectancy_usd = total_net_profit / total_trades if total_trades > 0 else 0.0
    max_dd = df_trades['max_drawdown'].max()
    return_pct = (total_net_profit / initial_balance) * 100.0

    print("\n=========================================================================================================")
    print(" 📊 BẢNG TỔNG HỢP HIỆU NĂNG V4 SYSTEM (CHIẾN LƯỢC 1 LỆNH DUY NHẤT R:R 1:2.0 TỪ 2020-2025)")
    print("=========================================================================================================")
    print(f" • Số Dư Ban Đầu                  : ${initial_balance:,.2f}")
    print(f" • Số Dư Cuối Kỳ (Sau 6 Năm)      : ${current_balance:,.2f}")
    print(f" • TỔNG LỢI NHUẬN NET (6 Năm)     : +${total_net_profit:,.2f} (+{return_pct:.2f}%)")
    print(f" • Win Rate (Tỷ lệ Thắng)          : {win_rate:.1f}% ({win_trades} Thắng / {loss_trades} Thua / {even_trades} Hòa)")
    print(f" • Profit Factor (Hệ Số Lợi Nhuận): {profit_factor:.2f}")
    print(f" • Kỳ Vọng Lợi Nhuận / Lệnh ($E)   : +${expectancy_usd:.2f} per trade")
    print(f" • Max Drawdown Lớn Nhất          : ${max_dd:,.2f}")
    print(f" • Tổng Số Lệnh Đã Thực Hiện      : {total_trades} lệnh (Trung bình {total_trades/6:.0f} lệnh/năm)")
    print("=========================================================================================================\n")

    # Báo cáo theo năm
    print(" 📅 BẢNG THỐNG KÊ HIỆU NĂNG TỪNG NĂM (2020 - 2025):")
    print(" | Năm   | Số Lệnh | Thắng | Thua | Net Profit ($)   | Win Rate (%) | Profit Factor |")
    print(" +-------+---------+-------+------+------------------+--------------+---------------+")

    df_trades['year'] = df_trades['date'].str[:4]
    for y in sorted(df_trades['year'].unique()):
        sub_y = df_trades[df_trades['year'] == y]
        y_total = len(sub_y)
        y_win = (sub_y['net_profit'] > 0).sum()
        y_loss = (sub_y['net_profit'] < 0).sum()
        y_pnl = sub_y['net_profit'].sum()
        y_wr = (y_win / y_total * 100.0) if y_total > 0 else 0.0
        
        g_prof = sub_y[sub_y['net_profit'] > 0]['net_profit'].sum()
        g_loss = abs(sub_y[sub_y['net_profit'] < 0]['net_profit'].sum())
        y_pf = (g_prof / g_loss) if g_loss > 0 else 99.9

        print(f" | {y:<5} | {y_total:<7} | {y_win:<5} | {y_loss:<4} | +${y_pnl:<15,.2f} | {y_wr:<12.1f} | {y_pf:<13.2f} |")

    print("=========================================================================================================\n")

    # Lưu báo cáo CSV
    report_csv = os.path.join(v4_dir, "v4_single_trade_results_2020_2025.csv")
    df_trades.to_csv(report_csv, index=False)
    print(f" 📂 Đã lưu báo cáo giao dịch chi tiết V4 System tại: {report_csv}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_v4_full_backtest()
