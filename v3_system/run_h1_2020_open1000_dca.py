"""
v3_system/run_h1_2020_open1000_dca.py
======================================
Script Backtest Kiểm Định Độc Lập H1 2020 (2020-01 -> 2020-06)
Cho Chiến Lược DCA Chốt Lời Cố Định Tại Giá Mở Cửa 10:00 AM (Open 10:00 AM Mean Reversion).

Được thiết kế bởi Senior Quantitative Researcher.
"""

import os
import sys
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "v3_system"))

from v3_data_pipeline import V3DataPipeline
from v3_open1000_dca_engine import V3Open1000DCAEngine


def run_h1_2020_open1000_dca_backtest():
    v3_dir = os.path.join(base_dir, "v3_system")
    os.makedirs(v3_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")
    if not os.path.exists(csv_2020):
        print(f"❌ Không tìm thấy file dữ liệu: {csv_2020}")
        return

    target_months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]

    print("\n=========================================================================================================")
    print(" 🚀 KIỂM ĐỊNH H1 2020: CHIẾN LƯỢC DCA CHỐT LỜI HỒI VỀ GIÁ MỞ CỬA 10:00 AM (OPEN 10:00 AM DCA)")
    print("=========================================================================================================")
    print(f" • File dữ liệu: {os.path.basename(csv_2020)}")
    print(f" • Phạm vi kiểm tra: 6 Tháng Đầu Năm 2020 ({', '.join(target_months)})")
    print(f" • Cấu hình DCA: Max Orders = 4 | Dynamic Step_0 = 1.0x ATR | Step Exp = 1.15 | Multiplier = 1.10x")
    print(f" • Vốn ban đầu: $10,000.00\n")

    pipeline = V3DataPipeline(base_dir)
    print(" 📥 Đang nạp và xử lý nến M1 H1 2020...")
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    feature_df, daily_m1_dict = pipeline.compute_daily_features(df_raw, target_months=target_months)

    total_days = len(feature_df)
    trading_dates = feature_df['date'].tolist()
    print(f" -> Đã trích xuất thành công: {total_days} ngày giao dịch.\n")

    engine = V3Open1000DCAEngine(
        step_0_ratio=1.0,
        step_exp=1.15,
        max_orders=4,
        multiplier=1.10,
        base_lot=0.10
    )

    initial_balance = 10000.0
    current_balance = initial_balance

    daily_logs = []

    print("=========================================================================================================")
    print(" 📊 BẢNG LOG NHẬT KÝ CHI TIẾT TỪNG NGÀY (DCA HỒI VỀ GIÁ MỞ CỬA 10:00 AM)")
    print("=========================================================================================================")
    print(" | STT | Ngày VN    | Hướng | Giá Mở 10h | TP Target 10h | Dynamic Step_0 | Lệnh Khớp | Kết Quả     | Net PnL ($) | Số Dư ($)  |")
    print(" +-----+------------+-------+------------+---------------+----------------+-----------+-------------+-------------+------------+")

    stt = 0
    for date_str in trading_dates:
        row = feature_df[feature_df['date'] == date_str].iloc[0]
        obs_df, exec_df = daily_m1_dict[date_str]

        atr_14 = row['atr_14_m15']
        close_0959 = row['close_0959']
        daily_vwap = row['daily_vwap']
        bb_slope_m15 = row['bb_slope_m15']

        res = engine.simulate_day_open1000_dca(
            exec_m1=exec_df,
            atr_14=atr_14,
            close_0959=close_0959,
            daily_vwap=daily_vwap,
            bb_slope_m15=bb_slope_m15
        )

        if res['traded']:
            stt += 1
            pnl = res['net_profit']
            current_balance += pnl

            direction = res['direction']
            p10h = res['price_1000']
            tp10h = res['tp_price_1000']
            step_d = res['step_0_dollars']
            n_orders = res['num_orders']
            outcome = res['outcome']

            pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

            print(f" | {stt:<3} | {date_str:<10} | {direction:<5} | {p10h:<10.2f} | {tp10h:<13.2f} | ${step_d:<13.2f} | {n_orders:<9} | {outcome:<11} | {pnl_str:<11} | ${current_balance:<10,.2f} |")

            daily_logs.append({
                'stt': stt,
                'date': date_str,
                'month': date_str[:7],
                'direction': direction,
                'price_1000': p10h,
                'tp_price_1000': tp10h,
                'sl_price': res['sl_price'],
                'step_0_dollars': step_d,
                'num_orders': n_orders,
                'outcome': outcome,
                'hit_tp': res['hit_tp'],
                'hit_sl': res['hit_sl'],
                'net_profit': pnl,
                'max_drawdown': res['max_drawdown'],
                'balance': current_balance
            })

    print("=========================================================================================================\n")

    df_logs = pd.DataFrame(daily_logs)

    if len(df_logs) == 0:
        print("❌ Không có giao dịch nào xuất hiện!")
        return

    # TÍNH TOÁN HIỆU NĂNG TỔNG QUAN
    total_trades = len(df_logs)
    win_trades = (df_logs['net_profit'] > 0).sum()
    loss_trades = (df_logs['net_profit'] < 0).sum()
    even_trades = total_trades - win_trades - loss_trades

    total_net_profit = df_logs['net_profit'].sum()
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = df_logs[df_logs['net_profit'] > 0]['net_profit'].sum()
    gross_loss = abs(df_logs[df_logs['net_profit'] < 0]['net_profit'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.9

    expectancy = total_net_profit / total_trades
    max_dd = df_logs['max_drawdown'].max()
    roi = (total_net_profit / initial_balance) * 100.0

    print("=========================================================================================================")
    print(" 🏆 BẢNG TỔNG HỢP HIỆU NĂNG 6 THÁNG H1 2020 (DCA CHỐT LỜI HỒI VỀ GIÁ MỞ CỬA 10:00 AM)")
    print("=========================================================================================================")
    print(f" • Vốn Khởi Điểm                    : ${initial_balance:,.2f}")
    print(f" • Số Dư Cuối Kỳ (Sau 6 Tháng)      : ${current_balance:,.2f}")
    print(f" • TỔNG LỢI NHUẬN NET (6 Tháng H1)  : +${total_net_profit:,.2f} (+{roi:.2f}%)")
    print(f" • Tỷ Lệ Thắng (Win Rate 10h TP)     : {win_rate:.1f}% ({win_trades} Thắng 10h / {loss_trades} Dính SL / {even_trades} Hòa)")
    print(f" • Profit Factor (Hệ Số Lợi Nhuận)  : {profit_factor:.2f}")
    print(f" | Kỳ Vọng Lợi Nhuận / Lệnh ($E)     : +${expectancy:.2f}")
    print(f" • Max Drawdown Lớn Nhất            : ${max_dd:,.2f}")
    print("=========================================================================================================\n")

    # Thống kê từng tháng
    print(" 📅 THỐNG KÊ TỪNG THÁNG H1 2020:")
    print(" | Tháng   | Số Ngày Khớp | Thắng TP 10h | Dính SL | Win Rate (%) | Net Profit ($)   | Max DD ($)  |")
    print(" +---------+--------------+--------------+---------+--------------+------------------+-------------+")
    for m in target_months:
        sub_m = df_logs[df_logs['month'] == m]
        m_tot = len(sub_m)
        m_win = (sub_m['net_profit'] > 0).sum()
        m_loss = (sub_m['net_profit'] < 0).sum()
        m_pnl = sub_m['net_profit'].sum()
        m_wr = (m_win / m_tot * 100.0) if m_tot > 0 else 0.0
        m_dd = sub_m['max_drawdown'].max() if m_tot > 0 else 0.0

        pnl_fmt = f"+${m_pnl:,.2f}" if m_pnl >= 0 else f"-${abs(m_pnl):,.2f}"

        print(f" | {m:<7} | {m_tot:<12} | {m_win:<12} | {m_loss:<7} | {m_wr:<12.1f} | {pnl_fmt:<16} | ${m_dd:<11,.2f} |")

    print("=========================================================================================================\n")

    # Xuất file CSV báo cáo
    daily_csv = os.path.join(v3_dir, "h1_2020_open1000_dca_daily_log.csv")
    df_logs.to_csv(daily_csv, index=False)
    print(f" 📂 Đã lưu báo cáo chi tiết từng ngày chiến lược DCA 10:00 AM tại: {daily_csv}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_h1_2020_open1000_dca_backtest()
