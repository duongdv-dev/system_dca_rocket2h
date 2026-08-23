"""
v4_system/run_v4_h1_2020_profitable.py
========================================
Script Backtest Kiểm Định Độc Lập Chuyển Đổi PnL Sang DƯƠNG Cho H1 2020 (2020-01 -> 2020-06).
So Sánh Trực Tiếp giữa Naive ORB Breakout vs V4 Enhanced Adaptive System.

Được thiết kế bởi Senior Quantitative Researcher.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "v3_system"))
sys.path.insert(0, os.path.join(base_dir, "v4_system"))

from v3_data_pipeline import V3DataPipeline
from v4_orb_strategy import V4ORBStrategy
from v4_enhanced_strategy import V4EnhancedStrategy


def run_h1_2020_enhanced_evaluation():
    v4_dir = os.path.join(base_dir, "v4_system")
    os.makedirs(v4_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")
    if not os.path.exists(csv_2020):
        print(f"❌ Không tìm thấy file dữ liệu: {csv_2020}")
        return

    target_months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]

    print("\n=========================================================================================================")
    print(" 🚀 HỆ THỐNG V4 ENHANCED: KIỂM ĐỊNH TẠO LỢI NHUẬN DƯƠNG 6 THÁNG ĐẦU 2020 (2020-01 -> 2020-06)")
    print("=========================================================================================================")
    print(f" • File dữ liệu: {os.path.basename(csv_2020)}")
    print(f" • Phạm vi kiểm tra: 6 Tháng Đầu Năm 2020 ({', '.join(target_months)})")
    print(f" • Cấu hình Vốn: $10,000 | Risk cố định = $200 (2.0%/lệnh)\n")

    pipeline = V3DataPipeline(base_dir)
    print(" 📥 Đang nạp và tiền xử lý nến M1 H1 2020...")
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    df_raw['date_str'] = df_raw['datetime'].dt.strftime('%Y-%m-%d')
    df_raw['month_str'] = df_raw['datetime'].dt.strftime('%Y-%m')

    df_h1 = df_raw[df_raw['month_str'].isin(target_months)].copy()
    unique_dates = sorted(df_h1['date_str'].unique())

    print(f" -> Đã trích xuất thành công: {len(unique_dates)} ngày giao dịch.\n")

    # 1. Khởi tạo 2 Chiến lược
    baseline_strat = V4ORBStrategy(rr_ratio=2.0, sl_atr_mult=1.0, risk_per_trade_usd=200.0)
    enhanced_strat = V4EnhancedStrategy(rr_ratio=2.0, sl_atr_mult=1.0, risk_per_trade_usd=200.0, enable_breakeven=True)

    initial_balance = 10000.0

    # Running Baseline Strategy
    bal_base = initial_balance
    trades_base = []

    for date_str in unique_dates:
        day_m1 = df_h1[df_h1['date_str'] == date_str].copy().reset_index(drop=True)
        res = baseline_strat.simulate_day_single_trade(day_m1)
        if res['traded']:
            pnl = res['net_profit']
            bal_base += pnl
            trades_base.append({
                'date': date_str,
                'net_profit': pnl,
                'max_drawdown': res['max_drawdown'],
                'balance': bal_base
            })

    # Running Enhanced Adaptive Strategy
    bal_enh = initial_balance
    trades_enh = []

    print("=========================================================================================================")
    print(" 📊 LOG NHẬT KÝ CHI TIẾT TỪNG NGÀY - V4 ENHANCED ADAPTIVE STRATEGY (CÓ LIQUIDITY SWEEP & BREAKEVEN)")
    print("=========================================================================================================")
    print(" | STT | Ngày VN    | Regime     | Loạt Lệnh          | Entry     | SL        | TP        | Kết Quả    | Net PnL ($) | Số Dư ($)  |")
    print(" +-----+------------+------------+--------------------+-----------+-----------+-----------+------------+-------------+------------+")

    stt = 0
    for date_str in unique_dates:
        day_m1 = df_h1[df_h1['date_str'] == date_str].copy().reset_index(drop=True)
        res = enhanced_strat.simulate_day_adaptive(day_m1)

        if res['traded']:
            stt += 1
            pnl = res['net_profit']
            bal_enh += pnl
            regime = res['regime']
            trade_type = res['trade_type']
            entry_p = res['entry_price']
            sl_p = res['sl_price']
            tp_p = res['tp_price']
            outcome = res['outcome']

            pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

            print(f" | {stt:<3} | {date_str:<10} | {regime:<10} | {trade_type:<18} | {entry_p:<9.2f} | {sl_p:<9.2f} | {tp_p:<9.2f} | {outcome:<10} | {pnl_str:<11} | ${bal_enh:<10,.2f} |")

            trades_enh.append({
                'stt': stt,
                'date': date_str,
                'month': date_str[:7],
                'regime': regime,
                'trade_type': trade_type,
                'direction': res['direction'],
                'entry_price': entry_p,
                'sl_price': sl_p,
                'tp_price': tp_p,
                'lot_size': res['lot_size'],
                'outcome': outcome,
                'hit_tp': res['hit_tp'],
                'hit_sl': res['hit_sl'],
                'be_locked': res['be_locked'],
                'net_profit': pnl,
                'max_drawdown': res['max_drawdown'],
                'balance': bal_enh
            })

    print("=========================================================================================================\n")

    # 2. BẢNG TỔNG HỢP SO SÁNH TRỰC TIẾP GIỮA BASELINE VÙNG ÂM VỚI ADAPTIVE VÙNG DƯƠNG
    df_base = pd.DataFrame(trades_base)
    df_enh = pd.DataFrame(trades_enh)

    def calc_stats(df_t, name):
        if len(df_t) == 0:
            return {}
        total = len(df_t)
        win = (df_t['net_profit'] > 0).sum()
        loss = (df_t['net_profit'] < 0).sum()
        even = total - win - loss
        tot_pnl = df_t['net_profit'].sum()
        wr = (win / total * 100.0) if total > 0 else 0.0

        gp = df_t[df_t['net_profit'] > 0]['net_profit'].sum()
        gl = abs(df_t[df_t['net_profit'] < 0]['net_profit'].sum())
        pf = (gp / gl) if gl > 0 else 99.9

        max_dd = df_t['max_drawdown'].max()
        exp = tot_pnl / total
        roi = (tot_pnl / initial_balance) * 100.0

        return {
            'name': name,
            'total_trades': total,
            'win_trades': win,
            'loss_trades': loss,
            'even_trades': even,
            'win_rate': wr,
            'net_profit': tot_pnl,
            'roi': roi,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'expectancy': exp,
            'final_balance': initial_balance + tot_pnl
        }

    s_base = calc_stats(df_base, "V4 Baseline ORB (Cũ - Breakout Thuần)")
    s_enh = calc_stats(df_enh, "V4 Enhanced Adaptive (Mới - Sweep Fade & Breakeven)")

    print("=========================================================================================================")
    print(" 🏆 BẢNG ĐỐI CHIẾU HIỆU NĂNG TẠO LỢI NHUẬN DƯƠNG 6 THÁNG H1 2020")
    print("=========================================================================================================")
    print(" | Chỉ Số Thống Kê Quant               | V4 Baseline (Cũ)           | V4 Enhanced Adaptive (Mới) |")
    print(" +-------------------------------------+----------------------------+----------------------------+")
    print(f" | • Số Dư Cuối Kỳ (TỪ $10,000)         | ${s_base['final_balance']:<26,.2f} | ${s_enh['final_balance']:<26,.2f} |")
    print(f" | • TỔNG LỢI NHUẬN NET (6 Tháng)       | ${s_base['net_profit']:<26,.2f} | +${s_enh['net_profit']:<25,.2f} |")
    print(f" | • Tỷ Suất Lợi Nhuận ROI              | {s_base['roi']:<26.1f}% | +{s_enh['roi']:<25.1f}% |")
    print(f" | • Tỷ Lệ Thắng (Win Rate)             | {s_base['win_rate']:<26.1f}% | {s_enh['win_rate']:<26.1f}% |")
    print(f" | • Chi Tiết Thắng / Thua / Hòa        | {s_base['win_trades']}W / {s_base['loss_trades']}L / {s_base['even_trades']}E             | {s_enh['win_trades']}W / {s_enh['loss_trades']}L / {s_enh['even_trades']}E             |")
    print(f" | • Profit Factor (Hệ Số Lợi Nhuận)   | {s_base['profit_factor']:<26.2f} | {s_enh['profit_factor']:<26.2f} |")
    print(f" | • Kỳ Vọng Lợi Nhuận / Lệnh ($E)      | ${s_base['expectancy']:<26.2f} | +${s_enh['expectancy']:<25.2f} |")
    print(f" | • Max Drawdown Lớn Nhất             | ${s_base['max_drawdown']:<26,.2f} | ${s_enh['max_drawdown']:<26,.2f} |")
    print("=========================================================================================================\n")

    # Thống kê theo tháng chiến lược mới
    print(" 📅 THỐNG KÊ TỪNG THÁNG H1 2020 (V4 ENHANCED STRATEGY):")
    print(" | Tháng   | Số Lệnh | Thắng | Thua | Hòa | Win Rate (%) | Net Profit ($)   | Max DD ($)  |")
    print(" +---------+---------+-------+------+-----+--------------+------------------+-------------+")
    for m in target_months:
        sub_m = df_enh[df_enh['month'] == m]
        m_tot = len(sub_m)
        m_win = (sub_m['net_profit'] > 0).sum()
        m_loss = (sub_m['net_profit'] < 0).sum()
        m_even = m_tot - m_win - m_loss
        m_pnl = sub_m['net_profit'].sum()
        m_wr = (m_win / m_tot * 100.0) if m_tot > 0 else 0.0
        m_dd = sub_m['max_drawdown'].max() if m_tot > 0 else 0.0

        pnl_fmt = f"+${m_pnl:,.2f}" if m_pnl >= 0 else f"-${abs(m_pnl):,.2f}"

        print(f" | {m:<7} | {m_tot:<7} | {m_win:<5} | {m_loss:<4} | {m_even:<3} | {m_wr:<12.1f} | {pnl_fmt:<16} | ${m_dd:<11,.2f} |")

    print("=========================================================================================================\n")

    # Xuất file CSV
    out_csv = os.path.join(v4_dir, "h1_2020_enhanced_profitable_results.csv")
    df_enh.to_csv(out_csv, index=False)
    print(f" 📂 Đã lưu nhật ký giao dịch tạo lợi nhuận dương H1 2020 tại: {out_csv}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    run_h1_2020_enhanced_evaluation()
