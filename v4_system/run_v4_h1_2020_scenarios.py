"""
v4_system/run_v4_h1_2020_scenarios.py
======================================
Script Backtest Kiểm Định 6 Tháng Đầu Năm 2020 (2020-H1: 2020-01 -> 2020-06)
Cho Hệ Thống Giao Dịch V4 (Non-DCA Single Position Asian ORB Breakout)
Chạy Tất Cả Các Kịch Bản Tham Số & Xuất Log Nhận Ký Từng Ngày Dạng Bảng.

Được thiết kế bởi Senior Quantitative Researcher.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Đảm bảo import được module v3_system và v4_system
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "v3_system"))
sys.path.insert(0, os.path.join(base_dir, "v4_system"))

from v3_data_pipeline import V3DataPipeline
from v4_orb_strategy import V4ORBStrategy


def run_h1_2020_all_scenarios():
    v4_dir = os.path.join(base_dir, "v4_system")
    os.makedirs(v4_dir, exist_ok=True)

    csv_2020 = os.path.join(base_dir, "XAUUSD_2020_m1.csv")
    if not os.path.exists(csv_2020):
        print(f"❌ Không tìm thấy file dữ liệu: {csv_2020}")
        return

    target_months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]

    print("\n=========================================================================================================")
    print(" 🚀 HỆ THỐNG V4: KIỂM ĐỊNH 6 THÁNG ĐẦU NĂM 2020 (2020-01 -> 2020-06) TẤT CẢ CÁC KỊCH BẢN")
    print("=========================================================================================================")
    print(f" • File dữ liệu: {os.path.basename(csv_2020)}")
    print(f" • Phạm vi kiểm tra: 6 Tháng Đầu Năm 2020 ({', '.join(target_months)})")
    print(f" • Vốn ban đầu: $10,000 | Risk cố định/lệnh = $200 (2.0% tài khoản)\n")

    # 1. Định nghĩa các Kịch bản thử nghiệm
    scenarios = [
        {
            'id': 'KB1_Base_RR2.0_SL1.0',
            'name': 'Kịch Bản 1: Baseline Institutional (R:R 1:2.0 | SL 1.0x ATR)',
            'rr_ratio': 2.0,
            'sl_atr_mult': 1.0
        },
        {
            'id': 'KB2_HighWin_RR1.5_SL1.0',
            'name': 'Kịch Bản 2: High Win Rate (R:R 1:1.5 | SL 1.0x ATR)',
            'rr_ratio': 1.5,
            'sl_atr_mult': 1.0
        },
        {
            'id': 'KB3_HighExp_RR3.0_SL1.0',
            'name': 'Kịch Bản 3: High Expectancy (R:R 1:3.0 | SL 1.0x ATR)',
            'rr_ratio': 3.0,
            'sl_atr_mult': 1.0
        },
        {
            'id': 'KB4_TightSL_RR2.0_SL0.8',
            'name': 'Kịch Bản 4: Tight Stop Loss (R:R 1:2.0 | SL 0.8x ATR)',
            'rr_ratio': 2.0,
            'sl_atr_mult': 0.8
        },
        {
            'id': 'KB5_WideSL_RR2.0_SL1.2',
            'name': 'Kịch Bản 5: Wide Buffer SL (R:R 1:2.0 | SL 1.2x ATR)',
            'rr_ratio': 2.0,
            'sl_atr_mult': 1.2
        }
    ]

    # 2. Nạp Dữ Liệu
    pipeline = V3DataPipeline(base_dir)
    print(" 📥 Đang nạp và xử lý nến M1 6 tháng đầu năm 2020...")
    df_raw = pipeline.load_and_preprocess_file(csv_2020)
    df_raw['date_str'] = df_raw['datetime'].dt.strftime('%Y-%m-%d')
    df_raw['month_str'] = df_raw['datetime'].dt.strftime('%Y-%m')

    # Lọc đúng 6 tháng đầu 2020
    df_h1 = df_raw[df_raw['month_str'].isin(target_months)].copy()
    unique_dates = sorted(df_h1['date_str'].unique())

    print(f" -> Tổng số ngày giao dịch trích xuất được: {len(unique_dates)} ngày.\n")

    all_daily_logs = []
    scenario_summary_results = []

    initial_balance = 10000.0

    # 3. Vòng Lặp Chạy Từng Kịch Bản
    for sc in scenarios:
        sc_id = sc['id']
        sc_name = sc['name']
        rr = sc['rr_ratio']
        sl_mult = sc['sl_atr_mult']

        strategy = V4ORBStrategy(
            rr_ratio=rr,
            sl_atr_mult=sl_mult,
            risk_per_trade_usd=200.0
        )

        current_balance = initial_balance
        sc_trades = []

        print("=========================================================================================================")
        print(f" 📊 LOG NHẬT KÝ CHI TIẾT TỪNG NGÀY - {sc_name}")
        print("=========================================================================================================")
        print(" | STT | Ngày VN    | Hướng | Giá Entry | Giá SL    | Giá TP    | Lot Size | Kết Quả   | Net PnL ($) | Số Dư ($)  |")
        print(" +-----+------------+-------+-----------+-----------+-----------+----------+-----------+-------------+------------+")

        stt = 0
        for date_str in unique_dates:
            day_m1 = df_h1[df_h1['date_str'] == date_str].copy().reset_index(drop=True)
            res = strategy.simulate_day_single_trade(day_m1)

            if res['traded']:
                stt += 1
                pnl = res['net_profit']
                current_balance += pnl
                direction = res['direction']
                entry_p = res['entry_price']
                sl_p = res['sl_price']
                tp_p = res['tp_price']
                lot = res['lot_size']
                outcome = res['outcome']

                pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

                print(f" | {stt:<3} | {date_str:<10} | {direction:<5} | {entry_p:<9.2f} | {sl_p:<9.2f} | {tp_p:<9.2f} | {lot:<8.2f} | {outcome:<9} | {pnl_str:<11} | ${current_balance:<10,.2f} |")

                trade_record = {
                    'scenario_id': sc_id,
                    'scenario_name': sc_name,
                    'stt': stt,
                    'date': date_str,
                    'month': date_str[:7],
                    'direction': direction,
                    'entry_price': entry_p,
                    'sl_price': sl_p,
                    'tp_price': tp_p,
                    'lot_size': lot,
                    'outcome': outcome,
                    'hit_tp': res['hit_tp'],
                    'hit_sl': res['hit_sl'],
                    'net_profit': pnl,
                    'max_drawdown': res['max_drawdown'],
                    'balance': current_balance
                }
                sc_trades.append(trade_record)
                all_daily_logs.append(trade_record)
            else:
                # Ngày không xuất hiện tín hiệu breakout
                pass

        print("=========================================================================================================\n")

        # Thống kê kịch bản
        df_sc = pd.DataFrame(sc_trades)
        if len(df_sc) > 0:
            total_trades = len(df_sc)
            win_trades = (df_sc['net_profit'] > 0).sum()
            loss_trades = (df_sc['net_profit'] < 0).sum()
            eod_closed = total_trades - win_trades - loss_trades
            total_pnl = df_sc['net_profit'].sum()
            win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0

            gross_profit = df_sc[df_sc['net_profit'] > 0]['net_profit'].sum()
            gross_loss = abs(df_sc[df_sc['net_profit'] < 0]['net_profit'].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.9

            max_dd = df_sc['max_drawdown'].max()
            expectancy = total_pnl / total_trades
            return_pct = (total_pnl / initial_balance) * 100.0

            scenario_summary_results.append({
                'scenario_id': sc_id,
                'scenario_name': sc_name,
                'rr_ratio': rr,
                'sl_atr_mult': sl_mult,
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': loss_trades,
                'eod_closed': eod_closed,
                'win_rate': win_rate,
                'net_profit': total_pnl,
                'return_pct': return_pct,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'expectancy_usd': expectancy,
                'final_balance': current_balance
            })

    # 4. BẢNG SO SÁNH TỔNG HỢP TẤT CẢ CÁC KỊCH BẢN H1 2020
    df_summary = pd.DataFrame(scenario_summary_results)
    
    print("\n=========================================================================================================================")
    print(" 🏆 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 6 THÁNG ĐẦU NĂM 2020 (H1 2020) GIỮA CÁC KỊCH BẢN")
    print("=========================================================================================================================")
    print(" | ID Kịch Bản          | Tổng Lệnh | Thắng | Thua | Win Rate (%) | Net Profit ($) | ROI (%)  | Profit Factor | Max DD ($) | Expectancy ($) |")
    print(" +----------------------+-----------+-------+------+--------------+----------------+----------+---------------+------------+----------------+")

    for _, r in df_summary.iterrows():
        print(f" | {r['scenario_id']:<20} | {r['total_trades']:<9} | {r['win_trades']:<5} | {r['loss_trades']:<4} | {r['win_rate']:<12.1f} | +${r['net_profit']:<14,.2f} | +{r['return_pct']:<7.1f}% | {r['profit_factor']:<13.2f} | ${r['max_drawdown']:<10,.2f} | +${r['expectancy_usd']:<13.2f} |")

    print("=========================================================================================================================\n")

    # 5. LƯU FILE CSV BÁO CÁO
    daily_csv_path = os.path.join(v4_dir, "h1_2020_daily_trades_all_scenarios.csv")
    summary_csv_path = os.path.join(v4_dir, "h1_2020_scenarios_comparison_summary.csv")

    pd.DataFrame(all_daily_logs).to_csv(daily_csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    print(f" 📂 Đã lưu nhật ký giao dịch chi tiết từng ngày của 5 kịch bản tại: {daily_csv_path}")
    print(f" 📂 Đã lưu bảng tổng hợp so sánh hiệu năng các kịch bản tại     : {summary_csv_path}")
    print("=========================================================================================================================\n")


if __name__ == '__main__':
    run_h1_2020_all_scenarios()
