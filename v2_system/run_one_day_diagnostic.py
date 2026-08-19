"""
v2_system/run_one_day_diagnostic.py
===================================
Script Mô Phỏng 1 Ngày Chi Tiết Từng Phút (Single Day Diagnostic Inspector).
Được thiết kế bởi Senior Quantitative Researcher.

Mục đích:
Cho phép người dùng soi từng phút nến M1, từng lệnh rải, giá khớp, PnL trạng thái và chốt lời TP
cho DUY NHẤT 1 NGÀY BẤT KỲ để kiểm tra tính thực tế và tìm xem có lỗi ở đâu.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from grid_simulator import GridSimulator

def run_single_day_diagnostic(target_date: str = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_2024_m1.csv"))) + sorted(glob.glob(os.path.join(base_dir, "XAUUSD_2025_m1.csv")))

    if not test_files:
        test_files = sorted(glob.glob(os.path.join(base_dir, "XAUUSD_2023_m1.csv")))

    pipeline = DataPipeline(base_dir)
    feature_df, daily_m1_dict = pipeline.extract_daily_dataset(test_files)

    if feature_df.empty:
        print("❌ Không tìm thấy dữ liệu hợp lệ!")
        return

    # Nếu không chỉ định ngày, tự chọn ngày đầu tiên có nảy rải lệnh
    if target_date is None or target_date not in daily_m1_dict:
        target_date = feature_df['date'].iloc[0]

    row = feature_df[feature_df['date'] == target_date].iloc[0]
    obs_df, exec_df = daily_m1_dict[target_date]

    atr_14 = row['atr_14_m15']
    close_0959 = row['close_0959']
    daily_vwap = row['daily_vwap']
    vwap_dist_atr = row['vwap_dist_atr']
    bb_zscore = row['bb_zscore_m15']
    bb_slope = row['bb_slope_m15']
    morning_momentum = row['morning_momentum']
    price_1000 = exec_df['open'].iloc[0]

    direction = -1 if close_0959 >= daily_vwap else 1
    dir_str = "SELL (Kỳ vọng giảm về Open 10:00)" if direction == -1 else "BUY (Kỳ vọng tăng về Open 10:00)"

    print("\n=========================================================================================================")
    print(f" 🔍 KỊCH BẢN MÔ PHỎNG CHI TIẾT 1 NGÀY QUAN SÁT: {target_date}")
    print("=========================================================================================================")
    print(f" 1. CHỈ SỐ THỊ TRƯỜNG LÚC 09:59:59 AM:")
    print(f"    • Giá Open 10:00 AM (price_1000): ${price_1000:,.2f}")
    print(f"    • Giá Close 09:59 AM:            ${close_0959:,.2f}")
    print(f"    • Daily VWAP:                    ${daily_vwap:,.2f} (Lệch VWAP: {vwap_dist_atr:+.2f}x ATR)")
    print(f"    • ATR(14) M15:                   ${atr_14:,.2f}")
    print(f"    • BB Z-Score / Slope M15:        Z={bb_zscore:.2f} | Slope={bb_slope:.2f}")
    print(f"    • Động lượng phiên sáng:         Momentum = {morning_momentum:.2f}")
    print(f"    👉 Hướng Giao Dịch Quyết Định:   {dir_str}\n")

    # Đặt bộ tham số kiểm tra mẫu: Step_0 = 1.2x ATR, Step_Exp = 1.2, Max_Orders = 4, Multiplier = 1.2x
    step_0 = 1.2 * atr_14
    step_exp = 1.2
    max_orders = 4
    multiplier = 1.20
    spread_dollars = 0.25  # Spread $0.25 (25 pips)

    if direction == 1:
        tp_price = price_1000 + spread_dollars
    else:
        tp_price = price_1000 - spread_dollars

    trigger_prices = []
    curr_p = (price_1000 - step_0) if direction == 1 else (price_1000 + step_0)
    trigger_prices.append(curr_p)

    for i in range(1, max_orders):
        dist_i = step_0 * (step_exp ** i)
        curr_p = curr_p - dist_i if direction == 1 else curr_p + dist_i
        trigger_prices.append(curr_p)

    print(" 2. THÔNG SỐ LƯỚI DCA ĐÃ CẤU HÌNH KHỚP:")
    print(f"    • Mức giá Chốt Lời (TP Target cố định): ${tp_price:,.2f}")
    for i, tp_val in enumerate(trigger_prices, start=1):
        print(f"    • Trigger Price Order {i}: ${tp_val:,.2f}")

    print("\n 3. NHẬT KÝ THỰC THI TỪNG PHÚT (10:00:00 - 12:00:00):")
    print(" ---------------------------------------------------------------------------------------------------------")
    print(" | Phút | Thời Gian VN | Giá M1 High | Giá M1 Low | Lệnh Khớp Mới | Lớp Lệnh | PnL Trạng Thái ($) | Trạng Thái   |")
    print(" +------+--------------+-------------+------------+---------------+----------+--------------------+--------------+")

    orders_placed = []
    next_order_idx = 0
    closed = False
    hit_tp = False
    net_profit = 0.0
    contract_size = 100.0

    for m_idx, (t, row) in enumerate(exec_df.iterrows()):
        time_str = t.strftime('%H:%M:%S')
        high_t = row['high']
        low_t = row['low']
        close_t = row['close']
        new_order_str = "-"

        # Kích hoạt lệnh lưới
        while next_order_idx < len(trigger_prices):
            trig_p = trigger_prices[next_order_idx]
            triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

            if triggered:
                entry_p = (trig_p + spread_dollars / 2.0) if direction == 1 else (trig_p - spread_dollars / 2.0)
                lot_k = 0.1 * (multiplier ** next_order_idx)
                orders_placed.append({'price': entry_p, 'lot': lot_k})
                new_order_str = f"Order {next_order_idx+1} ({lot_k:.2f} lot)"
                next_order_idx += 1
            else:
                break

        floating_pnl = 0.0
        if len(orders_placed) > 0:
            floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * contract_size

        # In log nếu có sự kiện khớp lệnh hoặc cứ mỗi 15 phút
        status_str = "Đang chạy"
        
        # Kiểm tra TP
        if len(orders_placed) > 0 and ((direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price)):
            hit_tp = True
            closed = True
            net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * contract_size
            status_str = "🎉 HIT TP!"
            print(f" | {m_idx:<4} | {time_str:<12} | {high_t:<11.2f} | {low_t:<10.2f} | {new_order_str:<13} | {len(orders_placed):<8} | {net_profit:<18,.2f} | {status_str:<12} |")
            break

        if new_order_str != "-" or m_idx % 15 == 0 or m_idx == len(exec_df) - 1:
            print(f" | {m_idx:<4} | {time_str:<12} | {high_t:<11.2f} | {low_t:<10.2f} | {new_order_str:<13} | {len(orders_placed):<8} | {floating_pnl:<18,.2f} | {status_str:<12} |")

    if not closed and len(orders_placed) > 0:
        final_close = exec_df['close'].iloc[-1]
        net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * contract_size

    print(" ---------------------------------------------------------------------------------------------------------")
    print(" 4. TỔNG KẾT BÁO CÁO NGÀY:")
    print(f"    • Số lệnh khớp thực tế:  {len(orders_placed)} lệnh")
    print(f"    • Trạng thái cắn TP:     {hit_tp}")
    print(f"    • Lợi nhuận Net PnL:     {'+' if net_profit>=0 else ''}${net_profit:,.2f}")
    print("=========================================================================================================\n")


if __name__ == '__main__':
    # Chạy mô phỏng 1 ngày mẫu
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_single_day_diagnostic(target)
