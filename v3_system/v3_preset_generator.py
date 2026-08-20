"""
v3_system/v3_preset_generator.py
================================
Engine Sinh Preset & Mô Phỏng Thực Thi Linh Hoạt Theo Hướng Nảy Giá Thực Tế (Dynamic Price-Stretch Trigger).
Được thiết kế bởi Senior Quantitative Researcher.

Cải Tiến Hướng Giao Dịch (Dynamic Execution Direction):
1. PRESET CHỈ CHỨA CÁC THAM SỐ LƯỚI: [step_0_ratio, step_exp, max_orders, multiplier].
2. HƯỚNG BUY / SELL KHÔNG BỊ ÉP CỨNG BỞI VWAP 09:59!
3. Hướng được quyết định linh hoạt theo diễn biến giá thực tế sau 10:00 AM:
   - Nếu giá nảy GIẢM xuống dưới Open 10:00 đúng Step_0 -> Kích hoạt lệnh BUY (Kỳ vọng hồi lên Open 10:00).
   - Nếu giá nảy TĂNG lên trên Open 10:00 đúng Step_0 -> Kích hoạt lệnh SELL (Kỳ vọng hồi xuống Open 10:00).
   - Hoặc thuận xu hướng phiên sáng (Uptrend chỉ Buy Dip, Downtrend chỉ Sell Rally).
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class V3PresetGenerator:
    def __init__(self, contract_size: float = 100.0, spread_dollars: float = 0.25):
        self.contract_size = contract_size
        self.spread_dollars = spread_dollars

    @staticmethod
    def generate_540_candidate_presets() -> List[Dict[str, float]]:
        """
        Tạo không gian 192 candidate presets an toàn.
        """
        step_0_ratios = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4]
        step_exps = [1.1, 1.2, 1.3]
        max_orders_list = [3, 4]
        multipliers = [1.0, 1.05, 1.10, 1.15]

        presets = []
        for s0, se, mo, mult in itertools.product(step_0_ratios, step_exps, max_orders_list, multipliers):
            presets.append({
                'step_0_ratio': s0,
                'step_exp': se,
                'max_orders': mo,
                'multiplier': mult
            })
        return presets

    def simulate_day(
        self,
        exec_m1: pd.DataFrame,
        atr_14: float,
        close_0959: float,
        daily_vwap: float,
        params: Dict[str, float],
        bb_slope_m15: float = 0.0,
        base_lot: float = 0.1
    ) -> Dict[str, Any]:
        """
        Mô phỏng thực thi nến M1 (10:00 - 12:00) với Hướng giao dịch linh hoạt theo mốc nảy giá thực tế.
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']

        price_1000 = exec_m1['open'].iloc[0]

        # 2 Ngưỡng kích hoạt Order 1 linh hoạt:
        # Ngưỡng BUY : price_1000 - step_0 (Chờ giá giảm chạm ngưỡng -> Mở BUY hồi về Open 10:00)
        # Ngưỡng SELL: price_1000 + step_0 (Chờ giá tăng chạm ngưỡng -> Mở SELL hồi về Open 10:00)
        buy_trig_1 = price_1000 - step_0
        sell_trig_1 = price_1000 + step_0

        # Lọc bảo vệ xu hướng sáng:
        # Nếu dải M15 đang dốc lên mạnh (bb_slope > 0.08) -> Cấm mở SELL (Chỉ cho phép BUY khi giá dip)
        # Nếu dải M15 đang dốc xuống mạnh (bb_slope < -0.08) -> Cấm mở BUY (Chỉ cho phép SELL khi giá rally)
        allow_buy = (bb_slope_m15 >= -0.12)
        allow_sell = (bb_slope_m15 <= 0.12)

        direction = 0  # 1: BUY, -1: SELL
        orders_placed = []
        trigger_prices = []
        sl_price = 0.0
        tp_price = 0.0

        closed = False
        hit_tp = False
        hit_sl = False
        hit_minute = -1
        net_profit = 0.0
        max_drawdown = 0.0

        for m_idx, (t, row) in enumerate(exec_m1.iterrows()):
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # A. Nếu chưa có lệnh nào được mở -> Xác định hướng dựa trên mốc giá chạm FIRST!
            if direction == 0:
                buy_triggered = allow_buy and (low_t <= buy_trig_1)
                sell_triggered = allow_sell and (high_t >= sell_trig_1)

                if buy_triggered:
                    direction = 1
                    tp_price = price_1000 + self.spread_dollars
                    entry_p = buy_trig_1 + self.spread_dollars / 2.0
                    orders_placed.append({'price': entry_p, 'lot': base_lot})

                    # Tính danh sách trigger cho các lệnh BUY tiếp theo
                    curr_p = buy_trig_1
                    trigger_prices = [curr_p]
                    for i in range(1, max_orders):
                        curr_p = curr_p - step_0 * (step_exp ** i)
                        trigger_prices.append(curr_p)
                    sl_price = trigger_prices[-1] - 1.0 * atr_14

                elif sell_triggered:
                    direction = -1
                    tp_price = price_1000 - self.spread_dollars
                    entry_p = sell_trig_1 - self.spread_dollars / 2.0
                    orders_placed.append({'price': entry_p, 'lot': base_lot})

                    # Tính danh sách trigger cho các lệnh SELL tiếp theo
                    curr_p = sell_trig_1
                    trigger_prices = [curr_p]
                    for i in range(1, max_orders):
                        curr_p = curr_p + step_0 * (step_exp ** i)
                        trigger_prices.append(curr_p)
                    sl_price = trigger_prices[-1] + 1.0 * atr_14

            # B. Khi đã xác định được hướng và có ít nhất 1 lệnh
            elif len(orders_placed) > 0:
                # Kích hoạt các tầng lệnh nhồi tiếp theo
                next_idx = len(orders_placed)
                while next_idx < len(trigger_prices):
                    trig_p = trigger_prices[next_idx]
                    trig_hit = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)
                    if trig_hit:
                        entry_p = (trig_p + self.spread_dollars / 2.0) if direction == 1 else (trig_p - self.spread_dollars / 2.0)
                        lot_k = base_lot * (multiplier ** next_idx)
                        orders_placed.append({'price': entry_p, 'lot': lot_k})
                        next_idx += 1
                    else:
                        break

                floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size
                if floating_pnl < 0:
                    max_drawdown = max(max_drawdown, abs(floating_pnl))

                # Hard Stop Loss
                if (direction == 1 and low_t <= sl_price) or (direction == -1 and high_t >= sl_price):
                    hit_sl = True
                    closed = True
                    hit_minute = m_idx
                    net_profit = sum(o['lot'] * (sl_price - o['price'] if direction == 1 else o['price'] - sl_price) for o in orders_placed) * self.contract_size
                    break

                # Take Profit tại Price 10:00 AM
                if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                    hit_tp = True
                    closed = True
                    hit_minute = m_idx
                    net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                    break

        unclosed_at_12 = False
        if len(orders_placed) > 0 and not closed:
            unclosed_at_12 = True
            final_close = exec_m1['close'].iloc[-1]
            net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

        total_volume = sum(o['lot'] for o in orders_placed) if len(orders_placed) > 0 else 0.1
        pnl_points = net_profit / (self.contract_size * total_volume)
        dd_points = max_drawdown / (self.contract_size * total_volume)

        pnl_atr = pnl_points / (atr_14 + 1e-8)
        dd_atr = dd_points / (atr_14 + 1e-8)

        penalty_atr = 25.0 if unclosed_at_12 else (35.0 if hit_sl else 0.0)
        fitness_score = pnl_atr - (2.5 * dd_atr) - penalty_atr

        return {
            'direction': 'BUY' if direction == 1 else ('SELL' if direction == -1 else 'NONE'),
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'pnl_atr': pnl_atr,
            'dd_atr': dd_atr,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'hit_minute': hit_minute,
            'unclosed_at_12': unclosed_at_12,
            'fitness_score': fitness_score,
            'num_orders': len(orders_placed)
        }


if __name__ == '__main__':
    pass
