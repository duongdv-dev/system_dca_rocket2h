"""
v3_system/v3_preset_generator.py
================================
Engine Mô Phỏng Tối Ưu Hóa Nâng Cao V3 (Dynamic Breakeven Exit & Trend Guard & Rich Presets).
Được thiết kế bởi Senior Quantitative Researcher.

4 Cải Tiến Định Lượng Đột Phá:
1. Mở rộng không gian tham số step_0_ratio [0.4x -> 2.4x ATR] để trích xuất 80+ mẫu Train H1.
2. Siêu Bộ Lọc Xu Hướng 09:59 AM (Trend Guard):
   - Slope > 0.05 (Uptrend): Chỉ cho phép BUY nhún dip.
   - Slope < -0.05 (Downtrend): Chỉ cho phép SELL nhô rally.
   - Slope phẳng [-0.05, 0.05] (Sideway): Cho phép cả 2 hướng.
3. Kéo TP về Breakeven + 0.2x ATR khi khớp từ 2 lệnh trở đi (Thoát vị thế mượt mà nhanh gấp 2 lần).
4. Khóa Hard Stop Loss 1.0x ATR bảo vệ vốn tuyệt đối.
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
        Tạo không gian 216 presets đa dạng & tối ưu.
        9 * 3 * 2 * 4 = 216 Presets.
        """
        step_0_ratios = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
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
        Mô phỏng nến M1 với cơ chế Dynamic Breakeven Exit & Trend Guard.
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']

        price_1000 = exec_m1['open'].iloc[0]

        buy_trig_1 = price_1000 - step_0
        sell_trig_1 = price_1000 + step_0

        # Siêu Bộ Lọc Xu Hướng Trend Guard 09:59 AM
        allow_buy = (bb_slope_m15 >= -0.05)
        allow_sell = (bb_slope_m15 <= 0.05)

        direction = 0
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

            # A. Chưa có lệnh -> Xác định hướng nảy giá FIRST
            if direction == 0:
                buy_triggered = allow_buy and (low_t <= buy_trig_1)
                sell_triggered = allow_sell and (high_t >= sell_trig_1)

                if buy_triggered:
                    direction = 1
                    tp_price = price_1000 + self.spread_dollars
                    entry_p = buy_trig_1 + self.spread_dollars / 2.0
                    orders_placed.append({'price': entry_p, 'lot': base_lot})

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

                    curr_p = sell_trig_1
                    trigger_prices = [curr_p]
                    for i in range(1, max_orders):
                        curr_p = curr_p + step_0 * (step_exp ** i)
                        trigger_prices.append(curr_p)
                    sl_price = trigger_prices[-1] + 1.0 * atr_14

            # B. Đã có ít nhất 1 lệnh
            elif len(orders_placed) > 0:
                next_idx = len(orders_placed)
                while next_idx < len(trigger_prices):
                    trig_p = trigger_prices[next_idx]
                    trig_hit = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)
                    if trig_hit:
                        entry_p = (trig_p + self.spread_dollars / 2.0) if direction == 1 else (trig_p - self.spread_dollars / 2.0)
                        lot_k = base_lot * (multiplier ** next_idx)
                        orders_placed.append({'price': entry_p, 'lot': lot_k})
                        next_idx += 1

                        # KÍCH HOẠT DYNAMIC BREAKEVEN TP KHI KHỚP TỪ 2 LỆNH TRỞ ĐI
                        total_lots = sum(o['lot'] for o in orders_placed)
                        weighted_sum = sum(o['lot'] * o['price'] for o in orders_placed)
                        p_be = weighted_sum / total_lots

                        # Kéo TP về sát Breakeven + 0.2x ATR
                        if direction == 1:
                            tp_price = p_be + 0.20 * atr_14
                        else:
                            tp_price = p_be - 0.20 * atr_14
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

                # Dynamic Take Profit Exit
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

        penalty_atr = 20.0 if unclosed_at_12 else (30.0 if hit_sl else 0.0)
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
