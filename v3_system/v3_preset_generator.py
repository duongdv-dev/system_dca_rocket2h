"""
v3_system/v3_preset_generator.py
================================
Engine Sinh 540 Candidate Presets & Mô Phỏng Chấm Điểm Định Lượng (v3 Architecture).
Được thiết kế bởi Senior Quantitative Researcher.
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
        Tạo không gian 540 kịch bản tham số phong phú cho v3.
        9 * 4 * 3 * 5 = 540 Candidate Presets.
        """
        step_0_ratios = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        step_exps = [1.0, 1.1, 1.2, 1.3]
        max_orders_list = [3, 4, 5]
        multipliers = [1.0, 1.1, 1.2, 1.3, 1.4]

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
        base_lot: float = 0.1
    ) -> Dict[str, Any]:
        """
        Mô phỏng nến M1 (10:00 - 12:00) cho 1 kịch bản tham số cụ thể.
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']

        direction = -1 if close_0959 >= daily_vwap else 1
        price_1000 = exec_m1['open'].iloc[0]

        if direction == 1:
            tp_price = price_1000 + self.spread_dollars
        else:
            tp_price = price_1000 - self.spread_dollars

        trigger_prices = []
        curr_p = (price_1000 - step_0) if direction == 1 else (price_1000 + step_0)
        trigger_prices.append(curr_p)

        for i in range(1, max_orders):
            dist_i = step_0 * (step_exp ** i)
            curr_p = curr_p - dist_i if direction == 1 else curr_p + dist_i
            trigger_prices.append(curr_p)

        last_trig = trigger_prices[-1]
        sl_price = (last_trig - 1.5 * atr_14) if direction == 1 else (last_trig + 1.5 * atr_14)

        orders_placed = []
        next_order_idx = 0
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

            while next_order_idx < len(trigger_prices):
                trig_p = trigger_prices[next_order_idx]
                triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

                if triggered:
                    entry_p = (trig_p + self.spread_dollars / 2.0) if direction == 1 else (trig_p - self.spread_dollars / 2.0)
                    lot_k = base_lot * (multiplier ** next_order_idx)
                    orders_placed.append({'price': entry_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            if len(orders_placed) > 0:
                floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size

                if floating_pnl < 0:
                    max_drawdown = max(max_drawdown, abs(floating_pnl))

                if (direction == 1 and low_t <= sl_price) or (direction == -1 and high_t >= sl_price):
                    hit_sl = True
                    closed = True
                    hit_minute = m_idx
                    net_profit = sum(o['lot'] * (sl_price - o['price'] if direction == 1 else o['price'] - sl_price) for o in orders_placed) * self.contract_size
                    break

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
            'direction': 'SELL' if direction == -1 else 'BUY',
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
