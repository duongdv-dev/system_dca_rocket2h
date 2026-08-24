"""
v3_system/v3_open1000_dca_engine.py
====================================
Engine Mô Phỏng DCA Chuyên Biệt Chốt Lời Tại Giá Mở Cửa 10:00 AM (Open 10:00 AM Mean Reversion).
Đúng 100% Theo Ý Tưởng Cốt Lõi Của User:
"Chắc chắn giá quay về mở cửa 10h, DCA để tối ưu lợi nhuận".

Được thiết kế bởi Senior Quantitative Researcher.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class V3Open1000DCAEngine:
    def __init__(
        self,
        step_0_ratio: float = 1.0,
        step_exp: float = 1.15,
        max_orders: int = 4,
        multiplier: float = 1.05,
        contract_size: float = 100.0,
        spread_dollars: float = 0.25,
        base_lot: float = 0.10
    ):
        self.step_0_ratio = step_0_ratio
        self.step_exp = step_exp
        self.max_orders = max_orders
        self.multiplier = multiplier
        self.contract_size = contract_size
        self.spread_dollars = spread_dollars
        self.base_lot = base_lot

    def simulate_day_open1000_dca(
        self,
        exec_m1: pd.DataFrame,
        atr_14: float,
        close_0959: float,
        daily_vwap: float,
        bb_slope_m15: float = 0.0
    ) -> Dict[str, Any]:
        """
        Mô phỏng 1 ngày giao dịch với chiến thuật DCA chốt lời tại Giá Mở Cửa 10:00 AM (price_1000).
        """
        if len(exec_m1) < 10:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'insufficient_data'}

        price_1000 = exec_m1['open'].iloc[0]  # Giá Mở Cửa 10:00 AM VN

        step_0 = self.step_0_ratio * atr_14
        step_0 = max(1.5, step_0)  # Đảm bảo tối thiểu 1.5 giá

        # Siêu Bộ Lọc Xu Hướng Trend Guard (Lúc 09:59 AM)
        # UpTrend (Slope > 0.04) -> Ưu tiên BUY khi giá nhún xuống dưới 10h
        # DownTrend (Slope < -0.04) -> Ưu tiên SELL khi giá nhô lên trên 10h
        # Sideway (Slope trong [-0.04, 0.04]) -> Cho phép theo hướng lệch VWAP
        allow_buy = (bb_slope_m15 >= -0.04)
        allow_sell = (bb_slope_m15 <= 0.04)

        default_dir = -1 if close_0959 >= daily_vwap else 1

        direction = 0
        if default_dir == 1 and allow_buy:
            direction = 1
        elif default_dir == -1 and allow_sell:
            direction = -1
        elif allow_buy:
            direction = 1
        elif allow_sell:
            direction = -1
        else:
            direction = default_dir

        # MỤC TIÊU CHỐT LỜI (TP): CỐ ĐỊNH TẠI GIÁ MỞ CỬA 10:00 AM
        if direction == 1:
            tp_price = price_1000 + self.spread_dollars
            buy_trig_1 = price_1000 - step_0
            curr_p = buy_trig_1
            trigger_prices = [curr_p]
            for i in range(1, self.max_orders):
                curr_p = curr_p - step_0 * (self.step_exp ** i)
                trigger_prices.append(curr_p)
            sl_price = trigger_prices[-1] - 1.5 * atr_14
        else:
            tp_price = price_1000 - self.spread_dollars
            sell_trig_1 = price_1000 + step_0
            curr_p = sell_trig_1
            trigger_prices = [curr_p]
            for i in range(1, self.max_orders):
                curr_p = curr_p + step_0 * (self.step_exp ** i)
                trigger_prices.append(curr_p)
            sl_price = trigger_prices[-1] + 1.5 * atr_14

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

            # Khớp các lệnh DCA tiếp theo khi giá lệch xa
            while next_order_idx < len(trigger_prices):
                trig_p = trigger_prices[next_order_idx]
                triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

                if triggered:
                    entry_p = (trig_p + self.spread_dollars / 2.0) if direction == 1 else (trig_p - self.spread_dollars / 2.0)
                    lot_k = self.base_lot * (self.multiplier ** next_order_idx)
                    orders_placed.append({'price': entry_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            # Khi đã có ít nhất 1 lệnh khớp
            if len(orders_placed) > 0:
                floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size

                if floating_pnl < 0:
                    max_drawdown = max(max_drawdown, abs(floating_pnl))

                # Hard Stop Loss bảo vệ vốn
                if (direction == 1 and low_t <= sl_price) or (direction == -1 and high_t >= sl_price):
                    hit_sl = True
                    closed = True
                    hit_minute = m_idx
                    net_profit = sum(o['lot'] * (sl_price - o['price'] if direction == 1 else o['price'] - sl_price) for o in orders_placed) * self.contract_size
                    break

                # TAKE PROFIT CỐ ĐỊNH TẠI GIÁ MỞ CỬA 10:00 AM
                if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                    hit_tp = True
                    closed = True
                    hit_minute = m_idx
                    net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                    break

        # Nếu hết phiên chưa cắn TP 10h hoặc SL -> Đóng lệnh ở giá Close cuối ngày
        unclosed_at_eod = False
        if len(orders_placed) > 0 and not closed:
            unclosed_at_eod = True
            final_close = exec_m1['close'].iloc[-1]
            net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

        outcome = 'HIT_TP_10H' if hit_tp else ('HIT_SL' if hit_sl else ('CLOSED_EOD' if unclosed_at_eod else 'NO_TRADE'))

        return {
            'traded': len(orders_placed) > 0,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'price_1000': price_1000,
            'tp_price_1000': tp_price,
            'sl_price': sl_price,
            'step_0_dollars': step_0,
            'num_orders': len(orders_placed),
            'orders_placed': orders_placed,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'unclosed_at_eod': unclosed_at_eod,
            'outcome': outcome,
            'net_profit': net_profit,
            'max_drawdown': max_drawdown
        }
