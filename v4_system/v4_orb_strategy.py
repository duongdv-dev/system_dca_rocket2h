"""
v4_system/v4_orb_strategy.py
============================
Engine Giao Dịch Định Lượng 1 Lệnh Duy Nhất (Non-DCA Single Position Quant Strategy).
Chuyên biệt cho XAUUSD: Asian Range Breakout (ORB) & Liquidity Sweep System.

Được thiết kế bởi Senior Quantitative Researcher.

Nguyên lý:
1. Xác định Asian Range (06:00 - 13:59 VN).
2. Khi phiên Âu (14:00 - 18:00 VN) hoặc phiên Mỹ (19:30 - 23:00 VN) mở cửa:
   - Nếu giá phá vỡ Asian High -> Mở 1 LỆNH BUY DUY NHẤT.
   - Nếu giá phá vỡ Asian Low  -> Mở 1 LỆNH SELL DUY NHẤT.
3. Tỷ lệ Risk:Reward (R:R) cố định 1:2.0 hoặc 1:3.0 (SL = 1.0x ATR, TP = 2.0x hoặc 3.0x ATR).
4. KHÔNG DCA, KHÔNG NHỒI LỆNH, KHÔNG MARTINGALE.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class V4ORBStrategy:
    def __init__(
        self,
        rr_ratio: float = 2.0,
        sl_atr_mult: float = 1.0,
        contract_size: float = 100.0,
        spread_dollars: float = 0.25,
        risk_per_trade_usd: float = 200.0
    ):
        self.rr_ratio = rr_ratio
        self.sl_atr_mult = sl_atr_mult
        self.contract_size = contract_size
        self.spread_dollars = spread_dollars
        self.risk_per_trade_usd = risk_per_trade_usd

    def simulate_day_single_trade(
        self,
        day_m1: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Mô phỏng 1 ngày giao dịch với chiến lược 1 Lệnh Duy Nhất (Single Trade ORB).
        """
        if len(day_m1) < 100:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'insufficient_data'}

        day_m1 = day_m1.copy()
        day_m1['time_str'] = day_m1['datetime'].dt.strftime('%H:%M:%S')

        # 1. Asian Range (06:00:00 - 13:59:59 VN)
        asia_mask = (day_m1['time_str'] >= '06:00:00') & (day_m1['time_str'] <= '13:59:59')
        asia_m1 = day_m1[asia_mask]

        if len(asia_m1) < 30:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'thin_asian_session'}

        asia_high = asia_m1['high'].max()
        asia_low = asia_m1['low'].min()
        asia_mid = (asia_high + asia_low) / 2.0

        # Tính ATR(14) M15 từ nến Á
        asia_m1['m15_group'] = asia_m1['datetime'].dt.floor('15min')
        m15_df = asia_m1.groupby('m15_group').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).reset_index()

        tr = pd.concat([
            m15_df['high'] - m15_df['low'],
            (m15_df['high'] - m15_df['close'].shift(1)).abs(),
            (m15_df['low'] - m15_df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        atr_14 = max(1.0, atr_14)

        # 2. Phiên Giao Dịch Sóng Mạnh (14:00:00 - 23:00:00 VN)
        trade_mask = (day_m1['time_str'] >= '14:00:00') & (day_m1['time_str'] <= '23:00:00')
        trade_m1 = day_m1[trade_mask].copy().reset_index(drop=True)

        if len(trade_m1) < 10:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'no_trade_session'}

        direction = 0  # 1 = BUY, -1 = SELL
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        entry_idx = -1

        sl_distance = self.sl_atr_mult * atr_14
        tp_distance = sl_distance * self.rr_ratio

        # Tính Lot Size theo Risk cố định (VD $200)
        lot_size = self.risk_per_trade_usd / (sl_distance * self.contract_size + 1e-8)
        lot_size = max(0.01, round(lot_size, 2))

        # Tìm Tín Hiệu Breakout 1 Lệnh Duy Nhất
        for idx, row in trade_m1.iterrows():
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            if direction == 0:
                # Breakout High -> BUY
                if high_t > asia_high:
                    direction = 1
                    entry_price = asia_high + self.spread_dollars / 2.0
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + tp_distance
                    entry_idx = idx
                    break
                # Breakout Low -> SELL
                elif low_t < asia_low:
                    direction = -1
                    entry_price = asia_low - self.spread_dollars / 2.0
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - tp_distance
                    entry_idx = idx
                    break

        if direction == 0 or entry_idx < 0:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'no_breakout_signal'}

        # 3. Theo dõi 1 Lệnh Duy Nhất từ thời điểm Entry đến hết ngày
        post_entry_m1 = trade_m1.iloc[entry_idx+1:]
        hit_tp = False
        hit_sl = False
        net_profit = 0.0
        max_dd = 0.0

        for idx, row in post_entry_m1.iterrows():
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # Floating PnL & Drawdown
            if direction == 1:
                floating_pnl = (close_t - entry_price) * self.contract_size * lot_size
            else:
                floating_pnl = (entry_price - close_t) * self.contract_size * lot_size

            if floating_pnl < 0:
                max_dd = max(max_dd, abs(floating_pnl))

            # Hit Stop Loss
            if (direction == 1 and low_t <= sl_price) or (direction == -1 and high_t >= sl_price):
                hit_sl = True
                net_profit = -self.risk_per_trade_usd
                break

            # Hit Take Profit
            if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                hit_tp = True
                net_profit = self.risk_per_trade_usd * self.rr_ratio
                break

        # Nếu hết ngày chưa cắn TP/SL -> Đóng lệnh ở giá Close phiên
        if not hit_tp and not hit_sl and len(post_entry_m1) > 0:
            final_close = post_entry_m1['close'].iloc[-1]
            if direction == 1:
                net_profit = (final_close - entry_price) * self.contract_size * lot_size
            else:
                net_profit = (entry_price - final_close) * self.contract_size * lot_size

        outcome = 'HIT_TP' if hit_tp else ('HIT_SL' if hit_sl else 'CLOSED_EOD')

        return {
            'traded': True,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'lot_size': lot_size,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'outcome': outcome,
            'net_profit': net_profit,
            'max_drawdown': max_dd,
            'rr_ratio': self.rr_ratio,
            'sl_atr_mult': self.sl_atr_mult
        }


if __name__ == '__main__':
    pass
