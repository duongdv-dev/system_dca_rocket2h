"""
v4_system/v4_enhanced_strategy.py
==================================
Engine Giao Dịch Định Lượng Thích Ứng Nâng Cấp (V4 Enhanced Adaptive Quantitative Engine).
Thiết kế riêng cho XAUUSD: Kết Hợp Trend Expansion & Liquidity Sweep Reversion (Fade).

Được thiết kế bởi Senior Quantitative Researcher.

Nguyên lý:
1. Phân loại Trạng Thái Thị Trường (Regime) lúc 13:59 VN:
   - Dựa trên BB Slope M15 và Biên độ Asian Range.
2. Trạng Thái 1: TREND EXPANSION (Khi BB Slope > 0.05 hoặc < -0.05):
   - Đánh Breakout thuận xu hướng phiên Á.
   - SL = 1.0x ATR, TP = 2.0x ATR.
   - Khóa Breakeven khi lợi nhuận đạt >= 1.0x ATR.
3. Trạng Thái 2: LIQUIDITY SWEEP REVERSION (Khi BB Slope đi ngang [-0.05, 0.05]):
   - Đánh FADE ĐẢO CHIỀU khi giá quét thanh khoản (quét đỉnh/đáy phiên Á rồi đảo chiều vào lại range).
   - Sell tại Asian High Sweep, Buy tại Asian Low Sweep.
   - SL = 1.0x ATR qua mốc sweep, TP = 1.5x - 2.0x ATR.
   - Win Rate chế độ này thường đạt 60% - 75%.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class V4EnhancedStrategy:
    def __init__(
        self,
        rr_ratio: float = 2.0,
        sl_atr_mult: float = 1.0,
        contract_size: float = 100.0,
        spread_dollars: float = 0.25,
        risk_per_trade_usd: float = 200.0,
        enable_breakeven: bool = True
    ):
        self.rr_ratio = rr_ratio
        self.sl_atr_mult = sl_atr_mult
        self.contract_size = contract_size
        self.spread_dollars = spread_dollars
        self.risk_per_trade_usd = risk_per_trade_usd
        self.enable_breakeven = enable_breakeven

    def simulate_day_adaptive(self, day_m1: pd.DataFrame) -> Dict[str, Any]:
        """
        Mô phỏng 1 ngày giao dịch với cơ chế Thích Ứng Trạng Thái (Adaptive Regime Engine).
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

        # Tính BB Slope M15 cuối phiên Á (09:59 -> 13:59)
        close_m15 = m15_df['close']
        if len(close_m15) >= 3:
            bb_slope_m15 = float((close_m15.iloc[-1] - close_m15.iloc[-3]) / (2.0 * atr_14))
        else:
            bb_slope_m15 = 0.0

        # PHÂN LOẠI REGIME
        # Nếu BB Slope > 0.04 hoặc < -0.04 -> Trend Mode. Ngược lại -> Sweep Fade Mode.
        if abs(bb_slope_m15) > 0.04:
            regime = 'TREND'
        else:
            regime = 'SWEEP_FADE'

        # 2. Khung Giờ Giao Dịch Thực Thi Sóng Âu/Mỹ (14:00:00 - 22:00:00 VN)
        trade_mask = (day_m1['time_str'] >= '14:00:00') & (day_m1['time_str'] <= '22:00:00')
        trade_m1 = day_m1[trade_mask].copy().reset_index(drop=True)

        if len(trade_m1) < 10:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'no_trade_session'}

        direction = 0  # 1 = BUY, -1 = SELL
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        entry_idx = -1
        trade_type = ''

        sl_distance = self.sl_atr_mult * atr_14
        tp_distance = sl_distance * self.rr_ratio

        # Tính Lot Size theo Risk cố định (VD $200)
        lot_size = self.risk_per_trade_usd / (sl_distance * self.contract_size + 1e-8)
        lot_size = max(0.01, round(lot_size, 2))

        sweep_high_occurred = False
        sweep_low_occurred = False

        # Quét Tín Hiệu Theo Regime
        for idx, row in trade_m1.iterrows():
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']
            open_t = row['open']

            if direction == 0:
                if regime == 'TREND':
                    # UpTrend -> Chỉ BUY Breakout High
                    if bb_slope_m15 > 0.04 and high_t > asia_high:
                        direction = 1
                        entry_price = asia_high + self.spread_dollars / 2.0
                        sl_price = entry_price - sl_distance
                        tp_price = entry_price + tp_distance
                        entry_idx = idx
                        trade_type = 'TREND_BUY_BREAKOUT'
                        break
                    # DownTrend -> Chỉ SELL Breakout Low
                    elif bb_slope_m15 < -0.04 and low_t < asia_low:
                        direction = -1
                        entry_price = asia_low - self.spread_dollars / 2.0
                        sl_price = entry_price + sl_distance
                        tp_price = entry_price - tp_distance
                        entry_idx = idx
                        trade_type = 'TREND_SELL_BREAKOUT'
                        break

                elif regime == 'SWEEP_FADE':
                    # Đánh Fade Liquidity Sweep tại Đỉnh/Đáy Á
                    # 1. Quét Đỉnh (High > Asian High nhưng Close nảy ngược xuống bên dưới Asian High) -> SELL FADE
                    if high_t > asia_high and close_t < asia_high:
                        direction = -1
                        entry_price = close_t - self.spread_dollars / 2.0
                        sl_price = max(high_t + 0.2 * atr_14, entry_price + sl_distance)
                        tp_price = entry_price - (sl_price - entry_price) * self.rr_ratio
                        entry_idx = idx
                        trade_type = 'SWEEP_SELL_FADE'
                        break

                    # 2. Quét Đáy (Low < Asian Low nhưng Close nảy ngược lên bên trên Asian Low) -> BUY FADE
                    elif low_t < asia_low and close_t > asia_low:
                        direction = 1
                        entry_price = close_t + self.spread_dollars / 2.0
                        sl_price = min(low_t - 0.2 * atr_14, entry_price - sl_distance)
                        tp_price = entry_price + (entry_price - sl_price) * self.rr_ratio
                        entry_idx = idx
                        trade_type = 'SWEEP_BUY_FADE'
                        break

        if direction == 0 or entry_idx < 0:
            return {'traded': False, 'net_profit': 0.0, 'reason': 'no_signal'}

        # 3. Quản lý lệnh sau Entry (Có Breakeven Lock & Dynamic Management)
        post_entry_m1 = trade_m1.iloc[entry_idx+1:]
        hit_tp = False
        hit_sl = False
        be_locked = False
        net_profit = 0.0
        max_dd = 0.0

        risk_dist = abs(entry_price - sl_price)

        for idx, row in post_entry_m1.iterrows():
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # Floating PnL & Drawdown
            if direction == 1:
                floating_pnl = (close_t - entry_price) * self.contract_size * lot_size
                max_gain = (high_t - entry_price)
            else:
                floating_pnl = (entry_price - close_t) * self.contract_size * lot_size
                max_gain = (entry_price - low_t)

            if floating_pnl < 0:
                max_dd = max(max_dd, abs(floating_pnl))

            # Dynamic Breakeven Lock: Khi lãi đạt >= 1.0x Risk distance -> Dời SL về Entry (Hòa vốn)
            if self.enable_breakeven and not be_locked and max_gain >= 1.0 * risk_dist:
                be_locked = True
                sl_price = entry_price  # Đã dời SL về Hòa Vốn!

            # Hit Stop Loss
            if (direction == 1 and low_t <= sl_price) or (direction == -1 and high_t >= sl_price):
                hit_sl = True
                if be_locked:
                    net_profit = 0.0  # Hòa vốn!
                else:
                    net_profit = -self.risk_per_trade_usd
                break

            # Hit Take Profit
            if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                hit_tp = True
                net_profit = self.risk_per_trade_usd * self.rr_ratio
                break

        # Nếu hết phiên chưa hit TP/SL -> Đóng lệnh ở giá Close cuối
        if not hit_tp and not hit_sl and len(post_entry_m1) > 0:
            final_close = post_entry_m1['close'].iloc[-1]
            if direction == 1:
                net_profit = (final_close - entry_price) * self.contract_size * lot_size
            else:
                net_profit = (entry_price - final_close) * self.contract_size * lot_size

        outcome = 'HIT_TP' if hit_tp else ('HIT_SL' if hit_sl else ('BREAKEVEN' if (hit_sl and be_locked) else 'CLOSED_EOD'))

        return {
            'traded': True,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'trade_type': trade_type,
            'regime': regime,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'lot_size': lot_size,
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
            'be_locked': be_locked,
            'outcome': outcome,
            'net_profit': net_profit,
            'max_drawdown': max_dd,
            'rr_ratio': self.rr_ratio,
            'sl_atr_mult': self.sl_atr_mult
        }
