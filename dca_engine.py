import pandas as pd
import numpy as np
from typing import Dict, Any

class DCAEngine:
    """
    Engine giao dịch DCA Rocket 2h Tăng Trưởng Cao (High-Compound Yield Engine).
    Thiết kế riêng cho kỳ vọng lợi nhuận lớn (+100% đến +300%+ sau 2 năm)
    với cơ chế Dynamic Compounding Lot Sizing & High-Reward Breakeven Retracement.
    """
    def __init__(self, account_balance: float = 1000.0, base_risk_pct: float = 3.0):
        self.account_balance = account_balance
        self.base_risk_pct = base_risk_pct

    def simulate_day(self, 
                     trade_candles: pd.DataFrame, 
                     asian_atr_m15: float, 
                     asian_return_pct: float,
                     preset: Dict[str, Any],
                     current_equity: float = 1000.0) -> Dict[str, Any]:
        """
        Mô phỏng chuỗi DCA 2 tiếng cho 1 ngày với Lợi Nhuận Tăng Trưởng Cao (Compounding Equity).
        """
        if trade_candles.empty:
            return {'profit_usd': 0.0, 'max_drawdown_usd': 0.0, 'max_drawdown_pct': 0.0, 'orders_opened': 0, 'status': 'NO_DATA'}

        anchor_price = trade_candles.iloc[0]['open']
        
        # 1. Tính toán Khối lượng Lệnh Compounding theo Equity hiện tại
        # Với $1,000 balance -> Base Lot = 0.08 - 0.12 lot
        lot_scale = preset.get('lot_scale_mult', 1.5)
        base_lot = max(0.02, round((current_equity * 0.00008) * lot_scale, 2))
        
        # 2. Quyết định hướng giao dịch
        entry_rule = preset.get('entry_rule', 'REVERSAL_TO_10H')
        if entry_rule == 'ALWAYS_BUY':
            is_buy = True
        elif entry_rule == 'ALWAYS_SELL':
            is_buy = False
        elif entry_rule == 'FOLLOW_ASIAN_TREND':
            is_buy = (asian_return_pct >= 0)
        elif entry_rule == 'REVERSAL_TO_10H':
            is_buy = (asian_return_pct < 0)
        else:
            is_buy = True

        # 3. Tính toán các khoảng cách Động theo ATR
        step_atr_mult = preset.get('step_atr_mult', 1.2)
        base_step_dist = max(asian_atr_m15 * step_atr_mult, 1.0)  # Step tối thiểu $1.0 USD (10 pips)
        
        tp_atr_mult = preset.get('tp_atr_mult', 1.5)
        single_order_tp_dist = max(asian_atr_m15 * tp_atr_mult, 2.5)  # TP tối thiểu $2.5 USD cho lệnh đơn

        step_multiplier = preset.get('step_multiplier', 1.2)
        lot_multiplier = preset.get('lot_multiplier', 1.5)  # Martingale lot multiplier 1.5x
        max_dca_orders = preset.get('max_dca_orders', 5)
        max_loss_usd = preset.get('max_loss_usd', current_equity * 0.15)  # 15% Max Loss per sequence

        open_positions = []
        
        # Mở lệnh 1 lúc 10:00 AM Open
        open_positions.append({
            'price': anchor_price,
            'lot': base_lot
        })
        
        max_drawdown_usd = 0.0
        realized_pnl = 0.0
        status = 'TIMEOUT'
        last_entry_price = anchor_price

        for idx, row in trade_candles.iterrows():
            high_price = row['high']
            low_price = row['low']

            # Tính Floating PnL xấu nhất và tốt nhất trong nến M1 này
            float_pnl_worst = 0.0
            float_pnl_best = 0.0

            total_lots = sum(p['lot'] for p in open_positions)
            weighted_entry = sum(p['price'] * p['lot'] for p in open_positions) / total_lots

            for pos in open_positions:
                pos_lot = pos['lot']
                pos_price = pos['price']
                
                if is_buy:
                    worst_change = low_price - pos_price
                    best_change = high_price - pos_price
                else:
                    worst_change = pos_price - high_price
                    best_change = pos_price - low_price

                float_pnl_worst += worst_change * pos_lot * 100.0
                float_pnl_best += best_change * pos_lot * 100.0

            # Cập nhật Max Drawdown
            if float_pnl_worst < 0:
                drawdown_now = abs(float_pnl_worst)
                if drawdown_now > max_drawdown_usd:
                    max_drawdown_usd = drawdown_now

            # A. Kiểm tra Cắt Lỗ Khẩn Cấp (Emergency SL)
            if float_pnl_worst <= -max_loss_usd:
                realized_pnl = -max_loss_usd
                status = 'SL'
                break

            # B. Kiểm tra Chốt Lời (Take Profit)
            if len(open_positions) == 1:
                # Lệnh đơn: Ăn trọn sóng theo TP ATR (VD: $3.0 - $6.0 USD -> +$30 - +$60 USD PnL)
                if is_buy and high_price >= anchor_price + single_order_tp_dist:
                    realized_pnl = single_order_tp_dist * base_lot * 100.0
                    status = 'TP'
                    break
                elif not is_buy and low_price <= anchor_price - single_order_tp_dist:
                    realized_pnl = single_order_tp_dist * base_lot * 100.0
                    status = 'TP'
                    break
            else:
                # Chuỗi DCA: Chốt lời tại Giá Trung Bình + Offset (0.5 * ATR)
                # Với tổng volume nhồi lớn (0.08 + 0.12 + 0.18 = 0.38 lot), PnL chốt lời thu về +$80 - +$180 USD!
                tp_offset = max(asian_atr_m15 * 0.5, 0.8)
                target_tp_price = (weighted_entry + tp_offset) if is_buy else (weighted_entry - tp_offset)

                if is_buy and high_price >= target_tp_price:
                    realized_pnl = sum((target_tp_price - pos['price']) * pos['lot'] * 100.0 for pos in open_positions)
                    status = 'TP'
                    break
                elif not is_buy and low_price <= target_tp_price:
                    realized_pnl = sum((pos['price'] - target_tp_price) * pos['lot'] * 100.0 for pos in open_positions)
                    status = 'TP'
                    break

            # C. Kiểm tra Nhồi lệnh DCA tiếp theo
            if len(open_positions) < max_dca_orders:
                order_count = len(open_positions)
                current_step_dist = base_step_dist * (step_multiplier ** (order_count - 1))
                current_lot = round(base_lot * (lot_multiplier ** order_count), 2)

                if is_buy:
                    target_next_entry = last_entry_price - current_step_dist
                    if low_price <= target_next_entry:
                        open_positions.append({'price': target_next_entry, 'lot': current_lot})
                        last_entry_price = target_next_entry
                else:
                    target_next_entry = last_entry_price + current_step_dist
                    if high_price >= target_next_entry:
                        open_positions.append({'price': target_next_entry, 'lot': current_lot})
                        last_entry_price = target_next_entry

        # Timeout tại nến cuối cùng
        if status == 'TIMEOUT':
            last_close = trade_candles.iloc[-1]['close']
            if is_buy:
                realized_pnl = sum((last_close - pos['price']) * pos['lot'] * 100.0 for pos in open_positions)
            else:
                realized_pnl = sum((pos['price'] - last_close) * pos['lot'] * 100.0 for pos in open_positions)

        max_dd_pct = (max_drawdown_usd / current_equity) * 100.0

        return {
            'profit_usd': round(realized_pnl, 2),
            'max_drawdown_usd': round(max_drawdown_usd, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'orders_opened': len(open_positions),
            'status': status
        }
