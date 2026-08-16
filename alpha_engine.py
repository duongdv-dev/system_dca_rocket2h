import pandas as pd
import numpy as np
from typing import Dict, Any

class AlphaEngine:
    """
    Engine giao dịch High-Alpha cho Rocket 2h (10:00 - 12:00 VN Time).
    Quản lý vị thế với Tỷ lệ Risk/Reward Dương (1:2 đến 1:3).
    """
    def __init__(self, account_balance: float = 1000.0, risk_pct_per_trade: float = 1.5):
        self.account_balance = account_balance
        self.risk_pct_per_trade = risk_pct_per_trade

    def evaluate_setup_type(self, row_feature: pd.Series) -> str:
        """
        Xác định Setup giao dịch tiềm năng dựa trên đặc trưng phiên Á (08:00 - 10:00 AM VN).
        - BREAKOUT: Asian Range hẹp (Asian_Range / ATR <= 2.8)
        - REVERSAL: Asian Range kéo dãn quá mức (Asian_Range / ATR > 3.8)
        - NO_TRADE: Không rõ ràng
        """
        range_ratio = row_feature.get('asian_range_atr_ratio', 0.0)
        
        if range_ratio <= 2.8:
            return 'BREAKOUT'
        elif range_ratio > 3.8:
            return 'REVERSAL'
        else:
            return 'NO_TRADE'

    def simulate_day_setup(self, 
                           trade_candles: pd.DataFrame, 
                           asian_atr_m15: float, 
                           asian_return_pct: float,
                           setup_type: str,
                           tp_atr_mult: float = 3.0,
                           sl_atr_mult: float = 1.2) -> Dict[str, Any]:
        """
        Mô phỏng 1 ngày giao dịch theo Setup với Risk/Reward Dương.
        """
        if trade_candles.empty or setup_type == 'NO_TRADE':
            return {'profit_usd': 0.0, 'max_drawdown_usd': 0.0, 'status': 'NO_TRADE', 'is_win': False}

        anchor_price = trade_candles.iloc[0]['open']
        
        # Xác định hướng vào lệnh
        if setup_type == 'BREAKOUT':
            # Theo xu hướng phiên Á (Follow Asian Trend)
            is_buy = (asian_return_pct >= 0)
        elif setup_type == 'REVERSAL':
            # Ngược xu hướng phiên Á (Counter Asian Trend)
            is_buy = (asian_return_pct < 0)
        else:
            return {'profit_usd': 0.0, 'max_drawdown_usd': 0.0, 'status': 'NO_TRADE', 'is_win': False}

        # Khoảng cách SL và TP (Tính theo ATR)
        sl_dist = max(asian_atr_m15 * sl_atr_mult, 1.5)  # SL tối thiểu 15 pips ($1.5)
        tp_dist = max(asian_atr_m15 * tp_atr_mult, 3.5)  # TP tối thiểu 35 pips ($3.5) -> R:R >= 1:2.3

        # Tính toán Lot Size theo % Risk cố định (Mỗi lệnh rủi ro 1.5% = $15 cho $1000 account)
        risk_usd = self.account_balance * (self.risk_pct_per_trade / 100.0)
        # 1 lot = $100 per 1.0 USD move -> lot = risk_usd / (sl_dist * 100)
        lot_size = max(0.01, round(risk_usd / (sl_dist * 100.0), 2))

        # Điểm TP và SL
        if is_buy:
            tp_price = anchor_price + tp_dist
            sl_price = anchor_price - sl_dist
        else:
            tp_price = anchor_price - tp_dist
            sl_price = anchor_price + sl_dist

        max_drawdown_usd = 0.0
        realized_pnl = 0.0
        status = 'TIMEOUT'

        for idx, row in trade_candles.iterrows():
            high_price = row['high']
            low_price = row['low']

            # Tính PnL xấu nhất và tốt nhất trong nến M1
            if is_buy:
                worst_change = low_price - anchor_price
                best_change = high_price - anchor_price
            else:
                worst_change = anchor_price - high_price
                best_change = anchor_price - low_price

            float_pnl_worst = worst_change * lot_size * 100.0

            if float_pnl_worst < 0:
                dd_now = abs(float_pnl_worst)
                if dd_now > max_drawdown_usd:
                    max_drawdown_usd = dd_now

            # A. Kiểm tra Hit SL
            if (is_buy and low_price <= sl_price) or (not is_buy and high_price >= sl_price):
                realized_pnl = -risk_usd
                status = 'SL'
                break

            # B. Kiểm tra Hit TP (Reward = Risk * (tp_dist / sl_dist))
            if (is_buy and high_price >= tp_price) or (not is_buy and low_price <= tp_price):
                reward_usd = risk_usd * (tp_dist / sl_dist)
                realized_pnl = reward_usd
                status = 'TP'
                break

        # Hết 2 tiếng chưa chạm TP hay SL -> Đóng vị thế tại nến cuối cùng
        if status == 'TIMEOUT':
            last_close = trade_candles.iloc[-1]['close']
            if is_buy:
                pnl_dist = last_close - anchor_price
            else:
                pnl_dist = anchor_price - last_close
            realized_pnl = pnl_dist * lot_size * 100.0

        is_win = (realized_pnl > 0)

        return {
            'date': trade_candles.iloc[0]['date'] if 'date' in trade_candles.columns else '',
            'setup_type': setup_type,
            'is_buy': is_buy,
            'lot_size': lot_size,
            'sl_dist': round(sl_dist, 2),
            'tp_dist': round(tp_dist, 2),
            'profit_usd': round(realized_pnl, 2),
            'max_drawdown_usd': round(max_drawdown_usd, 2),
            'status': status,
            'is_win': is_win
        }
