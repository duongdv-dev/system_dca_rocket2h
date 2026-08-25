"""
v6_system/strategy_v1.py
------------------------
Engine giả lập chiến lược DCA Mean-Reversion Nâng Cao (Phase 3).
Hỗ trợ tham số Multiplier lũy tiến theo từng tầng DCA.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class StrategyV1:
    """Class đại diện cho chiến lược DCA Nâng Cao hỗ trợ Multiplier."""

    def __init__(
        self,
        step: float = 5.0,              # Khoảng cách giữa các tầng DCA ($)
        max_dca: int = 5,                # Số tầng DCA tối đa
        multiplier: float = 1.0,         # Hệ số nhân khối lượng mỗi tầng
        tp_dollars: Optional[float] = None, # Lợi nhuận giỏ lệnh chốt lời ($)
        lot_base: float = 0.01,          # Khối lượng tầng 1
        spread: float = 0.20,            # Spread ($0.20 = 2 pips XAUUSD)
        contract_size: float = 100.0     # 1.0 lot = 100 oz -> 0.01 lot = 1 oz
    ):
        self.step = step
        self.max_dca = max_dca
        self.multiplier = multiplier
        self.tp_dollars = tp_dollars if tp_dollars is not None else max(2.0, step * 0.5)
        self.lot_base = lot_base
        self.spread = spread
        self.contract_size = contract_size

    def _get_level_lot(self, level: int) -> float:
        """Tính khối lượng lot cho tầng level (1-indexed)."""
        return round(self.lot_base * (self.multiplier ** (level - 1)), 4)

    def run_daily_session(self, df_session: pd.DataFrame) -> Dict[str, Any]:
        """
        Mô phỏng 1 phiên giao dịch 10:00 -> 12:00 VN cho một ngày.
        """
        if df_session.empty:
            return {
                "date": None,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "NO_DATA",
                "max_level": 0,
                "trades_count": 0,
                "max_exposure_lots": 0.0
            }

        df_session = df_session.sort_values("dt_utc").reset_index(drop=True)
        date_str = str(df_session.iloc[0]["date_vn"])
        anchor_price = float(df_session.iloc[0]["open"])

        basket_direction: Optional[str] = None
        positions: List[Dict[str, Any]] = []
        next_sell_level = 1
        next_buy_level = 1

        cycle_finished = False
        exit_reason = "NO_TRADE"
        realized_pnl = 0.0
        max_level_reached = 0

        for idx, row in df_session.iterrows():
            current_open = float(row["open"])
            current_high = float(row["high"])
            current_low = float(row["low"])
            current_close = float(row["close"])
            is_last_candle = (idx == len(df_session) - 1)

            # 1. Kiểm tra kích hoạt tầng 1 nếu chưa có vị thế
            if basket_direction is None:
                trigger_sell_price = anchor_price + next_sell_level * self.step
                trigger_buy_price = anchor_price - next_buy_level * self.step

                if current_high >= trigger_sell_price:
                    basket_direction = "SELL"
                    positions.append({
                        "level": 1,
                        "type": "SELL",
                        "entry_price": trigger_sell_price,
                        "lot": self._get_level_lot(1)
                    })
                    next_sell_level += 1
                    max_level_reached = 1

                elif current_low <= trigger_buy_price:
                    basket_direction = "BUY"
                    positions.append({
                        "level": 1,
                        "type": "BUY",
                        "entry_price": trigger_buy_price,
                        "lot": self._get_level_lot(1)
                    })
                    next_buy_level += 1
                    max_level_reached = 1

            # 2. Nếu đã có giỏ lệnh active -> Kiểm tra nhồi thêm tầng DCA & kiểm tra Chốt lời TP
            if basket_direction is not None and not cycle_finished:
                if basket_direction == "SELL" and next_sell_level <= self.max_dca:
                    target_price = anchor_price + next_sell_level * self.step
                    if current_high >= target_price:
                        positions.append({
                            "level": next_sell_level,
                            "type": "SELL",
                            "entry_price": target_price,
                            "lot": self._get_level_lot(next_sell_level)
                        })
                        max_level_reached = next_sell_level
                        next_sell_level += 1

                elif basket_direction == "BUY" and next_buy_level <= self.max_dca:
                    target_price = anchor_price - next_buy_level * self.step
                    if current_low <= target_price:
                        positions.append({
                            "level": next_buy_level,
                            "type": "BUY",
                            "entry_price": target_price,
                            "lot": self._get_level_lot(next_buy_level)
                        })
                        max_level_reached = next_buy_level
                        next_buy_level += 1

                # Kiểm tra TP Basket Profit
                best_price_for_tp = current_low if basket_direction == "SELL" else current_high
                floating_pnl_best = self._calculate_basket_pnl(positions, best_price_for_tp)
                floating_pnl_close = self._calculate_basket_pnl(positions, current_close)

                if floating_pnl_best >= self.tp_dollars:
                    realized_pnl = self.tp_dollars
                    exit_reason = "TP_HIT"
                    cycle_finished = True
                    break

                # Force Close tại 12:00:00 VN
                if is_last_candle:
                    realized_pnl = floating_pnl_close
                    exit_reason = "FORCE_CLOSE_1200"
                    cycle_finished = True
                    break

        total_exposure = sum(p["lot"] for p in positions)

        return {
            "date": date_str,
            "traded": len(positions) > 0,
            "direction": basket_direction,
            "pnl": realized_pnl,
            "exit_reason": exit_reason,
            "max_level": max_level_reached,
            "trades_count": len(positions),
            "max_exposure_lots": round(total_exposure, 4),
            "anchor_price": anchor_price
        }

    def _calculate_basket_pnl(self, positions: List[Dict[str, Any]], current_price: float) -> float:
        """Tính PnL giỏ lệnh động hỗ trợ Lot size khác nhau theo từng tầng."""
        total_pnl = 0.0
        for pos in positions:
            entry = pos["entry_price"]
            pos_type = pos["type"]
            lot = pos["lot"]
            unit_size = lot * self.contract_size

            if pos_type == "SELL":
                pnl = (entry - (current_price + self.spread)) * unit_size
            else: # BUY
                pnl = (current_price - (entry + self.spread)) * unit_size

            total_pnl += pnl

        return total_pnl
