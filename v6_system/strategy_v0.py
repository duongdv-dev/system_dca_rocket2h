"""
v6_system/strategy_v0.py
------------------------
Engine giả lập chiến lược DCA Mean-Reversion Baseline V0.
Quy tắc:
1. Tại 10:00 VN: Anchor = Open Price. Max cycle/day = 1.
2. Giá tăng: Anchor + k * Step -> Mở SELL k (Lot = 0.01 * 1.0).
3. Giá giảm: Anchor - k * Step -> Mở BUY k (Lot = 0.01 * 1.0).
4. Chốt lời theo Basket Profit ($).
5. Force Close tại 12:00:00 VN.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class StrategyV0:
    """Class đại diện cho logic chiến lược DCA Baseline V0."""

    def __init__(
        self,
        step: float = 2.0,            # Khoảng cách giữa các tầng DCA ($)
        max_dca: int = 5,              # Số tầng DCA tối đa
        tp_dollars: float = 2.0,       # Lợi nhuận giỏ lệnh chốt lời ($)
        lot_size: float = 0.01,        # Khối lượng cố định mỗi lệnh
        spread: float = 0.20,          # Spread ($0.20 = 2 pips XAUUSD)
        contract_size: float = 100.0   # 1.0 lot = 100 oz -> 0.01 lot = 1 oz
    ):
        self.step = step
        self.max_dca = max_dca
        self.tp_dollars = tp_dollars
        self.lot_size = lot_size
        self.spread = spread
        self.unit_size = lot_size * contract_size  # 0.01 * 100 = 1 oz per trade

    def run_daily_session(self, df_session: pd.DataFrame) -> Dict[str, Any]:
        """
        Mô phỏng 1 phiên giao dịch 10:00 -> 12:00 VN cho một ngày.

        Args:
            df_session: DataFrame nến M1 từ 10:00:00 tới 12:00:00 VN của ngày đó.

        Returns:
            Dict chứa kết quả giao dịch của ngày (PnL, status, max_level_reached, trades_count, exit_reason).
        """
        if df_session.empty:
            return {
                "date": None,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "NO_DATA",
                "max_level": 0,
                "trades_count": 0
            }

        df_session = df_session.sort_values("dt_utc").reset_index(drop=True)
        date_str = str(df_session.iloc[0]["date_vn"])
        anchor_price = float(df_session.iloc[0]["open"])

        # Trạng thái phiên
        basket_direction: Optional[str] = None  # "SELL" hoặc "BUY"
        positions: List[Dict[str, Any]] = []    # Danh sách các lệnh trong giỏ
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

            # Nếu chưa có vị thế nào trong giỏ -> Kiểm tra kích hoạt tầng 1
            if basket_direction is None:
                # Kiểm tra giá tăng chạm Anchor + 1*Step -> Kích hoạt giỏ SELL
                trigger_sell_price = anchor_price + next_sell_level * self.step
                trigger_buy_price = anchor_price - next_buy_level * self.step

                # Ưu tiên mức biến động chạm trước trong nến
                if current_high >= trigger_sell_price:
                    basket_direction = "SELL"
                    positions.append({
                        "level": 1,
                        "type": "SELL",
                        "entry_price": trigger_sell_price,
                        "lot": self.lot_size
                    })
                    next_sell_level += 1
                    max_level_reached = 1

                elif current_low <= trigger_buy_price:
                    basket_direction = "BUY"
                    positions.append({
                        "level": 1,
                        "type": "BUY",
                        "entry_price": trigger_buy_price,
                        "lot": self.lot_size
                    })
                    next_buy_level += 1
                    max_level_reached = 1

            # Nếu đã có giỏ lệnh active -> Kiểm tra nhồi thêm tầng DCA & kiểm tra Chốt lời TP
            if basket_direction is not None and not cycle_finished:
                # 1. Kiểm tra mở thêm tầng DCA nếu giá đi tiếp
                if basket_direction == "SELL" and next_sell_level <= self.max_dca:
                    target_price = anchor_price + next_sell_level * self.step
                    if current_high >= target_price:
                        positions.append({
                            "level": next_sell_level,
                            "type": "SELL",
                            "entry_price": target_price,
                            "lot": self.lot_size
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
                            "lot": self.lot_size
                        })
                        max_level_reached = next_buy_level
                        next_buy_level += 1

                # 2. Kiểm tra Basket Profit TP dựa trên giá tốt nhất trong nến M1 (High/Low/Close)
                # Tính PnL floating giỏ lệnh tại current_close (hoặc giá tối ưu trong nến)
                best_price_for_tp = current_low if basket_direction == "SELL" else current_high
                
                floating_pnl_best = self._calculate_basket_pnl(positions, best_price_for_tp)
                floating_pnl_close = self._calculate_basket_pnl(positions, current_close)

                if floating_pnl_best >= self.tp_dollars:
                    realized_pnl = self.tp_dollars  # Chốt đúng TP target
                    exit_reason = "TP_HIT"
                    cycle_finished = True
                    break

                # 3. Force Close tại 12:00:00 (nến cuối cùng của phiên)
                if is_last_candle:
                    realized_pnl = floating_pnl_close
                    exit_reason = "FORCE_CLOSE_1200"
                    cycle_finished = True
                    break

        return {
            "date": date_str,
            "traded": len(positions) > 0,
            "direction": basket_direction,
            "pnl": realized_pnl,
            "exit_reason": exit_reason,
            "max_level": max_level_reached,
            "trades_count": len(positions),
            "anchor_price": anchor_price
        }

    def _calculate_basket_pnl(self, positions: List[Dict[str, Any]], current_price: float) -> float:
        """
        Tính toán tổng PnL thả nổi của giỏ lệnh tại mức giá hiện tại (đã trừ Spread).

        Lợi nhuận ($) = (Giá bán - Giá mua) * unit_size - spread_cost
        """
        total_pnl = 0.0
        for pos in positions:
            entry = pos["entry_price"]
            pos_type = pos["type"]

            if pos_type == "SELL":
                # Entry sell tại `entry`. Đóng sell ở `current_price + spread`
                pnl = (entry - (current_price + self.spread)) * self.unit_size
            else: # BUY
                # Entry buy tại `entry + spread`. Đóng buy ở `current_price`
                pnl = ((current_price) - (entry + self.spread)) * self.unit_size

            total_pnl += pnl

        return total_pnl
