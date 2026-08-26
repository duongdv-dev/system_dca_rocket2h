"""
v6_system/strategy_grid_ai.py
-----------------------------
Engine Chiến lược DCA Lựa Chọn Cấu Hình An Toàn Bằng AI (Phase 9).
Chỉ thực thi các cấu hình nằm trong tập Pre-defined Safe Grid Menu.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from v6_system.grid_selector import SafeGridSelector


class StrategyGridAI:
    """Class đại diện cho chiến lược DCA lựa chọn cấu hình an toàn bằng AI."""

    def __init__(
        self,
        selector: Optional[SafeGridSelector] = None,
        lot_base: float = 0.01,
        spread: float = 0.20,
        contract_size: float = 100.0
    ):
        self.selector = selector if selector is not None else SafeGridSelector()
        self.lot_base = lot_base
        self.spread = spread
        self.contract_size = contract_size

    def _get_level_lot(self, level: int, multiplier: float) -> float:
        """Tính khối lượng lot cho tầng level theo multiplier."""
        return round(self.lot_base * (multiplier ** (level - 1)), 4)

    def run_daily_session_grid_ai(self, df_session: pd.DataFrame) -> Dict[str, Any]:
        """
        Mô phỏng 1 phiên giao dịch 10:00 -> 12:00 VN với cấu hình được AI chọn từ danh mục an toàn.
        """
        if df_session.empty:
            return {
                "date": None,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "NO_DATA",
                "max_level": 0,
                "trades_count": 0,
                "config_name": "NONE"
            }

        df_session = df_session.sort_values("dt_utc").reset_index(drop=True)
        date_str = str(df_session.iloc[0]["date_vn"]) if "date_vn" in df_session.columns else str(df_session.iloc[0]["date"])
        anchor_price = float(df_session.iloc[0]["open"])

        # Lấy xác suất P(reversion) và ATR/ADX tại nến bắt đầu phiên
        p_revert = float(df_session.iloc[0]["prob_revert"]) if "prob_revert" in df_session.columns else 0.75
        current_atr = float(df_session.iloc[0]["atr"]) if "atr" in df_session.columns else 3.0
        current_adx = float(df_session.iloc[0]["adx"]) if "adx" in df_session.columns else 20.0

        # Lựa chọn cấu hình an toàn từ SafeGridSelector
        cfg = self.selector.select_optimal_config(p_revert, current_atr, current_adx)

        # Nếu AI quyết định SKIP -> BỎ PHIÊN
        if cfg["should_skip"]:
            return {
                "date": date_str,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "SKIPPED_BY_AI",
                "max_level": 0,
                "trades_count": 0,
                "config_name": cfg["name"],
                "anchor_price": anchor_price
            }

        step = cfg["step"]
        multiplier = cfg["multiplier"]
        max_dca = cfg["max_dca"]
        max_distance = cfg["max_distance"]
        tp_dollars = cfg["tp_dollars"]
        config_name = cfg["name"]

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

            # 1. Kiểm tra kích hoạt tầng 1
            if basket_direction is None:
                trigger_sell_price = anchor_price + next_sell_level * step
                trigger_buy_price = anchor_price - next_buy_level * step

                if current_high >= trigger_sell_price:
                    basket_direction = "SELL"
                    positions.append({
                        "level": 1,
                        "type": "SELL",
                        "entry_price": trigger_sell_price,
                        "lot": self._get_level_lot(1, multiplier)
                    })
                    next_sell_level += 1
                    max_level_reached = 1

                elif current_low <= trigger_buy_price:
                    basket_direction = "BUY"
                    positions.append({
                        "level": 1,
                        "type": "BUY",
                        "entry_price": trigger_buy_price,
                        "lot": self._get_level_lot(1, multiplier)
                    })
                    next_buy_level += 1
                    max_level_reached = 1

            # 2. Kiểm tra nhồi tầng DCA & Cắt lỗ Max Distance & Chốt lời TP
            if basket_direction is not None and not cycle_finished:
                if basket_direction == "SELL" and next_sell_level <= max_dca:
                    target_price = anchor_price + next_sell_level * step
                    if current_high >= target_price:
                        positions.append({
                            "level": next_sell_level,
                            "type": "SELL",
                            "entry_price": target_price,
                            "lot": self._get_level_lot(next_sell_level, multiplier)
                        })
                        max_level_reached = next_sell_level
                        next_sell_level += 1

                elif basket_direction == "BUY" and next_buy_level <= max_dca:
                    target_price = anchor_price - next_buy_level * step
                    if current_low <= target_price:
                        positions.append({
                            "level": next_buy_level,
                            "type": "BUY",
                            "entry_price": target_price,
                            "lot": self._get_level_lot(next_buy_level, multiplier)
                        })
                        max_level_reached = next_buy_level
                        next_buy_level += 1

                # A. Hard Stop Loss (Max Distance breach)
                if basket_direction == "SELL" and (current_high - anchor_price) >= max_distance:
                    realized_pnl = self._calculate_basket_pnl(positions, anchor_price + max_distance)
                    exit_reason = "STOP_LOSS_MAX_DIST"
                    cycle_finished = True
                    break

                elif basket_direction == "BUY" and (anchor_price - current_low) >= max_distance:
                    realized_pnl = self._calculate_basket_pnl(positions, anchor_price - max_distance)
                    exit_reason = "STOP_LOSS_MAX_DIST"
                    cycle_finished = True
                    break

                # B. TP Basket Profit
                best_price_for_tp = current_low if basket_direction == "SELL" else current_high
                floating_pnl_best = self._calculate_basket_pnl(positions, best_price_for_tp)
                floating_pnl_close = self._calculate_basket_pnl(positions, current_close)

                if floating_pnl_best >= tp_dollars:
                    realized_pnl = tp_dollars
                    exit_reason = "TP_HIT"
                    cycle_finished = True
                    break

                # C. Force Close 12:00 VN
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
            "config_name": config_name,
            "max_exposure_lots": round(total_exposure, 4),
            "anchor_price": anchor_price
        }

    def _calculate_basket_pnl(self, positions: List[Dict[str, Any]], current_price: float) -> float:
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
