"""
v6_system/strategy_adaptive.py
------------------------------
Engine Chiến lược Adaptive DCA (Phase 8).
Tự động thay đổi tham số mở lệnh theo từng phiên dựa trên Bộ Điều Khiển AdaptiveDCAController
và hỗ trợ Stop Loss cứng (Max Distance).
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from v6_system.adaptive_controller import AdaptiveDCAController


class StrategyAdaptiveDCA:
    """Class đại diện cho chiến lược DCA Thích Ứng theo mô hình AI."""

    def __init__(
        self,
        controller: Optional[AdaptiveDCAController] = None,
        lot_base: float = 0.01,
        spread: float = 0.20,
        contract_size: float = 100.0
    ):
        self.controller = controller if controller is not None else AdaptiveDCAController()
        self.lot_base = lot_base
        self.spread = spread
        self.contract_size = contract_size

    def _get_level_lot(self, level: int, multiplier: float) -> float:
        """Tính khối lượng lot cho tầng level theo multiplier."""
        return round(self.lot_base * (multiplier ** (level - 1)), 4)

    def run_daily_session_adaptive(self, df_session: pd.DataFrame) -> Dict[str, Any]:
        """
        Mô phỏng 1 phiên giao dịch 10:00 -> 12:00 VN có tham số Thích Ứng Động.
        """
        if df_session.empty:
            return {
                "date": None,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "NO_DATA",
                "max_level": 0,
                "trades_count": 0,
                "regime": "NONE"
            }

        df_session = df_session.sort_values("dt_utc").reset_index(drop=True)
        date_str = str(df_session.iloc[0]["date_vn"]) if "date_vn" in df_session.columns else str(df_session.iloc[0]["date"])
        anchor_price = float(df_session.iloc[0]["open"])

        # Lấy xác suất P(reversion) tại nến bắt đầu phiên 10:00 VN (hoặc nến 10:15)
        p_revert = float(df_session.iloc[0]["prob_revert"]) if "prob_revert" in df_session.columns else 0.75
        current_atr = float(df_session.iloc[0]["atr"]) if "atr" in df_session.columns else 3.0

        # Lấy cấu hình tham số thích ứng từ Controller
        cfg = self.controller.get_regime_config(p_revert, current_atr)

        # Nếu là Market C -> BỎ PHIÊN (SKIP)
        if cfg["should_skip"]:
            return {
                "date": date_str,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "SKIPPED_BY_AI",
                "max_level": 0,
                "trades_count": 0,
                "regime": cfg["regime"],
                "anchor_price": anchor_price
            }

        step = cfg["step"]
        multiplier = cfg["multiplier"]
        max_dca = cfg["max_dca"]
        max_distance = cfg["max_distance"]
        tp_dollars = cfg["tp_dollars"]

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

                # A. Kiểm tra Hard Stop Loss (Max Distance breach)
                if basket_direction == "SELL" and (current_high - anchor_price) >= max_distance:
                    # Cắt lỗ ở mức Max Distance
                    realized_pnl = self._calculate_basket_pnl(positions, anchor_price + max_distance)
                    exit_reason = "STOP_LOSS_MAX_DIST"
                    cycle_finished = True
                    break

                elif basket_direction == "BUY" and (anchor_price - current_low) >= max_distance:
                    realized_pnl = self._calculate_basket_pnl(positions, anchor_price - max_distance)
                    exit_reason = "STOP_LOSS_MAX_DIST"
                    cycle_finished = True
                    break

                # B. Kiểm tra TP Basket Profit
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
            "regime": cfg["regime"],
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
