"""
v6_system/strategy_ai.py
------------------------
Engine Chiến lược DCA kết hợp bộ lọc AI Filter (Phase 7).
Quy tắc:
1. Kiểm tra tín hiệu chạm tầng DCA.
2. Kiểm tra xác suất AI P(reversion) tại nến t:
   - Nếu P(reversion) >= prob_threshold -> Cho phép mở lệnh.
   - Nếu P(reversion) < prob_threshold -> Chặn / Không DCA thêm.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from v6_system.strategy_v1 import StrategyV1


class StrategyAIFilter(StrategyV1):
    """Class đại diện cho chiến lược DCA tích hợp bộ lọc AI Gatekeeper."""

    def __init__(
        self,
        step: float = 5.0,
        max_dca: int = 5,
        multiplier: float = 1.10,
        prob_threshold: float = 0.65,
        tp_dollars: Optional[float] = None,
        lot_base: float = 0.01,
        spread: float = 0.20
    ):
        super().__init__(
            step=step,
            max_dca=max_dca,
            multiplier=multiplier,
            tp_dollars=tp_dollars,
            lot_base=lot_base,
            spread=spread
        )
        self.prob_threshold = prob_threshold

    def run_daily_session_with_ai(self, df_session: pd.DataFrame, model_gatekeeper=None) -> Dict[str, Any]:
        """
        Mô phỏng 1 phiên giao dịch 10:00 -> 12:00 VN có kiểm tra bộ lọc AI.

        Args:
            df_session: DataFrame chứa nến M1 và các cột 25 features hoặc xác suất 'prob_revert'.
            model_gatekeeper: Lớp MLGatekeeper (nếu chưa pre-compute prob_revert).
        """
        if df_session.empty:
            return {
                "date": None,
                "traded": False,
                "pnl": 0.0,
                "exit_reason": "NO_DATA",
                "max_level": 0,
                "trades_count": 0,
                "blocked_by_ai": 0
            }

        df_session = df_session.sort_values("dt_utc").reset_index(drop=True)
        date_str = str(df_session.iloc[0]["date_vn"]) if "date_vn" in df_session.columns else str(df_session.iloc[0]["date"])
        anchor_price = float(df_session.iloc[0]["open"])

        basket_direction: Optional[str] = None
        positions: List[Dict[str, Any]] = []
        next_sell_level = 1
        next_buy_level = 1

        cycle_finished = False
        exit_reason = "NO_TRADE"
        realized_pnl = 0.0
        max_level_reached = 0
        blocked_by_ai_count = 0

        for idx, row in df_session.iterrows():
            current_open = float(row["open"])
            current_high = float(row["high"])
            current_low = float(row["low"])
            current_close = float(row["close"])
            is_last_candle = (idx == len(df_session) - 1)

            # Lấy xác suất P(reversion) từ AI
            p_revert = float(row["prob_revert"]) if "prob_revert" in row else 1.0

            # 1. Kiểm tra tầng 1
            if basket_direction is None:
                trigger_sell_price = anchor_price + next_sell_level * self.step
                trigger_buy_price = anchor_price - next_buy_level * self.step

                if current_high >= trigger_sell_price:
                    # Kiểm tra bộ lọc AI
                    if p_revert >= self.prob_threshold:
                        basket_direction = "SELL"
                        positions.append({
                            "level": 1,
                            "type": "SELL",
                            "entry_price": trigger_sell_price,
                            "lot": self._get_level_lot(1)
                        })
                        next_sell_level += 1
                        max_level_reached = 1
                    else:
                        blocked_by_ai_count += 1

                elif current_low <= trigger_buy_price:
                    if p_revert >= self.prob_threshold:
                        basket_direction = "BUY"
                        positions.append({
                            "level": 1,
                            "type": "BUY",
                            "entry_price": trigger_buy_price,
                            "lot": self._get_level_lot(1)
                        })
                        next_buy_level += 1
                        max_level_reached = 1
                    else:
                        blocked_by_ai_count += 1

            # 2. Kiểm tra nhồi thêm tầng DCA tiếp theo
            if basket_direction is not None and not cycle_finished:
                if basket_direction == "SELL" and next_sell_level <= self.max_dca:
                    target_price = anchor_price + next_sell_level * self.step
                    if current_high >= target_price:
                        if p_revert >= self.prob_threshold:
                            positions.append({
                                "level": next_sell_level,
                                "type": "SELL",
                                "entry_price": target_price,
                                "lot": self._get_level_lot(next_sell_level)
                            })
                            max_level_reached = next_sell_level
                            next_sell_level += 1
                        else:
                            blocked_by_ai_count += 1

                elif basket_direction == "BUY" and next_buy_level <= self.max_dca:
                    target_price = anchor_price - next_buy_level * self.step
                    if current_low <= target_price:
                        if p_revert >= self.prob_threshold:
                            positions.append({
                                "level": next_buy_level,
                                "type": "BUY",
                                "entry_price": target_price,
                                "lot": self._get_level_lot(next_buy_level)
                            })
                            max_level_reached = next_buy_level
                            next_buy_level += 1
                        else:
                            blocked_by_ai_count += 1

                # Kiểm tra TP Basket Profit
                best_price_for_tp = current_low if basket_direction == "SELL" else current_high
                floating_pnl_best = self._calculate_basket_pnl(positions, best_price_for_tp)
                floating_pnl_close = self._calculate_basket_pnl(positions, current_close)

                if floating_pnl_best >= self.tp_dollars:
                    realized_pnl = self.tp_dollars
                    exit_reason = "TP_HIT"
                    cycle_finished = True
                    break

                # Force Close 12:00 VN
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
            "blocked_by_ai": blocked_by_ai_count,
            "max_exposure_lots": round(total_exposure, 4),
            "anchor_price": anchor_price
        }
