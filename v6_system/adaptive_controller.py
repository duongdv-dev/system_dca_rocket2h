"""
v6_system/adaptive_controller.py
--------------------------------
Bộ Điều Khiển Tham Số Thích Ứng Theo AI (Phase 8 - Adaptive DCA Controller).
Phân loại trạng thái thị trường thành 3 chế độ (Regimes A, B, C) dựa trên xác suất hồi P(reversion).
"""

from typing import Dict, Any, Optional


class AdaptiveDCAController:
    """Class quản lý quy tắc điều khiển thích ứng các tham số DCA theo thời gian thực."""

    def __init__(
        self,
        high_prob_threshold: float = 0.80,
        mod_prob_threshold: float = 0.60
    ):
        self.high_prob_threshold = high_prob_threshold
        self.mod_prob_threshold = mod_prob_threshold

    def get_regime_config(self, prob_revert: float, current_atr: Optional[float] = None) -> Dict[str, Any]:
        """
        Quyết định tham số chiến lược theo xác suất P(reversion):
        - Market A (P >= 80%): Step = $4.0, Multiplier = 1.25, Max DCA = 6, Max Distance = $12.0
        - Market B (60% <= P < 80%): Step = $7.0, Multiplier = 1.10, Max DCA = 4, Max Distance = $15.0
        - Market C (P < 60%): SKIP (BỎ PHIÊN)
        """
        if prob_revert >= self.high_prob_threshold:
            return {
                "regime": "Market A (High Reversion)",
                "should_skip": False,
                "step": 4.0,
                "multiplier": 1.25,
                "max_dca": 6,
                "max_distance": 12.0,  # Hard Stop Loss distance ($)
                "tp_dollars": 3.0
            }
        elif prob_revert >= self.mod_prob_threshold:
            return {
                "regime": "Market B (Moderate Reversion)",
                "should_skip": False,
                "step": 7.0,
                "multiplier": 1.10,
                "max_dca": 4,
                "max_distance": 15.0,  # Hard Stop Loss distance ($)
                "tp_dollars": 4.0
            }
        else:
            return {
                "regime": "Market C (Trend / Skip)",
                "should_skip": True,
                "step": 10.0,
                "multiplier": 1.0,
                "max_dca": 0,
                "max_distance": 0.0,
                "tp_dollars": 0.0
            }
