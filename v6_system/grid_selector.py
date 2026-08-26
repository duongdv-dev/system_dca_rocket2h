"""
v6_system/grid_selector.py
--------------------------
Bộ Lựa Chọn Cấu Hình An Toàn V2 Từ Tập Pre-defined Grid (Phase 9/11 - Safe Grid Selector V2).
Thiết kế lại bộ tham số độ bền cao (High-Robustness Parameters):
1. Tăng Step (Step = $8.0 - $15.0) để loại bỏ nhiễu ngắn hạn.
2. Tăng TP (TP = $6.0 - $12.0) để làm suy giảm tác động của Spread & Slippage (Tỷ lệ ma sát < 10%).
3. Thắt chặt ngưỡng lọc AI (skip_threshold = 0.68) chỉ giao dịch phiên xác suất cao.
"""

from typing import Dict, Any, List, Optional


class SafeGridSelector:
    """Class quản lý danh mục cấu hình an toàn V2 và đánh giá chọn lựa bằng AI."""

    # Danh mục các bộ tham số V2 đã tối ưu độ bền ma sát (High Friction Robustness)
    SAFE_GRID_MENU: List[Dict[str, Any]] = [
        {
            "name": "Strategy V2-A (Wide Step Scalp)",
            "step": 8.0,
            "multiplier": 1.10,
            "max_dca": 3,
            "tp_dollars": 6.0,
            "max_distance": 26.0,
            "min_prob_required": 0.78
        },
        {
            "name": "Strategy V2-B (Optimal Plateau Deep)",
            "step": 10.0,
            "multiplier": 1.15,
            "max_dca": 4,
            "tp_dollars": 8.0,
            "max_distance": 42.0,
            "min_prob_required": 0.72
        },
        {
            "name": "Strategy V2-C (Wide Conservative)",
            "step": 12.0,
            "multiplier": 1.10,
            "max_dca": 3,
            "tp_dollars": 10.0,
            "max_distance": 38.0,
            "min_prob_required": 0.68
        },
        {
            "name": "Strategy V2-D (Ultra Defensive)",
            "step": 15.0,
            "multiplier": 1.05,
            "max_dca": 3,
            "tp_dollars": 12.0,
            "max_distance": 48.0,
            "min_prob_required": 0.65
        }
    ]

    def __init__(self, skip_threshold: float = 0.65):
        self.skip_threshold = skip_threshold

    def select_optimal_config(self, prob_revert: float, current_atr: float = 3.0, current_adx: float = 20.0) -> Dict[str, Any]:
        """
        Đánh giá trạng thái thị trường và chọn cấu hình tối ưu V2 từ danh mục an toàn.
        """
        # Nếu xác suất quá thấp -> BỎ PHIÊN (SKIP)
        if prob_revert < self.skip_threshold:
            return {
                "name": "Option E (SKIP / RISK HIGH)",
                "should_skip": True,
                "step": 12.0,
                "multiplier": 1.0,
                "max_dca": 0,
                "tp_dollars": 0.0,
                "max_distance": 0.0,
                "reason": f"Prob revert ({prob_revert:.2f}) < threshold ({self.skip_threshold:.2f})"
            }

        # Đánh giá điểm kỳ vọng cho từng cấu hình trong Menu V2
        best_cfg = None
        best_score = -float("inf")

        for cfg in self.SAFE_GRID_MENU:
            if prob_revert >= cfg["min_prob_required"]:
                step = cfg["step"]
                atr_ratio = step / (current_atr + 1e-6)
                volatility_fit = 1.0 / (1.0 + abs(atr_ratio - 3.0))
                
                score = prob_revert * 0.7 + volatility_fit * 0.3
                if score > best_score:
                    best_score = score
                    best_cfg = cfg

        if best_cfg is not None:
            result = best_cfg.copy()
            result["should_skip"] = False
            result["score"] = round(best_score, 4)
            return result

        # Fallback an toàn nếu không chọn được bộ cụ thể -> dùng Strategy V2-C
        fallback = self.SAFE_GRID_MENU[2].copy()
        fallback["should_skip"] = False
        fallback["name"] += " [Fallback]"
        return fallback
