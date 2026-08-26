"""
v6_system/grid_selector.py
--------------------------
Bộ Lựa Chọn Cấu Hình An Toàn Từ Tập Pre-defined Grid (Phase 9 - Safe Grid Selector).
Đảm bảo AI chỉ lựa chọn từ danh mục các bộ tham số đã được kiểm định an toàn (Strategy A, B, C, D)
và không tạo ra các tham số "điên rồ" bất thường.
"""

from typing import Dict, Any, List, Optional


class SafeGridSelector:
    """Class quản lý danh mục cấu hình an toàn và đánh giá chọn lựa bằng AI."""

    # Danh mục các bộ tham số an toàn đã được kiểm định từ Phase 4 Plateau
    SAFE_GRID_MENU: List[Dict[str, Any]] = [
        {
            "name": "Strategy A (Aggressive Scalp)",
            "step": 3.0,
            "multiplier": 1.05,
            "max_dca": 4,
            "tp_dollars": 2.0,
            "max_distance": 10.0,
            "min_prob_required": 0.82
        },
        {
            "name": "Strategy B (Balanced Steady)",
            "step": 5.0,
            "multiplier": 1.10,
            "max_dca": 5,
            "tp_dollars": 3.0,
            "max_distance": 15.0,
            "min_prob_required": 0.72
        },
        {
            "name": "Strategy C (Conservative Wide)",
            "step": 7.0,
            "multiplier": 1.15,
            "max_dca": 5,
            "tp_dollars": 5.0,
            "max_distance": 18.0,
            "min_prob_required": 0.62
        },
        {
            "name": "Strategy D (Defensive Deep)",
            "step": 10.0,
            "multiplier": 1.10,
            "max_dca": 4,
            "tp_dollars": 5.0,
            "max_distance": 20.0,
            "min_prob_required": 0.58
        }
    ]

    def __init__(self, skip_threshold: float = 0.55):
        self.skip_threshold = skip_threshold

    def select_optimal_config(self, prob_revert: float, current_atr: float = 3.0, current_adx: float = 20.0) -> Dict[str, Any]:
        """
        Đánh giá trạng thái thị trường và chọn cấu hình tối ưu từ danh mục an toàn.
        """
        # Nếu xác suất quá thấp -> BỎ PHIÊN (SKIP)
        if prob_revert < self.skip_threshold:
            return {
                "name": "Option E (SKIP / RISK HIGH)",
                "should_skip": True,
                "step": 10.0,
                "multiplier": 1.0,
                "max_dca": 0,
                "tp_dollars": 0.0,
                "max_distance": 0.0,
                "reason": f"Prob revert ({prob_revert:.2f}) < threshold ({self.skip_threshold:.2f})"
            }

        # Đánh giá điểm kỳ vọng cho từng cấu hình trong Menu
        best_cfg = None
        best_score = -float("inf")

        for cfg in self.SAFE_GRID_MENU:
            if prob_revert >= cfg["min_prob_required"]:
                # Tính điểm kỳ vọng dựa trên tính tương thích của biến động ATR/ADX với Step
                step = cfg["step"]
                # Điểm cao nhất khi Step cân bằng với ATR và ADX
                atr_ratio = step / (current_atr + 1e-6)
                volatility_fit = 1.0 / (1.0 + abs(atr_ratio - 1.5))
                
                score = prob_revert * 0.7 + volatility_fit * 0.3
                if score > best_score:
                    best_score = score
                    best_cfg = cfg

        if best_cfg is not None:
            result = best_cfg.copy()
            result["should_skip"] = False
            result["score"] = round(best_score, 4)
            return result

        # Fallback an toàn nếu không tìm thấy bộ khớp -> dùng Strategy D (Phòng thủ)
        fallback = self.SAFE_GRID_MENU[-1].copy()
        fallback["should_skip"] = False
        fallback["name"] += " [Fallback]"
        return fallback
