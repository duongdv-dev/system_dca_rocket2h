"""
v3_system/v3_preset_clusterer.py
=================================
Module Gom Nhóm & Phân Loại Các Presets Thành 6 Strategy Archetypes (v3 Architecture).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Phân loại 540 Candidate Presets thành 6 Archetypes giao dịch chuẩn.
2. Giảm nhiễu nhãn (Label Noise) cho mô hình Machine Learning.
3. Ánh xạ ngược từ Archetype ID về đại diện Preset tiêu biểu.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class V3PresetClusterer:
    ARCHETYPES = {
        0: {"name": "No_Trade_Pass", "desc": "Đứng ngoài bảo vệ vốn (Rủi ro cao hoặc không có tín hiệu)"},
        1: {"name": "Conservative_Scalp", "desc": "Lưới hẹp, Lot nhỏ, TP nhanh (An toàn)"},
        2: {"name": "Standard_DCA", "desc": "Khoảng cách lưới trung bình, nhồi lot cân bằng"},
        3: {"name": "Wide_Grid_Reversion", "desc": "Khoảng cách lưới rộng, chịu biến động lớn"},
        4: {"name": "Aggressive_Momentum", "desc": "Nhồi lot lớn bám sát sóng ngắn"},
        5: {"name": "Defensive_Breakeven", "desc": "Giãn khoảng cách cao, ưu tiên thoát hòa vốn nhanh"}
    }

    def __init__(self):
        pass

    @classmethod
    def classify_preset(cls, preset: Dict[str, float]) -> int:
        """
        Phân loại 1 Preset dựa trên các thông số định lượng:
        step_0_ratio, step_exp, max_orders, multiplier
        """
        if preset is None:
            return 0

        s0 = preset.get('step_0_ratio', 1.0)
        se = preset.get('step_exp', 1.2)
        mo = preset.get('max_orders', 3)
        mult = preset.get('multiplier', 1.0)

        if se >= 1.3:
            return 5  # Defensive_Breakeven
        elif s0 >= 1.6:
            return 3  # Wide_Grid_Reversion
        elif mult >= 1.12:
            return 4  # Aggressive_Momentum
        elif s0 <= 0.8 and mult <= 1.05:
            return 1  # Conservative_Scalp
        else:
            return 2  # Standard_DCA

    @classmethod
    def get_archetype_representative(cls, archetype_id: int, presets_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Lấy Preset đại diện tiêu biểu nhất cho Archetype ID.
        """
        if archetype_id == 0 or not presets_list:
            return None

        matching_presets = []
        for p in presets_list:
            if cls.classify_preset(p) == archetype_id:
                matching_presets.append(p)

        if not matching_presets:
            return presets_list[0]

        # Trả về preset trung tâm (median) của nhóm
        mid_idx = len(matching_presets) // 2
        return matching_presets[mid_idx]


if __name__ == '__main__':
    from v3_preset_generator import V3PresetGenerator
    presets = V3PresetGenerator.generate_540_candidate_presets()
    clusterer = V3PresetClusterer()
    
    counts = {i: 0 for i in range(6)}
    for p in presets:
        aid = clusterer.classify_preset(p)
        counts[aid] += 1
    
    print("📊 Phân bố 540 Presets vào 6 Strategy Archetypes:")
    for aid, name_dict in V3PresetClusterer.ARCHETYPES.items():
        print(f" • Archetype {aid} ({name_dict['name']}): {counts[aid]} presets")
