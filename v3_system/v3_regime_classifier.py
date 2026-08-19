"""
v3_system/v3_regime_classifier.py
==================================
Module Phân Loại Các Ngày Thành Các Nhóm Xu Hướng Thị Trường (v3 Architecture).
Được thiết kế bởi Senior Quantitative Researcher.

4 Nhóm Xu Hướng (Market Regimes):
1. Range_Sideway            : Biến động hẹp, dải M15 đi ngang, giá nằm sát VWAP.
2. Uptrend_Expansion        : Xu hướng tăng mở dải, M15 dốc lên, giá nằm trên VWAP.
3. Downtrend_Expansion      : Xu hướng giảm mở dải, M15 dốc xuống, giá nằm dưới VWAP.
4. High_Volatility_Outlier  : Biến động mạnh đột biến, nến sáng rộng > 2.2x ATR.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class V3RegimeClassifier:
    def __init__(self):
        pass

    @staticmethod
    def classify_day_regime(row: pd.Series) -> Dict[str, Any]:
        """
        Phân loại xu hướng 09:59 AM của 1 ngày dựa trên các chỉ số định lượng.
        """
        m_range_atr = row['morning_range_atr']
        m_momentum = row['morning_momentum']
        vwap_dist_atr = row['vwap_dist_atr']
        bb_slope = row['bb_slope_m15']
        bb_zscore = row['bb_zscore_m15']

        # Quy tắc phân loại định lượng:
        if m_range_atr > 2.2 or m_momentum > 0.65 or abs(bb_zscore) > 2.5:
            regime_id = 4
            regime_name = "High_Volatility_Outlier"
            regime_desc = "Biến động mạnh đột biến (Rủi ro cao)"
        elif bb_slope > 0.10 and vwap_dist_atr > 0.2:
            regime_id = 2
            regime_name = "Uptrend_Expansion"
            regime_desc = "Xu hướng tăng dốc lên (Uptrend)"
        elif bb_slope < -0.10 and vwap_dist_atr < -0.2:
            regime_id = 3
            regime_name = "Downtrend_Expansion"
            regime_desc = "Xu hướng giảm dốc xuống (Downtrend)"
        else:
            regime_id = 1
            regime_name = "Range_Sideway"
            regime_desc = "Đi ngang biến động hẹp (Sideway)"

        return {
            'regime_id': regime_id,
            'regime_name': regime_name,
            'regime_desc': regime_desc
        }

    def label_dataset_regimes(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gán nhãn nhóm xu hướng cho toàn bộ các ngày trong dataset.
        """
        df = feature_df.copy()
        regime_ids = []
        regime_names = []
        regime_descs = []

        for idx, row in df.iterrows():
            res = self.classify_day_regime(row)
            regime_ids.append(res['regime_id'])
            regime_names.append(res['regime_name'])
            regime_descs.append(res['regime_desc'])

        df['regime_id'] = regime_ids
        df['regime_name'] = regime_names
        df['regime_desc'] = regime_descs
        return df


if __name__ == '__main__':
    pass
