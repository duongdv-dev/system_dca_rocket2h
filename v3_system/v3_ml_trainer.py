"""
v3_system/v3_ml_trainer.py
==========================
Module Huấn Luyện AI LightGBM Học Ánh Xạ Từ Tín Hiệu R Sang Set Tham Số (v3 Architecture).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Nhận vào 6 đặc trưng liên tục R/ % đo lúc 09:59 AM.
2. Huấn luyện mô hình LightGBM Classifier để dự đoán Preset tối ưu cho ngày mới.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Any

class V3MLTrainer:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.model = None

    def train_lightgbm(self, train_df: pd.DataFrame) -> lgb.LGBMClassifier:
        """
        Huấn luyện mô hình LightGBM Classifier trên tập Train (6 tháng đầu).
        """
        valid_df = train_df[train_df['best_preset_id'] > 0].copy()
        
        if len(valid_df) < 10:
            print("⚠️ [V3MLTrainer] Tập Train có quá ít dữ liệu gán nhãn! Dùng fallback model.")
            return None

        X = valid_df[self.feature_cols].values
        y = valid_df['best_preset_id'].values

        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        self.model.fit(X, y)
        print(f"[V3MLTrainer] Huấn luyện xong Master LightGBM Model trên {len(valid_df)} mẫu dữ liệu Train!")
        return self.model

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán Preset ID cho tập Test (6 tháng sau Out-of-Sample).
        """
        if self.model is None:
            # Fallback nếu model chưa fit
            return np.ones(len(test_df), dtype=int)

        X = test_df[self.feature_cols].values
        preds = self.model.predict(X)
        return preds


if __name__ == '__main__':
    pass
