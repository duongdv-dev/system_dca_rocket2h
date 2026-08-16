import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

class AIPresetSelector:
    """
    Bộ chọn Preset AI Meta-Learner (AI Meta-Selector).
    Mỗi sáng lúc 10:00 AM, AI đọc các đặc trưng phiên Á và tự động CHỌN 1 PRESET TỐI ƯU NHẤT
    từ Preset Pool cho ngày hôm đó (hoặc chọn NO_TRADE để bảo vệ vốn).
    """
    def __init__(self, n_presets: int = 10):
        self.n_presets = n_presets
        if HAS_XGB:
            self.model = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                random_state=42
            )
        self.scaler = StandardScaler()
        self.feature_cols = ['asian_range_atr_ratio', 'asian_atr_m15', 'asian_return_pct', 'asian_body_ratio', 'asian_range']

    def train_selector(self, train_df: pd.DataFrame, train_labels: np.ndarray) -> float:
        """
        Huấn luyện AI Meta-Learner trên tập Train (2020 - 2023).
        train_labels: ID của Preset thắng lớn nhất ngày hôm đó.
        """
        X_train = train_df[self.feature_cols].fillna(0).values
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, train_labels)
        
        preds = self.model.predict(X_train_scaled)
        acc = (preds == train_labels).mean() * 100.0
        print(f"🤖 [AI Meta-Selector] Đã huấn luyện xong mô hình AI chọn Preset (Train Accuracy: {acc:.2f}%)")
        return acc

    def select_preset_for_days(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán và CHỌN PRESET TỐI ƯU NHẤT cho từng ngày trong tập Test (2024 - 2025).
        Returns:
            selected_preset_ids: Mảng chứa ID các Presets được AI chọn
        """
        X_test = test_df[self.feature_cols].fillna(0).values
        X_test_scaled = self.scaler.transform(X_test)

        selected_preset_ids = self.model.predict(X_test_scaled)
        return selected_preset_ids
