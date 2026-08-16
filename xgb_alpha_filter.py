import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

from sklearn.preprocessing import StandardScaler

class XGBAlphaFilter:
    """
    Bộ lọc Tín hiệu XGBoost Machine Learning.
    Huấn luyện trên đặc trưng phiên Á để dự đoán xác suất Thắng/Thua P(Win) của Setup lúc 10:00 AM.
    """
    def __init__(self, proba_threshold: float = 0.60):
        self.proba_threshold = proba_threshold
        if HAS_XGB:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
        self.scaler = StandardScaler()
        self.feature_cols = ['asian_range_atr_ratio', 'asian_atr_m15', 'asian_return_pct', 'asian_body_ratio', 'asian_range']

    def train_filter(self, train_df: pd.DataFrame, train_labels: np.ndarray) -> float:
        """
        Huấn luyện XGBoost trên tập Train (2020 - 2023).
        train_labels: 1 nếu lệnh thắng, 0 nếu lệnh thua.
        """
        X_train = train_df[self.feature_cols].fillna(0).values
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, train_labels)
        
        # Đánh giá Accuracy trên tập Train
        train_preds = self.model.predict(X_train_scaled)
        acc = (train_preds == train_labels).mean() * 100.0
        print(f"🤖 [XGBoost Filter] Đã huấn luyện xong model (Train Accuracy: {acc:.2f}%)")
        return acc

    def filter_days(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán xác suất P(Win) trên tập Test (2024 - 2025).
        Trả về: (pass_filter_mask, win_probabilities)
        """
        X_test = test_df[self.feature_cols].fillna(0).values
        X_test_scaled = self.scaler.transform(X_test)

        # Lấy xác suất của class 1 (Win)
        probas = self.model.predict_proba(X_test_scaled)[:, 1]
        pass_mask = (probas >= self.proba_threshold)

        return pass_mask, probas
