"""
v3_system/v3_ml_trainer.py
==========================
Module Huấn Luyện AI LightGBM Nâng Cấp (v3 Architecture - Financial Loss & Archetype Ranking).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Nhận các đặc trưng liên tục R và vi cấu trúc nến đo lúc 09:59 AM.
2. Gom nhãn về 6 Strategy Archetypes bằng V3PresetClusterer.
3. Áp dụng Financial Sample Weighting (Thưởng Fitness Score / Phạt Drawdown & SL).
4. Dự đoán Archetype ID & chọn Preset đại diện an toàn nhất.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
from v3_preset_clusterer import V3PresetClusterer

class V3MLTrainer:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.model = None

    def train_lightgbm(self, train_df: pd.DataFrame) -> lgb.LGBMClassifier:
        """
        Huấn luyện mô hình LightGBM Classifier trên tập Train với Financial Sample Weighting.
        """
        df = train_df.copy()

        # Ánh xạ từ best_preset_id -> best_archetype_id nếu chưa có
        if 'best_archetype_id' not in df.columns:
            from v3_preset_generator import V3PresetGenerator
            presets_list = V3PresetGenerator.generate_540_candidate_presets()
            
            archetype_ids = []
            for _, row in df.iterrows():
                pid = int(row.get('best_preset_id', 0))
                if pid > 0 and pid <= len(presets_list):
                    p = presets_list[pid - 1]
                    aid = V3PresetClusterer.classify_preset(p)
                else:
                    aid = 0
                archetype_ids.append(aid)
            df['best_archetype_id'] = archetype_ids

        # Tạo Financial Sample Weights (Thưởng Sharpe/Fitness, Phạt SL/Drawdown)
        weights = []
        for _, row in df.iterrows():
            fit = row.get('fitness_score', 0.0)
            hit_sl = row.get('hit_sl', False)
            dd_atr = row.get('dd_atr', 0.0)
            
            # Trọng số tài chính
            w = max(0.1, fit + 5.0)
            if hit_sl:
                w *= 0.1  # Phạt 10x nếu dính SL
            if dd_atr > 1.5:
                w *= 0.5  # Phạt nếu Drawdown lớn
            weights.append(w)

        X = df[self.feature_cols].values
        y = df['best_archetype_id'].values

        if len(np.unique(y)) < 2:
            print("⚠️ [V3MLTrainer] Tập Train chỉ chứa 1 nhãn Archetype! Dùng fallback model.")
            self.model = None
            return None

        self.model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        self.model.fit(X, y, sample_weight=np.array(weights))
        print(f"[V3MLTrainer] Huấn luyện xong Master LightGBM (Archetype Classifier) trên {len(df)} mẫu dữ liệu!")
        return self.model

    def predict(self, test_df: pd.DataFrame, confidence_threshold: float = 0.35) -> np.ndarray:
        """
        Dự đoán Archetype ID cho tập Test. Tự động đưa về 0 (No_Trade_Pass) nếu độ tin cậy thấp.
        """
        if self.model is None:
            return np.zeros(len(test_df), dtype=int)

        X = test_df[self.feature_cols].values
        probs = self.model.predict_proba(X)
        classes = self.model.classes_

        preds = []
        for idx, prob_row in enumerate(probs):
            max_prob_idx = np.argmax(prob_row)
            max_prob = prob_row[max_prob_idx]
            predicted_class = classes[max_prob_idx]

            # Kiểm tra No-Trade Guard khẩn cấp từ Regime Classifier
            force_no_trade = test_df.iloc[idx].get('force_no_trade', False) if 'force_no_trade' in test_df.columns else False

            if force_no_trade or max_prob < confidence_threshold:
                preds.append(0)  # No_Trade_Pass
            else:
                preds.append(predicted_class)

        return np.array(preds)


if __name__ == '__main__':
    pass

