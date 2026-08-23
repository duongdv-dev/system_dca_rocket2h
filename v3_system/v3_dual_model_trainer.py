"""
v3_system/v3_dual_model_trainer.py
===================================
Module Huấn Luyện AI 2 Giai Đoạn (Dual-Model Training Architecture).
Được thiết kế bởi Senior Quantitative Researcher.

Kiến trúc 2 Giai Đoạn:
- Stage 1: Risk Gatekeeper Model (Lọc An Toàn Vốn)
  Dự đoán xem 1 ngày có an toàn để bật lưới DCA không (Is Safe Trade Day).
  Áp dụng Asymmetric Loss / Sample Weights phạt nặng 10x nếu lỡ cho phép giao dịch vào ngày dính SL.

- Stage 2: Archetype Utility Ranker (Tối Ưu Kỳ Vọng Lợi Nhuận/Rủi Ro)
  Chỉ được gọi khi Stage 1 duyệt An Toàn.
  Dự đoán Archetype ID (1-5) tốt nhất dựa trên nhãn Robust Archetype Target.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
from v3_preset_clusterer import V3PresetClusterer

class V3DualModelTrainer:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.gatekeeper_model = None
        self.ranker_model = None

    def train_dual_models(self, train_df: pd.DataFrame) -> Tuple[lgb.LGBMClassifier, lgb.LGBMClassifier]:
        """
        Huấn luyện song song 2 mô hình (Risk Gatekeeper & Utility Ranker).
        """
        df = train_df.copy()

        # Kiểm tra sự tồn tại của nhãn robust
        if 'robust_is_safe' not in df.columns or 'robust_archetype_id' not in df.columns:
            raise ValueError("[V3DualModelTrainer] train_df phải chứa các cột nhãn 'robust_is_safe' và 'robust_archetype_id'")

        X = df[self.feature_cols].values
        y_safe = df['robust_is_safe'].values
        y_arch = df['robust_archetype_id'].values

        # =====================================================================
        # STAGE 1: RISK GATEKEEPER MODEL (BINARY CLASSIFIER)
        # =====================================================================
        # Trọng số tài chính bất đối ứng: Phạt nặng 10x nếu gán nhãn An Toàn sai vào ngày dính SL
        gatekeeper_weights = []
        for _, row in df.iterrows():
            is_safe = row['robust_is_safe']
            hit_sl = row.get('hit_sl', False)
            dd_atr = row.get('dd_atr', 0.0)

            if is_safe == 1:
                w = 1.0
            else:
                # Ngày không an toàn hoặc dính SL -> Tăng trọng số phạt gấp 8.0x
                w = 8.0 if hit_sl or dd_atr > 1.5 else 4.0
            gatekeeper_weights.append(w)

        if len(np.unique(y_safe)) >= 2:
            self.gatekeeper_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=8,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            self.gatekeeper_model.fit(X, y_safe, sample_weight=np.array(gatekeeper_weights))
            print("[V3DualModelTrainer] Stage 1 (Risk Gatekeeper) đã huấn luyện xong!")
        else:
            print("⚠️ [V3DualModelTrainer] Tập Train thiếu mẫu âm cho Gatekeeper. Dùng Fallback Gatekeeper.")
            self.gatekeeper_model = None

        # =====================================================================
        # STAGE 2: ARCHETYPE UTILITY RANKER (MULTICLASS ON SAFE DAYS)
        # =====================================================================
        safe_mask = (y_safe == 1) & (y_arch > 0)
        if np.sum(safe_mask) >= 5 and len(np.unique(y_arch[safe_mask])) >= 2:
            X_safe = X[safe_mask]
            y_safe_arch = y_arch[safe_mask]

            self.ranker_model = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=12,
                min_child_samples=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            self.ranker_model.fit(X_safe, y_safe_arch)
            print(f"[V3DualModelTrainer] Stage 2 (Archetype Ranker) đã huấn luyện xong trên {np.sum(safe_mask)} ngày an toàn!")
        else:
            print("⚠️ [V3DualModelTrainer] Tập Train không đủ mẫu an toàn cho Ranker. Dùng Fallback Ranker.")
            self.ranker_model = None

        return self.gatekeeper_model, self.ranker_model

    def predict(self, test_df: pd.DataFrame, safe_threshold: float = 0.55) -> np.ndarray:
        """
        Dự đoán kết hợp 2 giai đoạn:
        - Nếu Gatekeeper đánh giá Không An Toàn (< safe_threshold) -> Trả về 0 (No_Trade_Pass).
        - Ngược lại -> Gọi Ranker chọn Archetype ID (1-5).
        """
        n_samples = len(test_df)
        if self.gatekeeper_model is None:
            return np.zeros(n_samples, dtype=int)

        X_test = test_df[self.feature_cols].values

        # 1. Stage 1 Prediction
        safe_probs = self.gatekeeper_model.predict_proba(X_test)[:, 1]

        final_preds = []
        for idx in range(n_samples):
            # Kiểm tra No-Trade Guard bổ sung từ Regime Classifier
            force_no_trade = test_df.iloc[idx].get('force_no_trade', False) if 'force_no_trade' in test_df.columns else False
            p_safe = safe_probs[idx]

            if force_no_trade or p_safe < safe_threshold:
                final_preds.append(0)  # No_Trade_Pass
            else:
                # 2. Stage 2 Prediction
                if self.ranker_model is not None:
                    row_x = X_test[idx:idx+1]
                    arch_prob = self.ranker_model.predict_proba(row_x)[0]
                    classes = self.ranker_model.classes_
                    best_arch_idx = np.argmax(arch_prob)
                    final_preds.append(int(classes[best_arch_idx]))
                else:
                    final_preds.append(2)  # Standard_DCA Fallback

        return np.array(final_preds)


if __name__ == '__main__':
    pass
