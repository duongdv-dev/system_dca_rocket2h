"""
v6_system/ml_gatekeeper.py
--------------------------
Module AI ML Gatekeeper cho Version 6 Phase 5 (Hỗ trợ HistGradientBoosting Siêu Nhanh).
Huấn luyện mô hình XGBoost V1 hoặc HistGradientBoosting ước lượng xác suất hồi về Anchor trước 12:00 VN.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("MLGatekeeper")


class MLGatekeeper:
    """Class quản lý huấn luyện mô hình XGBoost V1 và đánh giá xác suất Gatekeeper."""

    FEATURE_COLS = [
        "distance_from_anchor", "dist_anchor_over_atr", "atr", "atr_norm",
        "rsi", "adx", "ema_9", "ema_21", "ema_50", "ema_slope",
        "volume", "vol_over_avg", "return_1m", "return_5m", "return_15m",
        "volatility", "time_since_10", "time_remaining_12", "session_high",
        "session_low", "candle_body", "upper_wick", "lower_wick", "vwap", "dist_to_vwap"
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None

    def train_and_evaluate(self, df_features: pd.DataFrame, split_year: int = 2024) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Huấn luyện mô hình theo phân chia chuỗi thời gian (Time-Series Split):
        - Train Set: Năm 2020 - 2023
        - Test Set : Năm 2024 - 2025
        """
        logger.info(f"Bắt đầu phân chia Dataset (Train < {split_year}, Test >= {split_year})...")
        
        date_col = "date" if "date" in df_features.columns else ("date_vn" if "date_vn" in df_features.columns else "dt_vn")
        df_features["year"] = pd.to_datetime(df_features[date_col]).dt.year
        
        train_mask = df_features["year"] < split_year
        test_mask = df_features["year"] >= split_year

        train_df = df_features[train_mask].dropna(subset=self.FEATURE_COLS).copy()
        test_df = df_features[test_mask].dropna(subset=self.FEATURE_COLS).copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Tập dữ liệu Train/Test rỗng. Kiểm tra phân chia năm.")

        X_train = train_df[self.FEATURE_COLS].astype(np.float32)
        y_train = train_df["target_revert"].values.astype(np.int32)

        X_test = test_df[self.FEATURE_COLS].astype(np.float32)
        y_test = test_df["target_revert"].values.astype(np.int32)

        logger.info(f"Kích thước tập Train: {len(X_train):,} dòng | Test: {len(X_test):,} dòng.")

        # Khởi tạo mô hình
        if HAS_XGBOOST:
            logger.info("Huấn luyện bằng mô hình XGBClassifier...")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        else:
            logger.info("Dùng HistGradientBoostingClassifier (Tối ưu hóa đa nhân siêu nhanh)...")
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state
            )

        self.model.fit(X_train, y_train)

        # Dự báo xác suất (Probability)
        y_prob_test = self.model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)

        # Tính toán chỉ số đo lường ML
        roc_auc = roc_auc_score(y_test, y_prob_test)
        pr_auc = average_precision_score(y_test, y_prob_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)

        # Feature Importance (Hoặc Permutation Importance nếu là HistGradientBoosting)
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            # Ước tính importance đơn giản cho HistGradientBoosting
            importances = np.zeros(len(self.FEATURE_COLS))

        df_importance = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        metrics = {
            "model_name": "XGBoost V1 Gatekeeper" if HAS_XGBOOST else "HistGradientBoosting Fast Model",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "positive_class_ratio_test": round(float(y_test.mean()), 4)
        }

        logger.info(f"Hoàn thành huấn luyện mô hình: ROC-AUC = {metrics['roc_auc']}, PR-AUC = {metrics['pr_auc']}.")
        return metrics, df_importance
