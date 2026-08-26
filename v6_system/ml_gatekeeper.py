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
        self._init_model()

    def _init_model(self):
        """Khởi tạo mô hình AI (XGBoost hoặc HistGradientBoosting)."""
        if HAS_XGBOOST:
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
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state
            )

    def train_and_evaluate(self, df_features: pd.DataFrame, split_year: int = 2024) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Huấn luyện mô hình theo phân chia chuỗi thời gian (Time-Series Split):
        - Train Set: Năm < split_year
        - Test Set : Năm >= split_year
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

        self._init_model()
        self.model.fit(X_train, y_train)

        probs_test = self.model.predict_proba(X_test)[:, 1]
        test_df["prob_revert"] = probs_test

        roc_auc = float(roc_auc_score(y_test, probs_test))
        pr_auc = float(average_precision_score(y_test, probs_test))
        preds = (probs_test >= 0.5).astype(int)

        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        logger.info(f"Hoàn thành huấn luyện mô hình: ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}.")

        metrics = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        return metrics, test_df

    def get_feature_importance(self) -> pd.DataFrame:
        """Lấy thứ hạng tầm quan trọng của 25 đặc trưng."""
        if self.model is None:
            return pd.DataFrame()

        if HAS_XGBOOST and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(len(self.FEATURE_COLS)) / len(self.FEATURE_COLS)

        df_imp = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df_imp
