"""
Machine Learning Gatekeeper Model Module (Giai đoạn 3)
------------------------------------------------------
Huấn luyện mô hình LightGBM Classifier sử dụng Purged Walk-Forward TimeSeries Cross-Validation,
tối ưu hóa theo Precision / PR-AUC để lọc các ngày phá ngưỡng (breakout), xuất thứ tự tầm quan trọng của đặc trưng,
đề xuất ngưỡng xác suất tối ưu để lọc lệnh trên EA MT5, và xuất mô hình sang định dạng ONNX.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PurgedWalkForwardCV:
    """
    Phân chia Validation theo chuỗi thời gian Purged Walk-Forward.
    Đảm bảo tính thứ tự thời gian nghiêm ngặt với vùng đệm (embargo/purge) để tránh rò rỉ dữ liệu (data leakage).
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 2):
        """
        Args:
            n_splits: Số lượng fold kiểm thử cuốn chiếu (walk-forward).
            purge_days: Số ngày giao dịch loại bỏ (purge) giữa tập train và test.
        """
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Tạo chỉ mục index cho tập train và test.

        Args:
            X: DataFrame đặc trưng đầu vào sắp xếp theo thứ tự thời gian.

        Returns:
            Danh sách các tuple (train_indices, test_indices).
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        splits = []

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_days
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples or test_start >= test_end:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits


class LightGBMGatekeeper:
    """Mô hình phân loại LightGBM Gatekeeper để lọc các ngày nguy cơ xu hướng mạnh."""

    FEATURE_COLS = [
        "asian_range", "asian_return", "asian_volume",
        "ny_range", "ny_return", "ny_volatility", "ny_trend",
        "atr_m5_14", "atr_m15_14",
        "dist_vwap", "dist_ema50", "dist_ema200",
        "is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday"
    ]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Siêu tham số LightGBM mặc định được tinh chỉnh ưu tiên Precision & PR-AUC.
        """
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 150,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model: Optional[lgb.LGBMClassifier] = None
        self.cv_results: List[Dict[str, Any]] = []

    def train_and_evaluate_cv(self, daily_df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Chạy Purged Walk-Forward Cross-Validation trên dataset hàng ngày.

        Args:
            daily_df: DataFrame hàng ngày chứa đặc trưng và mục tiêu `reverted`.
            n_splits: Số lượng fold walk-forward.

        Returns:
            Dictionary chứa các chỉ số CV và mức độ quan trọng của đặc trưng.
        """
        logger.info("Đang thực thi kiểm thử Purged Walk-Forward Cross-Validation (Giai đoạn 3)...")

        # Đảm bảo sắp xếp theo thời gian
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        X = daily_df[self.FEATURE_COLS]
        y = daily_df["reverted"].values

        cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=2)
        folds = cv.split(X)

        oof_preds = np.full(len(X), np.nan)
        fold_metrics = []
        feature_importances = np.zeros(len(self.FEATURE_COLS))

        for fold, (train_idx, test_idx) in enumerate(folds, 1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_test, y_test = X.iloc[test_idx], y[test_idx]

            # Fit LightGBM
            clf = lgb.LGBMClassifier(**self.params)
            clf.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            test_probs = clf.predict_proba(X_test)[:, 1]
            oof_preds[test_idx] = test_probs
            feature_importances += clf.feature_importances_ / len(folds)

            # Tính toán chỉ số fold ở ngưỡng mặc định 0.5
            test_preds = (test_probs >= 0.5).astype(int)
            prec = precision_score(y_test, test_preds, zero_division=0)
            rec = recall_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5

            prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_probs)
            pr_auc = auc(rec_curve, prec_curve)

            fold_metrics.append({
                "fold": fold,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            })
            logger.info(f" Fold {fold}/{len(folds)} - Độ chính xác (Precision): {prec*100:.2f}%, Độ bao phủ (Recall): {rec*100:.2f}%, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        # Huấn luyện mô hình cuối cùng trên toàn bộ dataset để triển khai thực tế
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)

        # Bảng Tầm quan trọng của Đặc trưng
        fi_df = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # Đánh giá dự đoán OOF theo các ngưỡng xác suất khác nhau
        oof_mask = ~np.isnan(oof_preds)
        oof_y = y[oof_mask]
        oof_p = oof_preds[oof_mask]

        threshold_analysis = self._evaluate_thresholds(oof_y, oof_p)

        self.cv_results = fold_metrics
        return {
            "fold_metrics": fold_metrics,
            "mean_precision": np.mean([m["precision"] for m in fold_metrics]),
            "mean_recall": np.mean([m["recall"] for m in fold_metrics]),
            "mean_roc_auc": np.mean([m["roc_auc"] for m in fold_metrics]),
            "mean_pr_auc": np.mean([m["pr_auc"] for m in fold_metrics]),
            "feature_importance": fi_df,
            "threshold_analysis": threshold_analysis
        }

    def _evaluate_thresholds(self, y_true: np.ndarray, y_probs: np.ndarray) -> pd.DataFrame:
        """
        Đánh giá Precision, Recall và Tần suất giao dịch qua các ngưỡng xác suất.

        Args:
            y_true: Nhãn thực tế.
            y_probs: Xác suất dự đoán đảo chiều.

        Returns:
            DataFrame phân tích theo ngưỡng.
        """
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        results = []

        baseline_win_rate = np.mean(y_true) * 100.0

        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            n_trades = np.sum(preds)
            if n_trades > 0:
                prec = precision_score(y_true, preds, zero_division=0) * 100.0
                rec = recall_score(y_true, preds, zero_division=0) * 100.0
                win_rate_boost = prec - baseline_win_rate
            else:
                prec = 0.0
                rec = 0.0
                win_rate_boost = 0.0

            results.append({
                "Ngưỡng_P(Revert)": th,
                "Tỷ_lệ_Thắng_WinRate(%)": round(prec, 2),
                "Tỷ_lệ_Bao_Phủ_Recall(%)": round(rec, 2),
                "Số_Lệnh_Được_Duyệt": int(n_trades),
                "Số_Lệnh_Bị_Lọc(%)": round((1.0 - n_trades / len(y_true)) * 100.0, 1) if len(y_true) > 0 else 0.0,
                "Mức_Tăng_WinRate_So_Gốc(%)": round(win_rate_boost, 2)
            })

        return pd.DataFrame(results)

    def print_ml_report(self, cv_results_dict: Dict[str, Any]) -> None:
        """In báo cáo tổng hợp Giai đoạn 3 định dạng tiếng Việt."""
        print("\n" + "=" * 70)
        print("GIAI ĐOẠN 3: BÁO CÁO ĐÁNH GIÁ MÔ HÌNH LIGHTGBM GATEKEEPER")
        print("=" * 70)
        print("Kết quả Purged Walk-Forward Cross-Validation:")
        print(f"  - Độ chính xác trung bình (Precision): {cv_results_dict['mean_precision'] * 100:.2f}%")
        print(f"  - Độ bao phủ trung bình (Recall)   : {cv_results_dict['mean_recall'] * 100:.2f}%")
        print(f"  - ROC-AUC Trung bình                : {cv_results_dict['mean_roc_auc']:.4f}")
        print(f"  - PR-AUC Trung bình                 : {cv_results_dict['mean_pr_auc']:.4f}")
        print("-" * 70)
        print("Bảng Tầm quan trọng của Đặc trưng (Feature Importance Ranking):")
        fi_df = cv_results_dict["feature_importance"]
        for _, row in fi_df.iterrows():
            print(f"  - {row['feature']:<20}: {row['importance']}")
        print("-" * 70)
        print("Khuyến nghị Ngưỡng Xác suất lọc lệnh cho Robot EA MT5:")
        print(cv_results_dict["threshold_analysis"].to_string(index=False))
        print("=" * 70 + "\n")


class ONNXExporter:
    """Xuất mô hình LightGBM sang định dạng ONNX để tích hợp trực tiếp vào Robot EA MetaTrader 5."""

    @staticmethod
    def export_to_onnx(
        model: lgb.LGBMClassifier,
        feature_cols: List[str],
        output_path: str
    ) -> bool:
        """
        Chuyển đổi mô hình LightGBM sang ONNX bằng skl2onnx hoặc onnxmltools.

        Args:
            model: Mô hình LightGBMClassifier đã train.
            feature_cols: Danh sách tên đặc trưng.
            output_path: Đường dẫn file .onnx đầu ra.

        Returns:
            True nếu xuất thành công, False nếu thất bại.
        """
        logger.info(f"Đang tiến hành xuất mô hình sang ONNX cho MT5 EA -> {output_path}")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(feature_cols)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_types)

            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"Xuất thành công mô hình ONNX: {output_path}")
            return True

        except Exception as err1:
            logger.warning(f"Chuyển đổi skl2onnx trực tiếp không thành công: {err1}. Đang thử qua onnxmltools...")
            try:
                import onnxmltools
                from onnxmltools.convert.common.data_types import FloatTensorType

                initial_types = [("input", FloatTensorType([None, len(feature_cols)]))]
                onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types)

                with open(output_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                logger.info(f"Xuất thành công mô hình ONNX qua onnxmltools: {output_path}")
                return True
            except Exception as err2:
                logger.error(f"Xuất ONNX yêu cầu thư viện skl2onnx hoặc onnxmltools: {err2}")
                logger.info("Mẫu code xuất ONNX thủ công để tích hợp MT5 EA:")
                print("\n" + "#" * 60)
                print("# CODE XUẤT MÔ HÌNH ONNX CHO ROBOT EA MT5")
                print("#" * 60)
                print("import joblib")
                print("import skl2onnx")
                print("from skl2onnx.common.data_types import FloatTensorType")
                print("# 1. Lưu mô hình dạng pkl")
                print("joblib.dump(model, 'v5_gatekeeper_model.pkl')")
                print("# 2. Chuyển đổi sang ONNX")
                print(f"initial_types = [('input', FloatTensorType([None, {len(feature_cols)}]))]")
                print("onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_types)")
                print(f"with open('{output_path}', 'wb') as f:")
                print("    f.write(onnx_model.SerializeToString())")
                print("#" * 60 + "\n")
                return False
