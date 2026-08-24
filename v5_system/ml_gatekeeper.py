"""
Machine Learning Gatekeeper Model Module (Stage 3)
--------------------------------------------------
Trains a LightGBM Classifier using Purged Walk-Forward TimeSeries Cross-Validation,
optimizes for Precision / PR-AUC to filter out breakout trend days, outputs feature importances,
recommends optimal probability thresholds for MT5 EA trade filtering, and exports to ONNX.
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
    Purged Group TimeSeries Walk-Forward Cross-Validator.
    Ensures strict temporal ordering with an embargo/purge window to prevent data leakage.
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 2):
        """
        Args:
            n_splits: Number of walk-forward folds.
            purge_days: Number of trading days to purge between train and test sets.
        """
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for training and test sets.

        Args:
            X: Input features DataFrame sorted chronologically.

        Returns:
            List of (train_indices, test_indices) tuples.
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
    """LightGBM Binary Classifier for filtering high-risk breakout days."""

    FEATURE_COLS = [
        "asian_range", "asian_return", "asian_volume",
        "ny_range", "ny_return", "ny_volatility", "ny_trend",
        "atr_m5_14", "atr_m15_14",
        "dist_vwap", "dist_ema50", "dist_ema200",
        "is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday"
    ]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Default LightGBM hyperparameters tuned for Precision & PR-AUC.
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
        Runs Purged Walk-Forward Cross-Validation on daily dataset.

        Args:
            daily_df: Daily DataFrame containing features and `reverted` target.
            n_splits: Number of walk-forward folds.

        Returns:
            Dictionary containing cross-validation metrics and feature importances.
        """
        logger.info("Executing Purged Walk-Forward Cross-Validation (Stage 3)...")

        # Ensure temporal sorting
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

            # Compute fold metrics at default 0.5 threshold
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
            logger.info(f"Fold {fold}/{len(folds)} - Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        # Train final model on entire dataset for production deployment
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)

        # Feature Importance DataFrame
        fi_df = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # Evaluate OOF predictions across probability thresholds
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
        Evaluates Precision, Recall, and Trade Frequency across different probability thresholds.

        Args:
            y_true: True binary targets.
            y_probs: Predicted probabilities.

        Returns:
            DataFrame of threshold metrics.
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
                "threshold": th,
                "precision_winrate_pct": prec,
                "recall_pct": rec,
                "trades_filtered_pct": (1.0 - n_trades / len(y_true)) * 100.0 if len(y_true) > 0 else 0.0,
                "n_passed_trades": n_trades,
                "winrate_boost_vs_baseline": win_rate_boost
            })

        return pd.DataFrame(results)

    def print_ml_report(self, cv_results_dict: Dict[str, Any]) -> None:
        """Prints styled summary report for Stage 3 ML Gatekeeper."""
        print("\n" + "=" * 65)
        print("STAGE 3: LIGHTGBM GATEKEEPER MODEL EVALUATION REPORT")
        print("=" * 65)
        print("Purged Walk-Forward Cross-Validation Results:")
        print(f"  - Mean Precision : {cv_results_dict['mean_precision'] * 100:.2f}%")
        print(f"  - Mean Recall    : {cv_results_dict['mean_recall'] * 100:.2f}%")
        print(f"  - Mean ROC-AUC   : {cv_results_dict['mean_roc_auc']:.4f}")
        print(f"  - Mean PR-AUC    : {cv_results_dict['mean_pr_auc']:.4f}")
        print("-" * 65)
        print("Feature Importance Ranking (LightGBM Split Count):")
        fi_df = cv_results_dict["feature_importance"]
        for _, row in fi_df.iterrows():
            print(f"  - {row['feature']:<20}: {row['importance']}")
        print("-" * 65)
        print("Probability Threshold Recommendations for MT5 EA Filtering:")
        print(cv_results_dict["threshold_analysis"].to_string(index=False))
        print("=" * 65 + "\n")


class ONNXExporter:
    """Exports trained LightGBM models to ONNX format for direct MT5 EA integration."""

    @staticmethod
    def export_to_onnx(
        model: lgb.LGBMClassifier,
        feature_cols: List[str],
        output_path: str
    ) -> bool:
        """
        Converts trained LightGBM model to ONNX using skl2onnx or onnxmltools.

        Args:
            model: Fitted LightGBMClassifier.
            feature_cols: List of feature names.
            output_path: Target path for .onnx file.

        Returns:
            True if export succeeded, False otherwise.
        """
        logger.info(f"Attempting ONNX export for MT5 integration -> {output_path}")

        try:
            # Attempt export via onnxmltools or skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_types = [("input", FloatTensorType([None, len(feature_cols)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_types)

            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"Successfully exported LightGBM model to ONNX: {output_path}")
            return True

        except Exception as err1:
            logger.warning(f"skl2onnx direct conversion failed: {err1}. Trying lightgbm native / onnxmltools...")
            try:
                import onnxmltools
                from onnxmltools.convert.common.data_types import FloatTensorType

                initial_types = [("input", FloatTensorType([None, len(feature_cols)]))]
                onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types)

                with open(output_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                logger.info(f"Successfully exported LightGBM model via onnxmltools to ONNX: {output_path}")
                return True
            except Exception as err2:
                logger.error(f"ONNX export requires skl2onnx or onnxmltools packages: {err2}")
                logger.info("ONNX Code snippet for MT5 EA deployment:")
                print("\n" + "#" * 60)
                print("# ONNX EXPORT CODE SNIPPET FOR MT5 INTEGRATION")
                print("#" * 60)
                print("import joblib")
                print("import skl2onnx")
                print("from skl2onnx.common.data_types import FloatTensorType")
                print("# 1. Save joblib model")
                print("joblib.dump(model, 'v5_gatekeeper_model.pkl')")
                print("# 2. Convert to ONNX")
                print(f"initial_types = [('input', FloatTensorType([None, {len(feature_cols)}]))]")
                print("onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_types)")
                print(f"with open('{output_path}', 'wb') as f:")
                print("    f.write(onnx_model.SerializeToString())")
                print("#" * 60 + "\n")
                return False
