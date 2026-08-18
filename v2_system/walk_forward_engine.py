"""
v2_system/walk_forward_engine.py
================================
Rolling Walk-Forward Optimization & Validation Module cho XAUUSD Grid/DCA (v2 - Fixed KeyError).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Thực hiện Rolling Walk-Forward 6 tháng Train -> 6 tháng Test (2020-2023).
2. Xử lý an toàn các trường hợp cửa sổ ngắn hoặc mỏng dữ liệu.
3. Xuất bảng tổng hợp 7 bán niên và trả về metrics Walk-Forward.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from grid_simulator import GridSimulator
from strategy_clustering import StrategyClustering
from ml_trainer import MLTrainer
from backtest_engine import OOSBacktestEngine

class WalkForwardEngine:
    def __init__(
        self,
        feature_cols: List[str],
        initial_balance: float = 10000.0,
        risk_pct_per_session: float = 0.02
    ):
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.risk_pct_per_session = risk_pct_per_session

    @staticmethod
    def get_half_year_period(date_str: str) -> str:
        """
        Xác định bán niên: YYYY-H1 (Tháng 1-6) hoặc YYYY-H2 (Tháng 7-12).
        """
        dt = pd.to_datetime(date_str)
        half = "H1" if dt.month <= 6 else "H2"
        return f"{dt.year}-{half}"

    def run_walk_forward_process(
        self,
        train_feature_df: pd.DataFrame,
        daily_m1_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = train_feature_df.copy()
        df['period'] = df['date'].apply(self.get_half_year_period)
        
        periods = sorted(df['period'].unique())
        print(f"\n[WalkForwardEngine] Tìm thấy {len(periods)} bán niên trong tập Train: {periods}")

        if len(periods) < 2:
            raise ValueError("Cần ít nhất 2 bán niên để chạy Walk-Forward!")

        wf_results = []
        wf_trades_list = []
        cumulative_balance = self.initial_balance

        print("\n=========================================================================================================")
        print(" 🔄 BẮT ĐẦU QUY TRÌNH ROLLING WALK-FORWARD OPTIMIZATION (TRAIN 6 THÁNG -> TEST 6 THÁNG)")
        print("=========================================================================================================")

        simulator = GridSimulator()

        for k in range(1, len(periods)):
            train_periods = periods[:k]
            val_period = periods[k]

            train_sub_df = df[df['period'].isin(train_periods)].copy().reset_index(drop=True)
            val_sub_df = df[df['period'] == val_period].copy().reset_index(drop=True)

            print(f"\n 📌 [CỬA SỔ WALK-FORWARD {k}/{len(periods)-1}]")
            print(f"    • Train Period: {train_periods[0]} -> {train_periods[-1]} ({len(train_sub_df)} ngày)")
            print(f"    • Test Period : {val_period} ({len(val_sub_df)} ngày)")

            labeled_train_df, best_params_df = simulator.evaluate_training_set(train_sub_df, daily_m1_dict)

            if len(best_params_df) < 3:
                print(f"    ⚠️ Tập train cửa sổ {k} có quá ít ngày có lãi ({len(best_params_df)} ngày). Dùng preset mặc định.")
                # Sử dụng fallback preset nếu mỏng ngày train
                preset_centroids = {
                    1: {'name': 'Lưới Hẹp (Narrow)', 'step_0_ratio': 1.2, 'step_exp': 1.15, 'max_orders': 3, 'multiplier': 1.15},
                    2: {'name': 'Tiêu Chuẩn (Standard)', 'step_0_ratio': 1.5, 'step_exp': 1.20, 'max_orders': 4, 'multiplier': 1.20},
                    3: {'name': 'Phòng Thủ (Defensive)', 'step_0_ratio': 1.8, 'step_exp': 1.25, 'max_orders': 4, 'multiplier': 1.25}
                }
                trainer = MLTrainer(feature_cols=self.feature_cols)
                # Fit dummy hoặc dùng rule
                val_preds = np.random.choice([1, 2, 3], size=len(val_sub_df))
            else:
                clustering = StrategyClustering(n_clusters=3, random_state=42)
                mapped_labels, preset_centroids = clustering.fit_clusters(best_params_df)
                train_target_df = clustering.assign_train_targets(labeled_train_df, best_params_df, mapped_labels)

                trainer = MLTrainer(feature_cols=self.feature_cols)
                trainer.train_lightgbm(train_target_df)

                val_preds = trainer.predict(val_sub_df)

            backtest_engine = OOSBacktestEngine(
                preset_centroids=preset_centroids,
                initial_balance=cumulative_balance,
                risk_pct_per_session=self.risk_pct_per_session
            )

            val_trades_df, val_metrics = backtest_engine.run_backtest(val_sub_df, val_preds, daily_m1_dict)

            cumulative_balance = val_metrics['final_balance']
            val_return_pct = val_metrics['total_return_pct']

            wf_results.append({
                'window': k,
                'train_period': f"{train_periods[0]}~{train_periods[-1]}",
                'val_period': val_period,
                'train_days': len(train_sub_df),
                'val_days': val_metrics['total_days'],
                'traded_days': val_metrics['traded_days'],
                'win_rate': val_metrics['win_rate'],
                'profit_factor': val_metrics['profit_factor'],
                'max_drawdown_pct': val_metrics['max_drawdown_pct'],
                'val_return_pct': val_return_pct,
                'end_balance': cumulative_balance
            })

            wf_trades_list.append(val_trades_df)

        wf_summary_df = pd.DataFrame(wf_results)

        print("\n=========================================================================================================")
        print(" 📊 BẢNG TỔNG HỢP KẾT QUẢ ROLLING WALK-FORWARD VALIDATION (2020 - 2023)")
        print("=========================================================================================================")
        print(" | CS | Train Period  | Test 6M  | Days | WinRate (%) | Profit Factor | Max DD (%) | 6M Return (%) | End Balance ($) |")
        print(" +----+---------------+----------+------+-------------+---------------+------------+---------------+-----------------+")

        for r in wf_results:
            print(f" | {r['window']:<2} | {r['train_period']:<13} | {r['val_period']:<8} | {r['val_days']:<4} | {r['win_rate']:<11.1f} | {r['profit_factor']:<13.2f} | {r['max_drawdown_pct']:<10.2f} | +{r['val_return_pct']:<13.2f} | ${r['end_balance']:<15,.2f} |")
        print("=========================================================================================================\n")

        overall_return_pct = ((cumulative_balance - self.initial_balance) / self.initial_balance) * 100.0
        avg_win_rate = wf_summary_df['win_rate'].mean() if 'win_rate' in wf_summary_df.columns else 0.0
        avg_pf = wf_summary_df['profit_factor'].mean() if 'profit_factor' in wf_summary_df.columns else 0.0

        print(f" 🏆 TỔNG KẾT WALK-FORWARD OUT-OF-SAMPLE (2020-2023):")
        print(f"   • Vốn ban đầu:                     ${self.initial_balance:,.2f}")
        print(f"   • Số dư lũy kế cuối kỳ 2023:      ${cumulative_balance:,.2f}")
        print(f"   • Tổng Lợi Nhuận Walk-Forward Net: +{overall_return_pct:.2f}% (+${cumulative_balance - self.initial_balance:,.2f})")
        print(f"   • Win Rate Trung Bình 6M:          {avg_win_rate:.1f}%")
        print(f"   • Profit Factor Trung Bình 6M:      {avg_pf:.2f}\n")

        all_wf_trades = pd.concat(wf_trades_list, ignore_index=True) if wf_trades_list else pd.DataFrame()

        metrics = {
            'final_balance': cumulative_balance,
            'overall_return_pct': overall_return_pct,
            'avg_win_rate': avg_win_rate,
            'avg_pf': avg_pf,
            'wf_summary': wf_summary_df
        }

        return all_wf_trades, metrics


if __name__ == '__main__':
    pass
