import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from dca_engine import DCAEngine

class PresetOptimizer:
    """
    Tự động phân nhóm trạng thái thị trường (Market Regimes) bằng K-Means/Silhouette Score,
    chạy Grid Search tìm kiếm các Presets ATR Dynamic tạo LỢI NHUẬN CAO THỰC TẾ trên tập Train (2020-2023)
    và kiểm định độc lập trên tập Test (2024-2025).
    """
    def __init__(self, account_balance: float = 1000.0, base_lot: float = 0.05):
        self.account_balance = account_balance
        self.dca_engine = DCAEngine(account_balance=account_balance, risk_pct_per_trade=1.5)

    def cluster_market_regimes(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str], KMeans, StandardScaler]:
        feature_cols = ['asian_range_atr_ratio', 'asian_atr_m15', 'asian_return_pct', 'asian_body_ratio']
        X = features_df[feature_cols].fillna(0).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        best_k = 3
        best_score = -1
        best_labels = None
        best_kmeans = None

        for k in [3, 4, 5]:
            if len(X_scaled) <= k:
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_kmeans = kmeans

        df_clustered = features_df.copy()
        df_clustered['regime_id'] = best_labels

        regime_names = {}
        for r_id in range(best_k):
            sub = df_clustered[df_clustered['regime_id'] == r_id]
            mean_ret = sub['asian_return_pct'].mean()
            mean_ratio = sub['asian_range_atr_ratio'].mean()

            if mean_ratio > 4.5 and abs(mean_ret) > 0.25:
                name = f"Regime {r_id}: High Volatility Trend"
            elif mean_ratio < 2.5:
                name = f"Regime {r_id}: Low Volatility Sideway"
            elif mean_ret > 0.1:
                name = f"Regime {r_id}: Moderate Bullish"
            elif mean_ret < -0.1:
                name = f"Regime {r_id}: Moderate Bearish"
            else:
                name = f"Regime {r_id}: Choppy / Normal"
            
            regime_names[r_id] = name

        print(f"-> Đã chọn K={best_k} Regimes (Silhouette Score: {best_score:.3f})")
        return df_clustered, regime_names, best_kmeans, scaler

    def optimize_presets(self, 
                         df_clustered: pd.DataFrame, 
                         trading_days_data: Dict[str, pd.DataFrame],
                         max_dd_target_pct: float = 10.0) -> Dict[str, Any]:
        """
        Grid Search tìm kiếm Presets tạo Lợi Nhuận Cao trên tập Train với Max Drawdown < 10%.
        """
        param_grid = []
        for entry_rule in ['REVERSAL_TO_10H', 'FOLLOW_ASIAN_TREND', 'ALWAYS_BUY', 'ALWAYS_SELL']:
            for step_atr_mult in [1.0, 1.5, 2.0, 2.5]:
                for tp_atr_mult in [0.8, 1.2, 1.8, 2.5]:
                    for lot_scale_mult in [1.0, 1.5, 2.0]:  # Tăng lot động để ăn lợi nhuận cao
                        for step_mult in [1.0, 1.2, 1.5]:
                            for lot_mult in [1.2, 1.5, 1.8]:
                                for max_orders in [3, 5, 7]:
                                    for max_loss in [60.0, 80.0, 100.0]:
                                        param_grid.append({
                                            'entry_rule': entry_rule,
                                            'step_atr_mult': step_atr_mult,
                                            'tp_atr_mult': tp_atr_mult,
                                            'lot_scale_mult': lot_scale_mult,
                                            'step_multiplier': step_mult,
                                            'lot_multiplier': lot_mult,
                                            'max_dca_orders': max_orders,
                                            'max_loss_usd': max_loss
                                        })

        regimes = sorted(df_clustered['regime_id'].unique())
        best_presets_summary = {}

        for r_id in regimes:
            regime_days = df_clustered[df_clustered['regime_id'] == r_id]
            print(f"⚡ High-Yield Grid Search cho Regime {r_id} ({len(regime_days)} ngày Train)...")

            candidate_results = []

            for preset in param_grid:
                daily_results = []
                cum_pnl = 0.0
                peak_pnl = 0.0
                max_dd_usd = 0.0

                for _, row in regime_days.iterrows():
                    date_str = row['date']
                    if date_str not in trading_days_data:
                        continue
                    
                    trade_df = trading_days_data[date_str]
                    res = self.dca_engine.simulate_day(
                        trade_candles=trade_df,
                        asian_atr_m15=row['asian_atr_m15'],
                        asian_return_pct=row['asian_return_pct'],
                        preset=preset
                    )
                    daily_results.append(res)

                    cum_pnl += res['profit_usd']
                    if cum_pnl > peak_pnl:
                        peak_pnl = cum_pnl
                    dd = peak_pnl - cum_pnl
                    if dd > max_dd_usd:
                        max_dd_usd = dd

                if not daily_results:
                    continue

                profits = [r['profit_usd'] for r in daily_results]
                wins = [p for p in profits if p > 0]
                losses = [abs(p) for p in profits if p < 0]

                win_rate = len(wins) / len(profits) * 100.0 if profits else 0.0
                total_profit = sum(profits)
                profit_factor = (sum(wins) / (sum(losses) + 1e-8)) if losses else 999.0
                max_dd_pct = (max_dd_usd / self.account_balance) * 100.0

                # Lọc các Presets có Max Drawdown < 10% và Tổng Lợi Nhuận Cao
                if max_dd_pct <= max_dd_target_pct and total_profit > 100.0:
                    candidate_results.append({
                        'preset': preset,
                        'total_profit_usd': round(total_profit, 2),
                        'profit_factor': round(profit_factor, 2),
                        'win_rate': round(win_rate, 2),
                        'max_drawdown_pct': round(max_dd_pct, 2),
                        'total_days': len(profits)
                    })

            # Sắp xếp ưu tiên Tổng Lợi Nhuận USD & Profit Factor
            candidate_results.sort(key=lambda x: (x['total_profit_usd'], x['profit_factor']), reverse=True)

            if candidate_results:
                best_preset_entry = candidate_results[0]
                best_presets_summary[f"Regime_{r_id}"] = best_preset_entry
            else:
                best_presets_summary[f"Regime_{r_id}"] = {
                    'preset': {'entry_rule': 'NO_TRADE'},
                    'total_profit_usd': 0.0,
                    'profit_factor': 0.0,
                    'win_rate': 0.0,
                    'max_drawdown_pct': 0.0,
                    'total_days': len(regime_days),
                    'note': 'NO_TRADE (Max DD constraint violated)'
                }

        return best_presets_summary
