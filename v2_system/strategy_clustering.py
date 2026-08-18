"""
v2_system/strategy_clustering.py
================================
Strategy Clustering Module cho Hệ Thống XAUUSD Grid/DCA (v2 - Fixed TP 10:00 Open).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Gom cụm K-Means (K=3) bộ tham số [step_0_ratio, step_exp, max_orders, multiplier].
2. TP cố định tại Giá Open 10:00 AM.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

class StrategyClustering:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.preset_centroids = {}  # {1: params, 2: params, 3: params}

    def fit_clusters(self, best_params_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
        """
        Thực hiện K-Means clustering trên [step_0_ratio, step_exp, max_orders, multiplier].
        """
        param_cols = ['step_0_ratio', 'step_exp', 'max_orders', 'multiplier']
        X_params = best_params_df[param_cols].values

        X_scaled = self.scaler.fit_transform(X_params)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        centroids_unscaled = self.scaler.inverse_transform(self.kmeans.cluster_centers_)

        raw_centroids = []
        for c_idx in range(self.n_clusters):
            c_dict = {col: centroids_unscaled[c_idx][i] for i, col in enumerate(param_cols)}
            c_dict['aggression_score'] = c_dict['step_0_ratio']
            c_dict['orig_idx'] = c_idx
            raw_centroids.append(c_dict)

        raw_centroids.sort(key=lambda x: x['step_0_ratio'])

        cluster_mapping = {}
        names = ["Lưới Hẹp (Narrow)", "Tiêu Chuẩn (Standard)", "Phòng Thủ (Defensive)"]
        
        for new_label, c_dict in enumerate(raw_centroids, start=1):
            orig_idx = c_dict['orig_idx']
            cluster_mapping[orig_idx] = new_label
            
            self.preset_centroids[new_label] = {
                'name': names[new_label - 1],
                'step_0_ratio': round(c_dict['step_0_ratio'], 2),
                'step_exp': round(c_dict['step_exp'], 2),
                'max_orders': int(round(c_dict['max_orders'])),
                'multiplier': round(c_dict['multiplier'], 2)
            }

        mapped_labels = np.array([cluster_mapping[lbl] for lbl in cluster_labels])

        best_params_df_copy = best_params_df.copy()
        best_params_df_copy['mapped_cluster'] = mapped_labels

        print("\n=========================================================================================================")
        print(" 🎯 KẾT QUẢ GOM CỤM K-MEANS (K=3) - TP CỐ ĐỊNH TẠI GIÁ OPEN 10:00 AM")
        print("=========================================================================================================")
        print(" | Preset ID | Tên Chiến Thuật         | Số Ngày Train | Mean PnL ($) | Mean DD ($) | Tham Số Tâm Cụm (Centroid) |")
        print(" +-----------+-------------------------+---------------+--------------+-------------+----------------------------+")

        for p_id in [1, 2, 3]:
            sub_c = best_params_df_copy[best_params_df_copy['mapped_cluster'] == p_id]
            mean_pnl = sub_c['net_profit'].mean() if not sub_c.empty else 0.0
            mean_dd = sub_c['max_drawdown'].mean() if not sub_c.empty else 0.0
            c_info = self.preset_centroids[p_id]
            c_str = f"S0:{c_info['step_0_ratio']} | Exp:{c_info['step_exp']} | MaxOrd:{c_info['max_orders']} | Mult:{c_info['multiplier']}"
            print(f" | Preset {p_id}  | {c_info['name']:<23} | {len(sub_c):<13} | {mean_pnl:<12,.2f} | {mean_dd:<11,.2f} | {c_str:<26} |")
        print("=========================================================================================================\n")

        return mapped_labels, self.preset_centroids

    def assign_train_targets(
        self,
        labeled_df: pd.DataFrame,
        best_params_df: pd.DataFrame,
        mapped_cluster_labels: np.ndarray
    ) -> pd.DataFrame:
        df = labeled_df.copy()
        df['target'] = 0

        best_params_df_copy = best_params_df.copy()
        best_params_df_copy['preset_cluster'] = mapped_cluster_labels

        date_to_cluster = dict(zip(best_params_df_copy['date'], best_params_df_copy['preset_cluster']))

        for i, row in df.iterrows():
            d = row['date']
            if d in date_to_cluster:
                df.at[i, 'target'] = date_to_cluster[d]

        target_counts = df['target'].value_counts().to_dict()
        print(f"[StrategyClustering] Bảng Phân Bổ Nhãn Đa Lớp Tập Train:")
        print(f"  • Class 0 (No-Trade):   {target_counts.get(0, 0)} ngày ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  • Class 1 (Lưới Hẹp):   {target_counts.get(1, 0)} ngày ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  • Class 2 (Tiêu Chuẩn): {target_counts.get(2, 0)} ngày ({target_counts.get(2, 0)/len(df)*100:.1f}%)")
        print(f"  • Class 3 (Phòng Thủ):  {target_counts.get(3, 0)} ngày ({target_counts.get(3, 0)/len(df)*100:.1f}%)\n")

        return df


if __name__ == '__main__':
    pass
