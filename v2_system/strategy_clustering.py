"""
v2_system/strategy_clustering.py
================================
Strategy Clustering Module cho Hệ Thống XAUUSD Grid/DCA (v2).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Nhận vào dữ liệu các ngày có kịch bản có lãi (Score > 0) cùng bộ tham số tốt nhất.
2. Dùng K-Means (K=3) gom cụm bộ tham số thành 3 cụm chiến thuật chuẩn:
   - Presets 1: Lưới Hẹp (Narrow / Aggressive)
   - Presets 2: Tiêu Chuẩn (Standard)
   - Presets 3: Phòng Thủ (Defensive)
3. Tính toán các tham số tâm cụm (Centroids) đại diện cho từng Preset.
4. Gán nhãn đa lớp (0: No-Trade, 1: Lưới hẹp, 2: Tiêu chuẩn, 3: Phòng thủ) cho tập Train.
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
        Thực hiện K-Means clustering trên không gian tham số tốt nhất.
        Các đặc trưng tham số: [step_0_ratio, step_exp, max_orders, multiplier, tp_be_ratio]
        """
        param_cols = ['step_0_ratio', 'step_exp', 'max_orders', 'multiplier', 'tp_be_ratio']
        X_params = best_params_df[param_cols].values

        # Scale đặc trưng trước khi K-Means
        X_scaled = self.scaler.fit_transform(X_params)
        
        # Fit K-Means
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Lấy tâm cụm dạng unscaled
        centroids_unscaled = self.scaler.inverse_transform(self.kmeans.cluster_centers_)

        # Sắp xếp các cụm theo độ rộng lưới (step_0_ratio * max_orders) để định danh rõ ràng:
        # Cụm 1: Lưới Hẹp (step_0_ratio nhỏ)
        # Cụm 2: Tiêu Chuẩn (trung bình)
        # Cụm 3: Phòng Thủ (step_0_ratio lớn, an toàn hơn)
        
        raw_centroids = []
        for c_idx in range(self.n_clusters):
            c_dict = {col: centroids_unscaled[c_idx][i] for i, col in enumerate(param_cols)}
            # Tính chỉ số độ hẹp/rộng
            c_dict['aggression_score'] = c_dict['step_0_ratio']
            c_dict['orig_idx'] = c_idx
            raw_centroids.append(c_dict)

        # Sắp xếp tăng dần theo step_0_ratio
        raw_centroids.sort(key=lambda x: x['step_0_ratio'])

        # Tạo map đổi tên cụm cũ -> cụm chuẩn 1, 2, 3
        cluster_mapping = {}
        names = ["Lưới Hẹp (Narrow)", "Tiêu Chuẩn (Standard)", "Phòng Thủ (Defensive)"]
        
        for new_label, c_dict in enumerate(raw_centroids, start=1):
            orig_idx = c_dict['orig_idx']
            cluster_mapping[orig_idx] = new_label
            
            # Làm tròn tham số tâm cụm cho sản xuất thực tế
            self.preset_centroids[new_label] = {
                'name': names[new_label - 1],
                'step_0_ratio': round(c_dict['step_0_ratio'], 2),
                'step_exp': round(c_dict['step_exp'], 2),
                'max_orders': int(round(c_dict['max_orders'])),
                'multiplier': round(c_dict['multiplier'], 2),
                'tp_be_ratio': round(c_dict['tp_be_ratio'], 2)
            }

        # Gán lại nhãn cụm 1, 2, 3
        mapped_labels = np.array([cluster_mapping[lbl] for lbl in cluster_labels])

        print("[StrategyClustering] Kết quả Gom cụm 3 Chiến thuật (Centroids):")
        for preset_id, p_info in self.preset_centroids.items():
            print(f"  Preset {preset_id} - {p_info['name']}: {p_info}")

        return mapped_labels, self.preset_centroids

    def assign_train_targets(
        self,
        labeled_df: pd.DataFrame,
        best_params_df: pd.DataFrame,
        mapped_cluster_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Kết hợp gán nhãn đa lớp (0, 1, 2, 3) cho toàn bộ tập dữ liệu Train.
        """
        df = labeled_df.copy()
        df['target'] = 0  # Mặc định 0: No-Trade

        # Gán nhãn cụm cho những ngày có lãi
        best_params_df_copy = best_params_df.copy()
        best_params_df_copy['preset_cluster'] = mapped_cluster_labels

        # Map lại vào DataFrame chính theo ngày
        date_to_cluster = dict(zip(best_params_df_copy['date'], best_params_df_copy['preset_cluster']))

        for i, row in df.iterrows():
            d = row['date']
            if d in date_to_cluster:
                df.at[i, 'target'] = date_to_cluster[d]

        target_counts = df['target'].value_counts().to_dict()
        print(f"[StrategyClustering] Phân bố nhãn đa lớp tập Train: {target_counts}")
        return df


if __name__ == '__main__':
    # Unit test cơ bản
    pass
