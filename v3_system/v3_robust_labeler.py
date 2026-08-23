"""
v3_system/v3_robust_labeler.py
==============================
Module Gán Nhãn Bền Vững Theo Cụm Trạng Thái Vi Cấu Trúc (Cluster-Based Robust Labeler).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Triệt tiêu hoàn toàn Hindsight Label Noise (hiện tượng chọn preset ngẫu nhiên thắng của 1 ngày).
2. Gom nhóm các ngày giao dịch trong tập Train thành K cụm vi cấu trúc (K-Means/GMM).
3. Đánh giá ma trận hiệu năng của 6 Archetypes trên TOÀN BỘ các ngày trong cụm.
4. Trả về nhãn `robust_archetype_id` và `robust_is_safe` bền vững cho từng ngày.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from v3_preset_generator import V3PresetGenerator
from v3_preset_clusterer import V3PresetClusterer

class V3RobustLabeler:
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.generator = V3PresetGenerator()
        self.presets_list = V3PresetGenerator.generate_540_candidate_presets()

    def calculate_day_preset_matrix(
        self,
        train_df: pd.DataFrame,
        train_m1_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> np.ndarray:
        """
        Tính ma trận utility score cho tất cả ngày x 540 presets.
        Kích thước: (N_days, 540)
        Utility = pnl_atr - 3.0 * dd_atr - (50.0 if hit_sl else 0.0)
        """
        n_days = len(train_df)
        n_presets = len(self.presets_list)
        utility_matrix = np.zeros((n_days, n_presets))

        for d_idx, (_, row) in enumerate(train_df.iterrows()):
            date_str = row['date']
            obs_df, exec_df = train_m1_dict[date_str]
            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']
            bb_slope_m15 = row['bb_slope_m15']

            for p_idx, p in enumerate(self.presets_list):
                res = self.generator.simulate_day(
                    exec_df, atr_14, close_0959, daily_vwap, p, bb_slope_m15=bb_slope_m15
                )
                if res['num_orders'] == 0:
                    u = 0.0
                else:
                    sl_penalty = 50.0 if res['hit_sl'] else 0.0
                    u = res['pnl_atr'] - (3.0 * res['dd_atr']) - sl_penalty
                utility_matrix[d_idx, p_idx] = u

        return utility_matrix

    def generate_robust_labels(
        self,
        train_df: pd.DataFrame,
        train_m1_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Tạo nhãn bền vững dựa trên Cụm Ngày (Cluster-Based Robust Labeling).
        """
        df = train_df.copy()
        X_feat = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X_feat)

        # 1. Phân cụm các ngày giao dịch
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        df['microstructure_cluster'] = cluster_labels

        print(f"[V3RobustLabeler] Đã phân cụm {len(df)} ngày thành {self.n_clusters} cụm vi cấu trúc thị trường.")

        # 2. Tính ma trận Utility (N_days x 540)
        utility_matrix = self.calculate_day_preset_matrix(df, train_m1_dict)

        # Map preset_index -> archetype_id
        preset_archetypes = np.array([V3PresetClusterer.classify_preset(p) for p in self.presets_list])

        robust_archetypes = []
        robust_is_safe = []

        # 3. Tính điểm trung bình của từng Archetype trong từng Cụm
        for d_idx, c_id in enumerate(cluster_labels):
            # Các ngày cùng cụm
            cluster_day_indices = np.where(cluster_labels == c_id)[0]

            # Với từng Archetype (1 đến 5), tính điểm trung bình của các preset thuộc Archetype đó trong cụm
            archetype_scores = {}
            for aid in range(1, 6):
                matching_preset_mask = (preset_archetypes == aid)
                if np.sum(matching_preset_mask) > 0:
                    # Utility trung bình của archetype này trong toàn bộ các ngày thuộc cụm
                    cluster_preset_utilities = utility_matrix[cluster_day_indices][:, matching_preset_mask]
                    mean_u = np.mean(cluster_preset_utilities)
                    archetype_scores[aid] = mean_u
                else:
                    archetype_scores[aid] = -999.0

            # Tìm Archetype tốt nhất cho cụm
            best_aid = max(archetype_scores, key=archetype_scores.get)
            best_cluster_u = archetype_scores[best_aid]

            # Nếu điểm utility trung bình của cụm < 0 -> Cụm này rủi ro cao -> Gán NO TRADE (0)
            if best_cluster_u < 0.2:
                robust_archetypes.append(0)
                robust_is_safe.append(0)
            else:
                robust_archetypes.append(best_aid)
                robust_is_safe.append(1)

        df['robust_archetype_id'] = robust_archetypes
        df['robust_is_safe'] = robust_is_safe

        safe_cnt = sum(robust_is_safe)
        print(f"[V3RobustLabeler] Hoàn tất gán nhãn bền vững: {safe_cnt}/{len(df)} ngày được duyệt AN TOÀN (Safe Trade Days).")
        return df


if __name__ == '__main__':
    pass
