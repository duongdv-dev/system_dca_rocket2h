"""
v6_system/label_builder.py
--------------------------
Module Xây Nhãn Nâng Cao cho AI (Phase 6 - Advanced AI Labeling System).
Quy tắc:
1. KHÔNG train nến tiếp theo tăng hay giảm.
2. TRAIN CHÍNH XÁC: Tại thời điểm t (với đặc trưng Distance, ATR, RSI, ADX...), nhìn về tương lai phiên [t+1, 12:00 VN]:
   - Y = 1: Nếu giá quay về Anchor / đạt Basket TP trước 12:00 VN.
   - Y = 0: Nếu giá không thể quay về Anchor / không đạt TP trước 12:00 VN.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("V6LabelBuilder")


class V6LabelBuilder:
    """Class xây dựng các loại nhãn AI chuyên sâu cho phiên 10:00 -> 12:00 VN."""

    def __init__(self, default_step: float = 5.0, default_tp: float = 2.0, max_adverse: float = 15.0):
        self.default_step = default_step
        self.default_tp = default_tp
        self.max_adverse = max_adverse

    def build_all_labels(self, df_session_features: pd.DataFrame) -> pd.DataFrame:
        """
        Xây dựng đồng thời 3 loại nhãn AI trên dữ liệu phiên:
        1. y_anchor       : Nhãn hồi về Anchor trước 12:00 VN
        2. y_basket_tp    : Nhãn đạt Basket TP trước 12:00 VN
        3. y_safe_revert : Nhãn hồi Anchor an toàn (không bị vi phạm rủi ro Max Adverse)
        """
        logger.info("Bắt đầu xây dựng 3 loại Nhãn AI chuyên sâu (Phase 6)...")
        df = df_session_features.copy()

        session_dfs = []
        grouped = df.groupby("date")

        for date_val, group in grouped:
            if group.empty:
                continue

            group = group.sort_values("dt_vn").copy()
            n = len(group)

            anchor_price = float(group.iloc[0]["anchor_price"]) if "anchor_price" in group.columns else float(group.iloc[0]["open"])

            close_vals = group["close"].values
            high_vals = group["high"].values
            low_vals = group["low"].values

            # Tương lai min low và max high bằng reverse cumulative
            rev_low = low_vals[::-1]
            rev_high = high_vals[::-1]

            cum_min_low = np.minimum.accumulate(rev_low)[::-1]
            cum_max_high = np.maximum.accumulate(rev_high)[::-1]

            future_min_low = np.roll(cum_min_low, -1)
            future_min_low[-1] = low_vals[-1]

            future_max_high = np.roll(cum_max_high, -1)
            future_max_high[-1] = high_vals[-1]

            # 1. Label: y_anchor
            above_mask = close_vals > anchor_price
            below_mask = close_vals < anchor_price
            equal_mask = ~above_mask & ~below_mask

            y_anchor = np.zeros(n, dtype=int)
            y_anchor[above_mask & (future_min_low <= anchor_price)] = 1
            y_anchor[below_mask & (future_max_high >= anchor_price)] = 1
            y_anchor[equal_mask] = 1

            # 2. Label: y_safe_revert (Hồi Anchor trước khi giá lệch quá max_adverse)
            # Nếu close > anchor: rủi ro xảy ra khi future_max_high > close + max_adverse
            # Nếu close < anchor: rủi ro xảy ra khi future_min_low < close - max_adverse
            y_safe = np.copy(y_anchor)

            breach_above = above_mask & (future_max_high > close_vals + self.max_adverse)
            breach_below = below_mask & (future_min_low < close_vals - self.max_adverse)
            
            y_safe[breach_above | breach_below] = 0

            # 3. Label: y_basket_tp (Đạt Target Profit giỏ lệnh)
            # Với nến t, nếu giá lệch > step thì cần quay về nấc TP tương ứng
            tp_target_sell = anchor_price + self.default_step - self.default_tp
            tp_target_buy = anchor_price - self.default_step + self.default_tp

            y_basket_tp = np.zeros(n, dtype=int)
            # Chiều Sell: giá hiện tại đã vọt qua Anchor+Step, cần future_min_low <= tp_target_sell
            y_basket_tp[above_mask & (future_min_low <= tp_target_sell)] = 1
            # Chiều Buy: giá hiện tại rơi dưới Anchor-Step, cần future_max_high >= tp_target_buy
            y_basket_tp[below_mask & (future_max_high >= tp_target_buy)] = 1
            y_basket_tp[equal_mask] = 1

            group["y_anchor"] = y_anchor
            group["y_basket_tp"] = y_basket_tp
            group["y_safe_revert"] = y_safe

            session_dfs.append(group)

        df_final = pd.concat(session_dfs, ignore_index=True)
        
        # Thống kê Class Balance
        for col in ["y_anchor", "y_basket_tp", "y_safe_revert"]:
            pos_cnt = int(df_final[col].sum())
            total = len(df_final)
            pct = (pos_cnt / total) * 100.0 if total > 0 else 0.0
            logger.info(f"Phân phối nhãn [{col:<14}]: Y=1 ({pos_cnt:,} / {total:,} -> {pct:.2f}%) | Y=0 ({total - pos_cnt:,})")

        return df_final
