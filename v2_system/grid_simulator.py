"""
v2_system/grid_simulator.py
===========================
Engine Mô Phỏng 72 Kịch Bản Grid/DCA và Gán Nhãn Cho Tập Train (v2 - Fixed & Robust).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Tạo không gian kịch bản tham số chuẩn hóa an toàn cho XAUUSD (Mean Reversion):
   - Step_0: [1.2x, 1.6x ATR] (Lưới đủ rộng chống trượt giá)
   - Step_Exp: [1.1, 1.3] (Mở rộng khoảng cách lưới theo cấp số nhân)
   - Max_Orders: [4, 5, 6] (Số tầng lệnh tối ưu)
   - Multiplier: [1.2, 1.5] (Martingale kiểm soát)
   - TP_BE: [0.5x, 0.8x ATR] (TP hợp lý che phủ spread & drawdown)
2. Chạy mô phỏng từng kịch bản trên nến M1 (10:00 - 12:00 VN) cho từng ngày.
3. Chấm điểm kịch bản theo Fitness Function nghiêm ngặt:
   Score = Net_Profit - (2.5 * Max_Drawdown) - (Penalty_500 nếu kẹt lệnh lúc 12:00).
4. Gán nhãn ngày:
   - Chỉ gán nhãn có lãi khi Score > 20.0 và hit TP trước 12:00.
   - Nếu không thỏa mãn -> Class 0 (No-Trade).
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class GridSimulator:
    def __init__(self, contract_size: float = 100.0):
        """
        :param contract_size: Kích thước hợp đồng XAUUSD (thông thường 100 oz / lot)
        """
        self.contract_size = contract_size

    @staticmethod
    def generate_parameter_grid() -> List[Dict[str, float]]:
        """
        Tạo danh sách các bộ tham số kịch bản grid an toàn cho Vàng (XAUUSD).
        3 * 2 * 3 * 2 * 2 = 72 kịch bản tham số tiêu chuẩn.
        """
        step_0_ratios = [1.2, 1.5, 1.8]
        step_exps = [1.1, 1.3]
        max_orders_list = [4, 5, 6]
        multipliers = [1.2, 1.5]
        tp_be_ratios = [0.5, 0.8]

        grid = []
        for s0, se, mo, mult, tp in itertools.product(step_0_ratios, step_exps, max_orders_list, multipliers, tp_be_ratios):
            grid.append({
                'step_0_ratio': s0,
                'step_exp': se,
                'max_orders': mo,
                'multiplier': mult,
                'tp_be_ratio': tp
            })
        return grid

    def simulate_day_scenario(
        self,
        exec_m1: pd.DataFrame,
        atr_14: float,
        close_0959: float,
        daily_vwap: float,
        params: Dict[str, float],
        base_lot: float = 0.1
    ) -> Dict[str, Any]:
        """
        Mô phỏng chi tiết 1 kịch bản trên nến M1 từ 10:00 đến 12:00.
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']
        tp_be_dist = params['tp_be_ratio'] * atr_14

        # Xác định hướng giao dịch Mean-Reversion
        direction = -1 if close_0959 >= daily_vwap else 1

        orders_placed = []
        price_1 = exec_m1['open'].iloc[0]
        orders_placed.append({'price': price_1, 'lot': base_lot})

        step_distances = [step_0 * (step_exp ** (i - 1)) for i in range(max_orders - 1)]

        next_trigger_prices = []
        curr_price = price_1
        for dist in step_distances:
            if direction == 1:
                curr_price = curr_price - dist
            else:
                curr_price = curr_price + dist
            next_trigger_prices.append(curr_price)

        next_order_idx = 0
        closed = False
        hit_tp = False
        net_profit = 0.0
        max_drawdown = 0.0

        for t, row in exec_m1.iterrows():
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # 1. Trigger lệnh mới
            while next_order_idx < len(next_trigger_prices):
                trig_p = next_trigger_prices[next_order_idx]
                triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

                if triggered:
                    lot_k = base_lot * (multiplier ** (next_order_idx + 1))
                    orders_placed.append({'price': trig_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            # 2. Tính giá Breakeven (BE)
            total_lot = sum(o['lot'] for o in orders_placed)
            price_be = sum(o['lot'] * o['price'] for o in orders_placed) / total_lot

            # 3. Tính giá Take Profit (TP)
            tp_price = price_be + tp_be_dist if direction == 1 else price_be - tp_be_dist

            # 4. Kiểm tra khớp TP
            if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                hit_tp = True
                closed = True
                net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                break

            # 5. Cập nhật Floating Drawdown
            floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size
            if floating_pnl < 0:
                max_drawdown = max(max_drawdown, abs(floating_pnl))

        # 6. Xử lý kẹt lệnh đến 12:00:00
        unclosed_at_12 = False
        if not closed:
            unclosed_at_12 = True
            final_close = exec_m1['close'].iloc[-1]
            net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

        # 7. Tính Fitness Score (Yêu cầu khắt khe hơn)
        penalty = 500.0 if unclosed_at_12 else 0.0
        fitness_score = net_profit - (2.5 * max_drawdown) - penalty

        return {
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'hit_tp': hit_tp,
            'unclosed_at_12': unclosed_at_12,
            'fitness_score': fitness_score,
            'num_orders': len(orders_placed)
        }

    def evaluate_training_set(
        self,
        feature_df: pd.DataFrame,
        daily_m1_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Mô phỏng các kịch bản cho tập Train (2020-2023).
        """
        param_grid = GridSimulator.generate_parameter_grid()
        print(f"[GridSimulator] Khởi tạo {len(param_grid)} kịch bản tham số an toàn...")

        labeled_records = []
        best_params_records = []

        for idx, row in feature_df.iterrows():
            date_str = row['date']
            if date_str not in daily_m1_dict:
                continue

            obs_df, exec_df = daily_m1_dict[date_str]
            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']

            # Lọc các ngày xu hướng quá mạnh sáng (tránh cản tàu)
            bb_zscore = row['bb_zscore_m15']
            momentum = row['morning_momentum']
            if abs(bb_zscore) > 2.5 or momentum > 0.8:
                # Ngày bùng nổ xu hướng sáng -> Đánh dấu No-Trade
                rec = row.to_dict()
                rec['best_fitness_score'] = -100.0
                rec['is_profitable'] = 0
                rec['step_0_ratio'] = np.nan
                rec['step_exp'] = np.nan
                rec['max_orders'] = np.nan
                rec['multiplier'] = np.nan
                rec['tp_be_ratio'] = np.nan
                rec['net_profit'] = 0.0
                rec['max_drawdown'] = 0.0
                labeled_records.append(rec)
                continue

            best_score = -float('inf')
            best_param = None
            best_res = None

            for p in param_grid:
                res = self.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
                if res['fitness_score'] > best_score:
                    best_score = res['fitness_score']
                    best_param = p
                    best_res = res

            # Điều kiện nghiêm ngặt để coi 1 ngày là có lãi chất lượng cao:
            # Score > 20.0 VÀ hit TP trước 12:00
            is_good_day = (best_score > 20.0) and (best_res is not None and best_res['hit_tp'])

            rec = row.to_dict()
            rec['best_fitness_score'] = best_score
            rec['is_profitable'] = 1 if is_good_day else 0

            if is_good_day and best_param is not None:
                rec['step_0_ratio'] = best_param['step_0_ratio']
                rec['step_exp'] = best_param['step_exp']
                rec['max_orders'] = best_param['max_orders']
                rec['multiplier'] = best_param['multiplier']
                rec['tp_be_ratio'] = best_param['tp_be_ratio']
                rec['net_profit'] = best_res['net_profit']
                rec['max_drawdown'] = best_res['max_drawdown']
                best_params_records.append(rec)
            else:
                rec['step_0_ratio'] = np.nan
                rec['step_exp'] = np.nan
                rec['max_orders'] = np.nan
                rec['multiplier'] = np.nan
                rec['tp_be_ratio'] = np.nan
                rec['net_profit'] = 0.0
                rec['max_drawdown'] = 0.0

            labeled_records.append(rec)

        labeled_df = pd.DataFrame(labeled_records)
        best_params_df = pd.DataFrame(best_params_records)
        
        prof_count = (labeled_df['is_profitable'] == 1).sum()
        print(f"[GridSimulator] Tổng số ngày train: {len(labeled_df)} | Ngày có kịch bản chất lượng cao (>20 score & Hit TP): {prof_count}")

        return labeled_df, best_params_df
