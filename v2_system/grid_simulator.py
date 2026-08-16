"""
v2_system/grid_simulator.py
===========================
Engine Mô Phỏng 72 Kịch Bản Grid/DCA và Gán Nhãn Cho Tập Train.
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Tạo không gian 72 kịch bản tham số (Grid Search):
   - Step_0: [0.8x, 1.2x ATR]
   - Step_Exp: [1.0, 1.2]
   - Max_Orders: [3, 4, 5, 6] (hoặc [3, 4, 5])
   - Multiplier: [1.0, 1.3, 1.5]
   - TP_BE: [0.4x, 0.7x ATR]
2. Chạy mô phỏng từng kịch bản trên nến M1 (10:00 - 12:00 VN) cho từng ngày.
3. Chấm điểm kịch bản theo Fitness Function:
   Score = Net_Profit - (2.0 * Max_Drawdown) - (Penalty_500 nếu kẹt lệnh lúc 12:00).
4. Gán nhãn ngày:
   - Nếu best Score <= 0 -> Class 0 (No-Trade).
   - Nếu best Score > 0 -> Lưu bộ tham số tốt nhất.
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
        Tạo danh sách các bộ tham số kịch bản grid.
        2 * 2 * 3 * 3 * 2 = 72 kịch bản tham số tiêu chuẩn.
        """
        step_0_ratios = [0.8, 1.2]
        step_exps = [1.0, 1.2]
        max_orders_list = [3, 4, 5]  # 3 options -> 2 * 2 * 3 * 3 * 2 = 72 scenarios
        multipliers = [1.0, 1.3, 1.5]
        tp_be_ratios = [0.4, 0.7]

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
        
        :param exec_m1: DataFrame nến M1 trong khung 10:00 - 12:00 (chứa open, high, low, close)
        :param atr_14: ATR M15 tính lúc 09:59
        :param close_0959: Giá đóng cửa 09:59
        :param daily_vwap: VWAP phiên sáng
        :param params: Bộ tham số grid (step_0_ratio, step_exp, max_orders, multiplier, tp_be_ratio)
        :param base_lot: Kích thước lot cơ sở
        :return: Dict chứa kết quả (net_profit, max_drawdown, hit_tp, unclosed_at_12, fitness_score)
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']
        tp_be_dist = params['tp_be_ratio'] * atr_14

        # Xác định hướng giao dịch theo Mean-Reversion:
        # Nếu giá 09:59 nằm trên VWAP -> Đang cao -> Đặt SELL kỳ vọng hồi về
        # Nếu giá 09:59 nằm dưới VWAP -> Đang thấp -> Đặt BUY kỳ vọng hồi về
        direction = -1 if close_0959 >= daily_vwap else 1  # 1: BUY, -1: SELL

        # Tính trước mức giá và khối lượng cho tối đa max_orders lệnh
        # Order 1 khớp ngay tại Open nến 10:00
        orders_placed = []
        
        # Mở lệnh 1 tại nến 10:00 Open
        price_1 = exec_m1['open'].iloc[0]
        orders_placed.append({'price': price_1, 'lot': base_lot})

        # Danh sách khoảng cách các lệnh tiếp theo
        # Lệnh k (k>=2) cách lệnh k-1 một khoảng Step_{k-2}
        step_distances = [step_0 * (step_exp ** (i - 1)) for i in range(max_orders - 1)]

        # Mức giá kích hoạt cho các lệnh tiếp theo (nếu giá đi ngược)
        next_trigger_prices = []
        curr_price = price_1
        for dist in step_distances:
            if direction == 1:  # BUY -> Đi ngược là giá giảm
                curr_price = curr_price - dist
            else:  # SELL -> Đi ngược là giá tăng
                curr_price = curr_price + dist
            next_trigger_prices.append(curr_price)

        next_order_idx = 0  # Chỉ số của lệnh tiếp theo chờ kích hoạt (trong next_trigger_prices)

        # Trạng thái trong phiên
        closed = False
        hit_tp = False
        net_profit = 0.0
        max_drawdown = 0.0
        peak_equity = 0.0

        # Lặp qua từng nến M1 (10:00 - 12:00)
        for t, row in exec_m1.iterrows():
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # 1. Kiểm tra kích hoạt lệnh mới (nếu chưa đạt max_orders)
            while next_order_idx < len(next_trigger_prices):
                trig_p = next_trigger_prices[next_order_idx]
                triggered = False
                if direction == 1 and low_t <= trig_p:  # BUY trigger khi Low <= Target
                    triggered = True
                elif direction == -1 and high_t >= trig_p:  # SELL trigger khi High >= Target
                    triggered = True

                if triggered:
                    lot_k = base_lot * (multiplier ** (next_order_idx + 1))
                    orders_placed.append({'price': trig_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            # 2. Tính giá hòa vốn Breakeven (BE) hiện tại
            total_lot = sum(o['lot'] for o in orders_placed)
            price_be = sum(o['lot'] * o['price'] for o in orders_placed) / total_lot

            # 3. Tính giá Take Profit (TP) hiện tại
            if direction == 1:
                tp_price = price_be + tp_be_dist
            else:
                tp_price = price_be - tp_be_dist

            # 4. Kiểm tra cắn TP trong nến M1 này
            if direction == 1 and high_t >= tp_price:
                # Khớp TP lệnh BUY
                hit_tp = True
                closed = True
                net_profit = sum(o['lot'] * (tp_price - o['price']) for o in orders_placed) * self.contract_size
                break
            elif direction == -1 and low_t <= tp_price:
                # Khớp TP lệnh SELL
                hit_tp = True
                closed = True
                net_profit = sum(o['lot'] * (o['price'] - tp_price) for o in orders_placed) * self.contract_size
                break

            # 5. Cập nhật Floating PnL & Max Drawdown tại Close nến t
            if direction == 1:
                floating_pnl = sum(o['lot'] * (close_t - o['price']) for o in orders_placed) * self.contract_size
            else:
                floating_pnl = sum(o['lot'] * (o['price'] - close_t) for o in orders_placed) * self.contract_size

            # Drawdown là mức lỗ trạng thái lớn nhất
            if floating_pnl < 0:
                max_drawdown = max(max_drawdown, abs(floating_pnl))

        # 6. Xử lý kẹt lệnh đến 12:00:00 (Chưa cắn TP)
        unclosed_at_12 = False
        if not closed:
            unclosed_at_12 = True
            final_close = exec_m1['close'].iloc[-1]
            if direction == 1:
                net_profit = sum(o['lot'] * (final_close - o['price']) for o in orders_placed) * self.contract_size
            else:
                net_profit = sum(o['lot'] * (o['price'] - final_close) for o in orders_placed) * self.contract_size

        # 7. Tính Fitness Score
        # Score = Net_Profit - 2.0 * Max_Drawdown - (500 nếu kẹt lệnh tới 12:00)
        penalty = 500.0 if unclosed_at_12 else 0.0
        fitness_score = net_profit - (2.0 * max_drawdown) - penalty

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
        Mô phỏng 72 kịch bản cho tất cả các ngày trong tập Train (2020-2023).
        
        :return: (labeled_df, best_params_df)
        """
        param_grid = GridSimulator.generate_parameter_grid()
        print(f"[GridSimulator] Khởi tạo {len(param_grid)} kịch bản tham số...")

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

            best_score = -float('inf')
            best_param = None
            best_res = None

            for p in param_grid:
                res = self.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
                if res['fitness_score'] > best_score:
                    best_score = res['fitness_score']
                    best_param = p
                    best_res = res

            # Gán nhãn:
            # Score <= 0 -> 0 (No-Trade)
            # Score > 0 -> Sẽ được gom cụm K-Means ở bước tiếp theo
            rec = row.to_dict()
            rec['best_fitness_score'] = best_score
            rec['is_profitable'] = 1 if best_score > 0 else 0

            if best_score > 0 and best_param is not None:
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
        print(f"[GridSimulator] Tổng số ngày train: {len(labeled_df)} | Ngày có kịch bản có lãi (>0 score): {prof_count}")

        return labeled_df, best_params_df
