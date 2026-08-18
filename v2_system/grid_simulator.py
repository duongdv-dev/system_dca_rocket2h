"""
v2_system/grid_simulator.py
===========================
Engine Mô Phỏng Kịch Bản Grid/DCA & Chấm Điểm Fitness (v2 - True Intraday Stretch & Fixed 10:00 Open TP).
Được thiết kế bởi Senior Quantitative Researcher.

QUY TẮC CỐT LÕI ĐÍCH THỰC CỦA USER:
1. Giá Tham Chiếu & TP Target: `price_1000` là Giá Open 10:00 AM.
   - TP Target của TOÀN BỘ CÁC LỆNH LUÔN CỐ ĐỊNH TẠI `price_1000`.
2. Điểm Vào Lệnh 1 (Order 1 Trigger):
   - Lệnh 1 KHÔNG MỞ NGAY LÚC 10:00:00 AM!
   - Hệ thống chờ giá nảy giãn ra xa `price_1000` đúng khoảng cách `Step_0` (Step_0 = step_0_ratio * ATR_14):
     + Lệnh SELL: Kích hoạt khi High M1 >= price_1000 + Step_0 (Giá chạm price_1000 + Step_0).
     + Lệnh BUY : Kích hoạt khi Low M1 <= price_1000 - Step_0 (Giá chạm price_1000 - Step_0).
3. Rải Lệnh Tiếp Theo (Order 2, 3, 4...):
   - Nếu giá tiếp tục nảy đi xa hơn, rải tiếp Order 2, 3, 4 theo khoảng cách `Step_k`.
4. Chốt Lời (Take Profit Exit):
   - Khi giá hồi quay trở lại chạm đúng `price_1000`:
     + Tất cả các lệnh đã khớp (Order 1, Order 2, Order 3...) ĐỀU ĐÓNG TẠI `price_1000` VÀ TẤT CẢ ĐỀU ĐẠT LỢI NHUẬN DƯƠNG LỚN!
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

class GridSimulator:
    def __init__(self, contract_size: float = 100.0):
        self.contract_size = contract_size

    @staticmethod
    def generate_parameter_grid() -> List[Dict[str, float]]:
        """
        Tạo không gian 32 kịch bản tham số chuẩn theo đúng quy tắc giãn `Step_0` trước khi mở Order 1.
        """
        step_0_ratios = [0.6, 1.0, 1.4, 1.8]
        step_exps = [1.1, 1.25]
        max_orders_list = [3, 4]
        multipliers = [1.15, 1.3]

        grid = []
        for s0, se, mo, mult in itertools.product(step_0_ratios, step_exps, max_orders_list, multipliers):
            grid.append({
                'step_0_ratio': s0,
                'step_exp': se,
                'max_orders': mo,
                'multiplier': mult
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
        Mô phỏng 1 ngày theo đúng quy tắc: Chờ giá giãn đủ `Step_0` mới khớp Order 1, TP cố định tại Open 10:00.
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']

        # Hướng Mean Reversion: 09:59 >= VWAP -> SELL; 09:59 < VWAP -> BUY
        direction = -1 if close_0959 >= daily_vwap else 1

        price_1000 = exec_m1['open'].iloc[0] # Giá Open 10:00 AM
        tp_price = price_1000               # TP LUÔN CỐ ĐỊNH TẠI GIÁ OPEN 10:00 AM!

        # Tính danh sách mức giá kích hoạt cho Order 1, Order 2, Order 3, Order 4...
        # Order 1 khớp tại (price_1000 + step_0) đối với SELL hoặc (price_1000 - step_0) đối với BUY
        trigger_prices = []
        curr_p = price_1000 - step_0 if direction == 1 else price_1000 + step_0
        trigger_prices.append(curr_p)

        for i in range(1, max_orders):
            dist_i = step_0 * (step_exp ** i)
            curr_p = curr_p - dist_i if direction == 1 else curr_p + dist_i
            trigger_prices.append(curr_p)

        orders_placed = []
        next_order_idx = 0
        closed = False
        hit_tp = False
        hit_minute = -1
        net_profit = 0.0
        max_drawdown = 0.0

        for m_idx, (t, row) in enumerate(exec_m1.iterrows()):
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # 1. Kích hoạt các lệnh (Từ Order 1 trở đi khi giá chạm trigger price)
            while next_order_idx < len(trigger_prices):
                trig_p = trigger_prices[next_order_idx]
                triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

                if triggered:
                    lot_k = base_lot * (multiplier ** next_order_idx)
                    orders_placed.append({'price': trig_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            # 2. Nếu đã có ít nhất 1 lệnh được mở
            if len(orders_placed) > 0:
                floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size

                if floating_pnl < 0:
                    max_drawdown = max(max_drawdown, abs(floating_pnl))

                # 3. Kiểm tra cắn TP cố định tại Giá Open 10:00 AM (price_1000)
                if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                    hit_tp = True
                    closed = True
                    hit_minute = m_idx
                    net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                    break

        # 4. Xử lý Hard Exit 12:00:00 (Nếu đã khớp lệnh nhưng chưa cắn TP)
        unclosed_at_12 = False
        if len(orders_placed) > 0 and not closed:
            unclosed_at_12 = True
            final_close = exec_m1['close'].iloc[-1]
            net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

        total_volume = sum(o['lot'] for o in orders_placed) if len(orders_placed) > 0 else 0.1
        pnl_points = net_profit / (self.contract_size * total_volume)
        dd_points = max_drawdown / (self.contract_size * total_volume)

        pnl_atr = pnl_points / (atr_14 + 1e-8)
        dd_atr = dd_points / (atr_14 + 1e-8)

        penalty_atr = 15.0 if unclosed_at_12 else 0.0
        fitness_score = pnl_atr - (2.5 * dd_atr) - penalty_atr

        return {
            'direction': 'SELL' if direction == -1 else 'BUY',
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'pnl_atr': pnl_atr,
            'dd_atr': dd_atr,
            'hit_tp': hit_tp,
            'hit_minute': hit_minute,
            'unclosed_at_12': unclosed_at_12,
            'fitness_score': fitness_score,
            'num_orders': len(orders_placed)
        }

    def evaluate_training_set(
        self,
        feature_df: pd.DataFrame,
        daily_m1_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        param_grid = GridSimulator.generate_parameter_grid()
        print(f"\n[GridSimulator] Quét & chấm điểm {len(param_grid)} kịch bản (Order 1 chờ giãn Step_0, TP cố định Open 10:00)...")

        preset_perf_stats = {
            i: {
                'params': p,
                'total_score': 0.0,
                'total_pnl': 0.0,
                'total_dd': 0.0,
                'win_days': 0,
                'loss_days': 0,
                'total_days': 0
            } for i, p in enumerate(param_grid)
        }

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
            vwap_dist_atr = row['vwap_dist_atr']
            bb_zscore = row['bb_zscore_m15']
            bb_slope = row['bb_slope_m15']
            morning_momentum = row['morning_momentum']

            # Trend Safety Filter
            is_trend_sell_conflict = (bb_slope > 0.15 and vwap_dist_atr > 0)
            is_trend_buy_conflict = (bb_slope < -0.15 and vwap_dist_atr < 0)
            
            if abs(vwap_dist_atr) < 0.50 or abs(bb_zscore) > 2.6 or morning_momentum > 0.65 or is_trend_sell_conflict or is_trend_buy_conflict:
                rec = row.to_dict()
                rec['best_fitness_score'] = -100.0
                rec['is_profitable'] = 0
                rec['step_0_ratio'] = np.nan
                rec['step_exp'] = np.nan
                rec['max_orders'] = np.nan
                rec['multiplier'] = np.nan
                rec['net_profit'] = 0.0
                rec['max_drawdown'] = 0.0
                labeled_records.append(rec)
                continue

            best_score = -float('inf')
            best_param = None
            best_res = None

            for i, p in enumerate(param_grid):
                res = self.simulate_day_scenario(exec_df, atr_14, close_0959, daily_vwap, p)
                
                preset_perf_stats[i]['total_score'] += res['fitness_score']
                preset_perf_stats[i]['total_pnl'] += res['net_profit']
                preset_perf_stats[i]['total_dd'] = max(preset_perf_stats[i]['total_dd'], res['max_drawdown'])
                preset_perf_stats[i]['total_days'] += 1
                if res['net_profit'] > 0:
                    preset_perf_stats[i]['win_days'] += 1
                else:
                    preset_perf_stats[i]['loss_days'] += 1

                if res['fitness_score'] > best_score:
                    best_score = res['fitness_score']
                    best_param = p
                    best_res = res

            is_good_day = (best_score > 0.0) and (best_res is not None and best_res['num_orders'] > 0)

            rec = row.to_dict()
            rec['best_fitness_score'] = best_score
            rec['is_profitable'] = 1 if is_good_day else 0

            if is_good_day and best_param is not None:
                rec['step_0_ratio'] = best_param['step_0_ratio']
                rec['step_exp'] = best_param['step_exp']
                rec['max_orders'] = best_param['max_orders']
                rec['multiplier'] = best_param['multiplier']
                rec['net_profit'] = best_res['net_profit']
                rec['max_drawdown'] = best_res['max_drawdown']
                best_params_records.append(rec)
            else:
                rec['step_0_ratio'] = np.nan
                rec['step_exp'] = np.nan
                rec['max_orders'] = np.nan
                rec['multiplier'] = np.nan
                rec['net_profit'] = 0.0
                rec['max_drawdown'] = 0.0

            labeled_records.append(rec)

        labeled_df = pd.DataFrame(labeled_records)
        best_params_df = pd.DataFrame(best_params_records)

        ranked_presets = list(preset_perf_stats.values())
        ranked_presets.sort(key=lambda x: x['total_score'], reverse=True)

        print("\n=========================================================================================================")
        print(" 📊 BẢNG CHẤM ĐIỂM TOP 10 PRESETS (ORDER 1 CHỜ GIÃN STEP_0, TP CỐ ĐỊNH OPEN 10:00)")
        print("=========================================================================================================")
        print(" | Hạng | Step_0 | Step_Exp | Max_Orders | Multiplier | Tổng PnL ($) | Win Rate (%) | Max DD ($) |")
        print(" +------+--------+----------+------------+------------+--------------+--------------+------------+")

        for rank_idx, item in enumerate(ranked_presets[:10], start=1):
            p = item['params']
            w_rate = (item['win_days'] / max(1, item['total_days'])) * 100.0
            print(f" | {rank_idx:<4} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<10} | {p['multiplier']:<10.1f} | {item['total_pnl']:<12,.2f} | {w_rate:<12.1f} | {item['total_dd']:<10,.2f} |")
        print("=========================================================================================================\n")
        print(f"[GridSimulator] Tổng số ngày quan sát Train: {len(labeled_df)} ngày | Số ngày đạt Best Score > 0: {len(best_params_df)} ngày")

        return labeled_df, best_params_df
