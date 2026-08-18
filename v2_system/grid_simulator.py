"""
v2_system/grid_simulator.py
===========================
Engine Mô Phỏng Kịch Bản Grid/DCA & Chấm Điểm Fitness (v2 - Trend Alignment Alpha).
Được thiết kế bởi Senior Quantitative Researcher.

Nguyên lý Định Lượng Khắc Phục Âm Out-of-Sample:
1. Trend Alignment Rule: Tuyệt đối KHÔNG SELL ngược xu hướng tăng (bb_slope > 0.15) và KHÔNG BUY ngược xu hướng giảm (bb_slope < -0.15).
2. Lưới tham số an toàn chống nổ xu hướng Vàng 2024-2025:
   - Step_0: [1.2x, 1.6x ATR] (Lưới rộng hơn chống quét nến)
   - Step_Exp: [1.1, 1.25]
   - Max_Orders: [3, 4] (Khóa trần tối đa 4 lệnh, tránh tích lũy lot quá lớn)
   - Multiplier: [1.15, 1.25] (Hệ số nhồi nhẹ nhàng)
   - TP_BE: [0.35x, 0.5x ATR] (Chốt lời cực nhanh khi có nhịp hồi)
3. Trend Safety Filter: Bỏ qua ngày có động lượng phiên sáng mạnh (morning_momentum > 0.65).
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
        Tạo không gian 32 kịch bản tham số an toàn & bảo vệ vốn.
        2 * 2 * 2 * 2 * 2 = 32 kịch bản.
        """
        step_0_ratios = [1.2, 1.6]
        step_exps = [1.1, 1.25]
        max_orders_list = [3, 4]
        multipliers = [1.15, 1.25]
        tp_be_ratios = [0.35, 0.5]

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
        Mô phỏng 100% ĐỘC LẬP cho 1 ngày cụ thể.
        """
        step_0 = params['step_0_ratio'] * atr_14
        step_exp = params['step_exp']
        max_orders = int(params['max_orders'])
        multiplier = params['multiplier']
        tp_be_dist = params['tp_be_ratio'] * atr_14

        direction = -1 if close_0959 >= daily_vwap else 1

        orders_placed = []
        price_1 = exec_m1['open'].iloc[0]
        orders_placed.append({'price': price_1, 'lot': base_lot})

        step_distances = [step_0 * (step_exp ** (i - 1)) for i in range(max_orders - 1)]

        next_trigger_prices = []
        curr_price = price_1
        for dist in step_distances:
            curr_price = curr_price - dist if direction == 1 else curr_price + dist
            next_trigger_prices.append(curr_price)

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

            # 1. Kích hoạt lệnh lưới
            while next_order_idx < len(next_trigger_prices):
                trig_p = next_trigger_prices[next_order_idx]
                triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

                if triggered:
                    lot_k = base_lot * (multiplier ** (next_order_idx + 1))
                    orders_placed.append({'price': trig_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            # 2. Giá Breakeven & Floating PnL
            total_lot = sum(o['lot'] for o in orders_placed)
            price_be = sum(o['lot'] * o['price'] for o in orders_placed) / total_lot

            floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size

            if floating_pnl < 0:
                max_drawdown = max(max_drawdown, abs(floating_pnl))

            # 3. Dynamic Trailing TP cho lệnh nhồi (Khi nhồi > 1 lệnh, thu TP về gần BE hơn nữa)
            if len(orders_placed) >= 2:
                current_tp_dist = tp_be_dist * 0.7
            else:
                current_tp_dist = tp_be_dist

            tp_price = price_be + current_tp_dist if direction == 1 else price_be - current_tp_dist

            # 4. Kiểm tra cắn TP
            if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                hit_tp = True
                closed = True
                hit_minute = m_idx
                net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                break

        # 5. Xử lý Hard Exit 12:00
        unclosed_at_12 = False
        if not closed:
            unclosed_at_12 = True
            final_close = exec_m1['close'].iloc[-1]
            net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

        total_volume = sum(o['lot'] for o in orders_placed)
        pnl_points = net_profit / (self.contract_size * total_volume)
        dd_points = max_drawdown / (self.contract_size * total_volume)

        pnl_atr = pnl_points / (atr_14 + 1e-8)
        dd_atr = dd_points / (atr_14 + 1e-8)

        penalty_atr = 25.0 if unclosed_at_12 else 0.0
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
        """
        Mô phỏng và chấm điểm tất cả các kịch bản tham số trên tập Train (2020-2023).
        Tích hợp Trend Safety Alignment Filter.
        """
        param_grid = GridSimulator.generate_parameter_grid()
        print(f"\n[GridSimulator] Bắt đầu quét & chấm điểm {len(param_grid)} kịch bản tham số an toàn...")

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

            # Trend Alignment Safety Filters:
            # 1. Không đánh ngược xu hướng dốc (bb_slope > 0.15 & close > vwap -> KHÔNG SELL; bb_slope < -0.15 & close < vwap -> KHÔNG BUY)
            # 2. Không đánh khi động lượng nến sáng quá mạnh (morning_momentum > 0.65)
            # 3. Yêu cầu giá lệch xa VWAP (abs(vwap_dist_atr) >= 0.5)
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
                rec['tp_be_ratio'] = np.nan
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

            is_good_day = (best_score > 0.0)

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

        ranked_presets = list(preset_perf_stats.values())
        ranked_presets.sort(key=lambda x: x['total_score'], reverse=True)

        print("\n=========================================================================================================")
        print(" 📊 BẢNG CHẤM ĐIỂM TOP 10 KỊCH BẢN THAM SỐ (PRESETS) XUẤT SẮC NHẤT TẬP TRAIN (2020-2023)")
        print("=========================================================================================================")
        print(" | Hạng | Step_0 | Step_Exp | Max_Orders | Multiplier | TP_BE Ratio | Tổng PnL ($) | Win Rate (%) | Max DD ($) |")
        print(" +------+--------+----------+------------+------------+-------------+--------------+--------------+------------+")

        for rank_idx, item in enumerate(ranked_presets[:10], start=1):
            p = item['params']
            w_rate = (item['win_days'] / max(1, item['total_days'])) * 100.0
            print(f" | {rank_idx:<4} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<10} | {p['multiplier']:<10.1f} | {p['tp_be_ratio']:<11.1f} | {item['total_pnl']:<12,.2f} | {w_rate:<12.1f} | {item['total_dd']:<10,.2f} |")
        print("=========================================================================================================\n")
        print(f"[GridSimulator] Tổng số ngày quan sát Train: {len(labeled_df)} ngày | Số ngày đạt Best Score > 0: {len(best_params_df)} ngày")

        return labeled_df, best_params_df
