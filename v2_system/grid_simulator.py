"""
v2_system/grid_simulator.py
===========================
Engine Mô Phỏng Kịch Bản Grid/DCA & Chấm Điểm Fitness Từng Ngày (v2).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Tạo không gian 48 kịch bản tham số chuẩn hóa cho XAUUSD (Mean Reversion):
   - Step_0: [1.0x, 1.4x, 1.8x ATR]
   - Step_Exp: [1.1, 1.25]
   - Max_Orders: [4, 5]
   - Multiplier: [1.2, 1.4]
   - TP_BE: [0.4x, 0.6x ATR]
2. Mô phỏng từng kịch bản trên nến M1 (10:00 - 12:00 VN) cho từng ngày.
3. Chấm điểm theo Fitness Function & In Log mẫu quy trình chấm điểm từng ngày.
4. Gán nhãn cho tập Train (0: No-Trade, hoặc Bộ tham số tốt nhất).
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
        Tạo danh sách các bộ tham số kịch bản grid an toàn & cao cấp.
        3 * 2 * 2 * 2 * 2 = 48 kịch bản tinh chỉnh.
        """
        step_0_ratios = [1.0, 1.4, 1.8]
        step_exps = [1.1, 1.25]
        max_orders_list = [4, 5]
        multipliers = [1.2, 1.4]
        tp_be_ratios = [0.4, 0.6]

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
        initial_tp_be_dist = params['tp_be_ratio'] * atr_14

        # Hướng Mean Reversion: 09:59 > VWAP -> SELL; 09:59 < VWAP -> BUY
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
        net_profit = 0.0
        max_drawdown = 0.0

        for m_idx, (t, row) in enumerate(exec_m1.iterrows()):
            high_t = row['high']
            low_t = row['low']
            close_t = row['close']

            # 1. Kích hoạt lệnh mới
            while next_order_idx < len(next_trigger_prices):
                trig_p = next_trigger_prices[next_order_idx]
                triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)

                if triggered:
                    lot_k = base_lot * (multiplier ** (next_order_idx + 1))
                    orders_placed.append({'price': trig_p, 'lot': lot_k})
                    next_order_idx += 1
                else:
                    break

            # 2. Tính giá Breakeven (BE) & PnL trạng thái hiện tại
            total_lot = sum(o['lot'] for o in orders_placed)
            price_be = sum(o['lot'] * o['price'] for o in orders_placed) / total_lot

            floating_pnl = sum(o['lot'] * (close_t - o['price'] if direction == 1 else o['price'] - close_t) for o in orders_placed) * self.contract_size

            if floating_pnl < 0:
                max_drawdown = max(max_drawdown, abs(floating_pnl))

            # 3. Dynamic Smart Exit Engine
            if m_idx < 75:
                tp_dist = initial_tp_be_dist
            elif m_idx < 105:
                tp_dist = initial_tp_be_dist * 0.25
                if floating_pnl > 0:
                    hit_tp = True
                    closed = True
                    net_profit = floating_pnl
                    break
            else:
                tp_dist = initial_tp_be_dist * 0.1
                if floating_pnl >= -10.0:
                    closed = True
                    net_profit = floating_pnl
                    break

            tp_price = price_be + tp_dist if direction == 1 else price_be - tp_dist

            # 4. Kiểm tra cắn TP chuẩn
            if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                hit_tp = True
                closed = True
                net_profit = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                break

        # 5. Cưỡng chế 12:00
        unclosed_at_12 = False
        if not closed:
            unclosed_at_12 = True
            final_close = exec_m1['close'].iloc[-1]
            net_profit = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

        penalty = 300.0 if unclosed_at_12 else 0.0
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
        Mô phỏng và chấm điểm tất cả các kịch bản tham số trên toàn bộ ngày trong tập Train (2020-2023).
        In Log mẫu quy trình chấm điểm từng ngày.
        """
        param_grid = GridSimulator.generate_parameter_grid()
        print(f"\n[GridSimulator] Bắt đầu đánh giá {len(param_grid)} kịch bản tham số trên tập Train (2020-2023)...")

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
        sample_day_logs = []

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

            # Lọc Alpha: Bắt buộc lệch VWAP >= 0.5 ATR và không nổ xu hướng quá mức
            if abs(vwap_dist_atr) < 0.50 or abs(bb_zscore) > 2.6:
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

            is_good_day = (best_score > 15.0) and (best_res is not None and best_res['net_profit'] > 0)

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

            # Lưu mẫu 5 ngày tiêu biểu để in log giải thích quy trình chấm điểm từng ngày
            if len(sample_day_logs) < 6 and is_good_day:
                sample_day_logs.append({
                    'date': date_str,
                    'vwap_dist_atr': vwap_dist_atr,
                    'bb_zscore': bb_zscore,
                    'best_param': best_param,
                    'pnl': best_res['net_profit'],
                    'dd': best_res['max_drawdown'],
                    'score': best_score
                })

        labeled_df = pd.DataFrame(labeled_records)
        best_params_df = pd.DataFrame(best_params_records)

        # ----- IN LOG MẪU QUY TRÌNH CHẤM ĐIỂM TỪNG NGÀY TRONG TẬP TRAIN -----
        print("\n=========================================================================================================")
        print(" 🔍 VÍ DỤ MINH HỌA QUY TRÌNH MÔ PHỎNG & CHẤM ĐIỂM TỪNG NGÀY GIAO DỊCH (TẬP TRAIN)")
        print("=========================================================================================================")
        print(" [Công thức chấm điểm]: Score = Net_Profit - (2.5 * Max_Drawdown) - (Penalty_300 nếu kẹt lệnh 12:00)")
        print(" +------------+---------------+-----------+-----------------------------------+----------+---------+----------+")
        print(" | Ngày Train | Lệch VWAP/ATR | BB ZScore | Tham Số Thắng Nhất (Best Preset)  | PnL ($)  | DD ($)  | Score    |")
        print(" +------------+---------------+-----------+-----------------------------------+----------+---------+----------+")
        for s in sample_day_logs:
            p = s['best_param']
            p_str = f"S0:{p['step_0_ratio']} | Exp:{p['step_exp']} | Ord:{int(p['max_orders'])} | Mult:{p['multiplier']}"
            print(f" | {s['date']} | {s['vwap_dist_atr']:<13.2f} | {s['bb_zscore']:<9.2f} | {p_str:<33} | {s['pnl']:<8.2f} | {s['dd']:<7.2f} | {s['score']:<8.1f} |")
        print("=========================================================================================================\n")

        # ----- IN LOG BẢNG CHẤM ĐIỂM TOP 10 KỊCH BẢN THAM SỐ XUẤT SẮC NHẤT -----
        ranked_presets = list(preset_perf_stats.values())
        ranked_presets.sort(key=lambda x: x['total_score'], reverse=True)

        print("=========================================================================================================")
        print(" 📊 BẢNG CHẤM ĐIỂM TOP 10 KỊCH BẢN THAM SỐ (PRESETS) XUẤT SẮC NHẤT TẬP TRAIN (2020-2023)")
        print("=========================================================================================================")
        print(" | Hạng | Step_0 | Step_Exp | Max_Orders | Multiplier | TP_BE Ratio | Tổng PnL ($) | Win Rate (%) | Max DD ($) |")
        print(" +------+--------+----------+------------+------------+-------------+--------------+--------------+------------+")

        for rank_idx, item in enumerate(ranked_presets[:10], start=1):
            p = item['params']
            w_rate = (item['win_days'] / max(1, item['total_days'])) * 100.0
            print(f" | {rank_idx:<4} | {p['step_0_ratio']:<6.1f} | {p['step_exp']:<8.2f} | {int(p['max_orders']):<10} | {p['multiplier']:<10.1f} | {p['tp_be_ratio']:<11.1f} | {item['total_pnl']:<12,.2f} | {w_rate:<12.1f} | {item['total_dd']:<10,.2f} |")
        print("=========================================================================================================\n")
        print(f"[GridSimulator] Tổng số ngày quan sát Train: {len(labeled_df)} ngày | Số ngày đạt Alpha (>15 score & PnL > 0): {len(best_params_df)} ngày")

        return labeled_df, best_params_df
