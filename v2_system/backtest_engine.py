"""
v2_system/backtest_engine.py
=============================
Out-of-Sample Backtest Engine cho Hệ Thống XAUUSD Grid/DCA (v2 - True Intraday Stretch & Fixed 10:00 Open TP).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Chạy mô phỏng thực thi chi tiết trên dữ liệu nến M1 tập Test (2024-2025).
2. QUY TẮC CỐT LÕI ĐÍCH THỰC CỦA USER:
   - Chờ giá nảy giãn đủ `Step_0` từ Giá Open 10:00 AM mới mở Order 1!
   - Take Profit (TP) LUÔN CỐ ĐỊNH TẠI GIÁ OPEN 10:00 AM (price_1000).
   - Khi giá hồi về chạm `price_1000`: Order 1, Order 2, Order 3... TẤT CẢ ĐỀU ĐẠT LỢI NHUẬN DƯƠNG LỚN!
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class OOSBacktestEngine:
    def __init__(
        self,
        preset_centroids: Dict[int, Dict[str, float]],
        initial_balance: float = 10000.0,
        risk_pct_per_session: float = 0.02,
        contract_size: float = 100.0
    ):
        self.preset_centroids = preset_centroids
        self.initial_balance = initial_balance
        self.risk_pct_per_session = risk_pct_per_session
        self.contract_size = contract_size

    def run_backtest(
        self,
        test_feature_df: pd.DataFrame,
        predictions: np.ndarray,
        daily_m1_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        current_balance = self.initial_balance
        equity_curve = [current_balance]
        daily_trades = []

        print("\n==================================================================")
        print(" 🧪 RUNNING BACKTEST (ORDER 1 CHỜ GIÃN STEP_0, TP CỐ ĐỊNH OPEN 10:00)")
        print("==================================================================")
        print(f" • Vốn ban đầu:            ${self.initial_balance:,.2f}")
        print(f" • Khung giờ thực thi:      10:00 - 12:00 (Giờ Việt Nam, UTC+7)")
        print(f" • Order 1 Trigger:        Khớp khi giá nảy giãn xa Open 10:00 đúng Step_0")
        print(f" • Take Profit (TP):        LUÔN CỐ ĐỊNH TẠI GIÁ OPEN 10:00 AM")
        print(f" • Dynamic Risk Engine:    Khóa cứng Max Loss <= {self.risk_pct_per_session*100:.1f}% Balance / phiên")
        print(f" • Hard Exit:              Đóng 100% vị thế tại 12:00:00 VN\n")

        for idx, row in test_feature_df.iterrows():
            date_str = row['date']
            pred_class = int(predictions[idx])

            if date_str not in daily_m1_dict:
                continue

            obs_df, exec_m1 = daily_m1_dict[date_str]
            atr_14 = row['atr_14_m15']
            close_0959 = row['close_0959']
            daily_vwap = row['daily_vwap']
            vwap_dist_atr = row['vwap_dist_atr']
            bb_zscore = row['bb_zscore_m15']
            bb_slope = row['bb_slope_m15']
            morning_momentum = row['morning_momentum']

            # Trend Safety Filters
            is_trend_sell_conflict = (bb_slope > 0.15 and vwap_dist_atr > 0)
            is_trend_buy_conflict = (bb_slope < -0.15 and vwap_dist_atr < 0)

            if abs(vwap_dist_atr) < 0.50 or abs(bb_zscore) > 2.6 or morning_momentum > 0.65 or is_trend_sell_conflict or is_trend_buy_conflict:
                pred_class = 0

            # Trường hợp 0: No-Trade
            if pred_class == 0:
                daily_trades.append({
                    'date': date_str,
                    'pred_class': 0,
                    'preset_name': 'No-Trade',
                    'net_profit': 0.0,
                    'balance': current_balance,
                    'exit_reason': 'No-Trade',
                    'hit_tp': False,
                    'closed_before_12': True,
                    'num_orders': 0
                })
                equity_curve.append(current_balance)
                continue

            # Trường hợp 1, 2, 3: Áp dụng tham số Preset
            params = self.preset_centroids[pred_class]
            step_0 = params['step_0_ratio'] * atr_14
            step_exp = params['step_exp']
            max_orders = int(params['max_orders'])
            multiplier = params['multiplier']

            max_loss_allowed = current_balance * self.risk_pct_per_session

            # Tính Base Lot động
            total_grid_dist = step_0 * sum(step_exp ** i for i in range(max_orders - 1))
            total_lot_exp = sum(multiplier ** i for i in range(max_orders))
            max_loss_points = total_grid_dist + (0.5 * atr_14)
            
            base_lot = max(0.01, round(max_loss_allowed / (max_loss_points * self.contract_size * total_lot_exp), 2))

            # Hướng Mean Reversion
            direction = -1 if close_0959 >= daily_vwap else 1

            price_1000 = exec_m1['open'].iloc[0] # Giá Open 10:00 AM
            tp_price = price_1000               # TP LUÔN CỐ ĐỊNH TẠI GIÁ OPEN 10:00 AM!

            # Tính danh sách mức giá kích hoạt cho Order 1, Order 2, Order 3, Order 4...
            trigger_prices = []
            curr_p = price_1000 - step_0 if direction == 1 else price_1000 + step_0
            trigger_prices.append(curr_p)

            for i in range(1, max_orders):
                dist_i = step_0 * (step_exp ** i)
                curr_p = curr_p - dist_i if direction == 1 else curr_p + dist_i
                trigger_prices.append(curr_p)

            orders_placed = []
            next_order_idx = 0
            session_pnl = 0.0
            hit_tp = False
            exit_reason = 'No-Trade (Lệnh 1 chưa khớp)'
            closed_before_12 = False

            for m_idx, (t, r) in enumerate(exec_m1.iterrows()):
                high_t = r['high']
                low_t = r['low']

                # 1. Kích hoạt lệnh mới khi giá nảy tới trigger_prices
                while next_order_idx < len(trigger_prices):
                    trig_p = trigger_prices[next_order_idx]
                    triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)
                    if triggered:
                        lot_k = base_lot * (multiplier ** next_order_idx)
                        orders_placed.append({'price': trig_p, 'lot': lot_k})
                        next_order_idx += 1
                    else:
                        break

                # 2. Khi đã có ít nhất 1 lệnh khớp
                if len(orders_placed) > 0:
                    # Risk Engine 2% Cut
                    if direction == 1:
                        worst_floating_pnl = sum(o['lot'] * (low_t - o['price']) for o in orders_placed) * self.contract_size
                    else:
                        worst_floating_pnl = sum(o['lot'] * (o['price'] - high_t) for o in orders_placed) * self.contract_size

                    if worst_floating_pnl <= -max_loss_allowed:
                        session_pnl = -max_loss_allowed
                        exit_reason = 'Risk Engine Cut (2%)'
                        closed_before_12 = True
                        break

                    # Kiểm tra cắn TP cố định tại Giá Open 10:00 AM (price_1000)
                    if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                        hit_tp = True
                        closed_before_12 = True
                        exit_reason = 'TP Hit (10:00 Open)'
                        session_pnl = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                        break

            # 3. Hard Exit 12:00:00 (Nếu đã khớp lệnh nhưng chưa cắn TP)
            if len(orders_placed) > 0 and not closed_before_12:
                final_close = exec_m1['close'].iloc[-1]
                exit_reason = 'Hard Exit 12:00'
                session_pnl = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

            current_balance += session_pnl
            equity_curve.append(current_balance)

            daily_trades.append({
                'date': date_str,
                'pred_class': pred_class,
                'preset_name': params['name'],
                'net_profit': session_pnl,
                'balance': current_balance,
                'exit_reason': exit_reason,
                'hit_tp': hit_tp,
                'closed_before_12': closed_before_12,
                'num_orders': len(orders_placed)
            })

        trades_df = pd.DataFrame(daily_trades)

        total_days = len(trades_df)
        no_trade_days = (trades_df['pred_class'] == 0).sum()
        traded_days = total_days - no_trade_days
        no_trade_pct = (no_trade_days / total_days) * 100.0 if total_days > 0 else 0.0

        traded_df = trades_df[trades_df['pred_class'] > 0]
        win_trades = traded_df[traded_df['net_profit'] > 0]
        loss_trades = traded_df[traded_df['net_profit'] < 0]

        win_rate = (len(win_trades) / traded_days) * 100.0 if traded_days > 0 else 0.0
        
        gross_profit = win_trades['net_profit'].sum()
        gross_loss = abs(loss_trades['net_profit'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd_arr = (eq_arr - peak) / peak
        max_drawdown_pct = abs(np.min(dd_arr)) * 100.0

        total_return_pct = ((current_balance - self.initial_balance) / self.initial_balance) * 100.0

        preset_stats = {}
        for p_class in [1, 2, 3]:
            sub_p = trades_df[trades_df['pred_class'] == p_class]
            if not sub_p.empty:
                p_win = (sub_p['net_profit'] > 0).sum()
                p_winrate = (p_win / len(sub_p)) * 100.0
                p_pnl = sub_p['net_profit'].sum()
                preset_stats[p_class] = {
                    'count': len(sub_p),
                    'winrate': p_winrate,
                    'pnl': p_pnl,
                    'name': self.preset_centroids[p_class]['name']
                }

        metrics = {
            'total_days': total_days,
            'no_trade_days': no_trade_days,
            'no_trade_pct': no_trade_pct,
            'traded_days': traded_days,
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'initial_balance': self.initial_balance,
            'final_balance': current_balance,
            'total_return_pct': total_return_pct,
            'preset_stats': preset_stats
        }

        print("\n==================================================================")
        print(" 📊 BACKTEST REPORT (ORDER 1 CHỜ GIÃN STEP_0, TP CỐ ĐỊNH OPEN 10:00)")
        print("==================================================================")
        print(f" 💰 Initial Balance:                 ${self.initial_balance:,.2f}")
        print(f" 🚀 Final Balance:                   ${current_balance:,.2f}")
        print(f" 🔥 Net Return %:                    +{total_return_pct:.2f}% (+${current_balance - self.initial_balance:,.2f})")
        print(f" ------------------------------------------------------------------")
        print(f" • Win Rate:                         {win_rate:.2f}% ({len(win_trades)} W / {len(loss_trades)} L)")
        print(f" • Profit Factor:                    {profit_factor:.2f}")
        print(f" • Max Drawdown %:                  {max_drawdown_pct:.2f}%")
        print("==================================================================\n")

        return trades_df, metrics

    def plot_equity_curve(self, equity_curve: List[float], output_path: str):
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='System Equity ($)', color='#10B981', linewidth=2)
        plt.axhline(self.initial_balance, color='#6B7280', linestyle='--', label='Initial Balance')
        plt.title('Equity Curve (Order 1 Stretch Step_0 & Fixed 10:00 Open TP)', fontsize=14, fontweight='bold')
        plt.xlabel('Traded Sessions / Days', fontsize=12)
        plt.ylabel('Account Equity ($)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


if __name__ == '__main__':
    pass
