"""
v2_system/backtest_engine.py
=============================
Out-of-Sample Backtest Engine cho Hệ Thống XAUUSD Grid/DCA (v2 - Fixed & High Performance).
Được thiết kế bởi Senior Quantitative Researcher.

Chức năng:
1. Chạy mô phỏng thực thi chi tiết trên dữ liệu nến M1 tập Test (2024-2025).
2. Tích hợp 4 lớp quản trị rủi ro & vận hành sản xuất:
   - Morning Trend Safety Filter: Bỏ qua các ngày xu hướng mở cửa quá mạnh.
   - Dynamic Risk Engine: Khóa cứng rủi ro Max Loss <= 2% Balance cho mỗi phiên.
   - Time-Decay TP: Thu hẹp TP về sát Breakeven dần sau 11:30 VN.
   - Hard Exit: Cưỡng chế đóng 100% vị thế lúc 12:00:00.
3. Xuất báo cáo hiệu năng đầy đủ đến console stdout.
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
        """
        :param preset_centroids: Dictionary chứa tham số các preset {1: params, 2: params, 3: params}
        :param initial_balance: Số dư tài khoản ban đầu ($)
        :param risk_pct_per_session: Giới hạn rủi ro tối đa mỗi phiên (% balance)
        :param contract_size: Kích thước hợp đồng (100 oz / lot)
        """
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
        """
        Chạy mô phỏng Out-of-Sample cho từng ngày trong tập Test.
        """
        current_balance = self.initial_balance
        equity_curve = [current_balance]
        daily_trades = []

        print("\n==================================================================")
        print(" 🧪 BẮT ĐẦU RUN OUT-OF-SAMPLE BACKTEST (2024 - 2025)")
        print("==================================================================")
        print(f" • Vốn ban đầu:            ${self.initial_balance:,.2f}")
        print(f" • Khung giờ thực thi:      10:00 - 12:00 (Giờ Việt Nam, UTC+7)")
        print(f" • Dynamic Risk Engine:    Khóa cứng Max Loss <= {self.risk_pct_per_session*100:.1f}% Balance / phiên")
        print(f" • Time-Decay TP Engine:   Thu hẹp TP sau 11:30 VN (phút 90 đến 120)")
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

            # Bộ lọc bảo vệ xu hướng sáng: Nếu dải BB căng đét hoặc động lượng sáng cực đại -> Ép về No-Trade
            bb_zscore = row['bb_zscore_m15']
            momentum = row['morning_momentum']
            if abs(bb_zscore) > 2.5 or momentum > 0.8:
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

            # Trường hợp 1, 2, 3: Áp dụng tham số Preset tâm cụm tương ứng
            params = self.preset_centroids[pred_class]
            step_0 = params['step_0_ratio'] * atr_14
            step_exp = params['step_exp']
            max_orders = int(params['max_orders'])
            multiplier = params['multiplier']
            initial_tp_be_dist = params['tp_be_ratio'] * atr_14

            # Giới hạn lỗ tối đa cho phiên này (2% Balance hiện tại)
            max_loss_allowed = current_balance * self.risk_pct_per_session

            # Tính toán Base Lot động dựa trên rủi ro 2%
            total_grid_dist = step_0 * sum(step_exp ** i for i in range(max_orders - 1))
            total_lot_exp = sum(multiplier ** i for i in range(max_orders))
            max_loss_points = total_grid_dist + initial_tp_be_dist
            
            base_lot = max(0.01, round(max_loss_allowed / (max_loss_points * self.contract_size * total_lot_exp), 2))

            # Xác định hướng giao dịch Mean Reversion
            direction = -1 if close_0959 >= daily_vwap else 1

            # Khởi tạo lệnh 1 tại Open nến 10:00
            price_1 = exec_m1['open'].iloc[0]
            orders_placed = [{'price': price_1, 'lot': base_lot}]

            # Mức giá kích hoạt lệnh tiếp theo
            step_distances = [step_0 * (step_exp ** (i - 1)) for i in range(max_orders - 1)]
            next_trigger_prices = []
            curr_p = price_1
            for dist in step_distances:
                curr_p = curr_p - dist if direction == 1 else curr_p + dist
                next_trigger_prices.append(curr_p)

            next_order_idx = 0
            session_pnl = 0.0
            hit_tp = False
            exit_reason = 'Hard Exit 12:00'
            closed_before_12 = False

            # Vòng lặp từng nến M1 trong 2 tiếng
            total_candles = len(exec_m1)
            for m_idx, (t, r) in enumerate(exec_m1.iterrows()):
                high_t = r['high']
                low_t = r['low']

                # 1. Kiểm tra kích hoạt lệnh lưới mới
                while next_order_idx < len(next_trigger_prices):
                    trig_p = next_trigger_prices[next_order_idx]
                    triggered = (direction == 1 and low_t <= trig_p) or (direction == -1 and high_t >= trig_p)
                    if triggered:
                        lot_k = base_lot * (multiplier ** (next_order_idx + 1))
                        orders_placed.append({'price': trig_p, 'lot': lot_k})
                        next_order_idx += 1
                    else:
                        break

                # 2. Tính giá Breakeven (BE) hiện tại
                total_lot = sum(o['lot'] for o in orders_placed)
                price_be = sum(o['lot'] * o['price'] for o in orders_placed) / total_lot

                # 3. Time-Decay TP Engine: Thu hẹp TP sau 11:30 VN (phút 90 trở đi)
                if m_idx >= 90:
                    decay_ratio = 1.0 - 0.8 * ((m_idx - 90) / max(1, (total_candles - 90)))
                    current_tp_be_dist = initial_tp_be_dist * max(0.1, decay_ratio)
                else:
                    current_tp_be_dist = initial_tp_be_dist

                tp_price = price_be + current_tp_be_dist if direction == 1 else price_be - current_tp_be_dist

                # 4. Kiểm tra Risk Engine: Floating Drawdown cắn Max Loss (2% Balance)
                if direction == 1:
                    worst_floating_pnl = sum(o['lot'] * (low_t - o['price']) for o in orders_placed) * self.contract_size
                else:
                    worst_floating_pnl = sum(o['lot'] * (o['price'] - high_t) for o in orders_placed) * self.contract_size

                if worst_floating_pnl <= -max_loss_allowed:
                    session_pnl = -max_loss_allowed
                    exit_reason = 'Risk Engine Cut (2%)'
                    closed_before_12 = True
                    break

                # 5. Kiểm tra cắn Take Profit
                if (direction == 1 and high_t >= tp_price) or (direction == -1 and low_t <= tp_price):
                    hit_tp = True
                    closed_before_12 = True
                    exit_reason = 'TP Hit' if m_idx < 90 else 'Time-Decay TP Hit'
                    session_pnl = sum(o['lot'] * (tp_price - o['price'] if direction == 1 else o['price'] - tp_price) for o in orders_placed) * self.contract_size
                    break

            # 6. Nếu hết nến (12:00:00) chưa đóng -> Hard Exit tại giá Close 12:00
            if not closed_before_12:
                final_close = exec_m1['close'].iloc[-1]
                session_pnl = sum(o['lot'] * (final_close - o['price'] if direction == 1 else o['price'] - final_close) for o in orders_placed) * self.contract_size

            # Cập nhật số dư tài khoản
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

        # ----- TÍNH TOÁN BÁO CÁO HIỆU NĂNG CHI TIẾT -----
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

        # Max Drawdown % trên đường cong vốn
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd_arr = (eq_arr - peak) / peak
        max_drawdown_pct = abs(np.min(dd_arr)) * 100.0

        total_return_pct = ((current_balance - self.initial_balance) / self.initial_balance) * 100.0

        # Thống kê chi tiết theo từng Preset Chiến Thuật
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

        # Đếm các sự kiện vận hành
        risk_cuts = (trades_df['exit_reason'] == 'Risk Engine Cut (2%)').sum()
        decay_tp_hits = (trades_df['exit_reason'] == 'Time-Decay TP Hit').sum()
        normal_tp_hits = (trades_df['exit_reason'] == 'TP Hit').sum()
        hard_exits = (trades_df['exit_reason'] == 'Hard Exit 12:00').sum()

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
            'preset_stats': preset_stats,
            'risk_cuts': risk_cuts,
            'decay_tp_hits': decay_tp_hits,
            'normal_tp_hits': normal_tp_hits,
            'hard_exits': hard_exits
        }

        print("\n==================================================================")
        print(" 📊 BÁO CÁO KẾT QUẢ KHI CHẠY BACKTEST 2 NĂM 2024 - 2025 (OUT-OF-SAMPLE)")
        print("==================================================================")
        print(f" 💰 Vốn ban đầu (Initial Balance):   ${self.initial_balance:,.2f}")
        print(f" 🚀 Số dư cuối kỳ (Final Balance):   ${current_balance:,.2f}")
        print(f" 🔥 Tổng lợi nhuận Net Return %:    +{total_return_pct:.2f}% (+${current_balance - self.initial_balance:,.2f})")
        print(f" ------------------------------------------------------------------")
        print(f" • Tổng số ngày quan sát:            {total_days} ngày")
        print(f" • Tỷ lệ ngày No-Trade (Bảo vệ vốn): {no_trade_pct:.2f}% ({no_trade_days}/{total_days} ngày)")
        print(f" • Số ngày vào lệnh thực tế:         {traded_days} ngày")
        print(f" • Win Rate (Tỷ lệ thắng):           {win_rate:.2f}% ({len(win_trades)} thắng / {len(loss_trades)} thua)")
        print(f" • Profit Factor (Hệ số Lời/Lỗ):    {profit_factor:.2f}")
        print(f" • Max Drawdown % (Peak-to-Trough):  {max_drawdown_pct:.2f}%")
        print(f" ------------------------------------------------------------------")
        print(" 📌 HIỆU NĂNG THEO TỪNG PRESET CHIẾN THUẬT:")
        for p_class, p_data in preset_stats.items():
            print(f"    - Preset {p_class} [{p_data['name']}]: {p_data['count']} phiên | WinRate: {p_data['winrate']:.1f}% | Net PnL: +${p_data['pnl']:,.2f}")
        print(f" ------------------------------------------------------------------")
        print(" 🛡️ PHÂN BỐ LÝ DO ĐÓNG VỊ THẾ:")
        print(f"    - Chốt lời tiêu chuẩn (TP Hit):    {normal_tp_hits} phiên")
        print(f"    - Chốt lời Time-Decay sau 11:30:  {decay_tp_hits} phiên")
        print(f"    - Cắt lỗ khẩn cấp Risk Engine 2%: {risk_cuts} phiên")
        print(f"    - Hard Exit cưỡng chế lúc 12:00:  {hard_exits} phiên")
        print("==================================================================\n")

        return trades_df, metrics

    def plot_equity_curve(self, equity_curve: List[float], output_path: str):
        """
        Vẽ và lưu đồ thị Đường Cong Vốn (Equity Curve).
        """
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='System Equity ($)', color='#10B981', linewidth=2)
        plt.axhline(self.initial_balance, color='#6B7280', linestyle='--', label='Initial Balance')
        plt.title('Out-of-Sample Equity Curve (2024-2025) - XAUUSD Intraday Grid v2', fontsize=14, fontweight='bold')
        plt.xlabel('Traded Sessions / Days', fontsize=12)
        plt.ylabel('Account Equity ($)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[OOSBacktestEngine] Đã xuất đồ thị Equity Curve -> {output_path}")


if __name__ == '__main__':
    pass
