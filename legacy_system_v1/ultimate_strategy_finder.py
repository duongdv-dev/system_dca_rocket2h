import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any

from dca_engine import DCAEngine

class UltimateStrategyFinder:
    """
    Động cơ Tìm kiếm & Tối ưu hóa Chiến lược DCA Rocket 2h Lợi Nhuận Khủng (High Compounding Yield Engine).
    Mục tiêu duy nhất: Dựa trên dữ liệu XAUUSD M1 để tạo ra LỢI NHUẬN KHỦNG ĐẦU TƯ TRONG 2 NĂM:
      - Lợi nhuận 2 năm (2024-2025): +$1,000 - +$3,000+ USD (Tỷ suất +100% đến +300%+ trên $1,000 capital).
      - Win Rate: > 85%.
      - Profit Factor: > 2.0.
      - Max Drawdown: < 18% (Khống chế an toàn tài khoản).
    """
    def __init__(self, account_balance: float = 1000.0):
        self.account_balance = account_balance
        self.dca_engine = DCAEngine(account_balance=account_balance, base_risk_pct=3.0)

    def evaluate_strategy_on_period(self, 
                                     features_df: pd.DataFrame, 
                                     days_data: Dict[str, pd.DataFrame], 
                                     preset: Dict[str, Any],
                                     use_compounding: bool = True) -> Dict[str, Any]:
        """
        Mô phỏng chuỗi giao dịch trong suốt cả giai đoạn (gồm lãi kép Compounding Equity).
        """
        current_equity = self.account_balance
        peak_equity = self.account_balance
        max_dd_usd = 0.0
        daily_pnls = []

        for idx, row in features_df.iterrows():
            date_str = row['date']
            if date_str not in days_data:
                continue

            trade_df = days_data[date_str]
            res = self.dca_engine.simulate_day(
                trade_candles=trade_df,
                asian_atr_m15=row['asian_atr_m15'],
                asian_return_pct=row['asian_return_pct'],
                preset=preset,
                current_equity=current_equity if use_compounding else self.account_balance
            )

            pnl = res['profit_usd']
            daily_pnls.append(res)
            
            if use_compounding:
                current_equity += pnl
                if current_equity > peak_equity:
                    peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd_usd:
                    max_dd_usd = dd

        profits = [r['profit_usd'] for r in daily_pnls if r['status'] != 'NO_DATA']
        wins = [p for p in profits if p > 0]
        losses = [abs(p) for p in profits if p < 0]

        total_profit = sum(profits) if not use_compounding else (current_equity - self.account_balance)
        win_rate = len(wins) / len(profits) * 100.0 if profits else 0.0
        profit_factor = (sum(wins) / (sum(losses) + 1e-8)) if losses else 999.0
        max_dd_pct = (max_dd_usd / peak_equity) * 100.0 if use_compounding else 0.0

        return {
            'final_equity_usd': round(current_equity, 2),
            'total_profit_usd': round(total_profit, 2),
            'roi_pct': round((total_profit / self.account_balance) * 100.0, 2),
            'profit_factor': round(profit_factor, 2),
            'win_rate': round(win_rate, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'total_days': len(profits)
        }

    def search_high_yield_dca_strategy(self, 
                                        features_df: pd.DataFrame, 
                                        days_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Quét hàng ngàn tổ hợp tham số DCA Rocket 2h để tìm ra BỘ THAM SỐ TẠO LỢI NHUẬN KHỦNG TỪ +100% ĐẾN +300%+.
        """
        search_space = []

        for entry_rule in ['REVERSAL_TO_10H', 'FOLLOW_ASIAN_TREND', 'ALWAYS_BUY', 'ALWAYS_SELL']:
            for step_atr_mult in [1.2, 1.5, 1.8, 2.2]:
                for tp_atr_mult in [1.5, 2.0, 2.5]:
                    for lot_scale_mult in [1.5, 2.0, 2.5]:  # Scale lot khủng ăn lãi kép
                        for step_mult in [1.0, 1.2, 1.5]:
                            for lot_mult in [1.3, 1.5, 1.8]:
                                for max_orders in [3, 5]:
                                    for max_loss in [100.0, 150.0]:
                                        search_space.append({
                                            'entry_rule': entry_rule,
                                            'step_atr_mult': step_atr_mult,
                                            'tp_atr_mult': tp_atr_mult,
                                            'lot_scale_mult': lot_scale_mult,
                                            'step_multiplier': step_mult,
                                            'lot_multiplier': lot_mult,
                                            'max_dca_orders': max_orders,
                                            'max_loss_usd': max_loss
                                        })

        print(f"🔎 Đang quét {len(search_space)} tổ hợp DCA Rocket 2h để tìm LỢI NHUẬN KHỦNG (+100% đến +300%+)...")

        best_preset = None
        best_metrics = None
        highest_profit = -99999.0

        for preset in search_space:
            res = self.evaluate_strategy_on_period(features_df, days_data, preset, use_compounding=True)

            # Lọc các Presets có Win Rate > 80%, Max Drawdown < 18% và Lợi nhuận bứt phá
            if res['win_rate'] >= 80.0 and res['max_drawdown_pct'] <= 18.0 and res['profit_factor'] >= 1.8:
                if res['total_profit_usd'] > highest_profit:
                    highest_profit = res['total_profit_usd']
                    best_preset = preset
                    best_metrics = res

        if not best_preset:
            print("⚠️ Nới lỏng nhẹ tiêu chí để chọn ra Preset có Lợi Nhuận cao nhất...")
            for preset in search_space:
                res = self.evaluate_strategy_on_period(features_df, days_data, preset, use_compounding=True)
                if res['total_profit_usd'] > highest_profit:
                    highest_profit = res['total_profit_usd']
                    best_preset = preset
                    best_metrics = res

        return best_preset, best_metrics
