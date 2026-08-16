import pandas as pd
import numpy as np
from typing import Dict, List, Any

from alpha_engine import AlphaEngine
from dca_engine import DCAEngine

class PresetPool:
    """
    Quản lý Tập hợp Danh mục các Presets Đa dạng (Diverse Preset Pool).
    Mô phỏng tất cả Presets trên toàn bộ các ngày để tạo ma trận hiệu suất PnL.
    """
    def __init__(self, account_balance: float = 1000.0):
        self.account_balance = account_balance
        self.alpha_engine = AlphaEngine(account_balance=account_balance, risk_pct_per_trade=1.5)
        self.dca_engine = DCAEngine(account_balance=account_balance, risk_pct_per_trade=1.5)
        
        # Định nghĩa Pool gồm 10 Presets đại diện cho các trường phái khác nhau
        self.pool = [
            # Preset 0: Breakout Aggressive (R:R 1:3)
            {'id': 0, 'name': 'Breakout_Aggressive', 'type': 'ALPHA', 'setup': 'BREAKOUT', 'tp_mult': 3.5, 'sl_mult': 1.2},
            # Preset 1: Breakout Moderate (R:R 1:2)
            {'id': 1, 'name': 'Breakout_Moderate', 'type': 'ALPHA', 'setup': 'BREAKOUT', 'tp_mult': 2.5, 'sl_mult': 1.2},
            # Preset 2: Reversal Scalp (R:R 1:2)
            {'id': 2, 'name': 'Reversal_Scalp', 'type': 'ALPHA', 'setup': 'REVERSAL', 'tp_mult': 3.0, 'sl_mult': 1.5},
            # Preset 3: DCA Reversal Tight (Step 1.0*ATR, TP Breakeven)
            {'id': 3, 'name': 'DCA_Reversal_Tight', 'type': 'DCA', 'entry_rule': 'REVERSAL_TO_10H', 'step_atr_mult': 1.0, 'step_multiplier': 1.0, 'lot_multiplier': 1.2, 'max_dca_orders': 5, 'max_loss_usd': 80.0, 'tp_atr_mult': 1.0},
            # Preset 4: DCA Trend Follower (Step 1.5*ATR, Lot Mult 1.3)
            {'id': 4, 'name': 'DCA_Trend_Follower', 'type': 'DCA', 'entry_rule': 'FOLLOW_ASIAN_TREND', 'step_atr_mult': 1.5, 'step_multiplier': 1.2, 'lot_multiplier': 1.3, 'max_dca_orders': 5, 'max_loss_usd': 80.0, 'tp_atr_mult': 1.5},
            # Preset 5: DCA Wide Step Protection (Step 2.0*ATR, Lot Mult 1.5)
            {'id': 5, 'name': 'DCA_Wide_Protection', 'type': 'DCA', 'entry_rule': 'REVERSAL_TO_10H', 'step_atr_mult': 2.0, 'step_multiplier': 1.2, 'lot_multiplier': 1.5, 'max_dca_orders': 3, 'max_loss_usd': 60.0, 'tp_atr_mult': 1.2},
            # Preset 6: Always Buy Scalp
            {'id': 6, 'name': 'Always_Buy_Scalp', 'type': 'DCA', 'entry_rule': 'ALWAYS_BUY', 'step_atr_mult': 1.2, 'step_multiplier': 1.0, 'lot_multiplier': 1.2, 'max_dca_orders': 5, 'max_loss_usd': 80.0, 'tp_atr_mult': 1.2},
            # Preset 7: Always Sell Scalp
            {'id': 7, 'name': 'Always_Sell_Scalp', 'type': 'DCA', 'entry_rule': 'ALWAYS_SELL', 'step_atr_mult': 1.2, 'step_multiplier': 1.0, 'lot_multiplier': 1.2, 'max_dca_orders': 5, 'max_loss_usd': 80.0, 'tp_atr_mult': 1.2},
            # Preset 8: Breakout Conservative (R:R 1:1.5)
            {'id': 8, 'name': 'Breakout_Conservative', 'type': 'ALPHA', 'setup': 'BREAKOUT', 'tp_mult': 2.0, 'sl_mult': 1.3},
            # Preset 9: NO_TRADE (Đứng ngoài hoàn toàn bảo vệ vốn)
            {'id': 9, 'name': 'NO_TRADE', 'type': 'NONE'}
        ]

    def evaluate_preset_on_day(self, preset: Dict[str, Any], feature_row: pd.Series, trade_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Mô phỏng 1 Preset cụ thể trên 1 ngày giao dịch.
        """
        p_type = preset['type']

        if p_type == 'NONE' or trade_df.empty:
            return {'profit_usd': 0.0, 'max_drawdown_usd': 0.0, 'status': 'NO_TRADE', 'preset_id': preset['id']}

        if p_type == 'ALPHA':
            res = self.alpha_engine.simulate_day_setup(
                trade_candles=trade_df,
                asian_atr_m15=feature_row['asian_atr_m15'],
                asian_return_pct=feature_row['asian_return_pct'],
                setup_type=preset['setup'],
                tp_atr_mult=preset['tp_mult'],
                sl_atr_mult=preset['sl_mult']
            )
        else:  # DCA
            res = self.dca_engine.simulate_day(
                trade_candles=trade_df,
                asian_atr_m15=feature_row['asian_atr_m15'],
                asian_return_pct=feature_row['asian_return_pct'],
                preset=preset
            )

        res['preset_id'] = preset['id']
        return res

    def build_performance_matrix(self, features_df: pd.DataFrame, trading_days_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Xây dựng Ma trận Hiệu suất PnL cho tất cả Presets trên toàn bộ các ngày.
        Returns:
            matrix_df: DataFrame kích thước (n_days, n_presets) chứa PnL USD
            best_preset_labels: Nhãn Preset_ID xuất sắc nhất cho từng ngày
        """
        records = []
        best_labels = []

        for idx, row in features_df.iterrows():
            date_str = row['date']
            if date_str not in trading_days_data:
                continue

            trade_df = trading_days_data[date_str]
            day_pnls = {}
            best_pnl = -9999.0
            best_p_id = 9  # Mặc định NO_TRADE

            for preset in self.pool:
                p_id = preset['id']
                res = self.evaluate_preset_on_day(preset, row, trade_df)
                pnl = res['profit_usd']
                day_pnls[f"Preset_{p_id}"] = pnl

                # Chọn Preset có lợi nhuận cao nhất cho ngày đó (nếu tất cả đều lỗ thì chọn NO_TRADE = 0.0)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_p_id = p_id

            # Nếu lợi nhuận tốt nhất vẫn <= 0 -> gán nhãn NO_TRADE (Preset 9)
            if best_pnl <= 0.0:
                best_p_id = 9

            day_pnls['date'] = date_str
            records.append(day_pnls)
            best_labels.append(best_p_id)

        matrix_df = pd.DataFrame(records)
        return matrix_df, np.array(best_labels)
