"""
v6_system/run_phase2.py
-----------------------
Runner script thực thi Phase 2 - Strategy Baseline V0.
Chạy backtest trên dữ liệu M1 2020-2025, thống kê các thông số và đưa ra kết luận
xem ý tưởng gốc DCA 10:00-12:00 VN có EDGE / HOẠT ĐỘNG hay không.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.strategy_v0 import StrategyV0
from v6_system.backtester_v0 import BacktesterV0
from v6_system.config import BASE_DIR, OUTPUT_DIR


def main():
    print("=" * 75)
    print("      VERSION 6 - PHASE 2: STRATEGY BASELINE V0 BACKTEST (2020 - 2025)")
    print("=" * 75)

    # 1. Nạp và làm sạch dữ liệu Phase 1
    print("\n[Bước 1/3] Nạp & Chuẩn hóa dữ liệu M1 2020-2025...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    # 2. Định nghĩa các cấu hình Baseline V0 thử nghiệm
    presets = [
        {"name": "Preset 1 (Step=$2, MaxDCA=5, TP=$2)", "step": 2.0, "max_dca": 5, "tp_dollars": 2.0},
        {"name": "Preset 2 (Step=$3, MaxDCA=5, TP=$3)", "step": 3.0, "max_dca": 5, "tp_dollars": 3.0},
        {"name": "Preset 3 (Step=$5, MaxDCA=5, TP=$5)", "step": 5.0, "max_dca": 5, "tp_dollars": 5.0},
        {"name": "Preset 4 (Step=$2, MaxDCA=10, TP=$5)", "step": 2.0, "max_dca": 10, "tp_dollars": 5.0},
    ]

    backtester = BacktesterV0(initial_capital=1000.0)
    all_summaries = []

    print("\n[Bước 2/3] Thực thi Backtest các cấu hình V0...")
    for p in presets:
        strat = StrategyV0(
            step=p["step"],
            max_dca=p["max_dca"],
            tp_dollars=p["tp_dollars"],
            lot_size=0.01,
            spread=0.20
        )

        df_res, summary = backtester.run_backtest(df_clean, strat)
        summary["preset_name"] = p["name"]
        summary["step"] = p["step"]
        summary["max_dca"] = p["max_dca"]
        summary["tp_dollars"] = p["tp_dollars"]
        all_summaries.append(summary)

        # Export kết quả chi tiết từng ngày của preset tiêu chuẩn (Preset 1)
        if p["step"] == 2.0 and p["max_dca"] == 5 and p["tp_dollars"] == 2.0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            df_res.to_csv(os.path.join(OUTPUT_DIR, "phase2_daily_trades_preset1.csv"), index=False)

    # Export báo cáo tổng hợp
    summary_path = os.path.join(OUTPUT_DIR, "phase2_baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # 3. Báo cáo kết quả so sánh
    print("\n" + "=" * 75)
    print("                KẾT QUẢ SO SÁNH PHẦN TRĂM & HIỆU NĂNG V0")
    print("=" * 75)
    header = f"{'Preset Name':<35} | {'Trades':<7} | {'WinRate':<8} | {'Net PnL ($)':<12} | {'PF':<6} | {'MaxDD (%)':<9}"
    print(header)
    print("-" * 85)

    best_preset = None
    best_pnl = -float("inf")

    for s in all_summaries:
        pname = s["preset_name"]
        tdays = s.get("traded_days", 0)
        wr = f"{s.get('win_rate', 0)}%"
        pnl = f"${s.get('net_profit', 0):.2f}"
        pf = str(s.get("profit_factor", "0"))
        mdd = f"{s.get('max_drawdown_pct', 0)}%"

        print(f"{pname:<35} | {tdays:<7} | {wr:<8} | {pnl:<12} | {pf:<6} | {mdd:<9}")

        if s.get("net_profit", 0) > best_pnl:
            best_pnl = s.get("net_profit", 0)
            best_preset = s

    print("=" * 75)

    # Kết luận trả lời câu hỏi cốt lõi
    print("\n" + "*" * 75)
    print("           ĐÁNH GIÁ CỐT LÕI: STRATEGY DCA CƠ BẢN CÓ HOẠT ĐỘNG KHÔNG?")
    print("*" * 75)
    if best_preset and best_preset.get("net_profit", 0) > 0:
        print(f"-> KẾT LUẬN: Ý tưởng DCA Baseline V0 CÓ NĂNG LỰC TẠO LỢI NHUẬN (EDGE DƯƠNG)!")
        print(f"   Preset hiệu quả nhất: {best_preset['preset_name']}")
        print(f"   Net Profit: ${best_preset['net_profit']:.2f} (Return: {best_preset['return_pct']}%)")
        print(f"   Win Rate: {best_preset['win_rate']}% | Profit Factor: {best_preset['profit_factor']}")
        print(f"   Max Drawdown: {best_preset['max_drawdown_pct']}% (${best_preset['max_drawdown_dollars']})")
        print(f"   Chốt lời thành công (TP Hit): {best_preset['tp_hit_count']} phiên | Force Close 12:00: {best_preset['force_close_count']} phiên")
    else:
        print("-> KẾT LUẬN: Strategy DCA cơ bản V0 chưa có Edge dương trên bộ tham số cố định.")
        print("   Cần thêm bộ lọc điều kiện thị trường (Trend / Volatility / AI Gatekeeper) ở các Phase sau.")

    print("*" * 75 + "\n")


if __name__ == "__main__":
    main()
