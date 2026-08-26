"""
v6_system/run_phase8.py
-----------------------
Runner script thực thi Phase 8 - Adaptive DCA (Điều Khiển Tham Số Thích Ứng Theo AI).
Backtest so sánh 3 hệ thống:
1. Baseline V0 (Raw DCA)
2. Static Plateau DCA (Phase 4 Optimal Params + Stop Loss)
3. Adaptive DCA (Phase 8 Dynamic Regimes A/B/C + Stop Loss)
"""

import sys
import os
import json
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.feature_engineering import V6FeatureEngineer
from v6_system.label_builder import V6LabelBuilder
from v6_system.ml_gatekeeper import MLGatekeeper
from v6_system.strategy_v1 import StrategyV1
from v6_system.strategy_adaptive import StrategyAdaptiveDCA
from v6_system.adaptive_controller import AdaptiveDCAController
from v6_system.config import BASE_DIR, OUTPUT_DIR


def evaluate_system_performance(df_res: pd.DataFrame, system_name: str, initial_capital: float = 1000.0) -> Dict[str, Any]:
    """Tính toán các chỉ số đo lường rủi ro & hiệu năng."""
    traded_df = df_res[df_res["traded"] == True].copy()
    total_days = len(df_res)
    traded_days = len(traded_df)
    skipped_days = int((df_res["exit_reason"] == "SKIPPED_BY_AI").sum())

    if traded_days == 0:
        return {"system_name": system_name, "error": "Không có giao dịch."}

    net_pnl = float(traded_df["pnl"].sum())
    wins = traded_df[traded_df["pnl"] > 0]
    losses = traded_df[traded_df["pnl"] < 0]

    win_count = len(wins)
    losing_days_count = len(losses)
    win_rate = (win_count / traded_days) * 100.0

    gross_profit = float(wins["pnl"].sum())
    gross_loss = float(abs(losses["pnl"].sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Equity Curve & Drawdown
    traded_df["equity"] = initial_capital + traded_df["pnl"].cumsum()
    traded_df["peak"] = traded_df["equity"].cummax()
    traded_df["drawdown"] = traded_df["equity"] - traded_df["peak"]
    traded_df["drawdown_pct"] = (traded_df["drawdown"] / traded_df["peak"]) * 100.0

    max_dd_dollars = float(abs(traded_df["drawdown"].min()))
    max_dd_pct = float(abs(traded_df["drawdown_pct"].min()))

    # Tail Risk (Worst Day & Worst Month)
    worst_day_pnl = float(traded_df["pnl"].min())
    date_col = "date" if "date" in traded_df.columns else "date_vn"
    traded_df["month"] = pd.to_datetime(traded_df[date_col]).dt.to_period("M")
    monthly_pnl = traded_df.groupby("month")["pnl"].sum()
    worst_month_pnl = float(monthly_pnl.min()) if not monthly_pnl.empty else 0.0

    max_dca_reached = int(traded_df["max_level"].max()) if not traded_df.empty else 0
    avg_profit_per_traded = float(net_pnl / traded_days)

    fc_count = int((traded_df["exit_reason"] == "FORCE_CLOSE_1200").sum())
    sl_count = int((traded_df["exit_reason"] == "STOP_LOSS_MAX_DIST").sum())

    return {
        "system_name": system_name,
        "total_days": total_days,
        "traded_days": traded_days,
        "skipped_days": skipped_days,
        "net_profit": round(net_pnl, 2),
        "return_pct": round((net_pnl / initial_capital) * 100.0, 2),
        "win_rate": round(win_rate, 2),
        "losing_days_count": losing_days_count,
        "profit_factor": round(profit_factor, 2),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dca_reached": max_dca_reached,
        "avg_profit_per_traded": round(avg_profit_per_traded, 2),
        "worst_day_pnl": round(worst_day_pnl, 2),
        "worst_month_pnl": round(worst_month_pnl, 2),
        "force_close_count": fc_count,
        "stop_loss_count": sl_count
    }


def main():
    print("=" * 90)
    print("       VERSION 6 - PHASE 8: ADAPTIVE DCA & DYNAMIC PARAMETER CONTROL")
    print("=" * 90)

    # 1. Nạp và trích xuất dữ liệu
    print("\n[Bước 1/4] Nạp dữ liệu M1 & Trích xuất 25 Features...")
    engineer = V6DataEngineer()
    df_raw = engineer.load_all_years(data_dir=BASE_DIR)
    df_clean, _ = engineer.audit_and_clean_data(df_raw)

    fe = V6FeatureEngineer()
    df_features = fe.extract_session_features_and_targets(df_clean)

    label_builder = V6LabelBuilder()
    df_labeled = label_builder.build_all_labels(df_features)

    # 2. Huấn luyện mô hình XGBoost V1 Gatekeeper
    print("\n[Bước 2/4] Huấn luyện AI Gatekeeper & Dự báo xác suất P(reversion)...")
    gatekeeper = MLGatekeeper(random_state=42)
    df_labeled["target_revert"] = df_labeled["y_anchor"]
    gatekeeper.train_and_evaluate(df_labeled, split_year=2024)

    X_all = df_labeled[MLGatekeeper.FEATURE_COLS].astype(np.float32)
    df_labeled["prob_revert"] = gatekeeper.model.predict_proba(X_all)[:, 1]

    # 3. Thử nghiệm 3 hệ thống khác nhau
    print("\n[Bước 3/4] Chạy Backtest So Sánh 3 Hệ Thống (Baseline V0, Static Plateau, Adaptive DCA)...")

    # System 1: Baseline V0 (Unoptimized Raw DCA)
    strat_v0 = StrategyV1(step=5.0, max_dca=5, multiplier=1.10, tp_dollars=2.0)
    rec_v0 = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_v0.run_daily_session(group)
        rec_v0.append(res)
    perf_v0 = evaluate_system_performance(pd.DataFrame(rec_v0), "1. Baseline V0 (Raw DCA)")

    # System 2: Phase 4 Static Plateau DCA (Optimum Step=$8, Mult=1.15, TP=$5.0)
    strat_static = StrategyV1(step=8.0, max_dca=5, multiplier=1.15, tp_dollars=5.0)
    rec_static = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_static.run_daily_session(group)
        rec_static.append(res)
    perf_static = evaluate_system_performance(pd.DataFrame(rec_static), "2. Phase 4 Static Plateau DCA")

    # System 3: Phase 8 Adaptive DCA (Dynamic Regimes A/B/C + Stop Loss)
    strat_adaptive = StrategyAdaptiveDCA()
    rec_adaptive = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_adaptive.run_daily_session_adaptive(group)
        rec_adaptive.append(res)
    perf_adaptive = evaluate_system_performance(pd.DataFrame(rec_adaptive), "3. Phase 8 Adaptive DCA (AI Controlled)")

    all_perfs = [perf_v0, perf_static, perf_adaptive]

    # 4. Xuất kết quả
    print("\n[Bước 4/4] Xuất kết quả báo cáo Phase 8 Adaptive DCA...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_path = os.path.join(OUTPUT_DIR, "phase8_adaptive_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_perfs, f, ensure_ascii=False, indent=2)

    df_comp = pd.DataFrame(all_perfs)
    csv_path = os.path.join(OUTPUT_DIR, "phase8_comparison.csv")
    df_comp.to_csv(csv_path, index=False)

    # In bảng tổng hợp so sánh
    print("\n" + "=" * 120)
    print("                BẢNG TỔNG HỢP HIỆU NĂNG: BASELINE V0 vs STATIC PLATEAU vs ADAPTIVE DCA")
    print("=" * 120)
    hdr = f"{'System Name':<35} | {'Net PnL':<10} | {'Return':<8} | {'PF':<6} | {'MaxDD($)':<9} | {'WinRate':<8} | {'LossDays':<9} | {'Skip':<6}"
    print(hdr)
    print("-" * 120)

    for p in all_perfs:
        s_name = p["system_name"]
        pnl = f"${p['net_profit']:.1f}"
        ret = f"{p['return_pct']}%"
        pf = str(p['profit_factor'])
        mdd = f"${p['max_dd_dollars']:.1f}"
        wr = f"{p['win_rate']}%"
        ld = str(p['losing_days_count'])
        sk = str(p.get('skipped_days', 0))
        print(f"{s_name:<35} | {pnl:<10} | {ret:<8} | {pf:<6} | {mdd:<9} | {wr:<8} | {ld:<9} | {sk:<6}")

    print("=" * 120)

    print("\n" + "*" * 85)
    print("             TỔNG KẾT ĐÁNH GIÁ PHASE 8 ADAPTIVE DCA")
    print("*" * 85)
    print(f"Hệ thống Adaptive DCA đạt Net Profit : ${perf_adaptive['net_profit']} (Return: {perf_adaptive['return_pct']}%)")
    print(f"Profit Factor                         : {perf_adaptive['profit_factor']}")
    print(f"Win Rate                              : {perf_adaptive['win_rate']}%")
    print(f"Max Drawdown                          : ${perf_adaptive['max_dd_dollars']} ({perf_adaptive['max_dd_pct']}%)")
    print(f"Số phiên AI chọn SKIP (Market C)      : {perf_adaptive['skipped_days']} phiên")
    print("*" * 85 + "\n")


if __name__ == "__main__":
    main()
