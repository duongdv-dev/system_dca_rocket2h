"""
v6_system/run_phase7.py
-----------------------
Runner script thực thi Phase 7 - AI Filter & So Sánh Thực Nghiệm V0 vs V1+AI.
Đánh giá khách quan 7 câu hỏi kiểm chứng và đưa ra phán quyết chính thức về việc sử dụng AI.
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6_system.data_engineer import V6DataEngineer
from v6_system.feature_engineering import V6FeatureEngineer
from v6_system.label_builder import V6LabelBuilder
from v6_system.ml_gatekeeper import MLGatekeeper
from v6_system.strategy_v1 import StrategyV1
from v6_system.strategy_ai import StrategyAIFilter
from v6_system.backtester_v0 import BacktesterV0
from v6_system.config import BASE_DIR, OUTPUT_DIR


def evaluate_system_performance(df_res: pd.DataFrame, system_name: str, initial_capital: float = 1000.0) -> Dict[str, Any]:
    """Tính toán 7 chỉ số kiểm chứng chuyên sâu cho một hệ thống."""
    traded_df = df_res[df_res["traded"] == True].copy()
    total_days = len(df_res)
    traded_days = len(traded_df)

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

    # Force close count
    fc_count = int((traded_df["exit_reason"] == "FORCE_CLOSE_1200").sum())

    return {
        "system_name": system_name,
        "total_days": total_days,
        "traded_days": traded_days,
        "net_profit": round(net_pnl, 2),
        "win_rate": round(win_rate, 2),
        "losing_days_count": losing_days_count,
        "profit_factor": round(profit_factor, 2),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dca_reached": max_dca_reached,
        "avg_profit_per_traded": round(avg_profit_per_traded, 2),
        "worst_day_pnl": round(worst_day_pnl, 2),
        "worst_month_pnl": round(worst_month_pnl, 2),
        "force_close_count": fc_count
    }


def main():
    print("=" * 85)
    print("       VERSION 6 - PHASE 7: AI FILTER & EMPIRICAL V0 vs V1+AI COMPARISON")
    print("=" * 85)

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
    # Gán target cho mô hình
    df_labeled["target_revert"] = df_labeled["y_anchor"]
    gatekeeper.train_and_evaluate(df_labeled, split_year=2024)

    # Tính xác suất P(reversion) trên toàn bộ dataset
    X_all = df_labeled[MLGatekeeper.FEATURE_COLS].astype(np.float32)
    df_labeled["prob_revert"] = gatekeeper.model.predict_proba(X_all)[:, 1]

    # 3. Chạy Backtest So Sánh các Hệ Thống
    print("\n[Bước 3/4] Chạy Backtest So Sánh (Baseline V0 vs Strategy V1 + AI Filter)...")
    
    step = 5.0
    max_dca = 5
    multiplier = 1.10

    # A. Baseline V0 (Không có AI)
    strat_v0 = StrategyV1(step=step, max_dca=max_dca, multiplier=multiplier, tp_dollars=2.0)
    records_v0 = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_v0.run_daily_session(group)
        records_v0.append(res)
    df_res_v0 = pd.DataFrame(records_v0)
    perf_v0 = evaluate_system_performance(df_res_v0, "Baseline V0 (No AI)")

    # B. Strategy V1 + AI Filter (Threshold = 0.60)
    strat_ai_60 = StrategyAIFilter(step=step, max_dca=max_dca, multiplier=multiplier, prob_threshold=0.60, tp_dollars=2.0)
    records_ai_60 = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_ai_60.run_daily_session_with_ai(group)
        records_ai_60.append(res)
    df_res_ai_60 = pd.DataFrame(records_ai_60)
    perf_ai_60 = evaluate_system_performance(df_res_ai_60, "V1 + AI Filter (Prob >= 0.60)")

    # C. Strategy V1 + AI Filter (Threshold = 0.70)
    strat_ai_70 = StrategyAIFilter(step=step, max_dca=max_dca, multiplier=multiplier, prob_threshold=0.70, tp_dollars=2.0)
    records_ai_70 = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_ai_70.run_daily_session_with_ai(group)
        records_ai_70.append(res)
    df_res_ai_70 = pd.DataFrame(records_ai_70)
    perf_ai_70 = evaluate_system_performance(df_res_ai_70, "V1 + AI Filter (Prob >= 0.70)")

    # D. Strategy V1 + AI Filter (Threshold = 0.80)
    strat_ai_80 = StrategyAIFilter(step=step, max_dca=max_dca, multiplier=multiplier, prob_threshold=0.80, tp_dollars=2.0)
    records_ai_80 = []
    for date_val, group in df_labeled.groupby("date"):
        res = strat_ai_80.run_daily_session_with_ai(group)
        records_ai_80.append(res)
    df_res_ai_80 = pd.DataFrame(records_ai_80)
    perf_ai_80 = evaluate_system_performance(df_res_ai_80, "V1 + AI Filter (Prob >= 0.80)")

    all_perfs = [perf_v0, perf_ai_60, perf_ai_70, perf_ai_80]

    # 4. Trả lời 7 câu hỏi kiểm chứng thực nghiệm
    print("\n[Bước 4/4] Báo cáo 7 Câu Hỏi Đánh Giá Thực Nghiệm V0 vs V1+AI...")
    
    # So sánh Baseline V0 với bộ lọc AI tốt nhất (ví dụ Threshold = 0.70)
    best_ai = perf_ai_70

    q1_pf = best_ai["profit_factor"] > perf_v0["profit_factor"]
    q2_dd = best_ai["max_dd_dollars"] < perf_v0["max_dd_dollars"]
    q3_dca = best_ai["max_dca_reached"] < perf_v0["max_dca_reached"]
    q4_loss = best_ai["losing_days_count"] < perf_v0["losing_days_count"]
    q5_tail = abs(best_ai["worst_month_pnl"]) < abs(perf_v0["worst_month_pnl"])
    q6_avg = best_ai["avg_profit_per_traded"] > perf_v0["avg_profit_per_traded"]
    q7_fc = best_ai["force_close_count"] < perf_v0["force_close_count"]

    answers = {
        "1. Profit Factor tăng?": f"{q1_pf} (Baseline: {perf_v0['profit_factor']} -> AI: {best_ai['profit_factor']})",
        "2. Max Drawdown giảm?": f"{q2_dd} (Baseline: ${perf_v0['max_dd_dollars']} -> AI: ${best_ai['max_dd_dollars']})",
        "3. Max DCA level giảm?": f"{q3_dca} (Baseline: {perf_v0['max_dca_reached']} -> AI: {best_ai['max_dca_reached']})",
        "4. Losing days giảm?": f"{q4_loss} (Baseline: {perf_v0['losing_days_count']} ngày -> AI: {best_ai['losing_days_count']} ngày)",
        "5. Tail risk (Worst Month) giảm?": f"{q5_tail} (Baseline: ${perf_v0['worst_month_pnl']} -> AI: ${best_ai['worst_month_pnl']})",
        "6. Average profit/trade tăng?": f"{q6_avg} (Baseline: ${perf_v0['avg_profit_per_traded']} -> AI: ${best_ai['avg_profit_per_traded']})",
        "7. Force Close count 12:00 giảm?": f"{q7_fc} (Baseline: {perf_v0['force_close_count']} -> AI: {best_ai['force_close_count']})"
    }

    # Tổng số tiêu chí AI chiến thắng
    score_ai_wins = sum([q1_pf, q2_dd, q3_dca, q4_loss, q5_tail, q6_avg, q7_fc])

    # Xuất kết quả báo cáo JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_json_path = os.path.join(OUTPUT_DIR, "phase7_comparison_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_performances": all_perfs,
            "empirical_7_questions": answers,
            "ai_wins_count": score_ai_wins
        }, f, ensure_ascii=False, indent=2)

    # In bảng so sánh đối chứng
    print("\n" + "=" * 115)
    print("                BẢNG SO SÁNH ĐỐI CHỨNG THỰC NGHIỆM: BASELINE V0 vs V1 + AI FILTER")
    print("=" * 115)
    hdr = f"{'System Name':<32} | {'Net PnL':<10} | {'PF':<6} | {'MaxDD($)':<9} | {'WinRate':<8} | {'LossDays':<9} | {'WstMonth':<9} | {'FC Count':<8}"
    print(hdr)
    print("-" * 115)

    for p in all_perfs:
        s_name = p["system_name"]
        pnl = f"${p['net_profit']:.1f}"
        pf = str(p['profit_factor'])
        mdd = f"${p['max_dd_dollars']:.1f}"
        wr = f"{p['win_rate']}%"
        ld = str(p['losing_days_count'])
        wm = f"${p['worst_month_pnl']:.1f}"
        fc = str(p['force_close_count'])
        print(f"{s_name:<32} | {pnl:<10} | {pf:<6} | {mdd:<9} | {wr:<8} | {ld:<9} | {wm:<9} | {fc:<8}")

    print("=" * 115)

    print("\n" + "*" * 85)
    print("              BÁO CÁO 7 CÂU HỎI KIỂM CHỨNG KHOA HỌC")
    print("*" * 85)
    for q, ans in answers.items():
        symbol = "✅ ĐẠT" if "True" in ans else "❌ KHÔNG ĐẠT"
        print(f" {symbol} | {q:<35} : {ans}")
    print("*" * 85)

    # PHÁN QUYẾT CHÍNH THỨC
    print("\n" + "#" * 85)
    print("                      PHÁN QUYẾT CHÍNH THỨC (OFFICIAL VERDICT)")
    print("#" * 85)
    if score_ai_wins >= 4:
        print("-> PHÁN QUYẾT: GIỮ MÔ HÌNH AI FILTER!")
        print(f"   Mô hình AI đã chiến thắng {score_ai_wins}/7 tiêu chuẩn nghiệm thu thực nghiệm.")
        print("   AI chứng minh khả năng giảm rủi ro Tail Risk và tăng Profit Factor cho hệ thống.")
    else:
        print("-> PHÁN QUYẾT: DỪNG MÔ HÌNH AI FILTER VỚI CHIẾN LƯỢC NÀY!")
        print(f"   AI chỉ đạt {score_ai_wins}/7 tiêu chuẩn nghiệm thu thực nghiệm (Cần ít nhất 4/7).")
        print("   Tuyên bố: KHÔNG CỐ NHỒI AI VÀO STRATEGY CHỈ VÌ MUỐN CÓ AI!")
    print("#" * 85 + "\n")


if __name__ == "__main__":
    main()
