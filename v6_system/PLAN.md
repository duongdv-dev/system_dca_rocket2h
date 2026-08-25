# Version 6 System Plan & Roadmap

Hệ thống giao dịch tự động XAUUSD Version 6 (`v6_system`) được thiết kế theo kiến trúc đa giai đoạn (Multi-Phase Architecture) nâng cao, tập trung vào tính tin cậy của dữ liệu, quy trình kiểm định nghiêm ngặt và tối ưu hóa phiên giao dịch.

---

## Tổng Quan Các Phase (Multi-Phase Roadmap)

```
Phase 1: Data Engineering (Hoàn thành)
  └── Đọc & kiểm định dữ liệu thô Dukascopy 2020-2025
  └── Kiểm tra Timeframe, Timezone (Asia/Ho_Chi_Minh), OHLC, Volume, Duplicate, Gap, DST
  └── Trích xuất phiên 10:00 -> 12:00 VN & các chỉ số (anchor_price, candle 10:00/12:00, high/low/close)

Phase 2: Strategy Baseline V0 (Hoàn thành)
  └── Đánh giá ý tưởng DCA gốc có Edge hay không (không tối ưu hóa, không AI)
  └── Anchor 10:00 VN, Step cố định, Multiplier = 1.0, Lot = 0.01
  └── TP = Basket Profit ($), Force Close = 12:00 VN, Max cycle/day = 1

Phase 3: DCA Optimization & Multi-Metric Risk Scorecard (Hoàn thành)
  └── Grid Search 448 tổ hợp tham số: DCA Step (3..15), Max DCA (2..8), Multiplier (1.00..1.40)
  └── Đánh giá 10 chỉ số đo lường rủi ro: Profit Factor, Max DD, Recovery Factor, Win Rate, Average Trade, Worst Day, Worst Month, Max DCA, Max Exposure, Force Close %

Phase 4: Robustness Testing & Parameter Plateau Detection (Hoàn thành)
  └── Tìm kiếm Vùng Cao Nguyên Tham Số (Parameter Plateau) thay vì chọn cực đại đơn lẻ dễ Overfit
  └── Phân tích ma trận 2D Heatmap & Thuật toán Lân cận 3x3 (Neighborhood Stability)
  └── Phân loại ký hiệu trực quan: [+++] Gold Plateau, [++] Stable Region, [+] Acceptable, [-] Danger Zone

Phase 5: XGBoost V1 Probability Model (AI Gatekeeper) (Hoàn thành / Đang thực thi)
  └── AI không điều khiển DCA, chỉ ước lượng xác suất hồi về Anchor trước 12:00 VN
  └── Trích xuất 25 Đặc trưng Kỹ thuật (DistanceFromAnchor, ATR, RSI, ADX, EMAs, Volume Ratio, VWAP, Wicks)
  └── Phân chia Time-Series (Train 2020-2023 | Test 2024-2025), tính ROC-AUC, PR-AUC & Feature Importance

Phase 6: Execution Engine & Live Risk Control (Kế hoạch tiếp theo)
  └── Tích hợp AI Gatekeeper với DCA Strategy Engine
  └── Kết nối Live/Demo Execution (MT5 / FIX API)
```

---

## Chi Tiết Phase 1 — Data Engineering

### 1. Mục tiêu
Biến dữ liệu nến M1 XAUUSD thô từ Dukascopy (giai đoạn 2020–2025) thành dataset sạch, đáng tin cậy để phục vụ backtest và phân tích phiên giao dịch.

---

## Chi Tiết Phase 2 — Strategy Baseline V0

### 1. Mục tiêu Cốt lõi
Kiểm chứng ý tưởng DCA Mean-Reversion nguyên bản trên dữ liệu 2020-2025.

---

## Chi Tiết Phase 3 — DCA Optimization & Multi-Metric Scorecard

### 1. Mục tiêu
Thực hiện Grid Search trên 448 tổ hợp tham số và đánh giá Scorecard 10 chỉ số đo lường rủi ro.

---

## Chi Tiết Phase 4 — Robustness Testing & Plateau Detection

### 1. Mục tiêu Cốt lõi
Xác định **Vùng Cao Nguyên Tham Số (Parameter Plateau)** ổn định để tránh điểm cực đại cô lập (Overfitting Peak/Cliff).

---

## Chi Tiết Phase 5 — XGBoost V1 Probability Model (AI Gatekeeper)

### 1. Mục tiêu & Vai trò của AI
Mô hình AI **XGBoost V1** đóng vai trò làm Gatekeeper đưa ra xác suất:
*"Tại thời điểm hiện tại t, xác suất giá sẽ hồi về Anchor trước 12:00 VN là bao nhiêu?"*

### 2. Danh sách 25 Đặc trưng (Features)
- `distance_from_anchor`, `dist_anchor_over_atr`
- `atr`, `atr_norm`
- `rsi`, `adx`
- `ema_9`, `ema_21`, `ema_50`, `ema_slope`
- `volume`, `vol_over_avg`
- `return_1m`, `return_5m`, `return_15m`, `volatility`
- `time_since_10`, `time_remaining_12`
- `session_high`, `session_low`
- `candle_body`, `upper_wick`, `lower_wick`
- `vwap`, `dist_to_vwap`

### 3. Đánh giá Mô hình ML
- Phân chia dữ liệu: **Train (2020–2023)** vs **Test (2024–2025)**.
- Chỉ số đánh giá: **ROC-AUC**, **PR-AUC**, **Precision**, **Recall**, **F1-Score**.
- Xuất thứ hạng tầm quan trọng đặc trưng (Feature Importance Ranking).

---

## Hướng Dẫn Vận Hành Code Version 6

### Phase 1: Data Engineering
```bash
python3 v6_system/test_phase1.py
python3 v6_system/run_phase1.py
```

### Phase 2: Strategy Baseline V0
```bash
python3 v6_system/test_phase2.py
python3 v6_system/run_phase2.py
```

### Phase 3: DCA Optimization
```bash
python3 v6_system/test_phase3.py
python3 v6_system/run_phase3.py
```

### Phase 4: Robustness Testing & Plateau Analysis
```bash
python3 v6_system/test_phase4.py
python3 v6_system/run_phase4.py
```

### Phase 5: XGBoost V1 Probability Model
```bash
python3 v6_system/test_phase5.py
python3 v6_system/run_phase5.py
```

### Output Files (`v6_system/output/`)
- `daily_sessions_10_12.csv` (Dataset phiên Phase 1)
- `clean_m1_2020_2025.csv` (Dataset M1 sạch Phase 1)
- `data_quality_report.json` (Báo cáo chất lượng Phase 1)
- `phase2_baseline_summary.json` (Báo cáo kết quả Baseline Phase 2)
- `phase3_optimization_matrix.csv` (Ma trận 448 tổ hợp tham số Phase 3)
- `phase3_top_parameters.json` (Báo cáo Top 10 bộ tham số tối ưu Phase 3)
- `phase4_heatmaps.txt` (Ma trận Parameter Plateau Heatmap ASCII Phase 4)
- `phase4_plateau_report.json` (Báo cáo vùng tham số Plateau Phase 4)
- `phase5_model_metrics.json` (Báo cáo chỉ số mô hình XGBoost V1 Phase 5)
- `phase5_feature_importance.csv` (Thứ hạng 25 đặc trưng XGBoost Phase 5)
- `phase5_features_sample.csv` (Mẫu dữ liệu 25 đặc trưng & Target Phase 5)
