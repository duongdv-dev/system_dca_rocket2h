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

Phase 4: Robustness Testing & Parameter Plateau Detection (Hoàn thành / Đang thực thi)
  └── Tìm kiếm Vùng Cao Nguyên Tham Số (Parameter Plateau) thay vì chọn cực đại đơn lẻ dễ Overfit
  └── Phân tích ma trận 2D Heatmap & Thuật toán Lân cận 3x3 (Neighborhood Stability)
  └── Phân loại ký hiệu trực quan: [+++] Gold Plateau, [++] Stable Region, [+] Acceptable, [-] Danger Zone

Phase 5: Execution Engine & Risk Management
  └── Quản lý vốn nâng cao, trailing stop, partial close
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

### 2. Thuật Toán Plateau Score & Lân cận 3x3
- Với mỗi điểm tham số `(Step, Multiplier)`:
  - Tính $\text{Mean}_{\text{neighbors}}$ và $\text{Std}_{\text{neighbors}}$ của lân cận $3 \times 3$.
  - $\text{Plateau Score} = \frac{0.6 \times \text{Mean}_{\text{RF}} + 0.4 \times \text{Mean}_{\text{PF}}}{1.0 + \text{Std}_{\text{RF}}}$.
- Phân loại ma trận Heatmap:
  - `+++` : Vùng Cao Nguyên Vàng (Gold Plateau - Rất ổn định)
  - `++`  : Vùng Ổn Định (Stable Region)
  - `+`   : Vùng Chấp Nhận Đang Phát Triển (Acceptable)
  - `-`   : Vùng Nguy Hiểm / Vực Thẫm (Danger Zone / Cliff)

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

### Output Files (`v6_system/output/`)
- `daily_sessions_10_12.csv` (Dataset phiên Phase 1)
- `clean_m1_2020_2025.csv` (Dataset M1 sạch Phase 1)
- `data_quality_report.json` (Báo cáo chất lượng Phase 1)
- `phase2_baseline_summary.json` (Báo cáo kết quả Baseline Phase 2)
- `phase3_optimization_matrix.csv` (Ma trận 448 tổ hợp tham số Phase 3)
- `phase3_top_parameters.json` (Báo cáo Top 10 bộ tham số tối ưu Phase 3)
- `phase4_heatmaps.txt` (Ma trận Parameter Plateau Heatmap ASCII Phase 4)
- `phase4_plateau_report.json` (Báo cáo vùng tham số Plateau Phase 4)
