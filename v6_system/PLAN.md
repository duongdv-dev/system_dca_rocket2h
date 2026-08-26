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

Phase 5: XGBoost V1 Probability Model (AI Gatekeeper) (Hoàn thành)
  └── AI không điều khiển DCA, chỉ ước lượng xác suất hồi về Anchor trước 12:00 VN
  └── Trích xuất 25 Đặc trưng Kỹ thuật (DistanceFromAnchor, ATR, RSI, ADX, EMAs, Volume Ratio, VWAP, Wicks)
  └── Phân chia Time-Series (Train 2020-2023 | Test 2024-2025), tính ROC-AUC, PR-AUC & Feature Importance

Phase 6: Advanced AI Labeling System (Xây Label cho AI) (Hoàn thành)
  └── Không train nến tiếp theo tăng/giảm. Train nhãn hồi phiên: Y=1 nếu quay về Anchor / đạt TP trước 12:00, Y=0 nếu không.
  └── Hỗ trợ 3 bộ nhãn chuyên sâu: y_anchor, y_basket_tp, y_safe_revert.
  └── So sánh hiệu năng dự báo mô hình AI Gatekeeper trên từng loại nhãn.

Phase 7: AI Filter & Empirical V0 vs V1+AI Comparison (Hoàn thành)
  └── Tích hợp mô hình AI Filter kiểm soát xác suất P(reversion) >= Threshold trước khi mở lệnh DCA.
  └── So sánh đối chứng 7 chỉ số khoa học giữa Baseline V0 vs Strategy V1 + AI Filter.
  └── Đưa ra phán quyết chính thức: Giữ AI nếu cải thiện rõ rệt, hoặc DỪNG AI nếu không tạo edge mới.

Phase 8: Adaptive DCA (Điều Khiển Tham Số Thích Ứng Theo AI) (Hoàn thành)
  └── AI chủ động điều khiển tham số động theo 3 Chế độ Thị trường: Market A (P>=80%), Market B (60%<=P<80%), Market C (P<60% -> SKIP).
  └── Tích hợp Stop Loss cứng (Max Distance) khống chế hoàn toàn rủi ro Tail Risk.
  └── So sánh tổng thể hiệu năng Baseline V0 vs Static Plateau vs Adaptive DCA.

Phase 9: AI Parameter Selection (Lựa Chọn Cấu Hình An Toàn Từ Pre-defined Grid) (Hoàn thành)
  └── AI không tự tạo ra tham số ngẫu nhiên "điên rồ" (như Step=6.37).
  └── AI lựa chọn bộ tham số tối ưu từ Danh Mục An Toàn Được Duyệt Trước (Strategy A, B, C, D) hoặc SKIP.
  └── Đảm bảo tính an toàn 100% và khống chế tối đa rủi ro vận hành.

Phase 10: Walk-Forward Testing & Generalization Verification (Hoàn thành / Đang thực thi)
  └── Kiểm thử cuộn thời gian 4 Folds cuộn cửa sổ mở rộng (Expanding Windows): Test 2022, 2023, 2024, 2025 OOS.
  └── Ghép nối kết quả Out-of-Sample thực tế và tính chỉ số Walk-Forward Efficiency (WFE >= 0.70).
  └── Khẳng định năng lực khái quát hóa của hệ thống trên dữ liệu tương lai chưa từng thấy.
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

---

## Chi Tiết Phase 6 — Advanced AI Labeling System (Xây Label cho AI)

### 1. Mục tiêu & Quy tắc Gán Nhãn
- `Y = 1`: Giá đạt basket TP / quay về Anchor trước 12:00 VN.
- `Y = 0`: Không đạt trước 12:00 VN.

---

## Chi Tiết Phase 7 — AI Filter & Empirical V0 vs V1+AI Comparison

### 1. Mục tiêu & Cơ chế AI Filter
Đánh giá thực nghiệm 7 tiêu chí khoa học giữa Baseline V0 vs V1 + AI Filter.

---

## Chi Tiết Phase 8 — Adaptive DCA (Điều Khiển Tham Số Thích Ứng Theo AI)

### 1. Mục tiêu
Cho phép AI chủ động phân loại trạng thái thị trường và gán tham số động theo 3 Chế độ Thị trường A, B, C.

---

## Chi Tiết Phase 9 — AI Parameter Selection (Lựa Chọn Cấu Hình An Toàn)

### 1. Mục tiêu An Toàn
AI lựa chọn bộ tham số từ **Danh Mục Cấu Hình An Toàn Đã Được Duyệt Trước (Pre-defined Safe Grid Menu)** (Strategy A, B, C, D, SKIP).

---

## Chi Tiết Phase 10 — Walk-Forward Testing & Generalization Verification

### 1. Mục tiêu & Cấu trúc 4 Folds (Expanding Windows)
- **Fold 1**: Train `2020–2021` | Out-of-Sample Test: `2022`
- **Fold 2**: Train `2020–2022` | Out-of-Sample Test: `2023`
- **Fold 3**: Train `2020–2023` | Out-of-Sample Test: `2024`
- **Fold 4**: Train `2020–2024` | Out-of-Sample Test: `2025`

### 2. Chỉ số Nghiệm Thu Khái Quát Hóa
- **Walk-Forward Efficiency Ratio (WFE)**: $\frac{\text{Out-of-Sample Profit Factor}}{\text{In-Sample Profit Factor}} \ge 0.70$.
- Kết nối đường cong tài sản Out-of-Sample thực tế của 4 năm 2022–2025.

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

### Phase 6: Advanced AI Labeling System
```bash
python3 v6_system/test_phase6.py
python3 v6_system/run_phase6.py
```

### Phase 7: AI Filter & V0 vs V1+AI Comparison
```bash
python3 v6_system/test_phase7.py
python3 v6_system/run_phase7.py
```

### Phase 8: Adaptive DCA
```bash
python3 v6_system/test_phase8.py
python3 v6_system/run_phase8.py
```

### Phase 9: AI Parameter Selection (Safe Pre-defined Grid Menu)
```bash
python3 v6_system/test_phase9.py
python3 v6_system/run_phase9.py
```

### Phase 10: Walk-Forward Testing (Expanding Windows)
```bash
python3 v6_system/test_phase10.py
python3 v6_system/run_phase10.py
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
- `phase6_label_analysis.json` (Báo cáo so sánh hiệu năng AI trên 3 loại nhãn Phase 6)
- `phase6_features_labeled.csv` (Dataset đặc trưng và 3 loại nhãn AI Phase 6)
- `phase7_comparison_report.json` (Báo cáo so sánh đối chứng 7 chỉ số V0 vs V1+AI Phase 7)
- `phase8_adaptive_summary.json` (Báo cáo tổng hợp hiệu năng Adaptive DCA Phase 8)
- `phase8_comparison.csv` (Bảng so sánh 3 hệ thống V0 vs Static vs Adaptive Phase 8)
- `phase9_grid_ai_summary.json` (Báo cáo tổng hợp hiệu năng Safe Grid AI Selector Phase 9)
- `phase9_selection_distribution.csv` (Tần suất phân phối các cấu hình an toàn Phase 9)
- `phase10_walk_forward_summary.json` (Báo cáo tổng hợp kiểm thử 4 Folds Walk-Forward Phase 10)
- `phase10_oos_trades.csv` (Chi tiết nhật ký giao dịch Out-of-Sample thực tế 2022-2025 Phase 10)
