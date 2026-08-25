# Version 6 System Plan & Roadmap

Hệ thống giao dịch tự động XAUUSD Version 6 (`v6_system`) được thiết kế theo kiến trúc đa giai đoạn (Multi-Phase Architecture) nâng cao, tập trung vào tính tin cậy của dữ liệu, quy trình kiểm định nghiêm ngặt và tối ưu hóa phiên giao dịch.

---

## Tổng Quan Các Phase (Multi-Phase Roadmap)

```
Phase 1: Data Engineering (Hoàn thành)
  └── Đọc & kiểm định dữ liệu thô Dukascopy 2020-2025
  └── Kiểm tra Timeframe, Timezone (Asia/Ho_Chi_Minh), OHLC, Volume, Duplicate, Gap, DST
  └── Trích xuất phiên 10:00 -> 12:00 VN & các chỉ số (anchor_price, candle 10:00/12:00, high/low/close)

Phase 2: Strategy Baseline V0 (Hoàn thành / Đang thực thi)
  └── Đánh giá ý tưởng DCA gốc có Edge hay không (không tối ưu hóa, không AI)
  └── Anchor 10:00 VN, Step cố định, Multiplier = 1.0, Lot = 0.01
  └── TP = Basket Profit ($), Force Close = 12:00 VN, Max cycle/day = 1

Phase 3: Backtest Engine & Advanced Performance Analytics
  └── Giả lập chi tiết Spread, Slippage, Swap, Commission
  └── Thống kê Sharpe, Sortino, Max Drawdown, Win Rate, Expectancy

Phase 4: AI / ML Gatekeeper & Parameter Optimization
  └── Xây dựng mô hình AI (XGBoost / LightGBM / Neural Net) để lọc tín hiệu nhiễu
  └── Tối ưu hóa siêu tham số (Hyperparameter Tuning)

Phase 5: Execution Engine & Risk Management
  └── Quản lý vốn nâng cao, trailing stop, partial close
  └── Kết nối Live/Demo Execution (MT5 / FIX API)
```

---

## Chi Tiết Phase 1 — Data Engineering

### 1. Mục tiêu
Biến dữ liệu nến M1 XAUUSD thô từ Dukascopy (giai đoạn 2020–2025) thành dataset sạch, đáng tin cậy để phục vụ backtest và phân tích phiên giao dịch.

### 2. Các Hạng Mục Kiểm Định (8 Steps Audit)

1. **Kiểm tra Timeframe**:
   - Xác minh khoảng cách thời gian giữa các nến M1 liên tiếp (chuẩn là 60 giây).
   - Xác định và thống kê các khoảng mất nến intraday (Missing candles).

2. **Kiểm tra Timezone & DST**:
   - Dữ liệu thô gốc Dukascopy là Unix Timestamp (epoch milliseconds UTC).
   - Chuyển đổi toàn bộ timestamp về múi giờ Việt Nam `Asia/Ho_Chi_Minh` (UTC+7).
   - Đảm bảo thời gian tăng đơn điệu (monotonic). UTC+7 cố định không bị ảnh hưởng bởi Daylight Saving Time (DST).

3. **Kiểm tra OHLC (Price Integrity)**:
   - Kiểm tra quan hệ logic giá: `High >= Open`, `High >= Low`, `High >= Close`, `Low <= Open`, `Low <= Close`.
   - Kiểm tra và loại bỏ các giá trị âm hoặc bằng 0 (`open <= 0`, `high <= 0`, `low <= 0`, `close <= 0`).

4. **Kiểm tra Volume**:
   - Kiểm tra `volume >= 0`. Phát hiện volume bằng 0 hoặc âm (chuẩn hóa volume âm về 0).

5. **Xử lý Duplicate**:
   - Phát hiện và loại bỏ các dòng bị trùng lặp `timestamp` UTC.

6. **Missing Candles Analysis**:
   - Tính toán tổng số nến M1 bị thiếu trong các đợt gián đoạn intraday.

7. **Gap Bất Thường (Abnormal Price Gaps)**:
   - Cảnh báo các khoảng nhảy giá bất thường trong cùng ngày: `|Open(t) - Close(t-1)| > $5.0`.

8. **Xuất Báo Cáo & Dataset Clean**:
   - Lưu báo cáo tổng hợp chất lượng dữ liệu dạng JSON.
   - Lưu trữ dataset M1 đã làm sạch và dataset phiên hàng ngày.

---

## Chi Tiết Phase 2 — Strategy Baseline V0

### 1. Mục tiêu Cốt lõi
Đây là phase quan trọng nhất nhằm thử nghiệm chiến lược đơn giản nhất ở dạng nguyên bản để kiểm chứng câu hỏi: **Strategy DCA cơ bản có Edge / hoạt động hay không?** (Chưa thực hiện tối ưu hóa hay AI).

### 2. Luật Giao Dịch V0

1. **Khởi tạo (10:00 VN)**:
   - Tại nến `10:00:00 VN`, xác định `Anchor = Open Price`.
   - Giới hạn: `Max cycle/day = 1`.

2. **Kích hoạt Lệnh DCA**:
   - **Giá tăng**: `Anchor + k * Step` -> Mở lệnh **SELL** tầng `k` (Khối lượng Lot = 0.01).
   - **Giá giảm**: `Anchor - k * Step` -> Mở lệnh **BUY** tầng `k` (Khối lượng Lot = 0.01).

3. **Tham số Cố định (V0 Baseline)**:
   - `Lot`: `0.01` (Cố định).
   - `Multiplier`: `1.0` (Cố định khối lượng tất cả tầng).
   - `Step`: Cố định (Ví dụ thử nghiệm $2.0, $3.0, $5.0).
   - `Max DCA`: Cố định (Ví dụ 3, 5, 10 tầng).
   - `TP`: `Basket Profit` ($) (Lợi nhuận giỏ lệnh).
   - `Force Close`: Đúng **12:00:00 VN** đóng 100% lệnh đang mở.

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

### Output Files (`v6_system/output/`)
- `daily_sessions_10_12.csv` (Dataset phiên hàng ngày Phase 1)
- `clean_m1_2020_2025.csv` (Dataset M1 sạch Phase 1)
- `data_quality_report.json` (Báo cáo chất lượng Phase 1)
- `phase2_baseline_summary.json` (Báo cáo kết quả so sánh Phase 2)
- `phase2_daily_trades_preset1.csv` (Chi tiết giao dịch từng ngày Phase 2)
