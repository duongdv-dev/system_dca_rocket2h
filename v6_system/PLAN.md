# Version 6 System Plan & Roadmap

Hệ thống giao dịch tự động XAUUSD Version 6 (`v6_system`) được thiết kế theo kiến trúc đa giai đoạn (Multi-Phase Architecture) nâng cao, tập trung vào tính tin cậy của dữ liệu, quy trình kiểm định nghiêm ngặt và tối ưu hóa phiên giao dịch.

---

## Tổng Quan Các Phase (Multi-Phase Roadmap)

```
Phase 1: Data Engineering (Hiện tại)
  └── Đọc & kiểm định dữ liệu thô Dukascopy 2020-2025
  └── Kiểm tra Timeframe, Timezone (Asia/Ho_Chi_Minh), OHLC, Volume, Duplicate, Gap, DST
  └── Trích xuất phiên 10:00 -> 12:00 VN & các chỉ số (anchor_price, candle 10:00/12:00, high/low/close)

Phase 2: Feature Engineering & Signal Generation
  └── Tính toán các chỉ báo phiên (Session Range, Anchor Deviation, Volatility Spikes)
  └── Xây dựng các quy tắc vào lệnh (Strategy Rules) dựa trên phiên 10:00 - 12:00 VN

Phase 3: Backtest Engine & Performance Analytics
  └── Giả lập khớp lệnh chi tiết (Spread, Slippage, Swap, Commission)
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

## 3. Cấu Trúc Session Output 10:00 → 12:00 VN

Dữ liệu tổng hợp phiên hàng ngày (`daily_sessions_10_12.csv`) bao gồm các cột:

- `date`: Ngày giao dịch (`YYYY-MM-DD` giờ VN)
- `anchor_price`: Giá Open tại nến 10:00 VN
- `open_1000`, `high_1000`, `low_1000`, `close_1000`, `volume_1000`: Chi tiết nến 10:00 VN
- `open_1200`, `high_1200`, `low_1200`, `close_1200`, `volume_1200`: Chi tiết nến 12:00 VN
- `session_high`: Giá High cao nhất trong khoảng 10:00 → 12:00 VN
- `session_low`: Giá Low thấp nhất trong khoảng 10:00 → 12:00 VN
- `session_close`: Giá Close phiên (Close nến 12:00 VN)
- `candle_count`: Số nến M1 thực tế trong phiên

---

## 4. Hướng Dẫn Vận Hành Code Phase 1

### Chạy Unit Test
```bash
python3 v6_system/test_phase1.py
```

### Chạy Pipeline Phase 1
```bash
python3 v6_system/run_phase1.py
```

### Kết quả đầu ra
File kết quả được lưu tự động tại `v6_system/output/`:
- `v6_system/output/daily_sessions_10_12.csv`
- `v6_system/output/clean_m1_2020_2025.csv`
- `v6_system/output/data_quality_report.json`
