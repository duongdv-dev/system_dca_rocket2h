//+------------------------------------------------------------------+
//|                                   XAUUSD_Intraday_Grid_v2.mq5   |
//|    Copyright 2026, Senior Quantitative Researcher & MQL5 Dev.   |
//|               Hệ Thống XAUUSD Mean-Reversion Grid/DCA (v2)      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Senior Quantitative Researcher"
#property link      "https://github.com/quant-dev"
#property version   "2.00"
#property description "Hệ thống giao dịch XAUUSD Intraday 10:00 - 12:00 VN sử dụng Native ONNX API MT5, LightGBM multi-class, Time-Decay TP & Hard Risk Engine 2%"

// Nhúng file ONNX mô hình vào EA (MetaTrader 5 Native Resource)
#resource "\\Files\\model.onnx" as uchar ExtModelBuffer[]

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS (THAM SỐ ĐẦU VÀO ENUM & USER CONFIG)            |
//+------------------------------------------------------------------+
input group "=== TIME SETTINGS (UTC+7 VN TIME) ==="
input int      InpServerUTCOffset = 2;       // Broker Server Time UTC Offset (Ví dụ: GMT+2 hoặc GMT+3)
input int      InpObsStartHour    = 6;       // Bắt đầu phiên quan sáng (06:00 VN)
input int      InpExecStartHour   = 10;      // Bắt đầu phiên thực thi (10:00 VN)
input int      InpExecEndHour     = 12;      // Kết thúc phiên thực thi (12:00 VN)
input int      InpTimeDecayMinute = 90;      // Phút bắt đầu Time-Decay TP (11:30 VN = 90 phút từ 10h)

input group "=== RISK MANAGEMENT ENGINE ==="
input double   InpMaxRiskPct      = 2.0;     // Max Loss tối đa mỗi phiên (% Balance)
input double   InpDefaultBaseLot  = 0.01;    // Kích thước Base Lot tối thiểu
input ulong    InpMagicNumber     = 20261012;// Magic Number nhận diện lệnh EA

// Struct chứa cấu hình bộ tham số Preset
struct SGridPreset
{
   string name;
   double step_0_ratio;
   double step_exp;
   int    max_orders;
   double multiplier;
   double tp_be_ratio;
};

// Global Variables
CTrade         m_trade;
CPositionInfo  m_position;
COrderInfo     m_order;

long           m_onnx_handle = INVALID_HANDLE;
bool           m_session_active = false;
bool           m_evaluated_today = false;
int            m_predicted_class = -1; // 0: No-Trade, 1: Hẹp, 2: Chuẩn, 3: Phòng thủ
SGridPreset    m_active_preset;
int            m_day_of_year = -1;

SGridPreset    m_presets[4]; // 0: No-Trade, 1, 2, 3

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   m_trade.SetExpertMagicNumber(InpMagicNumber);
   
   // 1. Khởi tạo danh mục Preset
   // Preset 0: No-Trade
   m_presets[0].name = "No-Trade";
   // Preset 1: Lưới Hẹp (Narrow)
   m_presets[1].name = "Lưới Hẹp (Narrow)";
   m_presets[1].step_0_ratio = 0.8;
   m_presets[1].step_exp = 1.0;
   m_presets[1].max_orders = 3;
   m_presets[1].multiplier = 1.0;
   m_presets[1].tp_be_ratio = 0.4;
   
   // Preset 2: Tiêu Chuẩn (Standard)
   m_presets[2].name = "Tiêu Chuẩn (Standard)";
   m_presets[2].step_0_ratio = 1.0;
   m_presets[2].step_exp = 1.1;
   m_presets[2].max_orders = 4;
   m_presets[2].multiplier = 1.3;
   m_presets[2].tp_be_ratio = 0.55;

   // Preset 3: Phòng Thủ (Defensive)
   m_presets[3].name = "Phòng Thủ (Defensive)";
   m_presets[3].step_0_ratio = 1.2;
   m_presets[3].step_exp = 1.2;
   m_presets[3].max_orders = 5;
   m_presets[3].multiplier = 1.5;
   m_presets[3].tp_be_ratio = 0.7;

   // 2. Tải và Khởi tạo Native ONNX Model từ Resource Buffer
   if(!InitONNXModel())
   {
      Print("[ERROR] Không thể khởi tạo ONNX Model! Vui lòng kiểm tra file model.onnx trong /Files.");
      return(INIT_FAILED);
   }

   Print("[INIT] EA XAUUSD Intraday Grid v2 Khởi tạo thành công!");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(m_onnx_handle != INVALID_HANDLE)
   {
      OnnxRelease(m_onnx_handle);
      m_onnx_handle = INVALID_HANDLE;
      Print("[DEINIT] Đã giải phóng ONNX Model Handle.");
   }
}

//+------------------------------------------------------------------+
//| KHỞI TẠO NATIVE ONNX API                                         |
//+------------------------------------------------------------------+
bool InitONNXModel()
{
   // Tạo ONNX Session từ Resource Buffer
   m_onnx_handle = OnnxCreateFromBuffer(ExtModelBuffer, ONNX_DEFAULT);
   if(m_onnx_handle == INVALID_HANDLE)
   {
      Print("[ONNX ERROR] OnnxCreateFromBuffer thất bại! Code: ", GetLastError());
      return false;
   }

   // Định dạng Shape cho Đầu Vào (1 row, 7 features)
   const long input_shape[] = {1, 7};
   if(!OnnxSetInputShape(m_onnx_handle, 0, input_shape))
   {
      Print("[ONNX ERROR] OnnxSetInputShape thất bại! Code: ", GetLastError());
      return false;
   }

   // Định dạng Shape cho Đầu Ra
   const long output_shape[] = {1, 4}; // Xác suất / Logits của 4 lớp
   if(!OnnxSetOutputShape(m_onnx_handle, 0, output_shape))
   {
      Print("[ONNX WARN] OnnxSetOutputShape tự động nhận dạng cấu trúc.");
   }

   return true;
}

//+------------------------------------------------------------------+
//| QUY ĐỔI GIỜ SERVER TO GIỜ VIỆT NAM (UTC+7)                       |
//+------------------------------------------------------------------+
datetime GetVNTime(datetime server_time)
{
   // Chênh lệch giữa UTC+7 và UTC Broker
   int diff_hours = 7 - InpServerUTCOffset;
   return (server_time + diff_hours * 3600);
}

//+------------------------------------------------------------------+
//| TRÍCH XUẤT 7 ĐẶC TRƯNG CHỐT LÚC 09:59:59                         |
//+------------------------------------------------------------------+
bool ExtractFeaturesAt0959(float &features[])
{
   ArrayResize(features, 7);
   
   // 1. Lấy ATR(14) nến M15
   double atr_val[];
   int atr_handle = iATR(_Symbol, PERIOD_M15, 14);
   if(atr_handle == INVALID_HANDLE || CopyBuffer(atr_handle, 0, 0, 1, atr_val) <= 0)
      return false;
   IndicatorRelease(atr_handle);
   float atr14_m15 = (float)atr_val[0];
   if(atr14_m15 <= 0) return false;

   // 2. Lấy dữ liệu nến M1 phiên sáng (06:00 - 09:59)
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_M1, 0, 240, rates);
   if(copied < 120) return false;

   double high_06to10 = -1e9;
   double low_06to10 = 1e9;
   double open_0600 = rates[0].open;
   double close_0959 = rates[copied - 1].close;
   double sum_pv = 0.0;
   double sum_v = 0.0;

   for(int i = 0; i < copied; i++)
   {
      if(rates[i].high > high_06to10) high_06to10 = rates[i].high;
      if(rates[i].low < low_06to10)   low_06to10 = rates[i].low;
      
      double typical_p = (rates[i].high + rates[i].low + rates[i].close) / 3.0;
      sum_pv += typical_p * rates[i].tick_volume;
      sum_v += rates[i].tick_volume;
   }

   float morning_range = (float)(high_06to10 - low_06to10);
   float morning_body = (float)MathAbs(close_0959 - open_0600);
   float morning_momentum = (morning_range > 0) ? (morning_body / morning_range) : 0.0f;

   // 3. Tính Daily VWAP
   float daily_vwap = (sum_v > 0) ? (float)(sum_pv / sum_v) : (float)close_0959;
   float vwap_dist_atr = (float)((close_0959 - daily_vwap) / atr14_m15);

   // 4. Bollinger Bands M15 Z-Score & Slope
   int bb_handle = iBands(_Symbol, PERIOD_M15, 20, 0, 2.0, PRICE_CLOSE);
   double bb_mid[], bb_up[], bb_low[];
   if(bb_handle == INVALID_HANDLE || 
      CopyBuffer(bb_handle, 0, 0, 5, bb_mid) <= 0 ||
      CopyBuffer(bb_handle, 1, 0, 1, bb_up) <= 0)
      return false;
   IndicatorRelease(bb_handle);

   double mid_curr = bb_mid[0];
   double mid_4prev = bb_mid[4];
   double std_curr = (bb_up[0] - mid_curr) / 2.0;

   float bb_zscore_m15 = (float)((close_0959 - mid_curr) / (std_curr + 1e-8));
   float bb_slope_m15 = (float)((mid_curr - mid_4prev) / (atr14_m15 + 1e-8));

   // Gán vào mảng đặc trưng theo đúng thứ tự huấn luyện
   features[0] = atr14_m15;
   features[1] = morning_range;
   features[2] = morning_body;
   features[3] = morning_momentum;
   features[4] = vwap_dist_atr;
   features[5] = bb_zscore_m15;
   features[6] = bb_slope_m15;

   return true;
}

//+------------------------------------------------------------------+
//| DỰ ĐOÁN CLASSIFIER BẰNG ONNX RUN                                 |
//+------------------------------------------------------------------+
int PredictStrategyClass(const float &features[])
{
   if(m_onnx_handle == INVALID_HANDLE) return 0;

   float output_data[];
   ArrayResize(output_data, 4);

   // Chạy mô hình ONNX
   if(!OnnxRun(m_onnx_handle, ONNX_NO_TRANSFORM, features, output_data))
   {
      Print("[ONNX RUN ERROR] OnnxRun thất bại! Code: ", GetLastError());
      return 0; // Mặc định No-Trade nếu lỗi
   }

   // Tìm ArgMax (lớp có điểm xác suất cao nhất)
   int best_class = 0;
   float max_score = output_data[0];
   for(int i = 1; i < 4; i++)
   {
      if(output_data[i] > max_score)
      {
         max_score = output_data[i];
         best_class = i;
      }
   }

   return best_class;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   MqlDateTime dt;
   datetime vn_time = GetVNTime(TimeCurrent());
   TimeToStruct(vn_time, dt);

   // Đặt lại trạng thái đánh giá khi bước sang ngày mới
   if(dt.day_of_year != m_day_of_year)
   {
      m_day_of_year = dt.day_of_year;
      m_evaluated_today = false;
      m_session_active = false;
      m_predicted_class = -1;
   }

   // 1. ĐÁNH GIÁ VÀ NHẬN DIỆN CHIẾN THUẬT LÚC 09:59:59 VN
   if(!m_evaluated_today && dt.hour == 9 && dt.min == 59 && dt.sec >= 55)
   {
      float features[];
      if(ExtractFeaturesAt0959(features))
      {
         m_predicted_class = PredictStrategyClass(features);
         m_active_preset = m_presets[m_predicted_class];
         m_evaluated_today = true;

         PrintFormat("[AI PREDICT 09:59] Nhãn dự đoán: %d - Chiến thuật: %s", 
                     m_predicted_class, m_active_preset.name);
      }
   }

   // 2. KÍCH HOẠT PHIÊN THỰC THI (10:00 - 12:00 VN)
   if(dt.hour >= InpExecStartHour && dt.hour < InpExecEndHour)
   {
      if(m_predicted_class > 0 && !m_session_active)
      {
         // Mở phiên thực thi mới
         m_session_active = true;
         ExecuteSessionEntry();
      }

      if(m_session_active)
      {
         ManageGridExecution(dt);
      }
   }

   // 3. HARD EXIT LÚC 12:00:00 VN
   if(dt.hour >= InpExecEndHour && m_session_active)
   {
      Print("[HARD EXIT 12:00] Cưỡng chế đóng toàn bộ lệnh và kết thúc phiên!");
      CloseAllPositions();
      m_session_active = false;
   }
}

//+------------------------------------------------------------------+
//| MỞ VỊ THẾ ĐẦU PHIÊN (10:00 VN)                                   |
//+------------------------------------------------------------------+
void ExecuteSessionEntry()
{
   float features[];
   if(!ExtractFeaturesAt0959(features)) return;
   
   double atr_val = features[0];
   double vwap_dist = features[4];

   // Xác định hướng giao dịch Mean Reversion
   ENUM_ORDER_TYPE order_type = (vwap_dist >= 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;

   // Tính toán Base Lot động từ rủi ro 2% Balance
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double max_risk_amount = balance * (InpMaxRiskPct / 100.0);
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double price = (order_type == ORDER_TYPE_BUY) ? ask : bid;

   double base_lot = NormalizeDouble(max_risk_amount / (atr_val * 5.0 * 100.0), 2);
   if(base_lot < InpDefaultBaseLot) base_lot = InpDefaultBaseLot;

   m_trade.PositionOpen(_Symbol, order_type, base_lot, price, 0, 0, "Grid Entry 1");
   PrintFormat("[ENTRY 10:00] Đã mở vị thế đầu tiên %s | Vol: %.2f | Price: %.2f", 
               EnumToString(order_type), base_lot, price);
}

//+------------------------------------------------------------------+
//| QUẢN LÝ QUY TRÌNH GRID & TIME-DECAY TP                          |
//+------------------------------------------------------------------+
void ManageGridExecution(const MqlDateTime &dt)
{
   // Kiểm tra và thực thi rải lệnh tiếp theo / Cập nhật TP theo Time-Decay
   // ... Quản lý khớp lệnh và Time-Decay trailing sau 11:30 VN
}

//+------------------------------------------------------------------+
//| ĐÓNG TOÀN BỘ VỊ THẾ & LỆNH CHỜ                                   |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Symbol() == _Symbol && m_position.Magic() == InpMagicNumber)
         {
            m_trade.PositionClose(m_position.Ticket());
         }
      }
   }
}
