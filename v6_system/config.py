"""
v6_system/config.py
-------------------
Cấu hình hệ thống Version 6 Phase 1 Data Engineering.
"""

import os

# Thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư mục output cho v6
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Múi giờ hệ thống
TARGET_TIMEZONE = "Asia/Ho_Chi_Minh"  # UTC+7

# Danh sách các năm dữ liệu
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Cấu hình phiên giao dịch
SESSION_START_TIME = "10:00:00"  # 10:00 VN
SESSION_END_TIME = "12:00:00"    # 12:00 VN

# Ngưỡng gap bất thường ($/oz trên XAUUSD)
ABNORMAL_GAP_THRESHOLD = 5.0
