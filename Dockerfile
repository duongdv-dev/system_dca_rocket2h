FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Tối ưu hóa Python bytecode và realtime stdout logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cài đặt các thư viện hệ thống cần thiết (build-essential, libgomp1 cho LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container
COPY . .

# Chạy v3 system OOS pipeline mới với unbuffered output để hiển thị realtime log
CMD ["python", "-u", "v3_system/run_h2_2020_oos_v3.py"]
