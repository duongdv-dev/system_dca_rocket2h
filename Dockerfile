FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Tối ưu hóa Python bytecode và output logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cài đặt các thư viện phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn dự án vào container
COPY . .

# Chạy script phân tích chính
CMD ["python", "run_analysis.py"]
