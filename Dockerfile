FROM python:3.10-slim

WORKDIR /app

# 1. Cài PyTorch bản CPU trước (chỉ khoảng ~150MB - 200MB thay vì 2.5GB)
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pyt

# 2. Cài các thư viện còn lại (Cùng với --no-cache-dir để xóa file rác)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
