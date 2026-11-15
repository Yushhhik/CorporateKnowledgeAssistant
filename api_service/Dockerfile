FROM python:3.11-slim
WORKDIR /app

# tesseract нужен для OCR картинок при индексации документов (русский язык обязателен)
RUN apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
    tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com --default-timeout=1000 -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]