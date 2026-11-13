FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --timeout 100 -r requirements.txt
COPY . .

CMD ["python", "-u", "bot.py"]