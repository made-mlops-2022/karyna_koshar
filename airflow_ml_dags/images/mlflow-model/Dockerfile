FROM python:3.8-slim-buster

WORKDIR /mlflow/

COPY requirements.txt .
RUN pip install --default-timeout=9000 --no-cache-dir -r requirements.txt

CMD mlflow server \
    --backend-store-uri sqlite:///:memory \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 8000