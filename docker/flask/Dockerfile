FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir flask==3.0.2 mlflow==2.12.2 pandas scikit-learn numpy pycaret requests
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app
EXPOSE 8000