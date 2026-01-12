FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=120 --retries 10 -r /app/requirements.txt

COPY . /app

ENTRYPOINT ["python", "-m", "src.predict"]
