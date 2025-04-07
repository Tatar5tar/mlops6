FROM python:3.12-slim

COPY requirements /app/
COPY /data /app/data
COPY /scripts /app/scripts
COPY /model /app/model
COPY /metrics /app/metrics

WORKDIR /app

RUN pip install -r requirements

EXPOSE 8005

CMD ["python", "scripts/main.py"]