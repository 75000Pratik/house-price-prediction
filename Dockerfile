FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ARG BUILD_SHA=local

ENV BUILD_SHA=$BUILD_SHA

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app.app:app"]