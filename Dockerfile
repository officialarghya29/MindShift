FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cerebro/ ./cerebro/
COPY evaluation/ ./evaluation/
COPY backend/ ./backend/
COPY models/saved/ ./models/saved/
COPY assets/ ./assets/

EXPOSE 8000
# Render (and most PaaS) route traffic to $PORT; default to 8000 locally.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
