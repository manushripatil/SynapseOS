FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=5050

EXPOSE 5050

# Run visual simulator (production)
CMD gunicorn -b 0.0.0.0:${PORT} --workers 1 --threads 4 visual.wsgi:app
