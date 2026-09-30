# Lightweight image running the fibre-arm allocation adjudication API.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application, tests and one-shot verification scripts.
COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level health gate: readiness only flips after the startup
# self-check proves the adjudication endpoint can serve requests.
HEALTHCHECK --interval=3s --timeout=3s --start-period=10s --retries=20 \
    CMD ["python", "scripts/healthcheck.py"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
