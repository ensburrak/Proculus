# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app


FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools && \
    pip wheel --wheel-dir /tmp/wheels -r requirements.txt


FROM base AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /tmp/wheels /tmp/wheels
COPY pyproject.toml requirements.txt .
RUN pip install --no-index --find-links=/tmp/wheels -r requirements.txt && \
    rm -rf /tmp/wheels

# NOTE: .dockerignore excludes .env and secrets - provide them via --env-file or Docker secrets.
COPY . .

RUN mkdir -p data logs metrics models runtime && \
    chown -R appuser:appuser /app /opt/venv

ENV OKX_USE_TESTNET=True \
    TELEGRAM_NOTIFICATIONS_ENABLED=1 \
    DOCKER_HEALTHCHECK_MODE=runtime \
    HEALTHCHECK_URL=http://127.0.0.1:8000/api/health

USER appuser

HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD ["python", "tools/docker_healthcheck.py"]

ENTRYPOINT ["tini", "--"]


FROM runtime-base AS bot-runtime
CMD ["python", "-m", "autotraderbot.cli", "runtime"]


FROM runtime-base AS api-runtime
ENV DOCKER_HEALTHCHECK_MODE=http
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
