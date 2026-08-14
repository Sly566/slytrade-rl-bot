# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /build/
COPY src /build/src

ARG EXTRAS="data,mt5"
RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels ".[${EXTRAS}]"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SLYTRADE_ENV=production \
    SLYTRADE_ALLOW_LIVE=0 \
    SLYTRADE_METRICS_PORT=9108 \
    SLYTRADE_METRICS_BIND=0.0.0.0

WORKDIR /app

# Keep the runtime image free of compilers and package-manager caches.
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.2" \
    && python -m pip install --no-cache-dir \
        "gymnasium>=0.29" \
        "stable-baselines3>=2.3" \
        "optuna>=3.6" \
        "mlflow>=2.14" \
    && rm -rf /wheels

COPY configs /app/configs
COPY src /app/src
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin slytrade \
    && mkdir -p /app/data /app/logs /app/state \
    && chown -R slytrade:slytrade /app

USER slytrade

# Liveness/readiness comes from the in-process metrics server (:9108), which the
# paper loop exposes on /healthz and /readyz.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9108/healthz', timeout=5).status == 200 else 1)"]

# Fail-closed entrypoint: refuses to boot if SLYTRADE_ALLOW_LIVE=1 without
# SLYTRADE_STAGE=demo. Default command runs the supervised paper loop.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["paper"]


FROM runtime AS production

# Production is deliberately fail-closed. Enabling live execution requires an
# explicit environment override plus all application deployment gates.
CMD ["paper"]


FROM runtime AS development

USER root
COPY tests /app/tests
COPY docs /app/docs
COPY .github /app/.github
RUN python -m pip install --no-cache-dir pytest pytest-cov ruff mypy types-PyYAML
USER slytrade
CMD ["doctor"]
