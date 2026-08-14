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
    SLYTRADE_ALLOW_LIVE=0

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

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin slytrade \
    && mkdir -p /app/data /app/logs \
    && chown -R slytrade:slytrade /app

USER slytrade

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-m", "slytrade.cli", "doctor"]

ENTRYPOINT ["python", "-m", "slytrade.cli"]
CMD ["doctor"]


FROM runtime AS production

# Production is deliberately fail-closed. Enabling live execution requires an
# explicit environment override plus all application deployment gates.
CMD ["doctor"]


FROM runtime AS development

USER root
COPY tests /app/tests
COPY docs /app/docs
COPY .github /app/.github
RUN python -m pip install --no-cache-dir pytest pytest-cov ruff mypy types-PyYAML
USER slytrade
CMD ["doctor"]
