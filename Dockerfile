FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "pandas>=2.2,<3" \
        "scikit-learn>=1.4,<2" \
        "pydantic>=2.6,<3" \
        "pyyaml>=6.0,<7" \
        "fastapi>=0.110,<1" \
        "uvicorn>=0.29,<1" \
    && python -m pip install --no-cache-dir --no-deps "xgboost-cpu==3.4.1" \
    && python -m pip install --no-cache-dir --no-deps /wheels/*.whl \
    && rm -rf /wheels

COPY src ./src
COPY configs ./configs
COPY models/xgboost_candidate_v1.joblib ./models/xgboost_candidate_v1.joblib
COPY models/feature_preprocessor_v1.joblib ./models/feature_preprocessor_v1.joblib

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
