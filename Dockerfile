FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GRIDTOEV_MODEL_PATH=/app/models/gridtoev_model_bundle.joblib \
    GRIDTOEV_DATASET_PATH=/app/data/processed/gridtoev_model_ready.csv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && addgroup --system gridtoev \
    && adduser --system --ingroup gridtoev gridtoev

COPY --chown=gridtoev:gridtoev models/gridtoev_model_bundle.joblib ./models/gridtoev_model_bundle.joblib
COPY --chown=gridtoev:gridtoev data/processed/gridtoev_model_ready.csv ./data/processed/gridtoev_model_ready.csv

USER gridtoev

EXPOSE 8000

CMD ["python", "-m", "gridtoev.server"]
