FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GRIDTOEV_MODEL_PATH=/app/models/gridtoev_model_bundle.joblib \
    GRIDTOEV_DATASET_PATH=/app/data/processed/gridtoev_model_ready.csv \
    GRIDTOEV_DAILY_MODEL_PATH=/app/models/v2/daily_curtailment_bundle.joblib \
    GRIDTOEV_DAILY_REPORT_PATH=/app/benchmarks/daily_curtailment_v2/evaluation.json \
    GRIDTOEV_ACTUALS_PATH=/app/data/processed/eirgrid_core_history_30min.csv.gz \
    GRIDTOEV_CONTRACT_PATH=/app/config/benchmark_contract.v2.json \
    GRIDTOEV_RELEASE_REPORT_PATH=/app/benchmarks/release_preflight/release_report.v2.json

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && addgroup --system gridtoev \
    && adduser --system --ingroup gridtoev gridtoev

COPY --chown=gridtoev:gridtoev models/gridtoev_model_bundle.joblib ./models/gridtoev_model_bundle.joblib
COPY --chown=gridtoev:gridtoev models/v2/daily_curtailment_bundle.joblib ./models/v2/daily_curtailment_bundle.joblib
COPY --chown=gridtoev:gridtoev benchmarks/daily_curtailment_v2/evaluation.json ./benchmarks/daily_curtailment_v2/evaluation.json
COPY --chown=gridtoev:gridtoev data/processed/gridtoev_model_ready.csv ./data/processed/gridtoev_model_ready.csv
COPY --chown=gridtoev:gridtoev data/processed/eirgrid_core_history_30min.csv.gz ./data/processed/eirgrid_core_history_30min.csv.gz
COPY --chown=gridtoev:gridtoev config/benchmark_contract.v2.json ./config/benchmark_contract.v2.json
COPY --chown=gridtoev:gridtoev benchmarks/release_preflight/release_report.v2.json ./benchmarks/release_preflight/release_report.v2.json

USER gridtoev

EXPOSE 8000

CMD ["python", "-m", "gridtoev.server"]
