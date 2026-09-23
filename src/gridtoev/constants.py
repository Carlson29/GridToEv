from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "gridtoev_model_ready.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gridtoev_model_bundle.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

MODEL_VERSION = "1.1.0"
SUPPORTED_FORECAST_HORIZONS = (30, 60)
INTERVAL_HOURS = 0.5
ROLLING_ORIGIN_WINDOWS = ((0.40, 0.60), (0.60, 0.80), (0.80, 1.00))

IDENTIFIER_COLUMNS = {
    "issue_timestamp_utc",
    "target_timestamp_utc",
}

TARGET_COLUMNS = {
    "dispatch_down_event",
    "dispatch_down_mwh",
    "curtailment_mwh",
    "constraint_mwh",
    "high_frequency_min_generation_mwh",
    "rocof_inertia_mwh",
    "snsp_curtailment_mwh",
    "transmission_constraint_mwh",
    "tso_test_mwh",
    "other_reduction_mwh",
    "recoverable_surplus_upper_bound_mwh",
    "recoverable_surplus_100mw_flex_mwh",
}

REGRESSION_TARGETS = (
    "dispatch_down_mwh",
    "curtailment_mwh",
    "constraint_mwh",
)
