from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    IDENTIFIER_COLUMNS,
    MODEL_VERSION,
    PROJECT_ROOT,
    SUPPORTED_FORECAST_HORIZONS,
    TARGET_COLUMNS,
)


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    learning_rate: float = 0.05
    max_iter: int = 180
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 42


@dataclass
class TrainingResult:
    bundle: dict[str, Any]
    metrics: dict[str, Any]
    artifact_path: Path
    metrics_path: Path
    metadata_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path | str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = IDENTIFIER_COLUMNS | TARGET_COLUMNS | {
        "forecast_horizon_minutes",
        "dispatch_down_mwh_latest_observed",
        "dispatch_down_event_latest_observed",
        "dispatch_down_mwh_lag_1",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data["issue_timestamp_utc"] = pd.to_datetime(
        data["issue_timestamp_utc"], errors="raise", utc=True
    )
    data["target_timestamp_utc"] = pd.to_datetime(
        data["target_timestamp_utc"], errors="raise", utc=True
    )
    data = data.sort_values(
        ["issue_timestamp_utc", "forecast_horizon_minutes"]
    ).reset_index(drop=True)

    invalid_horizons = sorted(
        set(data["forecast_horizon_minutes"]) - set(SUPPORTED_FORECAST_HORIZONS)
    )
    if invalid_horizons:
        raise ValueError(f"Unsupported forecast horizons: {invalid_horizons}")
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = IDENTIFIER_COLUMNS | TARGET_COLUMNS
    columns = [column for column in data.columns if column not in excluded]
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"Model features must be numeric: {non_numeric}")
    return columns


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave room for a test period")

    frame = data.copy()
    frame["issue_timestamp_utc"] = pd.to_datetime(
        frame["issue_timestamp_utc"], errors="raise", utc=True
    )
    unique_times = np.array(sorted(frame["issue_timestamp_utc"].unique()))
    if len(unique_times) < 10:
        raise ValueError("At least ten unique issue timestamps are required")

    train_end = max(1, int(len(unique_times) * train_fraction))
    validation_end = max(train_end + 1, int(len(unique_times) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(unique_times) - 1)

    train_times = set(unique_times[:train_end])
    validation_times = set(unique_times[train_end:validation_end])
    test_times = set(unique_times[validation_end:])

    train = frame.loc[frame["issue_timestamp_utc"].isin(train_times)].copy()
    validation = frame.loc[frame["issue_timestamp_utc"].isin(validation_times)].copy()
    test = frame.loc[frame["issue_timestamp_utc"].isin(test_times)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train, validation, test


def _classifier(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=config.learning_rate,
                    max_iter=config.max_iter,
                    max_leaf_nodes=config.max_leaf_nodes,
                    min_samples_leaf=config.min_samples_leaf,
                    l2_regularization=config.l2_regularization,
                    early_stopping=False,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _regressor(config: TrainingConfig, loss: str = "poisson", quantile: float | None = None) -> Pipeline:
    model = HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    )


def _build_models(config: TrainingConfig) -> dict[str, Pipeline]:
    return {
        "event_classifier": _classifier(config),
        # Dispatch-down is highly persistent. Predicting the change from the latest
        # observed interval is more stable than asking a tree to relearn persistence.
        "dispatch_down_regressor": _regressor(config, loss="squared_error"),
        "curtailment_regressor": _regressor(config),
        "constraint_regressor": _regressor(config),
        "dispatch_down_quantile_p10": _regressor(config, loss="quantile", quantile=0.10),
        "dispatch_down_quantile_p50": _regressor(config, loss="quantile", quantile=0.50),
        "dispatch_down_quantile_p90": _regressor(config, loss="quantile", quantile=0.90),
    }


def _fit_models(
    models: dict[str, Pipeline],
    data: pd.DataFrame,
    columns: list[str],
) -> dict[str, Pipeline]:
    X = data[columns]
    event_target = data["dispatch_down_event"].astype(int)
    if event_target.nunique() < 2:
        raise ValueError("Event classifier requires both positive and negative examples")
    sample_weights = compute_sample_weight(class_weight="balanced", y=event_target)
    models["event_classifier"].fit(X, event_target, model__sample_weight=sample_weights)

    target_mapping = {
        "dispatch_down_regressor": "dispatch_down_mwh",
        "curtailment_regressor": "curtailment_mwh",
        "constraint_regressor": "constraint_mwh",
        "dispatch_down_quantile_p10": "dispatch_down_mwh",
        "dispatch_down_quantile_p50": "dispatch_down_mwh",
        "dispatch_down_quantile_p90": "dispatch_down_mwh",
    }
    for model_name, target in target_mapping.items():
        training_target = data[target].clip(lower=0)
        if model_name == "dispatch_down_regressor" or model_name.startswith(
            "dispatch_down_quantile_"
        ):
            training_target = (
                data[target] - data["dispatch_down_mwh_latest_observed"]
            )
        models[model_name].fit(X, training_target)
    return models


def _select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.10, 0.90, 81):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = probabilities >= threshold
    result = {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "threshold": float(threshold),
        "event_rate": float(np.mean(y_true)),
    }
    if pd.Series(y_true).nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return result


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    predictions = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    absolute_error = np.abs(np.asarray(y_true, dtype=float) - predictions)
    denominator = float(np.abs(y_true).sum())
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "wape": float(absolute_error.sum() / denominator) if denominator > 1e-6 else None,
        "actual_total_mwh": float(np.asarray(y_true, dtype=float).sum()),
        "nonzero_rate": float((np.asarray(y_true, dtype=float) > 0).mean()),
    }


def _dispatch_ml_prediction(models: dict[str, Pipeline], data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    predicted_change = models["dispatch_down_regressor"].predict(data[columns])
    return np.maximum(
        data["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float) + predicted_change,
        0.0,
    )


def _dispatch_quantile_prediction(
    models: dict[str, Pipeline],
    data: pd.DataFrame,
    columns: list[str],
    quantile: str,
) -> np.ndarray:
    predicted_change = models[f"dispatch_down_quantile_{quantile}"].predict(data[columns])
    return np.maximum(
        data["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
        + predicted_change,
        0.0,
    )


def _dispatch_trend_prediction(
    data: pd.DataFrame,
    alpha_by_horizon: dict[str, float],
) -> np.ndarray:
    """Project the most recent observation with a damped, horizon-specific trend."""
    horizons = data["forecast_horizon_minutes"].astype(int)
    supported = {str(value) for value in horizons.unique()}
    missing = sorted(supported - set(alpha_by_horizon))
    if missing:
        raise ValueError(f"Missing dispatch trend settings for horizons: {missing}")

    latest = data["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
    previous = data["dispatch_down_mwh_lag_1"].to_numpy(dtype=float)
    alpha = horizons.astype(str).map(alpha_by_horizon).to_numpy(dtype=float)
    return np.maximum(latest + alpha * (latest - previous), 0.0)


def _select_dispatch_strategy(
    y_true: pd.Series,
    ml_prediction: np.ndarray,
    validation: pd.DataFrame,
    fold_ids: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Choose stable trend damping and ML weight independently by horizon.

    The ML correction is accepted only when it improves on the trend baseline
    in every supplied rolling-origin fold. This deliberately favours a robust
    short-history forecast over a validation-period win that may not persist.
    """
    alpha_by_horizon: dict[str, float] = {}
    ml_weight_by_horizon: dict[str, float] = {}
    truth = np.asarray(y_true, dtype=float)
    fold_ids = np.zeros(len(validation), dtype=int) if fold_ids is None else fold_ids

    for horizon in SUPPORTED_FORECAST_HORIZONS:
        mask = validation["forecast_horizon_minutes"].eq(horizon).to_numpy()
        if not mask.any():
            raise ValueError(f"Validation data has no {horizon}-minute rows")
        horizon_frame = validation.loc[mask]
        best_alpha = 0.0
        best_trend_mae = float("inf")
        for alpha in np.linspace(-0.50, 1.50, 81):
            trend = _dispatch_trend_prediction(
                horizon_frame,
                {str(horizon): float(alpha)},
            )
            score = float(mean_absolute_error(truth[mask], trend))
            if (score, float(alpha)) < (best_trend_mae, best_alpha):
                best_trend_mae = score
                best_alpha = float(alpha)

        trend = _dispatch_trend_prediction(
            horizon_frame,
            {str(horizon): best_alpha},
        )
        horizon_fold_ids = np.asarray(fold_ids)[mask]
        best_weight = 0.0
        best_score = best_trend_mae
        for ml_weight in np.linspace(0.25, 1.0, 4):
            blended = np.maximum(
                ml_weight * ml_prediction[mask] + (1.0 - ml_weight) * trend,
                0.0,
            )
            stable = all(
                mean_absolute_error(
                    truth[mask][horizon_fold_ids == fold],
                    blended[horizon_fold_ids == fold],
                )
                <= mean_absolute_error(
                    truth[mask][horizon_fold_ids == fold],
                    trend[horizon_fold_ids == fold],
                )
                for fold in np.unique(horizon_fold_ids)
            )
            score = float(mean_absolute_error(truth[mask], blended))
            if stable and score < best_score:
                best_score = score
                best_weight = float(ml_weight)

        alpha_by_horizon[str(horizon)] = best_alpha
        ml_weight_by_horizon[str(horizon)] = best_weight
    return alpha_by_horizon, ml_weight_by_horizon


def _rolling_dispatch_backtest(
    evaluation_models: dict[str, Pipeline],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    config: TrainingConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create expanding-window dispatch predictions without crossing time."""
    unique_times = np.array(sorted(train["issue_timestamp_utc"].unique()))
    frames: list[pd.DataFrame] = []
    predictions: list[np.ndarray] = []
    fold_ids: list[np.ndarray] = []

    for fold, (train_end_fraction, score_end_fraction) in enumerate(
        ((0.40, 0.60), (0.60, 0.80), (0.80, 1.00))
    ):
        train_end = max(1, int(len(unique_times) * train_end_fraction))
        score_end = max(train_end + 1, int(len(unique_times) * score_end_fraction))
        fit_times = set(unique_times[:train_end])
        score_times = set(unique_times[train_end:score_end])
        fit = train.loc[train["issue_timestamp_utc"].isin(fit_times)].copy()
        score = train.loc[train["issue_timestamp_utc"].isin(score_times)].copy()
        if fit.empty or score.empty:
            continue

        model = _regressor(config, loss="squared_error")
        residual = fit["dispatch_down_mwh"] - fit["dispatch_down_mwh_latest_observed"]
        model.fit(fit[columns], residual)
        latest = score["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
        prediction = np.maximum(latest + model.predict(score[columns]), 0.0)
        frames.append(score)
        predictions.append(prediction)
        fold_ids.append(np.full(len(score), fold, dtype=int))

    frames.append(validation)
    predictions.append(_dispatch_ml_prediction(evaluation_models, validation, columns))
    fold_ids.append(np.full(len(validation), len(fold_ids), dtype=int))
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(predictions),
        np.concatenate(fold_ids),
    )


def _blend_dispatch_predictions(
    data: pd.DataFrame,
    ml_prediction: np.ndarray,
    alpha_by_horizon: dict[str, float],
    ml_weight_by_horizon: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    trend = _dispatch_trend_prediction(data, alpha_by_horizon)
    weights = (
        data["forecast_horizon_minutes"]
        .astype(int)
        .astype(str)
        .map(ml_weight_by_horizon)
        .to_numpy(dtype=float)
    )
    blended = np.maximum(weights * ml_prediction + (1.0 - weights) * trend, 0.0)
    return blended, trend


def _evaluate_models(
    models: dict[str, Pipeline],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    config: TrainingConfig,
) -> tuple[dict[str, Any], float, dict[str, float], dict[str, float], float]:
    validation_probabilities = models["event_classifier"].predict_proba(validation[columns])[:, 1]
    threshold = _select_threshold(validation["dispatch_down_event"], validation_probabilities)
    test_probabilities = models["event_classifier"].predict_proba(test[columns])[:, 1]

    classification = {
        "validation": _classification_metrics(
            validation["dispatch_down_event"], validation_probabilities, threshold
        ),
        "test": _classification_metrics(test["dispatch_down_event"], test_probabilities, threshold),
        "latest_observation_baseline_test": _classification_metrics(
            test["dispatch_down_event"],
            test["dispatch_down_event_latest_observed"].to_numpy(dtype=float),
            0.5,
        ),
        "stale_persistence_baseline_test": _classification_metrics(
            test["dispatch_down_event"],
            test["dispatch_down_event_lag_1"].to_numpy(dtype=float),
            0.5,
        ),
    }

    strategy_frame, strategy_ml, strategy_fold_ids = _rolling_dispatch_backtest(
        models,
        train,
        validation,
        columns,
        config,
    )
    dispatch_trend_alpha, dispatch_ml_weight = _select_dispatch_strategy(
        strategy_frame["dispatch_down_mwh"],
        strategy_ml,
        strategy_frame,
        strategy_fold_ids,
    )
    test_ml = _dispatch_ml_prediction(models, test, columns)
    test_hybrid, test_trend = _blend_dispatch_predictions(
        test,
        test_ml,
        dispatch_trend_alpha,
        dispatch_ml_weight,
    )
    test_latest = test["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
    test_stale_persistence = test["dispatch_down_mwh_lag_1"].to_numpy(dtype=float)

    regression: dict[str, Any] = {
        "dispatch_down_mwh": _regression_metrics(test["dispatch_down_mwh"], test_hybrid),
        "dispatch_down_ml_only": _regression_metrics(test["dispatch_down_mwh"], test_ml),
        "dispatch_down_trend_baseline": _regression_metrics(
            test["dispatch_down_mwh"], test_trend
        ),
        "dispatch_down_latest_observation_baseline": _regression_metrics(
            test["dispatch_down_mwh"], test_latest
        ),
        "dispatch_down_stale_persistence_baseline": _regression_metrics(
            test["dispatch_down_mwh"], test_stale_persistence
        ),
        "dispatch_down_trend_alpha_by_horizon": dispatch_trend_alpha,
        "dispatch_down_ml_weight_by_horizon": dispatch_ml_weight,
    }
    for target, model_name in (
        ("curtailment_mwh", "curtailment_regressor"),
        ("constraint_mwh", "constraint_regressor"),
    ):
        regression[target] = _regression_metrics(
            test[target], models[model_name].predict(test[columns])
        )

    quantile_predictions = {
        quantile: _dispatch_quantile_prediction(models, test, columns, quantile)
        for quantile in ("p10", "p50", "p90")
    }
    validation_p10 = _dispatch_quantile_prediction(models, validation, columns, "p10")
    validation_p90 = _dispatch_quantile_prediction(models, validation, columns, "p90")
    validation_lower = np.minimum(validation_p10, validation_p90)
    validation_upper = np.maximum(validation_p10, validation_p90)
    validation_truth = validation["dispatch_down_mwh"].to_numpy(dtype=float)
    nonconformity = np.maximum.reduce(
        [validation_lower - validation_truth, validation_truth - validation_upper, np.zeros(len(validation))]
    )
    interval_adjustment = float(np.quantile(nonconformity, 0.80, method="higher"))

    lower = np.maximum(np.minimum.reduce(list(quantile_predictions.values())) - interval_adjustment, 0.0)
    upper = np.maximum.reduce(list(quantile_predictions.values())) + interval_adjustment
    quantiles = {
        "p10_pinball_loss": float(
            mean_pinball_loss(test["dispatch_down_mwh"], quantile_predictions["p10"], alpha=0.10)
        ),
        "p50_pinball_loss": float(
            mean_pinball_loss(test["dispatch_down_mwh"], quantile_predictions["p50"], alpha=0.50)
        ),
        "p90_pinball_loss": float(
            mean_pinball_loss(test["dispatch_down_mwh"], quantile_predictions["p90"], alpha=0.90)
        ),
        "p10_p90_empirical_coverage": float(
            ((test["dispatch_down_mwh"].to_numpy() >= lower) & (test["dispatch_down_mwh"].to_numpy() <= upper)).mean()
        ),
        "conformal_interval_adjustment_mwh": interval_adjustment,
    }
    return {
        "classification": classification,
        "regression": regression,
        "uncertainty": quantiles,
    }, threshold, dispatch_trend_alpha, dispatch_ml_weight, interval_adjustment


def _partition_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "issue_timestamp_min_utc": frame["issue_timestamp_utc"].min().isoformat(),
        "issue_timestamp_max_utc": frame["issue_timestamp_utc"].max().isoformat(),
    }


def train_and_save(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    artifact_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    config = config or TrainingConfig()
    dataset_path = Path(dataset_path).resolve()
    artifact_path = Path(artifact_path).resolve()
    metrics_path = Path(metrics_path).resolve() if metrics_path else artifact_path.with_suffix(".metrics.json")
    metadata_path = (
        Path(metadata_path).resolve() if metadata_path else artifact_path.with_suffix(".metadata.json")
    )

    data = load_dataset(dataset_path)
    columns = feature_columns(data)
    train, validation, test = chronological_split(
        data,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    evaluation_models = _fit_models(_build_models(config), train, columns)
    (
        metrics,
        threshold,
        dispatch_trend_alpha,
        dispatch_ml_weight,
        interval_adjustment,
    ) = _evaluate_models(
        evaluation_models,
        train,
        validation,
        test,
        columns,
        config,
    )

    # The saved production models see train + validation, while the test period remains untouched.
    final_training_data = pd.concat([train, validation], ignore_index=True)
    final_models = _fit_models(_build_models(config), final_training_data, columns)

    component_total = max(
        float(final_training_data["curtailment_mwh"].mean() + final_training_data["constraint_mwh"].mean()),
        1e-9,
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "trained_at_utc": trained_at,
        "dataset_path": (
            dataset_path.relative_to(PROJECT_ROOT).as_posix()
            if dataset_path.is_relative_to(PROJECT_ROOT)
            else dataset_path.name
        ),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_rows": int(len(data)),
        "feature_count": len(columns),
        "feature_columns": columns,
        "forecast_horizons_minutes": list(SUPPORTED_FORECAST_HORIZONS),
        "classification_threshold": threshold,
        "dispatch_trend_alpha_by_horizon": dispatch_trend_alpha,
        "dispatch_regression_ml_weight_by_horizon": dispatch_ml_weight,
        "prediction_interval_adjustment_mwh": interval_adjustment,
        "default_component_shares": {
            "curtailment": float(final_training_data["curtailment_mwh"].mean() / component_total),
            "constraint": float(final_training_data["constraint_mwh"].mean() / component_total),
        },
        "training_config": asdict(config),
        "partitions": {
            "train": _partition_metadata(train),
            "validation": _partition_metadata(validation),
            "test": _partition_metadata(test),
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    bundle = {
        "metadata": metadata,
        "metrics": metrics,
        "feature_columns": columns,
        "models": final_models,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path, compress=3)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainingResult(bundle, metrics, artifact_path, metrics_path, metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist all GridToEV models")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--max-iter", type=int, default=TrainingConfig.max_iter)
    args = parser.parse_args()

    result = train_and_save(
        dataset_path=args.dataset,
        artifact_path=args.artifact,
        metrics_path=args.metrics,
        metadata_path=args.metadata,
        config=TrainingConfig(max_iter=args.max_iter),
    )
    print(f"Saved model bundle: {result.artifact_path}")
    print(f"Saved metrics: {result.metrics_path}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
