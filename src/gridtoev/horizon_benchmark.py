from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarking import (
    BenchmarkContract,
    RollingOriginFold,
    _fold_predictions,
    build_rolling_origin_folds,
    load_benchmark_contract,
    validate_benchmark_dataset,
)
from .constants import DEFAULT_DATASET_PATH, PROJECT_ROOT, SUPPORTED_FORECAST_HORIZONS
from .training import (
    OperatingPolicy,
    TrainingConfig,
    _blend_dispatch_predictions,
    _regression_metrics,
    _regressor,
    chronological_split,
    feature_columns,
    load_dataset,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "horizon_models"
DEFAULT_BASELINE_REPORT = PROJECT_ROOT / "benchmarks" / "v1.1.0" / "benchmark_report.json"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_kind: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class CandidateBenchmarkResult:
    report: dict[str, Any]
    report_path: Path
    registry_path: Path
    predictions_path: Path


def default_candidate_specs() -> list[CandidateSpec]:
    """Standard-library candidates run before any optional boosting package."""
    return [
        CandidateSpec("ridge_residual", "ridge_residual", {"alpha": 10.0}),
        CandidateSpec("ridge_residual_light", "ridge_residual", {"alpha": 1.0}),
        CandidateSpec(
            "hist_gradient_boosting_residual",
            "hist_gradient_boosting_residual",
            {
                "learning_rate": 0.04,
                "max_iter": 220,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 3.0,
            },
        ),
        CandidateSpec(
            "hist_gradient_boosting_conservative",
            "hist_gradient_boosting_residual",
            {
                "learning_rate": 0.03,
                "max_iter": 180,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 35,
                "l2_regularization": 8.0,
            },
        ),
        CandidateSpec(
            "extra_trees_residual",
            "extra_trees_residual",
            {
                "n_estimators": 240,
                "min_samples_leaf": 3,
                "max_features": 0.75,
            },
        ),
        CandidateSpec(
            "extra_trees_residual_conservative",
            "extra_trees_residual",
            {
                "n_estimators": 240,
                "min_samples_leaf": 8,
                "max_features": 0.65,
            },
        ),
        CandidateSpec(
            "two_stage_event_quantity",
            "two_stage_event_quantity",
            {
                "n_estimators": 240,
                "min_samples_leaf": 4,
                "max_features": 0.75,
            },
        ),
    ]


def _pipeline_for_spec(spec: CandidateSpec, random_state: int) -> Pipeline:
    parameters = dict(spec.parameters)
    if spec.model_kind == "ridge_residual":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(**parameters)),
            ]
        )
    if spec.model_kind == "hist_gradient_boosting_residual":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        early_stopping=False,
                        random_state=random_state,
                        **parameters,
                    ),
                ),
            ]
        )
    if spec.model_kind == "extra_trees_residual":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_jobs=-1,
                        random_state=random_state,
                        **parameters,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported residual model kind: {spec.model_kind}")


class HorizonSpecificModel:
    """Fit an independent model for every supported forecast horizon."""

    def __init__(
        self,
        spec: CandidateSpec,
        columns: list[str],
        random_state: int = 42,
    ) -> None:
        self.spec = spec
        self.columns = list(columns)
        self.random_state = random_state
        self.models: dict[int, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "HorizonSpecificModel":
        for horizon in SUPPORTED_FORECAST_HORIZONS:
            subset = frame.loc[frame["forecast_horizon_minutes"].eq(horizon)]
            if subset.empty:
                raise ValueError(f"Training data has no {horizon}-minute rows")
            if self.spec.model_kind == "two_stage_event_quantity":
                self.models[horizon] = self._fit_two_stage(subset, horizon)
            else:
                model = _pipeline_for_spec(self.spec, self.random_state + horizon)
                residual = (
                    subset["dispatch_down_mwh"]
                    - subset["dispatch_down_mwh_latest_observed"]
                )
                model.fit(subset[self.columns], residual)
                self.models[horizon] = model
        return self

    def _fit_two_stage(self, subset: pd.DataFrame, horizon: int) -> dict[str, Any]:
        parameters = dict(self.spec.parameters)
        event = subset["dispatch_down_mwh"].gt(0).astype(int)
        event_model: Pipeline | None = None
        event_constant = float(event.iloc[0]) if event.nunique() == 1 else None
        if event_constant is None:
            event_model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        ExtraTreesClassifier(
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=self.random_state + horizon,
                            **parameters,
                        ),
                    ),
                ]
            )
            event_model.fit(subset[self.columns], event)

        positive = subset.loc[event.eq(1)]
        quantity_model: Pipeline | None = None
        quantity_constant = 0.0
        if len(positive) >= 2:
            quantity_model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_jobs=-1,
                            random_state=self.random_state + horizon + 1000,
                            **parameters,
                        ),
                    ),
                ]
            )
            quantity_model.fit(positive[self.columns], positive["dispatch_down_mwh"])
        elif len(positive) == 1:
            quantity_constant = float(positive["dispatch_down_mwh"].iloc[0])
        return {
            "event_model": event_model,
            "event_constant": event_constant,
            "quantity_model": quantity_model,
            "quantity_constant": quantity_constant,
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(frame), dtype=float)
        for horizon in SUPPORTED_FORECAST_HORIZONS:
            mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
            if not mask.any():
                continue
            if horizon not in self.models:
                raise ValueError(f"Model has not been fitted for horizon {horizon}")
            subset = frame.loc[mask]
            model = self.models[horizon]
            if self.spec.model_kind == "two_stage_event_quantity":
                prediction = self._predict_two_stage(model, subset)
            else:
                residual = model.predict(subset[self.columns])
                prediction = (
                    subset["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
                    + residual
                )
            result[mask] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
        return result

    def _predict_two_stage(self, model: dict[str, Any], subset: pd.DataFrame) -> np.ndarray:
        if model["event_model"] is None:
            probability = np.full(len(subset), model["event_constant"], dtype=float)
        else:
            probability = model["event_model"].predict_proba(subset[self.columns])[:, 1]
        if model["quantity_model"] is None:
            quantity = np.full(len(subset), model["quantity_constant"], dtype=float)
        else:
            quantity = model["quantity_model"].predict(subset[self.columns])
        return probability * np.maximum(quantity, 0.0)


def _metric_slice(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    return _regression_metrics(frame["dispatch_down_mwh"], predictions)


def evaluate_prediction_set(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    fold_names: Iterable[str],
) -> dict[str, Any]:
    predictions = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    fold_names = np.asarray(list(fold_names), dtype=object)
    if len(frame) != len(predictions) or len(frame) != len(fold_names):
        raise ValueError("Frame, predictions and fold names must have equal lengths")

    by_horizon: dict[str, Any] = {}
    for horizon in sorted(frame["forecast_horizon_minutes"].astype(int).unique()):
        mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
        by_horizon[str(horizon)] = _metric_slice(frame.loc[mask], predictions[mask])

    truth = frame["dispatch_down_mwh"].to_numpy(dtype=float)
    latest = frame["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
    positive_truth = truth[truth > 0]
    high_threshold = float(np.quantile(positive_truth, 0.75)) if len(positive_truth) else 0.0
    regimes = {
        "no_dispatch_down": truth <= 0,
        "positive_dispatch_down": truth > 0,
        "high_dispatch_down": truth >= high_threshold,
        "ramping_up": truth > latest,
        "ramping_down_or_flat": truth <= latest,
    }
    by_regime = {
        name: _metric_slice(frame.loc[mask], predictions[mask])
        for name, mask in regimes.items()
        if mask.any()
    }

    folds: list[dict[str, Any]] = []
    for name in dict.fromkeys(fold_names.tolist()):
        mask = fold_names == name
        folds.append(
            {
                "name": str(name),
                "rows": int(mask.sum()),
                "overall": _metric_slice(frame.loc[mask], predictions[mask]),
                "by_horizon": {
                    str(horizon): _metric_slice(
                        frame.loc[mask & frame["forecast_horizon_minutes"].eq(horizon).to_numpy()],
                        predictions[mask & frame["forecast_horizon_minutes"].eq(horizon).to_numpy()],
                    )
                    for horizon in sorted(frame["forecast_horizon_minutes"].astype(int).unique())
                    if (mask & frame["forecast_horizon_minutes"].eq(horizon).to_numpy()).any()
                },
            }
        )
    fold_mae = np.array([fold["overall"]["mae"] for fold in folds], dtype=float)
    return {
        "overall": _metric_slice(frame, predictions),
        "by_horizon": by_horizon,
        "by_event_regime": by_regime,
        "folds": folds,
        "dispersion": {
            "fold_mae_mean_mwh": float(fold_mae.mean()),
            "fold_mae_std_mwh": float(fold_mae.std(ddof=0)),
            "fold_mae_min_mwh": float(fold_mae.min()),
            "fold_mae_max_mwh": float(fold_mae.max()),
        },
    }


def assess_release_candidate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, float],
    *,
    maximum_horizon_regression: float = 0.0,
    maximum_fold_regression: float = 0.02,
) -> dict[str, Any]:
    baseline_mae = float(baseline["overall"]["mae"])
    candidate_mae = float(candidate["overall"]["mae"])
    aggregate_improvement = (baseline_mae - candidate_mae) / baseline_mae
    horizon_improvements = {
        horizon: (
            float(baseline["by_horizon"][horizon]["mae"])
            - float(candidate["by_horizon"][horizon]["mae"])
        )
        / float(baseline["by_horizon"][horizon]["mae"])
        for horizon in baseline["by_horizon"]
    }
    baseline_folds = {fold["name"]: fold for fold in baseline["folds"]}
    fold_improvements = {
        fold["name"]: (
            float(baseline_folds[fold["name"]]["overall"]["mae"])
            - float(fold["overall"]["mae"])
        )
        / float(baseline_folds[fold["name"]]["overall"]["mae"])
        for fold in candidate["folds"]
    }
    required_positive_folds = max(1, int(np.ceil(0.75 * len(fold_improvements))))
    positive_fold_count = sum(value > 0 for value in fold_improvements.values())
    checks = {
        "aggregate_improvement": aggregate_improvement
        >= float(thresholds["rolling_mae_improvement_min"]),
        "horizon_non_regression": min(horizon_improvements.values())
        >= -maximum_horizon_regression,
        "fold_stability": positive_fold_count >= required_positive_folds
        and min(fold_improvements.values()) >= -maximum_fold_regression,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "aggregate_mae_improvement_fraction": float(aggregate_improvement),
        "horizon_mae_improvement_fraction": horizon_improvements,
        "fold_mae_improvement_fraction": fold_improvements,
        "positive_fold_count": int(positive_fold_count),
        "required_positive_fold_count": int(required_positive_folds),
        "maximum_horizon_regression_fraction": maximum_horizon_regression,
        "maximum_fold_regression_fraction": maximum_fold_regression,
    }


def select_horizon_ensemble_weights(
    frame: pd.DataFrame,
    fold_names: Iterable[str],
    predictions_by_name: dict[str, np.ndarray],
    members: tuple[str, str],
    weight_grid: Iterable[float] | None = None,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Select two-member blend weights from development OOF predictions only."""
    first, second = members
    folds = np.asarray(list(fold_names), dtype=object)
    grid = list(weight_grid if weight_grid is not None else np.linspace(0.0, 1.0, 21))
    truth = frame["dispatch_down_mwh"].to_numpy(dtype=float)
    output = np.zeros(len(frame), dtype=float)
    weights: dict[str, dict[str, float]] = {}
    for horizon in sorted(frame["forecast_horizon_minutes"].astype(int).unique()):
        horizon_mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
        best: tuple[float, float] | None = None
        for first_weight in grid:
            blended = (
                float(first_weight) * predictions_by_name[first]
                + (1.0 - float(first_weight)) * predictions_by_name[second]
            )
            fold_scores = [
                mean_absolute_error(
                    truth[horizon_mask & (folds == fold)],
                    blended[horizon_mask & (folds == fold)],
                )
                for fold in np.unique(folds[horizon_mask])
            ]
            score = float(np.mean(fold_scores))
            key = (score, -float(first_weight))
            if best is None or key < (best[0], -best[1]):
                best = (score, float(first_weight))
        assert best is not None
        first_weight = best[1]
        output[horizon_mask] = np.maximum(
            first_weight * predictions_by_name[first][horizon_mask]
            + (1.0 - first_weight) * predictions_by_name[second][horizon_mask],
            0.0,
        )
        weights[str(horizon)] = {
            first: first_weight,
            second: 1.0 - first_weight,
        }
    return weights, output


def feature_family_columns(columns: list[str]) -> dict[str, list[str]]:
    always = [column for column in columns if column == "forecast_horizon_minutes"]

    def select(tokens: tuple[str, ...]) -> list[str]:
        selected = [
            column
            for column in columns
            if column in always or any(token in column.lower() for token in tokens)
        ]
        return list(dict.fromkeys(selected))

    groups = {
        "all_features": list(columns),
        "persistence_history": select(("dispatch_down_",)),
        "renewable_grid_state": select(
            ("wind", "solar", "renewable", "snsp", "oversupply", "headroom")
        ),
        "demand_market_interconnector": select(
            ("load", "demand", "price", "interconnector", "flow")
        ),
        "temporal_dynamics": select(
            ("_lag_", "_ramp_", "rolling_", "hour_", "weekday", "weekend", "month")
        ),
        "all_without_dispatch_history": [
            column for column in columns if "dispatch_down_" not in column.lower()
        ],
    }
    return {name: values for name, values in groups.items() if values}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_policy(path: Path) -> OperatingPolicy:
    report = json.loads(path.read_text(encoding="utf-8"))
    payload = report["selection"]["operating_policy"]
    return OperatingPolicy(
        classification_threshold=float(payload["classification_threshold"]),
        dispatch_trend_alpha_by_horizon={
            str(key): float(value)
            for key, value in payload["dispatch_trend_alpha_by_horizon"].items()
        },
        dispatch_ml_weight_by_horizon={
            str(key): float(value)
            for key, value in payload["dispatch_ml_weight_by_horizon"].items()
        },
        prediction_interval_adjustment_mwh=float(
            payload["prediction_interval_adjustment_mwh"]
        ),
    )


def _oof_frame(folds: list[RollingOriginFold]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.concat([fold.score for fold in folds], ignore_index=True)
    names = np.concatenate(
        [np.full(len(fold.score), fold.name, dtype=object) for fold in folds]
    )
    return frame, names


def _candidate_oof_predictions(
    spec: CandidateSpec,
    folds: list[RollingOriginFold],
    columns: list[str],
    random_state: int,
) -> tuple[np.ndarray, float, float]:
    predictions: list[np.ndarray] = []
    training_seconds = 0.0
    inference_seconds = 0.0
    for fold_index, fold in enumerate(folds):
        model = HorizonSpecificModel(spec, columns, random_state + fold_index * 100)
        started = perf_counter()
        model.fit(fold.fit)
        training_seconds += perf_counter() - started
        started = perf_counter()
        predictions.append(model.predict(fold.score))
        inference_seconds += perf_counter() - started
    return np.concatenate(predictions), training_seconds, inference_seconds


def _baseline_predictions(
    folds: list[RollingOriginFold],
    columns: list[str],
    policy: OperatingPolicy,
    config: TrainingConfig,
) -> np.ndarray:
    return np.concatenate(
        [_fold_predictions(fold, columns, policy, config) for fold in folds]
    )


def _report_with_metadata(
    metrics: dict[str, Any],
    *,
    kind: str,
    feature_group: str,
    feature_count: int,
    training_seconds: float,
    inference_seconds: float,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "feature_group": feature_group,
        "feature_count": int(feature_count),
        "settings": settings or {},
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        **metrics,
    }


def _registry_rows(
    reports: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
    contract: BenchmarkContract,
    dataset_hash: str,
) -> list[dict[str, Any]]:
    rows = []
    for name, report in reports.items():
        gate = gates.get(name)
        rows.append(
            {
                "run_id": f"{contract.contract_id}:{name}:{dataset_hash[:12]}",
                "contract_id": contract.contract_id,
                "candidate": name,
                "kind": report["kind"],
                "feature_group": report["feature_group"],
                "feature_count": report["feature_count"],
                "settings_json": json.dumps(report["settings"], sort_keys=True),
                "rolling_mae_mwh": report["overall"]["mae"],
                "rolling_rmse_mwh": report["overall"]["rmse"],
                "rolling_wape": report["overall"]["wape"],
                "rolling_30m_mae_mwh": report["by_horizon"]["30"]["mae"],
                "rolling_60m_mae_mwh": report["by_horizon"]["60"]["mae"],
                "fold_mae_std_mwh": report["dispersion"]["fold_mae_std_mwh"],
                "training_seconds": report["training_seconds"],
                "inference_seconds": report["inference_seconds"],
                "improvement_vs_v1_1": (
                    gate["aggregate_mae_improvement_fraction"] if gate else None
                ),
                "development_gate_passed": gate["passed"] if gate else None,
            }
        )
    return rows


def _final_test_thresholds(
    metrics: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    checks = {
        "mae": metrics["mae"] <= thresholds["final_test_mae_mwh_max"],
        "rmse": metrics["rmse"] <= thresholds["final_test_rmse_mwh_max"],
        "wape": metrics["wape"] <= thresholds["final_test_wape_max"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_horizon_benchmark(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    contract_path: Path | str = PROJECT_ROOT / "config" / "benchmark_contract.v1.json",
    baseline_report_path: Path | str = DEFAULT_BASELINE_REPORT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    candidate_specs: list[CandidateSpec] | None = None,
    *,
    run_feature_ablations: bool = True,
    random_state: int = 42,
) -> CandidateBenchmarkResult:
    """Compare horizon-specific candidates without replacing the production bundle."""
    started = perf_counter()
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir).resolve()
    contract = load_benchmark_contract(contract_path)
    validate_benchmark_dataset(dataset_path)
    data = load_dataset(dataset_path)
    columns = feature_columns(data)
    train, validation, final_test = chronological_split(
        data,
        train_fraction=contract.train_fraction,
        validation_fraction=contract.validation_fraction,
    )
    folds = build_rolling_origin_folds(train, validation, contract)
    development, fold_names = _oof_frame(folds)
    config = TrainingConfig(
        train_fraction=contract.train_fraction,
        validation_fraction=contract.validation_fraction,
        random_state=random_state,
    )
    policy = _load_frozen_policy(Path(baseline_report_path))

    prediction_table = development[
        [
            "issue_timestamp_utc",
            "target_timestamp_utc",
            "forecast_horizon_minutes",
            "dispatch_down_event",
            "dispatch_down_mwh",
        ]
    ].copy()
    prediction_table["fold"] = fold_names
    predictions: dict[str, np.ndarray] = {
        "v1.1.0": _baseline_predictions(folds, columns, policy, config),
        "latest_observation": development[
            "dispatch_down_mwh_latest_observed"
        ].to_numpy(dtype=float),
        "stale_persistence": development["dispatch_down_mwh_lag_1"].to_numpy(
            dtype=float
        ),
    }
    reports: dict[str, dict[str, Any]] = {}
    for name in ("v1.1.0", "latest_observation", "stale_persistence"):
        reports[name] = _report_with_metadata(
            evaluate_prediction_set(development, predictions[name], fold_names),
            kind="baseline",
            feature_group="current_model_feature_contract",
            feature_count=len(columns),
            training_seconds=0.0,
            inference_seconds=0.0,
            settings=(asdict(policy) if name == "v1.1.0" else {}),
        )
        prediction_table[name] = predictions[name]

    specs = candidate_specs or default_candidate_specs()
    spec_by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        candidate_prediction, fit_seconds, predict_seconds = _candidate_oof_predictions(
            spec, folds, columns, random_state
        )
        predictions[spec.name] = candidate_prediction
        reports[spec.name] = _report_with_metadata(
            evaluate_prediction_set(development, candidate_prediction, fold_names),
            kind="horizon_specific_candidate",
            feature_group="all_features",
            feature_count=len(columns),
            training_seconds=fit_seconds,
            inference_seconds=predict_seconds,
            settings={"model_kind": spec.model_kind, **spec.parameters},
        )
        prediction_table[spec.name] = candidate_prediction

    ranked_base = sorted(spec_by_name, key=lambda name: reports[name]["overall"]["mae"])
    ensemble_names: list[str] = []
    # Include the frozen model in the blend pool. A learned correction must prove that a
    # non-zero weight is useful; the grid is allowed to fall back to 100% v1.1.0.
    ensemble_pool = ["v1.1.0", *ranked_base[:4]]
    for first, second in combinations(ensemble_pool, 2):
        weights, ensemble_prediction = select_horizon_ensemble_weights(
            development,
            fold_names,
            predictions,
            (first, second),
        )
        name = f"ensemble__{first}__{second}"
        ensemble_names.append(name)
        predictions[name] = ensemble_prediction
        reports[name] = _report_with_metadata(
            evaluate_prediction_set(development, ensemble_prediction, fold_names),
            kind="oof_horizon_weighted_ensemble",
            feature_group="all_features",
            feature_count=len(columns),
            training_seconds=(
                reports[first]["training_seconds"] + reports[second]["training_seconds"]
            ),
            inference_seconds=(
                reports[first]["inference_seconds"] + reports[second]["inference_seconds"]
            ),
            settings={"members": [first, second], "weights_by_horizon": weights},
        )
        prediction_table[name] = ensemble_prediction

    baseline = reports["v1.1.0"]
    selectable_names = list(spec_by_name) + ensemble_names
    gates = {
        name: assess_release_candidate(
            reports[name], baseline, contract.release_thresholds
        )
        for name in selectable_names
    }
    passing = [name for name in selectable_names if gates[name]["passed"]]
    development_winner = min(
        passing or selectable_names,
        key=lambda name: reports[name]["overall"]["mae"],
    )

    best_individual = min(spec_by_name, key=lambda name: reports[name]["overall"]["mae"])
    ablations: dict[str, dict[str, Any]] = {}
    if run_feature_ablations:
        for group_name, group_columns in feature_family_columns(columns).items():
            if group_name == "all_features":
                group_report = reports[best_individual]
            else:
                group_prediction, fit_seconds, predict_seconds = _candidate_oof_predictions(
                    spec_by_name[best_individual], folds, group_columns, random_state
                )
                group_report = _report_with_metadata(
                    evaluate_prediction_set(development, group_prediction, fold_names),
                    kind="feature_family_ablation",
                    feature_group=group_name,
                    feature_count=len(group_columns),
                    training_seconds=fit_seconds,
                    inference_seconds=predict_seconds,
                    settings={
                        "base_candidate": best_individual,
                        "columns": group_columns,
                    },
                )
            ablations[group_name] = group_report

    final_test_report: dict[str, Any] = {
        "accessed": False,
        "reason": "Development gate did not pass; sealed final-test labels were not scored.",
        "metrics": None,
        "threshold_checks": None,
    }
    production_decision = "retain_v1.1.0"
    development_gate = gates[development_winner]
    if development_gate["passed"]:
        final_training = pd.concat([train, validation], ignore_index=True)
        if development_winner in spec_by_name:
            final_model = HorizonSpecificModel(
                spec_by_name[development_winner], columns, random_state
            ).fit(final_training)
            final_prediction = final_model.predict(final_test)
        else:
            settings = reports[development_winner]["settings"]
            first, second = settings["members"]

            def final_member_prediction(name: str, seed: int) -> np.ndarray:
                if name != "v1.1.0":
                    return HorizonSpecificModel(
                        spec_by_name[name], columns, seed
                    ).fit(final_training).predict(final_test)
                baseline_model = _regressor(config, loss="squared_error")
                baseline_model.fit(
                    final_training[columns],
                    final_training["dispatch_down_mwh"]
                    - final_training["dispatch_down_mwh_latest_observed"],
                )
                latest = final_test[
                    "dispatch_down_mwh_latest_observed"
                ].to_numpy(dtype=float)
                ml_prediction = np.maximum(
                    latest + baseline_model.predict(final_test[columns]), 0.0
                )
                blended, _ = _blend_dispatch_predictions(
                    final_test,
                    ml_prediction,
                    policy.dispatch_trend_alpha_by_horizon,
                    policy.dispatch_ml_weight_by_horizon,
                )
                return blended

            first_prediction = final_member_prediction(first, random_state)
            second_prediction = final_member_prediction(second, random_state + 1000)
            final_prediction = np.zeros(len(final_test), dtype=float)
            for horizon in SUPPORTED_FORECAST_HORIZONS:
                mask = final_test["forecast_horizon_minutes"].eq(horizon).to_numpy()
                weights = settings["weights_by_horizon"][str(horizon)]
                final_prediction[mask] = (
                    weights[first] * first_prediction[mask]
                    + weights[second] * second_prediction[mask]
                )
        final_metrics = evaluate_prediction_set(
            final_test,
            final_prediction,
            np.full(len(final_test), "final_test", dtype=object),
        )
        threshold_checks = _final_test_thresholds(
            final_metrics["overall"], contract.release_thresholds
        )
        final_test_report = {
            "accessed": True,
            "reason": "Development gate passed, so the frozen candidate was scored once.",
            "metrics": final_metrics,
            "threshold_checks": threshold_checks,
        }
        if threshold_checks["passed"]:
            production_decision = "candidate_eligible_for_release_integration"

    dataset_hash = _sha256(dataset_path)
    report = {
        "schema_version": 1,
        "issue": 8,
        "contract": {
            "contract_id": contract.contract_id,
            "baseline_model_version": contract.baseline_model_version,
            "release_thresholds": contract.release_thresholds,
            "folds": [fold.name for fold in folds],
        },
        "dataset": {
            "path": (
                dataset_path.relative_to(PROJECT_ROOT).as_posix()
                if dataset_path.is_relative_to(PROJECT_ROOT)
                else dataset_path.name
            ),
            "sha256": dataset_hash,
            "rows": int(len(data)),
            "feature_count": len(columns),
        },
        "upstream_feature_decisions": {
            "forecast_error_grid_headroom": "available code, no temporally overlapping labelled feature artifact",
            "semo_market_operational_signals": "excluded: zero overlap with January benchmark",
            "outage_constraint_signals": "excluded: development ablation failed on its feature branch",
            "regional_weather_signals": "excluded: development ablation failed on its feature branch",
        },
        "baselines": {name: reports[name] for name in ("v1.1.0", "latest_observation", "stale_persistence")},
        "candidates": {
            name: reports[name]
            for name in list(spec_by_name) + ensemble_names
        },
        "candidate_release_gates": gates,
        "feature_family_ablations": {
            "base_candidate": best_individual,
            "results": ablations,
        },
        "selection": {
            "development_winner": development_winner,
            "development_gate": development_gate,
            "production_decision": production_decision,
            "active_model": (
                development_winner
                if production_decision == "candidate_eligible_for_release_integration"
                else "v1.1.0"
            ),
            "optional_external_boosters_run": False,
            "optional_external_boosters_reason": (
                "Standard candidates were sufficient for the mandated comparison; optional dependencies "
                "were not added without demonstrated extra value."
            ),
        },
        "final_test": final_test_report,
        "runtime_seconds": float(perf_counter() - started),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.json"
    registry_path = output_dir / "experiment_registry.csv"
    predictions_path = output_dir / "development_oof_predictions.csv.gz"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    all_registry_reports = {
        **{name: reports[name] for name in reports},
        **{
            f"ablation__{name}": value
            for name, value in ablations.items()
            if name != "all_features"
        },
    }
    pd.DataFrame(
        _registry_rows(all_registry_reports, gates, contract, dataset_hash)
    ).to_csv(registry_path, index=False)
    prediction_table.to_csv(
        predictions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return CandidateBenchmarkResult(report, report_path, registry_path, predictions_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark horizon-specific dispatch-down models and stable OOF ensembles"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "config" / "benchmark_contract.v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-feature-ablations",
        action="store_true",
        help="Skip feature-family ablations for a quicker diagnostic run",
    )
    args = parser.parse_args()
    result = run_horizon_benchmark(
        dataset_path=args.dataset,
        contract_path=args.contract,
        output_dir=args.output_dir,
        run_feature_ablations=not args.skip_feature_ablations,
    )
    selection = result.report["selection"]
    print(f"Development winner: {selection['development_winner']}")
    print(f"Production decision: {selection['production_decision']}")
    print(f"Active model: {selection['active_model']}")
    print(f"Report: {result.report_path}")
    print(f"Registry: {result.registry_path}")
    print(f"OOF predictions: {result.predictions_path}")


if __name__ == "__main__":
    main()
