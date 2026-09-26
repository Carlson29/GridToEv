"""Evaluate an opt-in v2 point model without replacing the frozen v1 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .benchmarking import (
    _fold_report,
    _fit_with_mature_labels,
    build_rolling_origin_folds,
    load_benchmark_contract,
    validate_benchmark_dataset,
)
from .constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, PROJECT_ROOT
from .optional_v2 import (
    PHYSICAL_FEATURES,
    HorizonResidualModel,
    blend_predictions,
    candidate_features,
    select_blend_weights,
)
from .training import (
    TrainingConfig,
    _dispatch_trend_prediction,
    chronological_split,
    feature_columns,
    load_dataset,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "optional_v2_experiment.v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "optional_v2"
DEFAULT_ARTIFACT = PROJECT_ROOT / "models" / "v2" / "optional_v2_bundle.joblib"


@dataclass(frozen=True)
class FamilyScore:
    name: str
    physical: bool
    weights: dict[str, float]
    baseline_mae: float
    candidate_mae: float
    by_horizon: dict[str, dict[str, float]]
    by_fold: dict[str, dict[str, float]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mae(frame: pd.DataFrame, predictions: np.ndarray) -> float:
    return float(mean_absolute_error(frame["dispatch_down_mwh"], predictions))


def _family_score(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    name: str,
    physical: bool,
    weight_grid: list[float],
) -> FamilyScore:
    weights = select_blend_weights(frame, baseline, candidate, weight_grid)
    blended = blend_predictions(frame, baseline, candidate, weights)
    by_horizon = {}
    for horizon in (30, 60):
        mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
        by_horizon[str(horizon)] = {
            "v1_mae_mwh": _mae(frame.loc[mask], baseline[mask]),
            "v2_mae_mwh": _mae(frame.loc[mask], blended[mask]),
        }
    by_fold = {}
    for fold in frame["_fold"].unique():
        mask = frame["_fold"].eq(fold).to_numpy()
        by_fold[str(fold)] = {
            "v1_mae_mwh": _mae(frame.loc[mask], baseline[mask]),
            "v2_mae_mwh": _mae(frame.loc[mask], blended[mask]),
        }
    return FamilyScore(
        name=name,
        physical=physical,
        weights=weights,
        baseline_mae=_mae(frame, baseline),
        candidate_mae=_mae(frame, blended),
        by_horizon=by_horizon,
        by_fold=by_fold,
    )


def _eligible(score: FamilyScore, max_horizon_regression: float, minimum_improvement: float) -> bool:
    if score.candidate_mae >= score.baseline_mae * (1 - minimum_improvement) or all(weight == 0 for weight in score.weights.values()):
        return False
    return all(
        values["v2_mae_mwh"] <= values["v1_mae_mwh"] * (1 + max_horizon_regression)
        for values in score.by_horizon.values()
    )


def is_final_artifact_eligible(
    candidate_mae: float,
    baseline_mae: float,
    minimum_improvement: float,
    *,
    publication_verified: bool,
) -> bool:
    """Fail closed when source timestamps do not establish as-of availability."""

    return bool(
        publication_verified
        and candidate_mae < baseline_mae * (1 - minimum_improvement)
    )


def run(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    baseline_model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
    artifact_path: Path = DEFAULT_ARTIFACT,
) -> dict[str, object]:
    if artifact_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing optional v2 artifact: {artifact_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = load_benchmark_contract(PROJECT_ROOT / config["benchmark_contract_path"])
    if contract.label_maturity_policy != "target_before_score_issue":
        raise ValueError("Optional v2 requires the purged benchmark contract")
    validate_benchmark_dataset(dataset_path)
    if _sha256(dataset_path) != contract.frozen_baseline["dataset_sha256"]:
        raise ValueError("Dataset does not match the frozen v1 comparison")
    if _sha256(baseline_model_path) != contract.frozen_baseline["model_artifact_sha256"]:
        raise ValueError("v1 rollback bundle hash mismatch")
    data = load_dataset(dataset_path)
    base_columns = feature_columns(data)
    train, validation, final_test = chronological_split(
        data, contract.train_fraction, contract.validation_fraction
    )
    final_start = final_test["issue_timestamp_utc"].min()
    folds = build_rolling_origin_folds(
        train, validation, contract, final_test_start_utc=final_start
    )
    training_config = TrainingConfig(
        train_fraction=contract.train_fraction,
        validation_fraction=contract.validation_fraction,
    )
    development_frames: list[pd.DataFrame] = []
    baseline_parts: list[np.ndarray] = []
    candidate_parts: dict[str, list[np.ndarray]] = {
        "all_v1_features": [],
        "all_v1_features_plus_physical_interactions": [],
    }
    estimator = config["estimator"]
    for fold in folds:
        _, baseline = _fold_report(
            fold, base_columns, None, training_config, contract
        )
        scored = fold.score.copy()
        scored["_fold"] = fold.name
        development_frames.append(scored)
        baseline_parts.append(baseline)
        for name, physical in (
            ("all_v1_features", False),
            ("all_v1_features_plus_physical_interactions", True),
        ):
            columns = base_columns + (list(PHYSICAL_FEATURES) if physical else [])
            model = HorizonResidualModel(
                columns,
                n_estimators=int(estimator["n_estimators"]),
                min_samples_leaf=int(estimator["min_samples_leaf"]),
                max_features=float(estimator["max_features"]),
                random_state=int(estimator["random_state"]),
            )
            model.fit(candidate_features(fold.fit, physical=physical))
            candidate_parts[name].append(
                model.predict(candidate_features(fold.score, physical=physical))
            )
    development = pd.concat(development_frames, ignore_index=True)
    baseline_oof = np.concatenate(baseline_parts)
    baseline_mae = _mae(development, baseline_oof)
    expected = float(contract.frozen_baseline["rolling_mae_mwh"])
    if not np.isclose(baseline_mae, expected, atol=1e-9, rtol=0):
        raise ValueError(f"Purged v1 baseline mismatch: {baseline_mae} != {expected}")
    scores = [
        _family_score(
            development, baseline_oof, np.concatenate(candidate_parts[name]),
            name=name, physical=physical, weight_grid=config["candidate_weight_grid"],
        )
        for name, physical in (
            ("all_v1_features", False),
            ("all_v1_features_plus_physical_interactions", True),
        )
    ]
    tolerance = float(config["development_gate"]["maximum_single_horizon_regression_fraction"])
    minimum_development_improvement = float(config["development_gate"]["rolling_mae_improvement_min"])
    eligible = [
        score for score in scores
        if _eligible(score, tolerance, minimum_development_improvement)
    ]
    selected = min(eligible, key=lambda item: (item.candidate_mae, item.name)) if eligible else None
    report: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "benchmark_contract": contract.contract_id,
        "dataset_sha256": _sha256(dataset_path),
        "v1_bundle_sha256": _sha256(baseline_model_path),
        "development_rows": len(development),
        "development_v1_mae_mwh": baseline_mae,
        "families": [score.__dict__ for score in scores],
        "selected_family": selected.name if selected else None,
        "final_test_scored": False,
        "artifact_created": False,
    }
    if selected is not None:
        # Only the final-test start timestamp enters development selection.
        fit = pd.concat([train, validation], ignore_index=True)
        fit = _fit_with_mature_labels(fit, final_start, contract)
        columns = base_columns + (list(PHYSICAL_FEATURES) if selected.physical else [])
        final_model = HorizonResidualModel(
            columns,
            n_estimators=int(estimator["n_estimators"]),
            min_samples_leaf=int(estimator["min_samples_leaf"]),
            max_features=float(estimator["max_features"]),
            random_state=int(estimator["random_state"]),
        ).fit(candidate_features(fit, physical=selected.physical))
        baseline_bundle = joblib.load(baseline_model_path)
        metadata = baseline_bundle["metadata"]
        if any(float(value) != 0 for value in metadata["dispatch_regression_ml_weight_by_horizon"].values()):
            raise ValueError("Frozen v1 baseline needs its full ML prediction path")
        baseline_final = _dispatch_trend_prediction(
            final_test, metadata["dispatch_trend_alpha_by_horizon"]
        )
        baseline_final_mae = _mae(final_test, baseline_final)
        frozen_final_mae = float(contract.frozen_baseline["metrics"]["dispatch_down_mae_mwh"])
        if not np.isclose(baseline_final_mae, frozen_final_mae, atol=1e-9, rtol=0):
            raise ValueError("Final comparison does not reproduce frozen v1 point MAE")
        candidate_final = final_model.predict(
            candidate_features(final_test, physical=selected.physical)
        )
        blended_final = blend_predictions(
            final_test, baseline_final, candidate_final, selected.weights
        )
        final_mae = _mae(final_test, blended_final)
        report.update(
            {
                "final_test_scored": True,
                "final_test_rows": len(final_test),
                "final_test_v1_mae_mwh": baseline_final_mae,
                "final_test_v2_mae_mwh": final_mae,
                "final_test_improvement_fraction": (baseline_final_mae - final_mae) / baseline_final_mae,
            }
        )
        minimum_final_improvement = float(config["experimental_artifact_gate"]["final_test_mae_improvement_min"])
        if is_final_artifact_eligible(
            final_mae,
            baseline_final_mae,
            minimum_final_improvement,
            publication_verified=bool(config["source_publication_verified"]),
        ):
            artifact = {
                "metadata": {
                    "model_version": "2.0.0-experimental",
                    "experimental": True,
                    "dataset_sha256": _sha256(dataset_path),
                    "v1_bundle_sha256": _sha256(baseline_model_path),
                    "selected_family": selected.name,
                    "physical_interactions": selected.physical,
                    "feature_columns": columns,
                    "weights_by_horizon": selected.weights,
                    "development_mae_mwh": selected.candidate_mae,
                    "final_test_mae_mwh": final_mae,
                },
                "model": final_model,
            }
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, artifact_path, compress=3)
            report["artifact_created"] = True
            report["artifact_path"] = str(artifact_path)
            report["artifact_sha256"] = _sha256(artifact_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    report = run(
        dataset_path=args.dataset,
        baseline_model_path=args.baseline_model,
        config_path=args.config,
        output_dir=args.output_dir,
        artifact_path=args.artifact,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "families"}, indent=2))


if __name__ == "__main__":
    main()
