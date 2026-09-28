"""Evaluate Wind/Solar allocation of V2's *predicted* daily curtailment.

Serving equation for every candidate (conserves MWh by construction)::

    T_hat = V2 predicted_curtailment_mwh
    p_hat_wind = clamp(candidate(issue-time features), 0, 1)
    wind_hat = T_hat * p_hat_wind;  solar_hat = T_hat - wind_hat

Source labels are only training targets and evaluation truth. Scoring uses the
parent's out-of-sample predicted total; allocating the *actual* total is
reported separately as a clearly labelled diagnostic. Nothing here changes the
V1/V2 artifacts or any API route.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import daily_model
from .constants import PROJECT_ROOT


DEFAULT_SOURCE_LABELS = PROJECT_ROOT / "data" / "processed" / "eirgrid_source_curtailment_daily.csv"
DEFAULT_REPORT = PROJECT_ROOT / "benchmarks" / "daily_source_allocation_v2" / "evaluation.json"
BASELINES = ("wind_only", "training_monthly_share")
CANDIDATES = (
    "wind_only", "training_global_share", "training_monthly_share",
    "hgb_share", "logistic_share", "direct_source_regressors",
)
# Fixed before scoring the 2026 test: a learned allocator must beat the best
# baseline's combined source MAE and must not lose >5% on solar-active days.
SOLAR_ACTIVE_TOLERANCE = 1.05

# Experiment 2: physics-informed share. Every constant below was fixed before
# the experiment was scored and must not be tuned against its results.
PHYSICS_REPORT = PROJECT_ROOT / "benchmarks" / "daily_source_allocation_v2" / "physics_evaluation.json"
PHYSICS_CANDIDATE = "physics_share"
PHYSICS_ARTIFACT = PROJECT_ROOT / "models" / "v2" / "source_allocation_physics.json"
PHYSICS_MODEL_VERSION = "2.0.0-sources-experimental"
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "processed" / "eirgrid_core_history_30min.csv.gz"
# EirGrid system workbooks are republished monthly (for example V8, covering
# data through 31 August, was retrieved on 23 September). Month M-2 is therefore
# published before any issue day in month M.
CAPACITY_PUBLICATION_MONTHS = 2
# A full year always contains a summer peak, so the maximum measures installed
# capacity rather than the season's sunshine.
CAPACITY_WINDOW_DAYS = 365
WIND_CUT_IN_MS = 3.0
WIND_RATED_MS = 12.0
WIND_CURVE_EXPONENT = 1.5
# Fresh days never seen while designing experiment 2; the gate needs them.
CONFIRMATION_START = pd.Timestamp("2026-08-31T00:00:00Z")
MIN_CONFIRMATION_DAYS = 60
LABEL_COLUMNS = {"wind_curtailment_mwh", "solar_curtailment_mwh", "source_curtailment_total_mwh", "wind_share"}


def load_dataset(
    dataset_path: Path = daily_model.DEFAULT_DAILY_DATASET,
    labels_path: Path = DEFAULT_SOURCE_LABELS,
) -> pd.DataFrame:
    data = daily_model._validate_dataset(pd.read_csv(dataset_path))
    labels = pd.read_csv(labels_path)
    labels["issue_timestamp_utc"] = pd.to_datetime(labels.pop("target_date_utc"), utc=True)
    merged = data.merge(labels, on="issue_timestamp_utc", how="inner", validate="one_to_one")
    if len(merged) != len(data):
        raise ValueError("Every V2 day needs a complete Wind/Solar source label")
    if (merged["source_curtailment_total_mwh"] - merged["curtailment_mwh"]).abs().max() >= 0.05:
        raise ValueError("Source labels do not reconcile with the V2 daily label")
    return merged


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = daily_model.feature_columns(frame)
    leaked = LABEL_COLUMNS.intersection(columns) | {"curtailment_mwh", "curtailment_event"}.intersection(columns)
    if leaked:
        raise ValueError(f"Label columns cannot be features: {sorted(leaked)}")
    return columns


def capacity_proxies(history: pd.DataFrame, issue_days: pd.Series) -> pd.DataFrame:
    """Installed-capacity proxies known at each 00:00 UTC issue.

    Uses the maximum IE wind/solar availability over the 365 days ending on the
    last day of calendar month M-2, so only already-published months are read.
    """

    frame = history[["timestamp_utc", "eirgrid_ie_solar_availability_mw", "eirgrid_ie_wind_availability_mw"]].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    daily = frame.set_index("timestamp_utc").resample("D").max()
    trailing = daily.rolling(f"{CAPACITY_WINDOW_DAYS}D", min_periods=CAPACITY_WINDOW_DAYS // 2).max()
    days = pd.to_datetime(issue_days, utc=True)
    through = (
        days.dt.tz_localize(None).dt.to_period("M") - CAPACITY_PUBLICATION_MONTHS
    ).dt.end_time.dt.normalize().dt.tz_localize("UTC")
    looked_up = trailing.reindex(through.to_numpy())
    return pd.DataFrame({
        "solar_capacity_proxy_mw": looked_up["eirgrid_ie_solar_availability_mw"].to_numpy(),
        "wind_capacity_proxy_mw": looked_up["eirgrid_ie_wind_availability_mw"].to_numpy(),
        "capacity_proxy_through_utc": through.to_numpy(),
    }, index=issue_days.index)


def add_physics_features(data: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Forecast wind vs solar energy, each scaled by its published capacity proxy."""

    columns = feature_columns(data)
    result = data.join(capacity_proxies(history, data["issue_timestamp_utc"]))
    if result[["solar_capacity_proxy_mw", "wind_capacity_proxy_mw"]].isna().any().any():
        raise ValueError("Capacity proxy history does not cover every issue day")
    # Mean regional daily irradiance in kWh/m2 is peak-sun-hours per MW installed.
    sun_hours = result[[column for column in columns if column.endswith("_solar_sum_wm2h")]].mean(axis=1) / 1000
    wind_ms = result[[column for column in columns if column.endswith("_wind_mean_kmh")]].mean(axis=1) / 3.6
    capacity_factor = np.clip((wind_ms - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS), 0, 1) ** WIND_CURVE_EXPONENT
    result["solar_energy_proxy_mwh"] = result["solar_capacity_proxy_mw"] * sun_hours
    result["wind_energy_proxy_mwh"] = result["wind_capacity_proxy_mw"] * 24 * capacity_factor
    result["physics_log_ratio"] = np.log(
        (result["wind_energy_proxy_mwh"] + 1) / (result["solar_energy_proxy_mwh"] + 1)
    )
    return result


def fit_physics_share(fit: pd.DataFrame) -> LogisticRegression:
    """One-feature, MWh-weighted logistic calibration of the wind share."""

    positive = fit.loc[fit["source_curtailment_total_mwh"].gt(0)]
    stacked = pd.concat([positive[["physics_log_ratio"]]] * 2, ignore_index=True)
    outcome = np.r_[np.ones(len(positive)), np.zeros(len(positive))]
    mass = np.r_[positive["wind_curtailment_mwh"], positive["solar_curtailment_mwh"]]
    return LogisticRegression(C=1.0).fit(stacked, outcome, sample_weight=mass)


def _share(frame: pd.DataFrame) -> float:
    return float(frame["wind_curtailment_mwh"].sum() / frame["source_curtailment_total_mwh"].sum())


def fit_share_candidates(fit: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    """Fit every candidate on positive-curtailment days of ``fit`` only."""

    positive = fit.loc[fit["source_curtailment_total_mwh"].gt(0)]
    if positive.empty:
        raise ValueError("Need positive-curtailment days to learn shares")
    prior = _share(positive)
    monthly = {
        int(month): _share(group) for month, group in positive.groupby(positive["issue_timestamp_utc"].dt.month)
    }
    weights = positive["source_curtailment_total_mwh"]
    hgb = HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=8, min_samples_leaf=12,
        learning_rate=0.05, l2_regularization=3.0, random_state=42,
    ).fit(positive[columns], positive["wind_share"], sample_weight=weights)
    # MWh-weighted logistic share: each day contributes a wind and a solar row.
    stacked = pd.concat([positive[columns], positive[columns]], ignore_index=True)
    outcome = np.r_[np.ones(len(positive)), np.zeros(len(positive))]
    mass = np.r_[positive["wind_curtailment_mwh"], positive["solar_curtailment_mwh"]]
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(
        stacked, outcome, logisticregression__sample_weight=mass,
    )
    direct = {
        source: HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=150, max_leaf_nodes=8, min_samples_leaf=12,
            learning_rate=0.05, l2_regularization=3.0, random_state=42,
        ).fit(fit[columns], fit[f"{source}_curtailment_mwh"])
        for source in ("wind", "solar")
    }
    return {"prior": prior, "monthly": monthly, "hgb": hgb, "logistic": logistic, "direct": direct}


def predict_wind_shares(models: dict[str, object], frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    prior = float(models["prior"])  # type: ignore[arg-type]
    months = frame["issue_timestamp_utc"].dt.month
    wind = np.maximum(models["direct"]["wind"].predict(frame[columns]), 0)  # type: ignore[index]
    solar = np.maximum(models["direct"]["solar"].predict(frame[columns]), 0)  # type: ignore[index]
    total = wind + solar
    shares = {
        "wind_only": np.ones(len(frame)),
        "training_global_share": np.full(len(frame), prior),
        "training_monthly_share": months.map(models["monthly"]).fillna(prior).to_numpy(dtype=float),  # type: ignore[arg-type]
        "hgb_share": models["hgb"].predict(frame[columns]),  # type: ignore[union-attr]
        "logistic_share": models["logistic"].predict_proba(frame[columns])[:, 1],  # type: ignore[union-attr]
        "direct_source_regressors": np.divide(wind, total, out=np.full(len(frame), prior), where=total > 0),
    }
    return {name: np.clip(value, 0.0, 1.0) for name, value in shares.items()}


def allocate(total_mwh: np.ndarray, wind_share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a non-negative parent total; zero parent gives zero components."""

    total = np.maximum(np.asarray(total_mwh, dtype=float), 0.0)
    wind = total * np.clip(wind_share, 0.0, 1.0)
    return wind, total - wind


def score(frame: pd.DataFrame, total_mwh: np.ndarray, wind_share: np.ndarray) -> dict[str, object]:
    wind_hat, solar_hat = allocate(total_mwh, wind_share)
    wind = frame["wind_curtailment_mwh"].to_numpy(dtype=float)
    solar = frame["solar_curtailment_mwh"].to_numpy(dtype=float)
    wind_error = np.abs(wind_hat - wind)
    solar_error = np.abs(solar_hat - solar)
    solar_active = solar > 0
    curtailed = (wind + solar) > 0
    return {
        "rows": int(len(frame)),
        "wind_mae_mwh": float(wind_error.mean()),
        "solar_mae_mwh": float(solar_error.mean()),
        "combined_source_mae_mwh": float((wind_error + solar_error).mean()),
        "wind_bias_mwh": float((wind_hat - wind).mean()),
        "solar_bias_mwh": float((solar_hat - solar).mean()),
        "max_conservation_error_mwh": float(np.abs(wind_hat + solar_hat - np.maximum(total_mwh, 0)).max()),
        "curtailment_positive_days": int(curtailed.sum()),
        "curtailment_positive_combined_source_mae_mwh": float((wind_error + solar_error)[curtailed].mean()) if curtailed.any() else None,
        "solar_active_days": int(solar_active.sum()),
        "solar_active_solar_mae_mwh": float(solar_error[solar_active].mean()) if solar_active.any() else None,
        "predicted_solar_share_of_mwh": float(solar_hat.sum() / max((wind_hat + solar_hat).sum(), 1e-9)),
        "actual_solar_share_of_mwh": float(solar.sum() / max((wind + solar).sum(), 1e-9)),
    }


def _parent_totals(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, columns: list[str], artifact_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Out-of-sample V2 totals: train-only refit for 2025, saved artifact for 2026."""

    bundle = joblib.load(artifact_path)
    method = bundle["metadata"]["selected_amount_method"]
    if bundle["metadata"]["feature_columns"] != columns:
        raise ValueError("V2 artifact feature contract differs from the dataset")
    event = daily_model._classifier().fit(train[columns], train["curtailment_event"])
    amount = daily_model._regressors()[method]
    fit = train.loc[train["curtailment_event"].eq(1)] if method == "two_stage" else train
    amount.fit(fit[columns], fit["curtailment_mwh"])
    validation_total = daily_model._predict_amount(method, amount, validation[columns], event.predict_proba(validation[columns])[:, 1])
    trained_through = pd.Timestamp(bundle["metadata"]["trained_through_utc"])
    if test["issue_timestamp_utc"].min() <= trained_through:
        raise ValueError("Test days overlap the saved V2 artifact's training period")
    test_total = daily_model._predict_amount(
        method, bundle["amount_model"], test[columns], bundle["event_model"].predict_proba(test[columns])[:, 1],
    )
    return validation_total, test_total, method


def _slices(frame: pd.DataFrame, total: np.ndarray, shares: dict[str, np.ndarray], names: tuple[str, ...]) -> dict[str, dict]:
    quarter = frame["issue_timestamp_utc"].dt.year.astype(str) + "Q" + frame["issue_timestamp_utc"].dt.quarter.astype(str)
    result = {}
    for label, subset in frame.groupby(quarter, sort=True):
        positions = frame.index.get_indexer(subset.index)
        result[str(label)] = {
            name: {key: value for key, value in score(subset, total[positions], shares[name][positions]).items()
                   if key in ("rows", "combined_source_mae_mwh", "solar_mae_mwh", "solar_active_days", "actual_solar_share_of_mwh", "predicted_solar_share_of_mwh")}
            for name in names
        }
    return result


def _rounded(value: object) -> object:
    """Round floats so parallel-tree summation noise cannot change the report."""

    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(
    dataset_path: Path = daily_model.DEFAULT_DAILY_DATASET,
    labels_path: Path = DEFAULT_SOURCE_LABELS,
    artifact_path: Path = daily_model.DEFAULT_ARTIFACT,
    report_path: Path | None = DEFAULT_REPORT,
) -> dict:
    """Select on 2025 with a train-only parent, then score 2026 once."""

    data = load_dataset(dataset_path, labels_path)
    train, validation, test = (part.reset_index(drop=True) for part in daily_model.chronological_partitions(data))
    columns = feature_columns(data)
    validation_total, test_total, method = _parent_totals(train, validation, test, columns, artifact_path)

    validation_shares = predict_wind_shares(fit_share_candidates(train, columns), validation, columns)
    validation_scores = {name: score(validation, validation_total, share) for name, share in validation_shares.items()}
    selected = min(CANDIDATES, key=lambda name: (validation_scores[name]["combined_source_mae_mwh"], name))

    development = pd.concat([train, validation], ignore_index=True)
    test_shares = predict_wind_shares(fit_share_candidates(development, columns), test, columns)
    test_scores = {name: score(test, test_total, share) for name, share in test_shares.items()}
    actual_total = test["source_curtailment_total_mwh"].to_numpy(dtype=float)
    oracle_scores = {name: score(test, actual_total, share) for name, share in test_shares.items()}

    best_baseline = min(BASELINES, key=lambda name: test_scores[name]["combined_source_mae_mwh"])
    learned = selected not in BASELINES
    beats = learned and test_scores[selected]["combined_source_mae_mwh"] < test_scores[best_baseline]["combined_source_mae_mwh"]
    solar_ok = learned and (
        test_scores[selected]["solar_active_solar_mae_mwh"]
        <= SOLAR_ACTIVE_TOLERANCE * test_scores[best_baseline]["solar_active_solar_mae_mwh"]
    )
    conserved = all(item["max_conservation_error_mwh"] < 1e-6 for item in test_scores.values())
    gate = {
        "rule": (
            "Pass only if a learned allocator is selected on 2025 validation, beats the best test baseline's "
            "combined source MAE on the 2026 end-to-end test, stays within "
            f"{SOLAR_ACTIVE_TOLERANCE:.2f}x that baseline's solar-active solar MAE, and conserves the parent total."
        ),
        "selected_candidate_is_learned": learned,
        "best_test_baseline": best_baseline,
        "beats_best_baseline": bool(beats),
        "solar_active_not_materially_worse": bool(solar_ok),
        "conserves_parent_total": conserved,
        "labels_reconcile": True,
        "passed": bool(learned and beats and solar_ok and conserved),
    }
    report = {
        "experiment": "daily_source_allocation_v2",
        "status": "experimental_estimate_only" if not gate["passed"] else "passed_release_gate",
        "task": "Split V2's predicted UTC-day curtailment MWh into Wind and Solar",
        "parent_model": {
            "model_version": "2.0.0-daily-experimental",
            "selected_amount_method": method,
            "validation_totals": "Refit of the V2 recipe on the train partition only (out-of-sample for 2025)",
            "test_totals": "Saved V2 artifact, trained through 2025-12-31 (out-of-sample for 2026)",
            "validation_parent_mae_mwh": float(np.abs(validation_total - validation["curtailment_mwh"]).mean()),
            "test_parent_mae_mwh": float(np.abs(test_total - test["curtailment_mwh"]).mean()),
        },
        "serving_equation": "wind = T_hat * clamp(p_wind, 0, 1); solar = T_hat - wind",
        "feature_columns": columns,
        "inputs": {
            "dataset_sha256": _sha256(dataset_path),
            "source_labels_sha256": _sha256(labels_path),
            "parent_artifact_sha256": _sha256(artifact_path),
        },
        "date_ranges": {
            name: [part["issue_timestamp_utc"].min().date().isoformat(), part["issue_timestamp_utc"].max().date().isoformat()]
            for name, part in (("train", train), ("validation", validation), ("test", test))
        },
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "actual_solar_share_of_mwh": {
            name: round(1 - _share(part.loc[part["source_curtailment_total_mwh"].gt(0)]), 6)
            for name, part in (("train", train), ("validation", validation), ("test", test))
        },
        "candidates": list(CANDIDATES),
        "baselines": list(BASELINES),
        "selected_on_validation": selected,
        "validation_end_to_end": validation_scores,
        "test_end_to_end": test_scores,
        "test_by_quarter": _slices(test, test_total, test_shares, (selected, *BASELINES, "hgb_share", "logistic_share")),
        "test_oracle_total_diagnostic": {
            "notice": "Allocates the ACTUAL daily total, which is unknown at issue time. Isolates share skill only; never a serving score.",
            "scores": oracle_scores,
        },
        "release_gate": gate,
        "limitations": [
            "The parent's daily total error (about 1,600-1,700 MWh MAE) dominates source error; allocation choice changes end-to-end error by under 10%.",
            "Solar's share of curtailment grew from about 6% (2024 train) to 13% (2025) and 19% (2026 test); models fitted on past years under-allocate solar.",
            "V2 features start 2024-04-01, so only 275 training days precede the 2025 validation year.",
            "V1 half-hour allocation was not attempted: its model-ready table covers January 2026 only, with little solar curtailment.",
        ],
    }
    report = _rounded(report)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _gate(scores: dict[str, dict], candidate: str) -> dict[str, object]:
    best_baseline = min(BASELINES, key=lambda name: scores[name]["combined_source_mae_mwh"])
    beats = scores[candidate]["combined_source_mae_mwh"] < scores[best_baseline]["combined_source_mae_mwh"]
    solar_ok = (
        scores[candidate]["solar_active_solar_mae_mwh"]
        <= SOLAR_ACTIVE_TOLERANCE * scores[best_baseline]["solar_active_solar_mae_mwh"]
    )
    return {"best_baseline": best_baseline, "beats_best_baseline": bool(beats), "solar_active_not_materially_worse": bool(solar_ok)}


def evaluate_physics(
    dataset_path: Path = daily_model.DEFAULT_DAILY_DATASET,
    labels_path: Path = DEFAULT_SOURCE_LABELS,
    artifact_path: Path = daily_model.DEFAULT_ARTIFACT,
    history_path: Path = DEFAULT_HISTORY,
    report_path: Path | None = PHYSICS_REPORT,
    serving_artifact_path: Path = PHYSICS_ARTIFACT,
) -> dict:
    """Experiment 2: add the capacity-aware physics share and require fresh confirmation.

    The 2026-01..08 test was viewed while designing this candidate, so it cannot
    certify it. Release additionally needs >= MIN_CONFIRMATION_DAYS labelled days
    from CONFIRMATION_START onward, scored with allocators frozen on 2024-2025.
    """

    history = pd.read_csv(
        history_path,
        usecols=["timestamp_utc", "eirgrid_ie_solar_availability_mw", "eirgrid_ie_wind_availability_mw"],
    )
    data = add_physics_features(load_dataset(dataset_path, labels_path), history)
    published_before = data["issue_timestamp_utc"].dt.tz_localize(None).dt.to_period("M").dt.start_time.dt.tz_localize("UTC") - pd.DateOffset(months=1)
    if (data["capacity_proxy_through_utc"] >= published_before).any():
        raise ValueError("Capacity proxy reads a month that is not yet published at issue time")
    train, validation, later = (part.reset_index(drop=True) for part in daily_model.chronological_partitions(data))
    columns = feature_columns(data)
    validation_total, later_total, method = _parent_totals(train, validation, later, columns, artifact_path)
    in_test = later["issue_timestamp_utc"].lt(CONFIRMATION_START).to_numpy()
    test, confirmation = later.loc[in_test].reset_index(drop=True), later.loc[~in_test].reset_index(drop=True)
    test_total, confirmation_total = later_total[in_test], later_total[~in_test]
    candidates = (*CANDIDATES, PHYSICS_CANDIDATE)

    def shares(fit: pd.DataFrame, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        result = predict_wind_shares(fit_share_candidates(fit, columns), frame, columns)
        result[PHYSICS_CANDIDATE] = fit_physics_share(fit).predict_proba(frame[["physics_log_ratio"]])[:, 1]
        return result

    validation_scores = {name: score(validation, validation_total, share) for name, share in shares(train, validation).items()}
    selected = min(candidates, key=lambda name: (validation_scores[name]["combined_source_mae_mwh"], name))
    development = pd.concat([train, validation], ignore_index=True)
    test_shares = shares(development, test)
    test_scores = {name: score(test, test_total, share) for name, share in test_shares.items()}
    actual_total = test["source_curtailment_total_mwh"].to_numpy(dtype=float)
    oracle = {name: score(test, actual_total, share) for name, share in test_shares.items()}
    physics_model = fit_physics_share(development)

    learned = selected not in BASELINES
    if learned:
        test_gate = _gate(test_scores, selected)
    else:
        test_gate = {"best_baseline": None, "beats_best_baseline": False, "solar_active_not_materially_worse": False}
    confirmation_scores = None
    confirmation_gate: dict[str, object] = {
        "status": "pending", "rows": int(len(confirmation)), "required_rows": MIN_CONFIRMATION_DAYS,
    }
    if len(confirmation) >= MIN_CONFIRMATION_DAYS and learned:
        confirmation_scores = {
            name: score(confirmation, confirmation_total, share)
            for name, share in shares(development, confirmation).items()
        }
        confirmation_gate.update(_gate(confirmation_scores, selected))
        confirmation_gate["status"] = (
            "passed"
            if confirmation_gate["beats_best_baseline"] and confirmation_gate["solar_active_not_materially_worse"]
            else "failed"
        )
    conserved = all(item["max_conservation_error_mwh"] < 1e-6 for item in test_scores.values())
    test_passed = bool(
        learned and test_gate["beats_best_baseline"] and test_gate["solar_active_not_materially_worse"] and conserved
    )
    passed = bool(test_passed and confirmation_gate["status"] == "passed")
    if passed:
        status = "passed_release_gate"
    elif test_passed and confirmation_gate["status"] == "pending":
        status = "candidate_awaiting_fresh_confirmation"
    else:
        status = "experimental_estimate_only"

    report = {
        "experiment": "daily_source_allocation_v2_physics",
        "status": status,
        "task": "Split V2's predicted UTC-day curtailment MWh into Wind and Solar",
        "why_this_experiment": (
            "Experiment 1 (evaluation.json) failed because solar's share grew each year. EirGrid curtails roughly "
            "in proportion to each source's available output, so this candidate compares forecast solar and wind "
            "energy scaled by a published capacity proxy."
        ),
        "physics_candidate": {
            "name": PHYSICS_CANDIDATE,
            "formula": (
                "p_wind = 1 / (1 + exp(-(intercept + slope * ln((wind_energy + 1) / (solar_energy + 1))))); "
                "solar_energy = solar_capacity_proxy_mw * mean regional forecast irradiance (kWh/m2/day); "
                "wind_energy = wind_capacity_proxy_mw * 24 * clip((v - cut_in) / (rated - cut_in), 0, 1) ** exponent, "
                "v = mean regional forecast 100 m wind speed (m/s)"
            ),
            "fitted_on": "Positive-curtailment days 2024-04-01 to 2025-12-31, weighted by MWh",
            "intercept": float(physics_model.intercept_[0]),
            "slope": float(physics_model.coef_[0][0]),
            "fixed_constants": {
                "capacity_publication_months": CAPACITY_PUBLICATION_MONTHS,
                "capacity_window_days": CAPACITY_WINDOW_DAYS,
                "wind_cut_in_ms": WIND_CUT_IN_MS,
                "wind_rated_ms": WIND_RATED_MS,
                "wind_curve_exponent": WIND_CURVE_EXPONENT,
            },
            "capacity_rule": (
                "Maximum IE availability over the 365 days ending on the last day of calendar month M-2 for an "
                "issue day in month M. EirGrid republishes the system workbook monthly, so month M-2 is public by then."
            ),
        },
        "parent_model": {
            "selected_amount_method": method,
            "validation_parent_mae_mwh": float(np.abs(validation_total - validation["curtailment_mwh"]).mean()),
            "test_parent_mae_mwh": float(np.abs(test_total - test["curtailment_mwh"]).mean()),
        },
        "inputs": {
            "dataset_sha256": _sha256(dataset_path),
            "source_labels_sha256": _sha256(labels_path),
            "parent_artifact_sha256": _sha256(artifact_path),
            "capacity_history_sha256": _sha256(history_path),
        },
        "rows": {"train": len(train), "validation": len(validation), "test": len(test), "confirmation": len(confirmation)},
        "candidates": list(candidates),
        "baselines": list(BASELINES),
        "selected_on_validation": selected,
        "validation_end_to_end": validation_scores,
        "test_end_to_end": test_scores,
        "test_by_quarter": _slices(
            test, test_total, test_shares, tuple(dict.fromkeys((selected, *BASELINES, PHYSICS_CANDIDATE))),
        ),
        "test_oracle_total_diagnostic": {
            "notice": "Allocates the ACTUAL daily total, which is unknown at issue time. Isolates share skill only; never a serving score.",
            "scores": oracle,
        },
        "confirmation_end_to_end": confirmation_scores,
        "release_gate": {
            "rule": (
                "Selected on 2025 validation; must be a learned candidate that beats the best baseline's combined "
                f"source MAE and stays within {SOLAR_ACTIVE_TOLERANCE:.2f}x its solar-active solar MAE on the "
                f"2026-01..08 test AND again on >= {MIN_CONFIRMATION_DAYS} fresh days from "
                f"{CONFIRMATION_START.date().isoformat()}, which were unseen when this candidate was designed."
            ),
            "test_2026": {**test_gate, "conserves_parent_total": conserved, "holdout_viewed_during_design": True},
            "fresh_confirmation": confirmation_gate,
            "passed": passed,
        },
        "limitations": [
            "The 2026-01..08 test had been viewed in experiment 1 and in exploratory research, so it cannot certify this candidate on its own.",
            "The parent's daily total error still dominates; allocation improves Wind/Solar error but cannot fix a wrong total.",
            "The capacity proxy uses observed availability, not official installed capacity, and assumes EirGrid's monthly publication cadence holds.",
            "The share is a daily average; it does not describe which half-hours were curtailed.",
        ],
    }
    report = _rounded(report)
    if report_path is not None:
        # Frozen serving parameters: fitted on 2024-2025 only, so reruns that add
        # confirmation days reproduce this exact file and cannot peek at them.
        artifact = {
            "model_version": PHYSICS_MODEL_VERSION,
            "candidate": PHYSICS_CANDIDATE,
            "parent_model_version": "2.0.0-daily-experimental",
            "fitted_on": [
                development["issue_timestamp_utc"].min().date().isoformat(),
                development["issue_timestamp_utc"].max().date().isoformat(),
            ],
            "intercept": float(physics_model.intercept_[0]),
            "slope": float(physics_model.coef_[0][0]),
            "fixed_constants": report["physics_candidate"]["fixed_constants"],
            "parent_artifact_sha256": report["inputs"]["parent_artifact_sha256"],
        }
        serving_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        serving_artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        report["serving_artifact"] = serving_artifact_path.relative_to(PROJECT_ROOT).as_posix()             if serving_artifact_path.is_relative_to(PROJECT_ROOT) else serving_artifact_path.name
        report["serving_artifact_sha256"] = _sha256(serving_artifact_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
