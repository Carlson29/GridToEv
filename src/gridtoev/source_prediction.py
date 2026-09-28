"""Experimental live Wind/Solar split of V2's predicted daily curtailment.

The two physics parameters are frozen (fitted on 2024-2025 only) and verified
against their evaluation report before use. Every day after the design period
is therefore an unbiased prospective test: its prediction is fixed before the
outcome exists and is scored later by ``evaluate_physics``'s confirmation gate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .daily_curtailment import REGIONS, build_forecast_features, fetch_previous_runs
from .daily_model import EARLIEST_TARGET_DATE, DailyCurtailmentService
from .source_allocation import (
    DEFAULT_HISTORY,
    PHYSICS_ARTIFACT,
    PHYSICS_REPORT,
    add_physics_features,
    allocate,
)


logger = logging.getLogger(__name__)
CAPACITY_COLUMNS = ["timestamp_utc", "eirgrid_ie_solar_availability_mw", "eirgrid_ie_wind_availability_mw"]


class CapacityUnavailableError(ValueError):
    """The published capacity month needed for this date is not in the bundled archive."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourceAllocationService:
    def __init__(
        self,
        daily_service: DailyCurtailmentService,
        artifact_path: Path | str = PHYSICS_ARTIFACT,
        report_path: Path | str = PHYSICS_REPORT,
        history_path: Path | str = DEFAULT_HISTORY,
    ) -> None:
        self.daily_service = daily_service
        self.artifact_path = Path(artifact_path)
        self.report_path = Path(report_path)
        self.history_path = Path(history_path)
        self.artifact: dict[str, Any] | None = None
        self.report: dict[str, Any] | None = None
        self.history: pd.DataFrame | None = None

    def load(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if _sha256(self.artifact_path) != report["serving_artifact_sha256"]:
            raise ValueError("Source allocation artifact checksum does not match its evaluation report")
        artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        if artifact["parent_artifact_sha256"] != _sha256(self.daily_service.artifact_path):
            raise ValueError("Source allocation was evaluated against a different V2 artifact")
        history = pd.read_csv(self.history_path, usecols=CAPACITY_COLUMNS)
        history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True)
        self.artifact, self.report, self.history = artifact, report, history

    @property
    def capacity_through_utc(self) -> date:
        """Last day of published capacity history in the bundled archive."""
        assert self.history is not None
        return self.history["timestamp_utc"].max().date()

    def latest_predictable_date(self) -> date:
        """Issue month M needs capacity through month M-2, so the archive month + 2."""
        last = pd.Timestamp(self.capacity_through_utc)
        # A partially published month cannot be used; step back to the last complete one.
        complete = last.to_period("M") if (last + pd.Timedelta(days=1)).day == 1 else last.to_period("M") - 1
        return (complete + 2).end_time.date()

    def validation(self) -> dict[str, Any]:
        assert self.report is not None
        report = self.report
        selected = report["selected_on_validation"]
        baseline = report["release_gate"]["test_2026"]["best_baseline"]
        return {
            "status": report["status"],
            "fresh_confirmation": report["release_gate"]["fresh_confirmation"],
            "provisional_2026_test": {
                "note": "Provisional: 2026-01..08 was viewed while this method was designed, so these figures may be optimistic.",
                "combined_source_mae_mwh": report["test_end_to_end"][selected]["combined_source_mae_mwh"],
                "baseline_combined_source_mae_mwh": report["test_end_to_end"][baseline]["combined_source_mae_mwh"],
                "solar_active_solar_mae_mwh": report["test_end_to_end"][selected]["solar_active_solar_mae_mwh"],
                "baseline_solar_active_solar_mae_mwh": report["test_end_to_end"][baseline]["solar_active_solar_mae_mwh"],
                "baseline": baseline,
            },
        }

    def model_info(self) -> dict[str, Any]:
        if self.artifact is None:
            self.load()
        assert self.artifact is not None and self.report is not None
        return {
            "model_version": self.artifact["model_version"],
            "parent_model_version": self.artifact["parent_model_version"],
            "experimental": True,
            "method": "physics_share",
            "plain_language": (
                "When the grid must waste renewable power, EirGrid cuts wind and solar roughly in proportion to "
                "how much each could produce. This model estimates that from the day-ahead wind and sunshine "
                "forecast, scaled by how much wind and solar capacity is installed, and uses it to split V2's "
                "predicted daily total."
            ),
            "formula": self.report["physics_candidate"]["formula"],
            "intercept": self.artifact["intercept"],
            "slope": self.artifact["slope"],
            "fitted_on": self.artifact["fitted_on"],
            "fixed_constants": self.artifact["fixed_constants"],
            "capacity_rule": self.report["physics_candidate"]["capacity_rule"],
            "capacity_history_through_utc": self.capacity_through_utc.isoformat(),
            "requestable_date_min_utc": EARLIEST_TARGET_DATE.isoformat(),
            "requestable_date_max_utc": min(self.latest_predictable_date(), datetime.now(timezone.utc).date()).isoformat(),
            "validation": self.validation(),
            "limitations": self.report["limitations"],
        }

    def check_capacity(self, target_date: date) -> None:
        if target_date > self.latest_predictable_date():
            needed = (pd.Period(target_date, "M") - 2).end_time.date().isoformat()
            raise CapacityUnavailableError(
                f"{target_date.isoformat()} needs published capacity data through {needed}, but the bundled "
                f"EirGrid archive ends {self.capacity_through_utc.isoformat()}. The archive must be refreshed "
                "before this date can be split."
            )

    def split(self, features: pd.DataFrame, parent: dict[str, Any]) -> dict[str, Any]:
        """Split one V2 prediction; ``features`` is the same row V2 just used."""

        assert self.artifact is not None and self.history is not None
        self.check_capacity(pd.Timestamp(features.iloc[0]["issue_timestamp_utc"]).date())
        row = add_physics_features(features, self.history).iloc[0]
        ratio = float(row["physics_log_ratio"])
        wind_share = 1 / (1 + math.exp(-(self.artifact["intercept"] + self.artifact["slope"] * ratio)))
        total = float(parent["predicted_curtailment_mwh"])
        wind, solar = (float(value[0]) for value in allocate([total], [wind_share]))
        result = {
            "model_version": self.artifact["model_version"],
            "parent_model_version": parent["model_version"],
            "experimental": True,
            "validation_status": self.report["status"] if self.report else None,
            "target_date_utc": parent["target_date_utc"],
            "issue_timestamp_utc": parent["issue_timestamp_utc"],
            "forecast_max_available_at_utc": parent["forecast_max_available_at_utc"],
            "curtailment_event_probability": parent["curtailment_event_probability"],
            "predicted_curtailment_mwh": total,
            "predicted_wind_curtailment_mwh": wind,
            "predicted_solar_curtailment_mwh": solar,
            "reconciliation_error_mwh": abs(wind + solar - total),
            "predicted_wind_share_percent": round(100 * wind_share, 2),
            "predicted_solar_share_percent": round(100 * (1 - wind_share), 2),
            "capacity_proxy": {
                "solar_mw": float(row["solar_capacity_proxy_mw"]),
                "wind_mw": float(row["wind_capacity_proxy_mw"]),
                "published_data_through_utc": pd.Timestamp(row["capacity_proxy_through_utc"]).date().isoformat(),
            },
            "summary": (
                f"For {parent['target_date_utc']} (UTC), V2 predicts {total:,.1f} MWh of curtailment. "
                f"This experimental model expects about {100 * wind_share:.0f}% of it from wind "
                f"({wind:,.1f} MWh) and {100 * (1 - wind_share):.0f}% from solar ({solar:,.1f} MWh)."
            ),
            "compare_with_actual": f"/actuals/curtailment/sources?target_date_utc={parent['target_date_utc']}",
            "notice": (
                "Experimental and not yet confirmed on fresh data. The split can only be as good as V2's total, "
                "which can miss by thousands of MWh on a single day. Not a dispatch instruction."
            ),
        }
        # Audit trail for the prospective test: prediction fixed before the outcome is known.
        logger.info(
            "source_split_prediction target=%s issued_at=%s total=%.3f wind=%.3f solar=%.3f artifact=%s",
            result["target_date_utc"], datetime.now(timezone.utc).isoformat(), total, wind, solar,
            self.report["serving_artifact_sha256"] if self.report else None,
        )
        return result

    def predict_date(self, target_date: date) -> dict[str, Any]:
        if self.artifact is None:
            self.load()
        if target_date < EARLIEST_TARGET_DATE:
            raise ValueError(f"No complete archived forecast before {EARLIEST_TARGET_DATE.isoformat()}")
        if target_date > datetime.now(timezone.utc).date():
            raise ValueError("A future target day has not reached its 00:00 UTC issue time")
        self.check_capacity(target_date)  # before any network call, so it fails fast
        forecast = pd.concat(
            [fetch_previous_runs(region, target_date, target_date, cache=False) for region in REGIONS],
            ignore_index=True,
        )
        features = build_forecast_features(forecast)
        if len(features) != 1:
            raise ValueError("The archived forecast has missing hours or unavailable inputs for this day")
        parent = self.daily_service.predict_features(features)
        return self.split(features, parent)
