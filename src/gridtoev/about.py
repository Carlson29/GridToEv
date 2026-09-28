"""Small, machine-readable model facts for downstream About pages.

Scores and versions are taken from the loaded, verified model reports. Descriptions
are deliberately narrower than the full /model-info responses: they explain what
a consumer may say without promoting a historical replay to a live forecast.
"""

from __future__ import annotations

from typing import Any


REPO = "https://github.com/Carlson29/GridToEv"
EIRGRID_REPORTS = "https://www.eirgrid.ie/grid/system-and-renewable-data-reports"
ENTSOE_TRANSPARENCY = "https://transparency.entsoe.eu/"
OPEN_METEO_PREVIOUS_RUNS = "https://open-meteo.com/en/docs/previous-runs-api"


def _route(path: str, data_mode: str, explanation: str) -> dict[str, Any]:
    return {
        "path": path,
        "data_mode": data_mode,
        "explanation": explanation,
        "verified_live": False,
    }


def build_models_about(v1_info: dict[str, Any], v2_info: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize the two different targets without inventing a live data feed."""

    v1_test = v1_info["evaluation"]["test"]
    v1_components = v1_info["prediction_components"]
    v1 = {
        "model_id": "v1",
        "available": True,
        "model_version": v1_info["model_version"],
        "display_name": "V1 — short-horizon renewable dispatch-down",
        "experimental": False,
        "plain_language_summary": (
            "Estimates whether renewable energy will be turned down and how many "
            "MWh are at risk in one half-hour, 30 or 60 minutes after the issue time."
        ),
        "target": {
            "name": "renewable dispatch-down",
            "interval_minutes": 30,
            "forecast_horizons_minutes": [30, 60],
            "energy_unit": "MWh per target half-hour",
            "prediction_field": "predicted_dispatch_down_mwh",
            "component_fields": ["predicted_curtailment_mwh", "predicted_constraint_mwh"],
            "at_risk_formula": "predicted_curtailment_mwh + predicted_constraint_mwh",
            "interpretation": (
                "The served curtailment and constraint components are reconciled to "
                "predicted_dispatch_down_mwh. A 30-minute and 60-minute forecast "
                "for the same target are alternatives, not additive energy."
            ),
        },
        "data_sources": [
            {"name": "EirGrid", "role": "Observed Irish grid, renewable and dispatch-down data", "url": EIRGRID_REPORTS},
            {"name": "ENTSO-E Transparency Platform", "role": "Generation, load and price inputs", "url": ENTSOE_TRANSPARENCY},
        ],
        "prediction_routes": [
            _route("/predict/from-dataset", "historical_prediction", "Replays one stored V1 issue time; not today's grid."),
            _route("/predict/window/from-dataset", "historical_prediction", "Replays up to 24 hours of stored V1 issue times."),
            _route("/predict/latest", "historical_prediction", "Latest bundled dataset row, not the current clock time."),
            _route("/predict/v1/from-raw", "user_supplied_unverified", "Caller supplies source values and 48 prior half-hours; the API cannot verify their provenance."),
            _route("/predict/features", "user_supplied_unverified", "Caller supplies all engineered features; the API cannot verify their provenance."),
        ],
        "dataset_coverage": {
            "first_utc": v1_info["available_issue_timestamp_min_utc"],
            "last_utc": v1_info["available_issue_timestamp_max_utc"],
            "time_kind": "issue_timestamp_utc",
            "details_endpoint": "/dataset/info",
        },
        "estimators": {
            "event_classifier": v1_components["event_classifier"]["estimator"],
            "dispatch_down_regressor": v1_components["dispatch_down_regressor"]["estimator"],
            "curtailment_regressor": v1_components["curtailment_regressor"]["estimator"],
            "constraint_regressor": v1_components["constraint_regressor"]["estimator"],
        },
        "serving_policy": {
            "method": v1_components["dispatch_down_serving_policy"]["method"],
            "ml_weight_by_horizon": v1_info["evaluation"]["serving_ml_weight_by_horizon"],
            "explanation": (
                "The final dispatch-down MWh is a horizon-specific trend/ML blend; "
                "the fitted regressor's name alone does not describe the served amount."
            ),
        },
        "evaluation": {
            "test_mae_mwh": v1_test["dispatch_down_mae_mwh"],
            "metric_label": "Mean absolute error per half-hour (MWh)",
            "test_period_start_utc": v1_test["period_start_utc"],
            "test_period_end_utc": v1_test["period_end_utc"],
            "test_rows": v1_test["rows"],
            "details_endpoint": "/model-info",
            "public_evaluation_url": f"{REPO}/blob/release/shareable-prediction-api/models/training_metrics.json",
            "caveat": "Historical held-out score, not the error of a new forecast or proof of live-input accuracy.",
        },
        "uncertainty": {
            "kind": "p10_p90_interval",
            "response_fields": ["prediction_interval_p10_mwh", "prediction_interval_p50_mwh", "prediction_interval_p90_mwh"],
            "empirical_test_coverage": v1_test["interval_coverage_p10_p90"],
            "explanation": "Historical P10–P90 dispatch-down interval; it is not a guarantee for one prediction.",
        },
        "actuals_endpoint": "/actuals/v1",
        "limitations": [
            "No verified live V1 source collector is included in this API.",
            "The API's recoverable_surplus_mwh caps predicted dispatch-down by charging power only; a consuming backend must separately cap by available flexible demand.",
        ],
    }

    v2_test = v2_info["evaluation"]["test"] if v2_info is not None else None
    v2_dates = v2_info["evaluation"]["date_ranges"] if v2_info is not None else None
    v2_components = v2_info["prediction_components"] if v2_info is not None else None
    v2 = {
        "model_id": "v2",
        "available": v2_info is not None,
        "model_version": v2_info["model_version"] if v2_info is not None else None,
        "display_name": "V2 — experimental daily curtailment",
        "experimental": True,
        "plain_language_summary": (
            "Estimates whether any renewable curtailment will occur during a UTC day "
            "and the total curtailed energy for that whole day. It does not predict constraints."
        ),
        "target": {
            "name": "renewable curtailment",
            "interval_minutes": 1440,
            "forecast_horizons_minutes": [],
            "energy_unit": "MWh per complete UTC day",
            "prediction_field": "predicted_curtailment_mwh",
            "component_fields": ["predicted_curtailment_mwh"],
            "at_risk_formula": None,
            "interpretation": (
                "Daily curtailment only. V2 has no predicted constraint component, "
                "so do not apply V1's curtailment-plus-constraints formula or compare "
                "its daily MWh directly with V1's half-hour MWh."
            ),
        },
        "data_sources": [
            {"name": "EirGrid", "role": "Observed complete-day curtailment labels", "url": EIRGRID_REPORTS},
            {"name": "Open-Meteo Previous Runs", "role": "Archived GFS day-ahead regional weather forecasts", "url": OPEN_METEO_PREVIOUS_RUNS},
        ],
        "prediction_routes": [
            _route("/predict/curtailment/window", "historical_prediction", "Replays 1–7 days already in the bundled V2 model dataset."),
            _route("/predict/curtailment/day", "archived_day_ahead_forecast", "Fetches fixed-lead weather forecasts for a historical or current UTC day; it is not a live grid reading."),
            _route("/predict/curtailment/from-raw", "user_supplied_unverified", "Caller supplies 96 hourly forecasts; publication times cannot be independently verified."),
        ],
        "dataset_coverage": {
            "first_utc": min(bounds[0] for bounds in v2_dates.values())[:10] if v2_dates is not None else None,
            "last_utc": max(bounds[1] for bounds in v2_dates.values())[:10] if v2_dates is not None else None,
            "time_kind": "target_date_utc",
            "details_endpoint": "/dataset/daily-curtailment/coverage",
        },
        "estimators": {
            "event_classifier": v2_components["event_model"]["estimator"],
            "positive_day_amount_regressor": v2_components["amount_model"]["estimator"],
        } if v2_components is not None else {},
        "serving_policy": {
            "method": v2_components["amount_model"]["selected_method"],
            "ml_weight_by_horizon": None,
            "explanation": "The daily amount combines curtailment-event probability with a positive-day amount estimate.",
        } if v2_components is not None else None,
        "evaluation": {
            "test_mae_mwh": v2_test["daily_mae_mwh"],
            "metric_label": "Mean absolute error per full UTC day (MWh)",
            "test_period_start_utc": v2_dates["test"][0],
            "test_period_end_utc": v2_dates["test"][1],
            "test_rows": v2_test["rows"],
            "details_endpoint": "/model-info/daily-curtailment",
            "public_evaluation_url": f"{REPO}/blob/release/shareable-prediction-api/benchmarks/daily_curtailment_v2/evaluation.json",
            "caveat": "Historical held-out daily score; archived forecast publication times are inferred, not independently audited receipts.",
        } if v2_test is not None and v2_dates is not None else None,
        "uncertainty": {
            "kind": "event_probability_only",
            "response_fields": ["curtailment_event_probability"],
            "empirical_test_coverage": None,
            "explanation": "An event probability is returned; no calibrated MWh prediction interval is served for V2.",
        },
        "actuals_endpoint": "/actuals/daily-curtailment",
        "limitations": [
            "Experimental model; not a dispatch instruction or a verified live grid reading.",
            "The daily amount does not include a network-constraint estimate.",
        ],
    }
    return {
        "schema_version": "1.0",
        "models": [v1, v2],
        "integration_notes": {
            "same_target_horizons_are_alternatives": True,
            "data_mode_notice": "Historical replay, client-generated simulation, archived forecast and unverified raw inputs must be labelled separately; none is a verified live grid feed.",
            "data_mode_labels": {
                "historical_prediction": "Historical dataset prediction",
                "archived_day_ahead_forecast": "Archived day-ahead weather prediction",
                "user_supplied_unverified": "Prediction from unverified caller-supplied inputs",
                "simulated": "Simulated demo generated by the consuming app",
                "verified_live": "Verified live grid forecast — not currently offered by GridToEv",
            },
            "consumer_scenario_formula_url": "https://github.com/MarcoLadeira/SaveThePlanet/blob/main/backend/scenario.py",
            "consumer_scenario_notice": "EV-demand caps, charging-session equivalents, efficiency assumptions and feasible schedules belong to the consuming backend, not these model predictions.",
            "metric_comparison_notice": "V1 half-hour dispatch-down MAE and V2 full-day curtailment MAE have different targets and units; do not rank them by their numeric MWh scores.",
        },
    }
