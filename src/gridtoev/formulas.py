"""Human- and machine-readable equations for the *served* prediction outputs.

Tree ensembles cannot be reduced to a single coefficient equation. These steps
describe the exact post-estimator arithmetic in inference.py and daily_model.py;
fitted parameter values come from the loaded model, never from this module.
"""

from __future__ import annotations

from typing import Any

from .constants import INTERVAL_HOURS


def _step(step_id: str, expression: str, explanation: str) -> dict[str, str]:
    return {"id": step_id, "expression": expression, "explanation": explanation}


def v1_formulas(info: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    components = info["prediction_components"]
    return {
        "model_id": "v1",
        "model_version": info["model_version"],
        "target": "Renewable dispatch-down MWh in one target half-hour, 30 or 60 minutes after issue",
        "model_family": "Histogram gradient-boosted classifier and regressors, plus fitted trend/ML serving policy",
        "single_linear_equation_available": False,
        "estimator_functions": {
            "event_classifier.predict_proba(X)[1]": components["event_classifier"]["estimator"],
            "dispatch_down_regressor.predict(X)": components["dispatch_down_regressor"]["estimator"],
            "curtailment_regressor.predict(X)": components["curtailment_regressor"]["estimator"],
            "constraint_regressor.predict(X)": components["constraint_regressor"]["estimator"],
            **{
                f"dispatch_down_quantile_{q}.predict(X)": components[f"dispatch_down_quantile_{q}"]["estimator"]
                for q in ("p10", "p50", "p90")
            },
        },
        "estimator_equations": {
            "event_classifier": "p(X) = 1 / (1 + exp(-F(X))); F(X) = initial_log_odds + sum(fitted_tree_contribution_k(X))",
            "point_and_component_regressors": "y_hat(X) = initial_prediction + sum(fitted_tree_contribution_k(X))",
            "quantile_regressors": "q_hat(X) = initial_quantile_prediction + sum(fitted_tree_contribution_k(X))",
        },
        "symbols": {
            "X": "One engineered feature row known at issue time, in the saved feature-column order.",
            "h": "Forecast horizon, 30 or 60 minutes.",
            "latest_observed": "max(dispatch_down_mwh_latest_observed, 0), MWh from a completed past half-hour.",
            "previous_observed": "max(dispatch_down_mwh_lag_1, 0), MWh from the preceding completed half-hour.",
            "alpha[h]": "Fitted dispatch_trend_alpha_by_horizon for the selected horizon.",
            "w[h]": "Fitted dispatch_regression_ml_weight_by_horizon for the selected horizon.",
            "flexible_load_capacity_mw": "Caller-supplied flexible charging power capacity in MW.",
        },
        "fitted_parameters": {
            "classification_threshold": metadata["classification_threshold"],
            "dispatch_trend_alpha_by_horizon": metadata["dispatch_trend_alpha_by_horizon"],
            "dispatch_regression_ml_weight_by_horizon": metadata["dispatch_regression_ml_weight_by_horizon"],
            "default_component_shares": metadata["default_component_shares"],
            "prediction_interval_adjustment_mwh": metadata["prediction_interval_adjustment_mwh"],
            "target_interval_hours": INTERVAL_HOURS,
            "risk_level_cutoffs": {"medium": 0.4, "high": 0.7},
        },
        "steps": [
            _step("dispatch_down_probability", "dispatch_down_probability = event_classifier.predict_proba(X)[1]", "Classifier probability of any dispatch-down in the target half-hour."),
            _step("dispatch_down_event_prediction", "dispatch_down_event_prediction = dispatch_down_probability >= classification_threshold", "Boolean classification at the saved threshold; it does not zero the MWh estimate."),
            _step("risk_level", "risk_level = 'high' if dispatch_down_probability >= 0.70 else 'medium' if dispatch_down_probability >= 0.40 else 'low'", "Display band based on probability, distinct from the event classification threshold."),
            _step("trend_baseline", "trend_baseline = max(latest_observed + alpha[h] * (latest_observed - previous_observed), 0)", "Extrapolates the change between two completed past half-hours; h chooses the fitted alpha."),
            _step("ml_total", "ml_total = max(latest_observed + dispatch_down_regressor.predict(X), 0)", "The regressor predicts a change, not the final absolute MWh."),
            _step("predicted_dispatch_down_mwh", "predicted_dispatch_down_mwh = max(w[h] * ml_total + (1 - w[h]) * trend_baseline, 0)", "Final served total. A zero ML weight means only the trend baseline contributes to this amount."),
            _step("raw_components", "raw_curtailment = max(curtailment_regressor.predict(X), 0); raw_constraint = max(constraint_regressor.predict(X), 0)", "Independent nonnegative component estimates before reconciliation."),
            _step("predicted_curtailment_mwh", "if predicted_dispatch_down_mwh <= 0: predicted_curtailment_mwh = 0; elif raw_curtailment + raw_constraint > 0: predicted_curtailment_mwh = predicted_dispatch_down_mwh * raw_curtailment / (raw_curtailment + raw_constraint); else: predicted_curtailment_mwh = predicted_dispatch_down_mwh * default_component_shares['curtailment']", "Scales the raw components to the served total; uses the fitted default share only when both raw components are zero."),
            _step("predicted_constraint_mwh", "predicted_constraint_mwh = predicted_dispatch_down_mwh - predicted_curtailment_mwh", "The two served components sum to the served dispatch-down total."),
            _step("prediction_interval", "q = sorted([max(latest_observed + dispatch_down_quantile_p10.predict(X), 0), max(latest_observed + dispatch_down_quantile_p50.predict(X), 0), max(latest_observed + dispatch_down_quantile_p90.predict(X), 0)]); p10 = max(q[0] - prediction_interval_adjustment_mwh, 0); p50 = q[1]; p90 = q[2] + prediction_interval_adjustment_mwh", "Sorted quantile-model change forecasts, anchored to the latest past observation; saved interval adjustment widens the outer bounds."),
            _step("recoverable_surplus_mwh", f"recoverable_surplus_mwh = min(predicted_dispatch_down_mwh, flexible_load_capacity_mw * {INTERVAL_HOURS})", "Upper bound from half-hour power capacity only; actual flexible demand is not checked here."),
        ],
        "limitations": [
            "The fitted tree splits and leaf values are stored in the trusted model artifact; they have no short global regression coefficients or logistic formula.",
            "The tree-ensemble equations describe model structure, not exported tree splits, leaf values or an independent reimplementation of the saved estimators.",
            "These equations describe outputs after feature engineering, not how to reconstruct X from raw source observations.",
            "Scores on /model-info are historical test metrics, not exact error bounds for an individual result.",
        ],
    }


def v2_formulas(info: dict[str, Any]) -> dict[str, Any]:
    components = info["prediction_components"]
    method = components["amount_model"]["selected_method"]
    amount_estimator = components["amount_model"]["estimator"]
    amount_model_equation = (
        "amount_model.predict(X) = (1 / N) * sum(tree_k.predict(X) for k in 1..N)"
        if amount_estimator == "ExtraTreesRegressor" else
        "amount_model.predict(X) = initial_prediction + sum(fitted_tree_contribution_k(X))"
    )
    amount_expression = (
        "predicted_curtailment_mwh = curtailment_event_probability * amount_estimate_mwh"
        if method == "two_stage" else
        "predicted_curtailment_mwh = amount_estimate_mwh"
    )
    return {
        "model_id": "v2",
        "model_version": info["model_version"],
        "target": "Renewable curtailment MWh over one complete UTC day",
        "model_family": f"{components['event_model']['estimator']} event classifier and {components['amount_model']['estimator']} amount regressor",
        "single_linear_equation_available": False,
        "estimator_functions": {
            "event_model.predict_proba(X)[1]": components["event_model"]["estimator"],
            "amount_model.predict(X)": components["amount_model"]["estimator"],
        },
        "estimator_equations": {
            "event_classifier": "p(X) = 1 / (1 + exp(-F(X))); F(X) = initial_log_odds + sum(fitted_tree_contribution_k(X))",
            "amount_regressor": amount_model_equation,
        },
        "symbols": {
            "X": "One engineered day-ahead weather forecast feature row for the target UTC day.",
            "curtailment_event_probability": "Classifier probability of any curtailment during that day, from 0 to 1.",
            "amount_estimate_mwh": (
                "Nonnegative regressor prediction in MWh for the complete UTC day; trained on curtailment days only."
                if method == "two_stage" else
                "Nonnegative regressor prediction in MWh for the complete UTC day; trained on all days."
            ),
        },
        "fitted_parameters": {"selected_amount_method": method},
        "steps": [
            _step("curtailment_event_probability", "curtailment_event_probability = event_model.predict_proba(X)[1]", "Probability that curtailment occurs on the target day. The API does not serve a thresholded yes/no label for V2."),
            _step("amount_estimate_mwh", "amount_estimate_mwh = max(amount_model.predict(X), 0)", "Nonnegative daily amount estimate from the fitted regressor."),
            _step("predicted_curtailment_mwh", amount_expression, "The selected serving method combines event probability and the amount estimate." if method == "two_stage" else "The selected serving method uses the nonnegative amount estimate directly."),
        ],
        "limitations": [
            "These nonlinear tree ensembles have no short global regression coefficients or logistic formula; fitted splits and leaves are in the trusted model artifact.",
            "This predicts daily curtailment only, not constraints or half-hour dispatch-down.",
            "No calibrated daily MWh prediction interval or thresholded event label is served.",
        ],
    }
