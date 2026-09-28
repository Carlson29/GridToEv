"""Read-only, bounded export of the fitted numeric tree functions in loaded artifacts.

The manifest describes each estimator and its feature transform. A tree endpoint
returns one complete fitted piecewise function (thresholds and leaf values). The
existing /formulas routes describe the separate serving arithmetic around these
estimators. Nothing here retrains, alters, or substitutes for a model artifact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


V1_BASE = "/model-info/v1/fitted-formulas"
V2_BASE = "/model-info/daily-curtailment/fitted-formulas"

V1_TARGETS = {
    "event_classifier": (
        "dispatch_down_event: whether observed half-hour dispatch-down MWh exceeds zero",
        "dispatch_down_probability, a probability from 0 to 1",
        "probability",
        "Supplies the event probability and thresholded event label; it does not zero the MWh estimate.",
    ),
    "dispatch_down_regressor": (
        "dispatch_down_mwh minus the latest completed half-hour dispatch_down_mwh",
        "predicted change in total dispatch-down MWh",
        "MWh change",
        "Used in the candidate ML total; the saved horizon-specific weight determines its contribution to the served point total.",
    ),
    "curtailment_regressor": (
        "observed curtailment_mwh in the target half-hour",
        "raw curtailment MWh before nonnegative clipping and component reconciliation",
        "MWh",
        "After clipping, its ratio to the raw constraint estimate allocates the served total.",
    ),
    "constraint_regressor": (
        "observed constraint_mwh in the target half-hour",
        "raw constraint MWh before nonnegative clipping and component reconciliation",
        "MWh",
        "After clipping, its ratio to the raw curtailment estimate allocates the served total.",
    ),
    **{
        f"dispatch_down_quantile_{quantile}": (
            "dispatch_down_mwh minus the latest completed half-hour dispatch_down_mwh",
            f"predicted {quantile.upper()} change in total dispatch-down MWh",
            "MWh change",
            "Anchored to the latest observed MWh, then sorted with the other quantiles; the outer bounds receive a saved adjustment.",
        ) for quantile in ("p10", "p50", "p90")
    },
}


def _histogram_model(model: object) -> bool:
    return isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor))


def _inverse_link(model: object) -> str:
    if isinstance(model, ExtraTreesRegressor):
        return "identity"
    link = type(model._loss.link).__name__
    try:
        return {"LogitLink": "sigmoid", "LogLink": "exp", "IdentityLink": "identity"}[link]
    except KeyError as error:
        raise ValueError(f"Unsupported fitted link: {link}") from error


def _tree_count(model: object) -> int:
    if _histogram_model(model):
        if any(len(iteration) != 1 for iteration in model._predictors):
            raise ValueError("Only one-tree-per-iteration estimators can be exported")
        return len(model._predictors)
    if isinstance(model, ExtraTreesRegressor):
        return len(model.estimators_)
    raise ValueError(f"Unsupported fitted estimator: {type(model).__name__}")


def _describe_estimator(
    model: object, *, estimator_id: str, target: str,
    output: str, unit: str, serving_role: str, base: str,
    serving_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    count = _tree_count(model)
    link = _inverse_link(model)
    histogram = _histogram_model(model)
    aggregation = "sum" if histogram else "mean"
    baseline = float(model._baseline_prediction[0, 0]) if histogram else None
    if histogram:
        formula = f"raw_score(X) = {baseline:.17g} + sum(tree_k(X), k=0..{count - 1}); prediction(X) = {link}(raw_score(X))"
    else:
        formula = f"prediction(X) = sum(tree_k(X), k=0..{count - 1}) / {count}"
    return {
        "id": estimator_id,
        "estimator_class": type(model).__name__,
        "task": "classification" if isinstance(model, HistGradientBoostingClassifier) else "regression",
        "training_target": target,
        "prediction_output": output,
        "output_unit": unit,
        "serving_role": serving_role,
        "formula": formula,
        "baseline_raw_score": baseline,
        "aggregation": aggregation,
        "inverse_link": link,
        "tree_count": count,
        "tree_endpoint_template": f"{base}/{estimator_id}/trees/{{tree_index}}",
        "serving_weight_by_horizon": serving_weights,
    }


def _v1_parts(bundle: dict[str, Any], estimator_id: str) -> tuple[object, list[str]]:
    if estimator_id not in V1_TARGETS or estimator_id not in bundle["models"]:
        raise LookupError("Unknown V1 fitted estimator")
    pipeline = bundle["models"][estimator_id]
    if not isinstance(pipeline, Pipeline) or "imputer" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("V1 fitted estimator pipeline is incompatible with formula export")
    features = list(pipeline.named_steps["imputer"].get_feature_names_out(bundle["feature_columns"]))
    model = pipeline.named_steps["model"]
    if len(features) != model.n_features_in_:
        raise ValueError("V1 transformed feature order disagrees with fitted estimator")
    return model, features


def build_v1_fitted_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    columns = list(bundle["feature_columns"])
    first_pipeline = bundle["models"]["event_classifier"]
    imputer = first_pipeline.named_steps["imputer"]
    transformed = list(imputer.get_feature_names_out(columns))
    input_features = [
        {
            "name": name,
            "position": index,
            "missing_fill_value": float(imputer.statistics_[index]) if math.isfinite(float(imputer.statistics_[index])) else None,
        }
        for index, name in enumerate(columns)
    ]
    estimators = []
    point_weights = bundle["metadata"]["dispatch_regression_ml_weight_by_horizon"]
    for estimator_id, (target, output, unit, role) in V1_TARGETS.items():
        model, fitted_features = _v1_parts(bundle, estimator_id)
        if fitted_features != transformed:
            raise ValueError("V1 estimators do not share the declared transformed feature order")
        fitted_statistics = bundle["models"][estimator_id].named_steps["imputer"].statistics_
        if not np.array_equal(fitted_statistics, imputer.statistics_, equal_nan=True):
            raise ValueError("V1 estimators do not share the declared fitted imputation medians")
        weights = point_weights if estimator_id == "dispatch_down_regressor" else None
        estimators.append(_describe_estimator(
            model, estimator_id=estimator_id,
            target=target, output=output, unit=unit, serving_role=role,
            base=V1_BASE, serving_weights=weights,
        ))
    return {
        "schema_version": "1.0",
        "model_id": "v1",
        "model_version": bundle["metadata"]["model_version"],
        "target": "EirGrid dispatch-down event, curtailment and constraint in one target half-hour, at a 30- or 60-minute lead",
        "input_kind": "Numeric engineered features available at the issue timestamp, not raw source rows or the future target",
        "input_features": input_features,
        "transformed_features": transformed,
        "preprocessing": "In saved column order, SimpleImputer replaces missing engineered values with the listed training medians. Its fitted missing-indicator columns, if any, appear in transformed_features. Tree feature indices refer to transformed_features, not raw source fields.",
        "estimators": estimators,
        "serving_formulas_endpoint": "/model-info/v1/formulas",
        "raw_input_schema_endpoint": "/model-info/v1/raw-input-schema",
        "interpretation": (
            "Each tree is a complete numeric piecewise function. Sum its selected leaf values "
            "with baseline_raw_score, then apply inverse_link. Use /formulas for clipping, "
            "trend blending, component allocation and uncertainty calculations. "
            f"The loaded point-regressor serving weights are {point_weights}; "
            "they do not control the event, component or quantile estimators."
        ),
        "limitation": "These models predict observed dispatch-down; they do not estimate extra or avoided curtailment caused by EV charging.",
    }


def build_v2_fitted_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    columns = list(bundle["metadata"]["feature_columns"])
    method = bundle["metadata"]["selected_amount_method"]
    amount_target = (
        "observed full-day curtailment_mwh on positive-curtailment days only"
        if method == "two_stage" else "observed full-day curtailment_mwh on all development days"
    )
    amount_model = bundle["amount_model"]
    amount_tree_count = _tree_count(amount_model)
    event_role = (
        "Multiplies the nonnegative amount estimate in the selected two-stage serving method."
        if method == "two_stage" else "Reported as an event probability; the selected direct amount method does not multiply by it."
    )
    return {
        "schema_version": "1.0",
        "model_id": "v2",
        "model_version": bundle["metadata"]["model_version"],
        "target": "EirGrid renewable curtailment event and total curtailment MWh over one complete UTC day",
        "input_kind": "Numeric engineered day-ahead regional weather forecasts and calendar features for the target UTC day",
        "input_features": [{"name": name, "position": index, "missing_fill_value": None} for index, name in enumerate(columns)],
        "transformed_features": columns,
        "preprocessing": "No fitted imputer or scaling. The API rejects non-finite engineered weather features. Tree feature indices refer to input_features in this exact saved order.",
        "estimators": [
            _describe_estimator(
                bundle["event_model"], estimator_id="event_model",
                target="curtailment_event: whether observed full-day curtailment_mwh exceeds zero",
                output="curtailment_event_probability, a probability from 0 to 1",
                unit="probability", serving_role=event_role, base=V2_BASE,
            ),
            _describe_estimator(
                amount_model, estimator_id="amount_model",
                target=amount_target, output="amount estimate before nonnegative clipping and optional event-probability multiplication",
                unit="MWh", serving_role=f"Selected {method} amount estimator for the daily curtailment result.", base=V2_BASE,
            ),
        ],
        "serving_formulas_endpoint": "/model-info/daily-curtailment/formulas",
        "raw_input_schema_endpoint": "/model-info/daily-curtailment/raw-input-schema",
        "interpretation": (
            "Each tree is a complete numeric piecewise function. The classifier sums leaf contributions "
            "with its baseline and applies sigmoid. The amount model combines its "
            f"{amount_tree_count} trees using its listed aggregation and inverse link. "
            "Use /formulas for the selected final daily MWh calculation."
        ),
        "limitation": "This is observed daily curtailment, not extra or avoided curtailment caused by a proposed charging intervention.",
    }


def _histogram_nodes(model: object, tree_index: int, features: list[str]) -> list[dict[str, Any]]:
    predictor = model._predictors[tree_index][0]
    nodes = []
    for node_id, node in enumerate(predictor.nodes):
        leaf = bool(node["is_leaf"])
        if not leaf and bool(node["is_categorical"]):
            raise ValueError("Categorical histogram splits require a different export contract")
        feature_index = None if leaf else int(node["feature_idx"])
        nodes.append({
            "node_id": node_id,
            "is_leaf": leaf,
            "value": float(node["value"]) if leaf else None,
            "sample_count": int(node["count"]),
            "feature_index": feature_index,
            "feature_name": features[feature_index] if feature_index is not None else None,
            "threshold": float(node["num_threshold"]) if not leaf else None,
            "left_child": int(node["left"]) if not leaf else None,
            "right_child": int(node["right"]) if not leaf else None,
            "missing_go_to_left": bool(node["missing_go_to_left"]) if not leaf else None,
        })
    return nodes


def _extra_tree_nodes(model: ExtraTreesRegressor, tree_index: int, features: list[str]) -> list[dict[str, Any]]:
    tree = model.estimators_[tree_index].tree_
    nodes = []
    for node_id in range(tree.node_count):
        leaf = int(tree.children_left[node_id]) == -1
        feature_index = None if leaf else int(tree.feature[node_id])
        nodes.append({
            "node_id": node_id,
            "is_leaf": leaf,
            "value": float(tree.value[node_id, 0, 0]) if leaf else None,
            "sample_count": int(tree.n_node_samples[node_id]),
            "feature_index": feature_index,
            "feature_name": features[feature_index] if feature_index is not None else None,
            "threshold": float(tree.threshold[node_id]) if not leaf else None,
            "left_child": int(tree.children_left[node_id]) if not leaf else None,
            "right_child": int(tree.children_right[node_id]) if not leaf else None,
            "missing_go_to_left": bool(tree.missing_go_to_left[node_id]) if not leaf else None,
        })
    return nodes


def _export_tree(model: object, *, model_id: str, version: str, estimator_id: str,
                 tree_index: int, features: list[str]) -> dict[str, Any]:
    count = _tree_count(model)
    if tree_index < 0 or tree_index >= count:
        raise LookupError("Tree index is outside the fitted estimator")
    histogram = _histogram_model(model)
    nodes = _histogram_nodes(model, tree_index, features) if histogram else _extra_tree_nodes(model, tree_index, features)
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_version": version,
        "estimator_id": estimator_id,
        "tree_index": tree_index,
        "tree_count": count,
        "value_meaning": "Additive raw-score contribution; sum with the other trees and baseline before the inverse link." if histogram else "One tree's MWh estimate; average over every tree for the fitted amount model.",
        "branch_rule": "At each non-leaf node, follow left_child if transformed_features[feature_index] <= threshold, otherwise right_child. For a missing value, follow left_child only when missing_go_to_left is true. A leaf's value is the tree output.",
        "nodes": nodes,
    }


def export_v1_tree(bundle: dict[str, Any], estimator_id: str, tree_index: int) -> dict[str, Any]:
    model, features = _v1_parts(bundle, estimator_id)
    return _export_tree(model, model_id="v1", version=bundle["metadata"]["model_version"],
                        estimator_id=estimator_id, tree_index=tree_index, features=features)


def export_v2_tree(bundle: dict[str, Any], estimator_id: str, tree_index: int) -> dict[str, Any]:
    if estimator_id not in {"event_model", "amount_model"}:
        raise LookupError("Unknown V2 fitted estimator")
    model = bundle[estimator_id]
    features = list(bundle["metadata"]["feature_columns"])
    if len(features) != model.n_features_in_:
        raise ValueError("V2 feature order disagrees with fitted estimator")
    return _export_tree(model, model_id="v2", version=bundle["metadata"]["model_version"],
                        estimator_id=estimator_id, tree_index=tree_index, features=features)
