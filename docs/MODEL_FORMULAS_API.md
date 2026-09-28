# Model formula API

GridToEv exposes both the **actual fitted ML decision rules** and the equations used to turn their outputs into an API prediction. These are read-only explanations of the loaded artifact, not a second prediction interface. Use prediction routes to obtain a forecast. Neither model predicts an EV intervention's *extra* or *avoided* curtailment; that would require a separately validated counterfactual model.

| Model | Serving formula | Fitted ML formula manifest | Predicted target |
| --- | --- | --- | --- |
| V1 | `GET /model-info/v1/formulas` | `GET /model-info/v1/fitted-formulas` | Renewable dispatch-down in one target **half-hour**, issued 30 or 60 minutes earlier. |
| V2 | `GET /model-info/daily-curtailment/formulas` | `GET /model-info/daily-curtailment/fitted-formulas` | Renewable **curtailment only** over one complete UTC day. |

The four manifest/serving routes take **no path parameters, query parameters or request body**. Send `X-API-Key` if the deployment requires it. V1 returns `200` when loaded; V2 returns `503` if the optional daily model is disabled. All are linked in `GET /models/catalog` and `GET /models/about`. Keep the API key in your backend, not in browser JavaScript. The live API's base URL, once this change is deployed, is `https://gridtoev-api.onrender.com`.

## Actual fitted ML formulas: variables, targets and numeric trees

The `/fitted-formulas` manifests identify **every loaded classifier and regressor**. `input_features` lists the exact engineered variables and saved positions. `transformed_features` is the array addressed by each tree's `feature_index`; V1 first fills missing values with the listed fitted medians, while V2 rejects non-finite inputs. These are *not* the original EirGrid/ENTSO-E or hourly weather fields: use `raw_input_schema_endpoint` to understand the source inputs. For each estimator, `training_target` says what observed label it learned, `prediction_output` and `output_unit` say what its result means, `serving_role` says whether/how that result affects the API prediction, and `formula` gives the **actual loaded baseline, tree count, aggregation and inverse link**.

One complete fitted tree is returned by the estimator's `tree_endpoint_template`, for example:

```text
GET /model-info/v1/fitted-formulas/curtailment_regressor/trees/0
GET /model-info/daily-curtailment/fitted-formulas/amount_model/trees/0
```

Replace `0` with any zero-based index below that estimator's `tree_count`. A node with `is_leaf=false` tests `transformed_features[feature_index] <= threshold`: follow `left_child` if true, otherwise `right_child`. `missing_go_to_left` states the fitted missing-value direction (V1 normally imputes first; V2 rejects missing inputs). At a leaf, `value` is the exact fitted numeric output for that tree. Repeat for every tree in the estimator; the manifest's `formula` specifies how to combine them. A separate tree endpoint bounds response size and keeps the OpenAPI page usable. Unknown estimator or index returns `404`; disabled V2 returns `503`. The tree endpoints are read-only and require the same optional API key.

The loaded V1 event classifier has a fitted raw baseline and additive boosted trees:

```text
raw_event_score(X) = baseline + sum(event_tree_k(X))
dispatch_down_probability = sigmoid(raw_event_score(X))
```

The V1 point and quantile regressors use `baseline + sum(tree_k(X))` with the **identity** link; their target is a *change from the latest completed observation*, not absolute future MWh. The V1 curtailment and constraint regressors use the **Poisson log link**, which is important:

```text
raw_component_score(X) = baseline + sum(component_tree_k(X))
raw_component_mwh = exp(raw_component_score(X))
```

The loaded V2 event classifier likewise applies `sigmoid(baseline + sum(tree_k(X)))`. Its selected positive-day Extra Trees amount regressor averages **240 fitted trees**:

```text
conditional_amount_mwh(X) = sum(amount_tree_k(X) for k=0..239) / 240
predicted_daily_curtailment_mwh = event_probability(X) * max(conditional_amount_mwh(X), 0)
```

The last multiplication is serving arithmetic, also described by `/model-info/daily-curtailment/formulas`. If a later artifact selects the direct amount method, read its manifest and serving formula instead of assuming this two-stage equation. These rules are predictive associations, **not a causal formula for additional curtailment caused by EV demand**. Exporting the trees does not change or retrain either model.

For example, a backend can request the metadata as follows:

```python
import json
import os
from urllib.request import Request, urlopen

base = os.environ["GRID_TO_EV_API_BASE_URL"].rstrip("/")
key = os.environ["GRID_TO_EV_API_KEY"]

def get_model_formulas(model_id: str) -> dict:
    paths = {
        "v1": "/model-info/v1/formulas",
        "v2": "/model-info/daily-curtailment/formulas",
    }
    request = Request(base + paths[model_id], headers={"X-API-Key": key})
    with urlopen(request, timeout=20) as response:
        return json.load(response)

v1_formulas = get_model_formulas("v1")
for step in v1_formulas["steps"]:
    print(step["id"], step["expression"], step["explanation"])
```

## How to read the response

`model_id`, `model_version`, `target` and `model_family` identify which loaded model the *serving* equations describe. `estimator_functions` maps symbolic calls such as `event_classifier.predict_proba(X)[1]` to the fitted class. `estimator_equations` shows the link/ensemble form; `fitted_formulas_endpoint` links to the **numeric fitted trees** above. `symbols` defines inputs and intermediate quantities. `fitted_parameters` contains saved serving thresholds, weights, adjustments and/or selected method. `steps` is an ordered list of `{id, expression, explanation}` for calculations used by the API. `single_linear_equation_available` is `false` for both models: neither has one globally valid feature-coefficient equation.

The expressions are **human-readable, Python-like notation**, not executable snippets. `X` is already engineered into the feature order expected by the fitted artifact. To submit original source values, use the relevant `/raw-input-schema` and `/from-raw` routes; do not send `X` to a formula route. Read `model_version` and `fitted_parameters` at runtime rather than copying today's values into a dashboard: future retraining can change them.

## V1: classification and half-hour amount

V1 has separate event classification and energy estimation. In shortened notation (`h` is 30 or 60 minutes):

```text
p = event_classifier.predict_proba(X)[1]
dispatch_down_event_prediction = (p >= classification_threshold)
trend = max(latest_observed + alpha[h] * (latest_observed - previous_observed), 0)
ml_total = max(latest_observed + dispatch_down_regressor.predict(X), 0)
predicted_dispatch_down_mwh = max(w[h] * ml_total + (1 - w[h]) * trend, 0)
```

`latest_observed` and `previous_observed` are **past completed half-hour** dispatch-down observations, in MWh. The point regressor predicts a *change* from the latest observation. `alpha[h]` and `w[h]` are saved separately for the 30- and 60-minute horizons. The current artifact has `w[30] = 0` and `w[60] = 0`, so the served **point total** is the trend estimate, not the point regressor's estimate. This does **not** mean all V1 machine-learning models are unused: its event classifier, component regressors and quantile regressors still supply their respective outputs.

The current `classification_threshold` is `0.1`. This event yes/no result is separate from the probability display band: `risk_level` is low below `0.4`, medium from `0.4` to below `0.7`, and high from `0.7` upward. A positive MWh estimate can coexist with a `false` event label because the classifier does not zero the amount.

The curtailment and network-constraint regressors each produce a nonnegative *raw* component. If their sum is positive, the API divides the served total in that ratio; if both are zero, it uses the saved default curtailment share. Thus:

```text
predicted_curtailment_mwh + predicted_constraint_mwh
    = predicted_dispatch_down_mwh
recoverable_surplus_mwh
    = min(predicted_dispatch_down_mwh, flexible_load_capacity_mw * 0.5)
```

The `0.5` converts MW of flexible-load capacity to MWh over one half-hour. This is a **power-cap upper bound**, not confirmed recoverable EV energy: it does not know actual flexible demand or charging efficiency. The P10/P50/P90 fields come from three quantile regressors predicting changes from the latest observation; the values are sorted, and the saved interval adjustment widens the outer bounds. For the complete piecewise reconciliation and interval equations, use `steps` in the endpoint response.

## V2: classification and daily amount

V2's event classifier returns the probability that **any curtailment occurs during the target UTC day**. Its current `two_stage` amount regressor is trained on curtailment days and estimates a nonnegative full-day amount. The served calculation is:

```text
curtailment_event_probability = event_model.predict_proba(X)[1]
amount_estimate_mwh = max(amount_model.predict(X), 0)
predicted_curtailment_mwh = curtailment_event_probability * amount_estimate_mwh
```

For example, probability `0.4` and amount estimate `100 MWh` would yield `40 MWh`; those numbers are **illustrative, not a model prediction**. The currently fitted amount estimator is Extra Trees, which averages its trees. If a later artifact selects the `direct_median` method, the endpoint will show `predicted_curtailment_mwh = amount_estimate_mwh` instead and identify the loaded amount estimator. V2 does **not** return a thresholded event yes/no value, a network-constraint estimate or a calibrated daily MWh prediction interval.

Do not compare the numerical MWh of V1 and V2 predictions or their MAE scores directly: they predict different targets over different durations. A formula response neither fetches live data nor proves the provenance of caller-supplied inputs. Historical evaluation metrics on `/model-info` describe past test data, not the error of an individual prediction. For target definitions, sources, route data modes, test coverage and actuals lookup, see [the About-page API guide](MODEL_ABOUT_API.md).
