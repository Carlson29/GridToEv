# Model formula API

GridToEv exposes the equations used to turn each model's fitted estimator outputs into an API prediction. These are **explanations of the deployed calculations**, not a second prediction interface. Use the prediction routes to obtain a forecast; use the formula routes to explain how its fields are produced.

| Model | Formula route | Predicted target |
| --- | --- | --- |
| V1 | `GET /model-info/v1/formulas` | Renewable dispatch-down in one target **half-hour**, issued 30 or 60 minutes earlier. |
| V2 | `GET /model-info/daily-curtailment/formulas` | Renewable **curtailment only** over one complete UTC day. |

Both routes take **no path parameters, query parameters or request body**. Send `X-API-Key` if the deployment requires it. V1 returns `200` when the model is loaded; V2 returns `503` if the optional daily model is disabled. The routes are discoverable through each model's `formulas_endpoint` in `GET /models/catalog` and `GET /models/about`. Keep the API key in your backend, not in browser JavaScript. The live API's base URL, once this change is deployed, is `https://gridtoev-api.onrender.com`.

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

`model_id`, `model_version`, `target` and `model_family` identify which deployed model the equations describe. `estimator_functions` maps symbolic calls such as `event_classifier.predict_proba(X)[1]` to the actual fitted estimator class. `estimator_equations` shows the **general nonlinear tree-ensemble form** (for example, a boosted-tree score passed through a sigmoid for event probability). It does **not** export every fitted tree split and leaf value. `symbols` defines the inputs and intermediate quantities. `fitted_parameters` contains the saved serving threshold, weights, adjustment and/or selected amount method. `steps` is an ordered list of `{id, expression, explanation}` for the calculations used by the API. `limitations` states what cannot be concluded from these equations. `single_linear_equation_available` is `false` for both models: neither has one globally valid linear-regression or logistic-regression coefficient equation.

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
