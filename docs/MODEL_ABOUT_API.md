# Backend model facts for an About page

`GET /models/about` is a read-only, API-key-protected JSON contract (`schema_version: "1.0"`) for a consuming backend. It returns exactly two entries, keyed by `model_id` (`v1`, `v2`), so the backend can populate a short How It Works page without parsing prose from the large `/model-info` responses. Keep `GRIDTOEV_API_KEY` on the backend; never expose it in browser code. `/models/catalog` points to this route through each model's `about_endpoint`.

For final API output calculations, use `GET /model-info/v1/formulas` and `GET /model-info/daily-curtailment/formulas` (linked as `formulas_endpoint` in `/models/about` and `/models/catalog`). Their `fitted_formulas_endpoint` links expose every loaded estimator's actual numeric tree rules, ordered engineered inputs, training target and predicted output. The manifest routes take **no request body**; per-tree routes take an estimator ID and zero-based tree index. All require the same `X-API-Key` header when configured. V2 routes return 503 when that optional model is disabled. A backend can display serving `steps[].expression` beside `steps[].explanation`, but these strings are explanations, not executable code. See [the formula API guide](MODEL_FORMULAS_API.md) for the exact fitted-rule contract.

For route-by-route usage, a backend example, exact V1/V2 output calculations and display caveats, see the [model formula API guide](MODEL_FORMULAS_API.md).

V1's classifier uses a boosted-tree score passed through a sigmoid to produce `dispatch_down_probability`; `dispatch_down_event_prediction` compares that probability with the saved threshold. The point regressor predicts a **change** from the latest observed dispatch-down, but the served total blends this ML estimate with a trend baseline using a saved, horizon-specific weight. The current loaded artifact has `w[30] = w[60] = 0`: the served point total comes from the trend baseline, even though the fitted point regressor still exists. Curtailment and constraint regressors provide raw component proportions, reconciled to the served total. Quantile regressors plus a saved adjustment give the P10–P90 interval. A flexible-load power cap limits `recoverable_surplus_mwh` for one half-hour.

V2's boosted-tree classifier yields `curtailment_event_probability`. An Extra Trees regressor averages its fitted trees to estimate a nonnegative daily amount. For the current `two_stage` artifact, **predicted daily curtailment MWh = event probability × positive-day amount estimate**. V2 does not return a thresholded event label, constraints, or a calibrated daily MWh interval. If a later artifact selects a direct amount method, the endpoint reports that loaded method and its corresponding amount equation.

Neither model is linear or logistic regression with a short list of global feature coefficients. The `estimator_equations` are the mathematical *form* of their tree ensembles; the fitted splits and leaf values live in the saved model artifacts. The `steps` describe the exact arithmetic applied to the estimator outputs in this deployment. They do not expose or authenticate raw feature provenance, and they do not convert historical test MAE into a per-prediction accuracy claim.

Example backend request:

```python
import json
import os
from urllib.request import Request, urlopen

base = os.environ["GRID_TO_EV_API_BASE_URL"].rstrip("/")
key = os.environ["GRID_TO_EV_API_KEY"]
request = Request(f"{base}/models/about", headers={"X-API-Key": key})
with urlopen(request, timeout=20) as response:
    facts = json.load(response)
models = {item["model_id"]: item for item in facts["models"]}
v1 = models["v1"]
v2 = models["v2"]
```

The response deliberately separates facts that must not be conflated:

| Field | Use in the About page |
| --- | --- |
| `display_name`, `plain_language_summary`, `target` | What the model predicts, energy unit, target interval and (for V1) 30/60-minute *lead* times. V2's 1,440-minute interval is a whole day, not a 1,440-minute lead claim. |
| `data_sources[].url` | Direct links to EirGrid, ENTSO-E and Open-Meteo documentation, with each source's role. |
| `prediction_routes[].data_mode`, `verified_live`, `integration_notes.data_mode_labels` | Label historical dataset predictions, archived weather predictions, unverified raw inputs and app-generated simulated demos separately. No current GridToEv route is a verified live grid feed. |
| `dataset_coverage` | Bounds of the bundled prediction dataset, **not** observed-outcome coverage or an assertion that today has complete inputs. |
| `evaluation.test_mae_mwh`, `metric_label`, `test_period_*`, `public_evaluation_url` | A measured historical test score with its correct time scale and a public source link. `evaluation` is `null` if optional V2 is disabled. Do not claim this is the error of the next prediction or compare V1 half-hour and V2 daily MWh scores numerically. |
| `estimators`, `serving_policy` | Fitted model names **and** how the served amount is formed. V1's final dispatch-down amount is a trend/ML blend; the saved weights can make the trend baseline the full served amount. V2 combines event probability with a positive-day amount estimate. |
| `formulas_endpoint` | Per-model equations, loaded parameters, symbols and plain-English calculation steps. |
| `uncertainty` | V1 serves P10/P50/P90 dispatch-down values and reports historical P10–P90 coverage. V2 serves an event probability but no calibrated MWh amount interval. |
| `actuals_endpoint` | Where to look up an observed outcome after it has been published. Actuals are separate from predictions and may be pending. |

For V1, `target.at_risk_formula` is `predicted_curtailment_mwh + predicted_constraint_mwh`. Its two components are reconciled to `predicted_dispatch_down_mwh` for **one target half-hour**. A +30 and +60 forecast of that same target are alternative vintages, not two amounts to add. V2 predicts **daily curtailment only**; `at_risk_formula` is therefore `null` because it provides no constraint estimate. Do not use V2's daily MWh as a V1 half-hour value or imply that its score is directly comparable.

GridToEv reports model predictions, not confirmed recovered charging energy. The consuming app's [scenario calculation](https://github.com/MarcoLadeira/SaveThePlanet/blob/main/backend/scenario.py) owns the flexible-demand cap, charging-session equivalents, 100% efficiency assumption, and any feasible optimizer result. In particular, GridToEv's V1 `recoverable_surplus_mwh` applies the *power-capacity* limit only; it does not know the consuming app's flexible EV demand. Present these results as potential upper bounds or illustrative equivalents, never measured charging or scheduled EVs.

See `/model-info` and `/model-info/daily-curtailment` for full fitted-estimator details and metrics. If V2 is not loaded, its entry remains present with `available: false`, `model_version: null`, empty `estimators` and `evaluation: null`; its conceptual target and source descriptions remain visible.
