# Handoff: predict curtailment by renewable source

**Status (updated 2026-09-28):** Step 2 source labels are built and reconciled. Steps 3-5 were run for
the daily V2 model: **the release gate failed**, so no source-prediction route, serving artifact, or
V1 half-hour allocator was added. See [Results](#results-2026-09-28) at the end of this page. The
original plan below is kept unchanged for the next attempt.

## Goal and meaning

Given an existing prediction of **renewable curtailment MWh**, return a defensible breakdown such as `wind_curtailment_mwh` and `solar_curtailment_mwh`. These parts must be non-negative and sum to the parent prediction (within rounding). This is a forecast of **which renewable technology is likely to be curtailed**, not proof of which generator physically powered an EV. The grid is a delivery network, not a fuel category. Gas, coal, oil, and peat generation are possible context features, not components of renewable curtailment.

Keep the parent target straight:

- V1's `predicted_dispatch_down_mwh` includes curtailment **and** network constraints. Allocate V1's `predicted_curtailment_mwh` when the route says *curtailment by source*. A later, separately named feature could allocate all dispatch-down by technology, but must not mix the two totals.
- V2's `predicted_curtailment_mwh` covers one complete UTC day. A V2 source breakdown must sum to that daily prediction, not to a half-hour V1 result.
- `recoverable_surplus_mwh` is a capacity-capped potential, not delivered EV energy or a source attribution.

## What is verified already

1. `src/gridtoev/eirgrid_history.py::normalise_dispatch_frame` reads `UT_TYPE`, `JURISDICTION`, `HH_TIMESTAMP`, `GMT_OFFSET`, and `Sum of CURTAILMENTS_MWH` from EirGrid's `DD HH` worksheet. It keeps `technology_type` in the normalized rows. `build_core_history` then filters to `IE` and **sums across technology types**, so `data/processed/eirgrid_core_history_30min.csv.gz` retains the total but not the technology split. Do not infer source labels from EirGrid wind/solar generation or availability columns.
2. The official half-hour workbooks are already catalogued in `data/processed/extended_source_manifest.csv` and downloaded by `scripts/build_extended_data.py`. No new provider is required for the first label audit. Raw downloads are git-ignored; preserve the source URL, retrieval time, and SHA-256 in the manifest instead of committing Excel files.
3. A read-only inspection of the EirGrid `DD HH` worksheets on 2026-09-28 found these **Ireland (`IE`) technology row counts**, not yet a complete modelling-quality audit:

   | Workbook year | IE Wind rows | IE Solar rows | Interpretation to verify |
   | --- | ---: | ---: | --- |
   | 2021 | 17,520 | absent | Do not silently call absent solar rows zero. |
   | 2022 | 17,520 | absent | Same caution. |
   | 2023 | 17,520 | 13,202 | Solar coverage starts partway through the year. |
   | 2024 | 17,568 | 17,568 | Candidate complete wind/solar year. |
   | 2025 | 17,520 | 17,520 | Candidate complete wind/solar year. |
   | 2026 V9 | 11,662 | 11,662 | Partial-year archive, ending at the manifest's latest interval. |

   In the inspected 2026 IE rows, `Sum of DD_MWH`, `Sum of CURTAILMENTS_MWH`, and `Sum of CONSTRAINTS_MWH` had no nulls or negatives. The largest absolute row-level `DD - curtailment - constraint` difference was 0.03 MWh, below the pipeline's 0.05 MWh accounting tolerance. These checks do **not** replace a timestamp-level, multi-year reconciliation.
4. V1 currently emits total/curtailment/constraint predictions in `src/gridtoev/inference.py`; V2 emits daily curtailment in `src/gridtoev/daily_model.py`. The current processed targets and public actuals endpoints contain totals, not per-technology outcomes. V1's original combined model-ready table covers January 2026; V2's archived forecast-feature table covers a longer but different period. Check the current data coverage before selecting training windows.

Official sources: [EirGrid system and renewable reports](https://www.eirgrid.ie/grid/system-and-renewable-data-reports), [2026 DD half-hour workbook](https://cms.eirgrid.ie/sites/default/files/publications/DD-HH-2026-V9.xlsx), and [2024 annual report's source-category breakdown](https://cms.eirgrid.ie/sites/default/files/publications/Annual-Renewable-Constraint-and-Curtailment-Report-2024-V1.0.pdf). The annual monthly tables support the category definitions, but **monthly shares are not half-hour training labels**.

## Implementation sequence for the next agent

### 1. Freeze the prediction contract before modelling

The final feature needs **one new prediction endpoint for each model**: one for V1's 30/60-minute curtailment-by-source prediction and one for V2's daily curtailment-by-source prediction. Recommended order: build and validate source labels for both grains, prototype **daily V2 first** because it has a longer solar-era feature window, then attempt V1 only if matched issue-time rows and held-out skill suffice. Do not change either parent's fitted artifact. **Do not edit, replace, wrap, or extend any existing endpoint, request schema, response schema, default, validation rule, or route documentation.** Keep all existing V1/V2 prediction and actuals routes working exactly as before. Add the two new routes only after their respective evidence gates pass; an unvalidated route must stay unavailable.

For each output, define `jurisdiction=IE`, `target_timestamp_utc` or `target_date_utc`, `target_kind=curtailment`, `unit=MWh`, `source=Wind|Solar`, `model_version`, `label_coverage`, and whether the result is `predicted`, `observed`, or `unavailable`. Reserve an `other_or_unattributed_mwh` field only if the audited source taxonomy actually requires it. Never relabel imported electricity as a generator type.

### 2. Preserve and audit technology-level actuals

Extend the EirGrid processing path **before** `build_core_history` discards `technology_type`. Create a separate, deterministic, UTC half-hour table keyed by `(timestamp_utc, jurisdiction, technology_type)` with wind/solar `curtailment_mwh`, optionally constraint and total dispatch-down for diagnostics. Reuse the existing GMT-offset normalization and source manifests. Keep unreported technology intervals as *missing/unknown*, not automatic zero; a reported numeric zero remains zero. Reject unknown categories, duplicate keys, negative/non-finite labels, or accounting failures rather than silently coercing them.

For every complete IE timestamp, test that `wind_curtailment_mwh + solar_curtailment_mwh` equals the existing IE `curtailment_mwh` to the documented source precision/tolerance. Investigate any residual before training. Repeat for each year and document first/last valid timestamp, missing intervals, technology spelling changes, daylight-saving behavior, and whether the source workbook was revised. Check that the 2023 partial solar coverage is not mistaken for a low-solar training regime. Save a compact quality report and data dictionary; do not commit raw workbooks or personal credentials.

For daily labels, sum **48 consecutive complete UTC half-hours** per target day, matching `src/gridtoev/daily_curtailment.py`'s complete-day logic. Do not sum partial days, mix IE with NI, or use the daily outcome as a prediction input. Reconcile daily wind + solar to the existing V2 daily curtailment label.

### 3. Build leakage-safe training rows

Join a target interval/day's source labels to the **same issue-time features** that its parent model would have seen. For V1, use the existing 30/60-minute target shift and forecast-vintage/as-of rules; for V2, use the archived day-ahead weather forecast convention. Never feed target-time generation, target-time availability, the actual total curtailment, future-published forecasts, or any source curtailment label into a predictive feature. Actual total may be used **only** to construct a training share label and to calculate historical evaluation metrics; serving uses the parent's *predicted* total.

Start from 2024 onward for an unambiguous IE wind/solar label window unless the audit proves earlier solar absences truly mean zero. Report matched row counts and excluded dates for each horizon and model. Keep chronological train/validation/test partitions, purge any overlapping V1 target intervals, and leave the final test period untouched until model selection is fixed. Fit preprocessing only on training data.

### 4. Train a small allocation model without replacing V1/V2

Recommended first candidate: on rows where actual total curtailment is positive, learn the **wind share** `wind_actual / (wind_actual + solar_actual)` from issue-time features; solar share is the remainder. At prediction time set both source amounts to zero when the parent predicts zero, otherwise multiply the parent's predicted curtailment MWh by bounded shares. This guarantees non-negative, conserved amounts. Compare it with (a) a training-only seasonal/hour-of-day share baseline and (b) direct non-negative wind/solar MWh regressors followed by reconciliation. Do not assume a more complex model wins.

The train label and serving equation must be explicit:

```text
T_hat = existing parent predicted_curtailment_mwh
p_hat_wind = clamp(source_share_model(issue_time_features), 0, 1)
wind_hat_mwh = T_hat * p_hat_wind
solar_hat_mwh = T_hat - wind_hat_mwh
```

If the audited raw total includes another technology, model and conserve the full observed taxonomy instead of forcing it into solar. If the source labels do not reconcile or there is too little independent solar-era data, stop at a documented *experimental estimate* and do not market it as a trustworthy forecast.

### 5. Evaluate the **end-to-end** prediction

On held-out chronological data, obtain the parent's prediction as it would have existed at issue time, then allocate that **predicted** total. Do not score shares using the true total as if it were available at inference. Report wind and solar MAE (MWh), combined source MAE, source bias, total conservation error, and results conditional on actual curtailment and actual solar curtailment being positive. Include day/night, season, year, and 30-versus-60-minute slices where applicable. Compare against both the training-only prior-share baseline and a wind-only allocation baseline; report honest uncertainty or unsupported slices rather than hiding zero-heavy behavior behind one global MAE. Keep V1/V2 parent score and existing output byte-for-byte unchanged unless a separate authorized upgrade is proposed.

Release gate: source labels reconcile; no as-of or target leakage; source outputs conserve the predicted parent total; held-out source errors improve on the baselines without a material regression in solar-active periods; tests and deployment-style smoke checks pass. If this gate fails, keep the model behind an opt-in flag or off the live API and report why.

### 6. Expose only validated results and observed comparisons

After the gate passes, create **two entirely new, separately documented prediction endpoints**; do not modify any existing endpoint or make source attribution an opt-in field on an existing response. Suggested paths are V1 `POST /predict/v1/curtailment/sources` and V2 `POST /predict/curtailment/sources/day` (check route collisions before implementation). The V1 request must take an `issue_timestamp_utc` in ISO 8601 UTC form and `forecast_horizon_minutes` of 30 or 60, plus any additional *issue-time-available* raw values or a clearly named archived-dataset lookup selector that its feature builder actually requires. The V2 request must take a `target_date_utc` in `YYYY-MM-DD` UTC form, plus any additional archived day-ahead/issue-time inputs or validated raw values its feature builder actually requires. Choose and document the exact request contract after inspecting the parent feature builders; do not invent unavailable live data or silently fill required inputs from future observations. For each new route, document its model, target period, data coverage, required/optional fields with units and realistic examples, valid date/horizon rules, historical versus genuinely forward-looking mode, and errors for unavailable inputs or out-of-coverage dates.

Each new response should include the parent target and horizon/day, parent predicted **curtailment** total, predicted Wind and Solar MWh, sum/reconciliation, model and label versions, training cutoff, and plain-language limitations. If the audited taxonomy requires another category, include it explicitly rather than forcing it into Solar. Create *new* matched `/actuals/.../sources` and `/model-info/.../sources` routes only if needed for archived labelled outcomes and held-out metrics; never alter the existing actuals or model-info endpoints. Document `unavailable` when a date is outside complete label coverage; do not return a plausible zero for a missing source. Reuse the existing API-key and artifact-integrity conventions without changing the behavior of current routes.

### 7. Tests, docs, and deliverables

- Unit tests: `UT_TYPE` mapping, unique UTC keys and DST conversion, missing-versus-zero, 2023 partial solar, 48-row daily completeness, non-negativity, source/parent reconciliation, and unknown taxonomy.
- Leakage tests: forecast publication times and source observations cannot exceed issue time; the target's total/source labels are absent from feature columns; training and holdout intervals do not overlap.
- Model tests: zero-parent prediction returns zero components; positive predictions conserve MWh; artifacts reproduce fixed fixture predictions; held-out metrics and hashes match the saved report.
- API tests: both *new* request schemas/help text, auth, valid and missing dates, V1/V2 separation, unavailable/off behavior, historical actual-source retrieval if added, and snapshot/regression tests proving all existing endpoints and their schemas are unchanged.
- Deliverables: labelled table plus data dictionary/quality report, a reproducible build/training script and commented notebook, a versioned model artifact and benchmark report, API/docs changes only after the gate, and a PR that states which gates passed or failed.

## Suggested code entry points

| Concern | Existing entry point | Likely next change |
| --- | --- | --- |
| Official workbook catalog | `data/processed/extended_source_manifest.csv`, `scripts/build_extended_data.py` | Reuse downloads and hashes; audit technology types by year. |
| Half-hour source labels | `src/gridtoev/eirgrid_history.py` | Preserve a parallel technology-level label table before IE aggregation. |
| Daily target rollup | `src/gridtoev/daily_curtailment.py` | Sum only complete UTC days of source labels. |
| V1 / V2 predictions | `src/gridtoev/inference.py`, `src/gridtoev/daily_model.py` | Read parent predicted **curtailment** total, never dispatch-down total by accident. |
| Model training / evaluation | `scripts/train_models.py`, `scripts/train_daily_curtailment_model.py`, `src/gridtoev/benchmarking.py` | New independent allocator artifact/report with chronology and baselines. |
| API / tests | `src/gridtoev/api.py`, `tests/test_eirgrid_history.py`, `tests/test_daily_model.py`, `tests/test_api.py` | Add two new, separate source-prediction routes only after validation; preserve every existing endpoint unchanged. |

## Handoff boundary

This documentation PR records the investigation and exact implementation path; it is **not** the source-allocation implementation. The next agent should begin with Step 2's multi-year raw-label reconciliation and post its results before choosing model complexity. Do not merge an unvalidated source model into `release/shareable-prediction-api` merely because it produces numbers that add up.


## Results (2026-09-28)

### Step 2: audited source labels — passed

`python scripts/build_source_curtailment_labels.py` (module `src/gridtoev/source_curtailment.py`)
reuses the checksum-verified catalog downloads and writes:

- `data/processed/eirgrid_source_curtailment_30min.csv.gz`: 99,310 contiguous UTC half-hours,
  2021-01-01 to 2026-08-31, Wind/Solar curtailment plus constraint and dispatch-down diagnostics;
- `data/processed/eirgrid_source_curtailment_daily.csv`: 1,248 complete UTC days (2023-04-01 to 2026-08-30);
- `data/processed/eirgrid_source_curtailment_quality_report.json` and
  `data/processed/eirgrid_source_curtailment_data_dictionary.csv`.

The raw checks reject duplicate raw keys (instead of the shared normaliser's silent sum), IE
technologies other than Wind/Solar, and negative or non-finite labels. No violations were found. IE
has only `Wind` and `Solar`, so no `other_or_unattributed_mwh` field is needed. IE Solar is first
published at **2023-03-31T23:00Z** (01 April local); earlier intervals are missing and stay NaN, not zero.
The 2023-03-31 partial-solar day is excluded from daily labels.

| Check | Result |
| --- | --- |
| Complete half-hours: Wind + Solar vs core `curtailment_mwh` | 59,952 rows, max difference 2.3e-13 MWh |
| Wind-only half-hours (pre-solar) vs core total | 39,358 rows, max difference 5.7e-14 MWh |
| Daily Wind + Solar vs every V2 label day | 882 of 882 days, max difference 3.6e-12 MWh |

Solar's share of IE curtailment MWh: 2.1% (2023 from April), 4.9% (2024), 12.7% (2025), 19.1% (2026 to August).

### Steps 3-5: V2 daily allocation — release gate failed

`python scripts/evaluate_daily_source_allocation.py` (module `src/gridtoev/source_allocation.py`) writes
`benchmarks/daily_source_allocation_v2/evaluation.json`. It uses V2's own feature contract and
partitions: train 2024-04..12 (275 days), select on 2025 (365), test 2026-01..08 (242). Parent totals
are out-of-sample: a train-only refit of the V2 recipe for 2025 and the saved V2 artifact (trained
through 2025-12-31) for 2026. Every candidate uses the conserving equation in step 4 and was fixed
before the 2026 score.

| 2026 end-to-end test (MWh/day) | Combined source MAE | Solar MAE | Solar MAE, solar-active days |
| --- | ---: | ---: | ---: |
| Wind-only baseline | 1,989.1 | 438.9 | 907.7 |
| Training-only global share | 1,860.4 | 374.1 | 712.4 |
| **Training-only monthly share (selected on 2025)** | 1,820.6 | 329.7 | 633.1 |
| Gradient-boosted share | 1,819.3 | 337.1 | 649.6 |
| MWh-weighted logistic share | 1,820.6 | 331.9 | 637.7 |
| Direct Wind/Solar regressors, reconciled | 1,851.2 | 346.8 | 669.7 |

Gate outcome: the 2025 selection picked a baseline, and no learned allocator beats the monthly-share
baseline on 2026 by a meaningful margin (the best is 0.07% lower on combined MAE and 2.6% *worse* on
solar-active days). The status is therefore `experimental_estimate_only`, and a test asserts that
`/predict/v1/curtailment/sources` and `/predict/curtailment/sources/day` remain unregistered.

Why it failed, for the next attempt:

1. **The parent total error dominates.** V2's daily MAE is about 1,720 MWh on 2026, so allocation
   changes end-to-end source error by at most about 8%. Given the actual total (a diagnostic only, never a
   serving score), the logistic share cuts 2026 solar MAE from 224 to 200 MWh against the
   monthly baseline. Share skill exists but is lost inside the parent's error.
2. **Non-stationary solar share.** The share of solar in curtailment roughly doubles each year. All candidates predicted
   8-11% solar in 2026 against an actual 19%. Tree models cannot extrapolate this. A capacity-aware,
   issue-time-available feature (for example, installed solar MW or the previous month's published share,
   with EirGrid's publication lag checked) is the most promising next candidate.
3. **Short history.** V2 archived forecasts start 2024-04-01, leaving 275 training days.
4. **V1 was not attempted.** Its model-ready table covers January 2026 only, when solar
   curtailment is minimal, so there are too few matched issue-time rows.

Because the 2026 holdout has now been viewed, any new candidate should be selected on 2025 and confirmed on
data published after 2026-08-31 before any route is exposed. Existing endpoints, schemas, and the V1/V2
artifacts are unchanged; the full existing test suite still passes.
