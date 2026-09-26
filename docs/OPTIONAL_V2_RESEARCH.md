# Optional v2 research (not a released model)

This branch starts an **opt-in** point-forecast alternative. It does not replace
v1.1.0 or change the prediction API. The promotion requirement is strict: the
candidate must have a lower dispatch-down MAE than v1 on the same final period.

## Branch and feature review

| Source | Decision for this experiment | Reason |
| --- | --- | --- |
| `main` | Use its 119 issue-time features and frozen v1.1.0 benchmark | Aligned January labels and a reproducible, purged comparison |
| `feature/release-gates-api-integration` | Re-evaluate its horizon-specific Extra Trees idea; do not copy its model | Earlier 1.22% development gain used the older split, had unstable folds, and did not score the final holdout |
| `feature/multi-year-causal-model-v2` | Defer its historical rows | Its 24-hour source-publication assumption needs verification; its published comparison is to delayed persistence, not v1.1.0 |
| `feature/frequency-entsoe-outages` | Defer its frequency and outage signals | No frequency archive or aligned labelled outage rows are available in this repository |
| `release/shareable-prediction-api` and release-gate branches | Leave serving as v1 | They improve access/safety, not the evaluated point forecast; v2 must not affect live users until it qualifies |
| Forecast/SEMO candidate tables already on `main` | Defer | Their September issue rows have no overlap with the January dispatch-down labels used for this comparison |

The experiment also tests four derived physical interactions (unrealised wind,
wind/load acceleration, and SNSP headroom), calculated from the issue-time row.
The development score selected the version *without* those additions, so they
are not claimed as validated v2 features.

## Reproducible evaluation

Run `python -m gridtoev.optional_v2_experiment` with the project's normal
dependencies and data in place. The predeclared candidates, blend grid, and
promotion thresholds live in `config/optional_v2_experiment.v1.json`; detailed
scores and source hashes are in
`benchmarks/optional_v2/experiment_report.json`. Training uses only labels
mature at each issue time and chooses the candidate from purged development
folds before scoring the final January period.

| Metric | v1.1.0 | Selected candidate |
| --- | ---: | ---: |
| Development MAE (MWh) | 10.095 | 9.922 |
| Final-period MAE (MWh) | 11.825 | 11.931 |

The selected candidate **fails** the required final MAE comparison by about
0.89%; the experiment deliberately creates no v2 model artifact. The January
final period has now been examined for v2. Do not retune against it and present
that as an independent validation.

## Route to a usable v2

1. Obtain an independent later *labelled* period with the same as-of feature
   semantics, including verified publication times for any new feeds.
2. Freeze a revised candidate and test plan before looking at that period.
3. Promote an experimental artifact only if its MAE is strictly below v1 on
   the same period, with tolerable horizon-level regressions. Keep v1 as the
   default and expose v2 only through an explicit version selection.

This is a point-forecast experiment. Event probabilities, components, and
interval coverage need separate validation before a full v2 API release.
