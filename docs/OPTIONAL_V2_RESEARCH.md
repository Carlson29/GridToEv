# Optional v2 research (not a released model)

This branch starts an **opt-in** point-forecast alternative. It does not replace
v1.1.0 or change the prediction API. The promotion requirement is strict: the
candidate must have a lower dispatch-down MAE than v1 on the same final period.

## Branch and feature review

| Source | Decision for this experiment | Reason |
| --- | --- | --- |
| `main` | Use its 119 issue-time features and frozen v1.1.0 benchmark | Aligned January labels and a reproducible, purged comparison |
| `feature/release-gates-api-integration` | Re-evaluate its horizon-specific Extra Trees idea; do not copy its model | Earlier 1.22% development gain used the older split, had unstable folds, and did not score the final holdout |
| `feature/multi-year-causal-model-v2` | Incorporate its historical data builder, notebook, and research benchmark; add a same-row v1 comparison | The 24-hour observation cutoff is not verified publication timing, and its original 5.83% gain was against delayed persistence, not v1.1.0 |
| `feature/frequency-entsoe-outages` | Defer its frequency and outage signals | No frequency archive or aligned labelled outage rows are available in this repository |
| `release/shareable-prediction-api` and release-gate branches | Leave serving as v1 | They improve access/safety, not the evaluated point forecast; v2 must not affect live users until it qualifies |
| Forecast/SEMO candidate tables already on `main` | Defer | Their September issue rows have no overlap with the January dispatch-down labels used for this comparison |

The experiment also tests four derived physical interactions (unrealised wind,
wind/load acceleration, and SNSP headroom), calculated from the issue-time row.
The development score selected the version *without* those additions, so they
are not claimed as validated v2 features.

The incorporated multi-year research contains 99,310 half-hour source rows
(2021–August 2026), exact future-label joins, 39 delayed/calendar features,
monthly purged folds, and a sealed August holdout. Its original result remains
in `benchmarks/multi_year_causal/experiment_report.json`; August was **not**
scored. The separate, predeclared comparison in
`config/multi_year_v1_comparison.v1.json` trains its 24-hour-delayed candidate
only on labels ending before 2025, then aligns it to v1's exact January
development rows. Run `python -m gridtoev.history_v1_comparison` to reproduce
`benchmarks/optional_v2/multi_year_same_rows.json`.

| Same-row January development MAE | MWh |
| --- | ---: |
| Frozen v1.1.0 | 10.095 |
| Multi-year delayed candidate | 78.222 |

The multi-year candidate is therefore not a v1 improvement. No January final
test was spent on it. The published reports are historical and [EirGrid lists
the dispatch-down and quarter-hourly system reports as monthly
publications](https://cms.eirgrid.ie/sites/default/files/publications/Final-Stakeholder-Engagement-Plan-2025.pdf).
An observation timestamp and an assumed 24-hour delay do **not** prove the
record was available to a forecaster then. The builder is useful research, but
its columns cannot enter a serving v2 until as-of publication is evidenced.

An additional offline transition-model probe showed that near-current
dispatch-down observations make historical transfer much more competitive.
That is *not* promoted: the retrospective dispatch-down report does not
establish that those exact values existed at the issue time. Using them to
claim a leakage-free live v2 would be misleading.

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
0.89%; the experiment deliberately creates no v2 model artifact. Artifact
creation is also blocked while source-publication timing remains unverified.
The January
final period has now been examined for v2. Do not retune against it and present
that as an independent validation.

## Route to a usable v2

1. Obtain an independent later *labelled* period and archive timestamped live
   feature snapshots. Verify that every source field existed before issue time;
   historical report timestamps alone are insufficient.
2. Freeze a revised candidate and test plan before looking at that period.
3. Promote an experimental artifact only if its MAE is strictly below v1 on
   the same period, with tolerable horizon-level regressions. Keep v1 as the
   default and expose v2 only through an explicit version selection.

This is a point-forecast experiment. Event probabilities, components, and
interval coverage need separate validation before a full v2 API release.
