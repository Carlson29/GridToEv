# Issue #1: label-mature rolling benchmark

## Why a v2 contract is necessary

The frozen v1.1.0 benchmark checked that training issue times preceded score issue
times, but that is insufficient for a 60-minute label. At the train/validation
boundary, the last training label targets 2026-01-22 23:00 UTC while validation
starts at 22:30 UTC. The same overlap occurs in all three rolling folds and at
the validation/final-test boundary. Those labels would not exist when the first
forecast in the next period was issued.

`config/benchmark_contract.v2.json` keeps the same source dataset, 70/15/15 time
periods, release thresholds, and v1.1.0 rollback hashes. It adds the explicit
`target_before_score_issue` policy. Each fold discards fit rows whose target
time is at or after its first score issue time, and score rows whose targets
would mature after the next boundary. Train and validation are likewise
purged before policy selection and the final-test handoff. Each rolling fold
chooses its operating settings from an inner split of its own earlier fit
period, never from the later outer validation period. The report records the
cutoffs and exclusions. Three training and three scoring rows are excluded
at each relevant boundary in the January sample.

Run from an installed environment:

```powershell
python scripts/run_purged_benchmark.py
python -m unittest discover -s tests -v
```

The command verifies the pinned dataset/model/metadata checksums and frozen v2
metrics. It refits evaluation models in memory and writes only to
`benchmarks/v2-purged/`; it never replaces the production bundle.
`python scripts/run_benchmark.py` remains the
legacy v1 reproduction command.

## Measured impact

| Final-test measure | Historical v1 | Purged v2 |
| --- | ---: | ---: |
| Dispatch-down MAE | 11.8250 MWh | 11.8250 MWh |
| Event average precision | 0.999795 | 0.999783 |
| P50 pinball loss | 6.9976 | 7.0128 |
| P10–P90 coverage | 74.48% | 71.69% |
| Rolling-origin MAE | 9.8717 MWh | 10.0947 MWh |

The unchanged final-test point MAE is not evidence that the old split was safe: the
selected point forecast had zero weight on the fitted regressor, while the
classification, interval, and rolling results did change. The v2 evaluation still falls
short of the sprint's release targets. The final test is scored only after
the operating policy is selected from purged development data.

Issue #8's horizon-model comparison and issue #9's calibration used the v1
contract. Their metrics cannot be promoted against the v2 baseline without a
fresh selection and evaluation under the purged folds. Keep v1.1.0 in service
and preserve those historical reports; do not silently relabel them as v2.
