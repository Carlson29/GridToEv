# Issue #9: interval calibration and component forecasts

This experiment uses the 1,634 development out-of-fold predictions produced by issue #8. It is
stacked on `feature/horizon-model-ensembles` because that branch has not been merged into `main`.
The active total forecast is still v1.1.0: issue #8 did not find a replacement that passed its
development release gate.

## Selection and final-test boundary

The runner validates the issue #8 dataset hash, fold contract, unique issue/horizon keys, and that
every calibration row predates the frozen final-test period. It compares four methods by calibrating
on earlier scored folds and evaluating on later folds:

- one global symmetric absolute-residual radius;
- separate symmetric radii for 30 and 60 minutes;
- separate lower and upper signed-residual offsets by horizon; and
- symmetric radii based on the most recent available fold.

The selection rule minimizes forward-fold violations of the target 78–90% coverage band, then
distance from 80% coverage, then average interval width. It selects the horizon-specific asymmetric
method, fits its offsets on development predictions, and only then loads final-test labels.
P50 is the active central forecast. P10/P90 are clipped to be nonnegative and ordered around P50.

## Measured outcome

| Final-test measure | Candidate | Deployed v1.1.0 interval |
| --- | ---: | ---: |
| P50 pinball loss | 5.9125 | 6.6904 |
| P10–P90 coverage | 68.91% | 77.26% |
| Average width | 22.28 MWh | 34.11 MWh |

The candidate meets the P50 target of at most 6.0, but **fails** the required 78–90% interval
coverage. It is excluded from production. The deployed bundle's measured interval result differs
from the frozen benchmark report (74.48% coverage) because the deployed bundle was refitted on
train plus validation, whereas the frozen benchmark evaluated models fitted on train only.

The final-test result is a one-time evaluation of the policy selected on development data. Do not
retune the candidate against these final-test labels. Better interval coverage requires additional
labelled periods and a new predeclared holdout evaluation.

## Curtailment and constraint components

The labelled development data spans only 25.44 days. Curtailment has 231 positive rows spread
across 8 days; constraint has 992 positive rows across 20 days. The refit gate requires at least
90 days of history, 100 positive rows and 20 positive days for **each** component. Therefore this
branch does not refit either component model.

The existing component predictions are clipped at zero, rescaled to the central total forecast,
and checked on the final test. Curtailment plus constraint equals total dispatch-down MWh to
floating-point precision (maximum difference `2.84e-14` MWh).

## Rebuild and outputs

From an installed project environment:

```powershell
python scripts/run_calibration_benchmark.py
```

The runner writes:

- `benchmarks/calibrated_intervals/calibration_report.json`
- `benchmarks/calibrated_intervals/interval_policy.json`
- `benchmarks/calibrated_intervals/development_interval_comparison.csv`

The notebook `notebooks/09_review_calibration_and_components.ipynb` reviews the committed results
without running another final-test evaluation. Issue #9 remains open because the required final-test
coverage was not achieved.
