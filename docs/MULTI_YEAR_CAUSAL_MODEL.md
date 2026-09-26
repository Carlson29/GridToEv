# Multi-year causal point-forecast experiment

This branch tests a reproducible path to more training data without using the
short September 2026 forecast/SEMO tables as if they had historical labels.
It uses the 99,310-row EirGrid half-hour history (2021-01 through 2026-08),
producing 198,521 issue/horizon examples after the observation delay and exact
future-label joins. Source fields are lagged at least 24 hours, then joined by
timestamp; calendar fields are known at issue time. The builder checks unique
keys and prevents feature timestamps crossing the delay cutoff.

The 24-hour delay is an **assumption**, not verified publication metadata for
every historical report. This is a research candidate, not a live feature
pipeline. Before any deployment, obtain the actual publication/update times,
rebuild the as-of dataset with those vintages, and connect the same fields to
a reliable live feed. The current v1.1.0 API cannot consume this feature set.

`config/multi_year_causal_experiment.v1.json` froze the model and gates before
scoring. Development folds are May, June and July 2026; August is the sealed
final month. Each fit excludes labels whose target time reaches the score
month. Each score excludes labels crossing the next monthly boundary. The
candidate is a fixed horizon-specific histogram gradient booster; its same-row
comparator is the last observation at least 24 hours old.

Run from an installed environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m gridtoev.history_experiment
python -m pytest tests/test_history_model.py tests/test_history_experiment_artifacts.py -q
```

The committed report in `benchmarks/multi_year_causal/experiment_report.json`
shows 5.83% aggregate development MAE improvement, with both horizons better
but one of three months worse. The required 15% aggregate and fold-stability
gates fail. August was not accessed. The model was **not** saved, promoted, or
deployed. This percentage is against a delayed-observation comparator on a
different dataset; it is **not** a measured improvement over deployed v1.1.0.

The next justified step is not to retune on these same three months. Acquire
verified historical publication vintages or an independently timestamped live
source, predeclare a new independent evaluation period, and compare against a
v2-comparable serving baseline. A production candidate also needs the event,
interval, component, and live-contract gates, which this point-only experiment
does not supply.
