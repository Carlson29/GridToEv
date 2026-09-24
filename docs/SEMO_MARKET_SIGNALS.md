# SEMO market and operational signals

Issue #5 adds publication-time-safe SEMO signals that can explain conditions immediately before
renewable dispatch-down: wind and demand forecasts, the real-time operational schedule, physical
notifications (PN), forecast imbalance, interconnector transfer limits, lagged imbalance outcomes,
and SSII/SIFF.

## Rebuild the artifacts

From the repository root, with the project environment active, run:

```powershell
python scripts/build_semo_market_data.py --days 2 --workers 12
```

`--days` controls the SEMO catalog window ending on the current UTC date. Downloads are stored below
`data/raw/semo/market_signals/<dataset>/<year>/<month>/`, checksum-verified, and reused on later runs.
Raw upstream files are intentionally ignored by Git; the manifest makes the committed processed
artifacts traceable to their exact source URLs and SHA-256 hashes.

The commented notebook `notebooks/05_build_semo_market_signals.ipynb` provides the same workflow plus
coverage and leakage checks.

## Inputs

The builder requests these official SEMO static-report families:

- 15-minute and four-day rolling wind forecasts;
- daily load forecasts;
- RTIC operational schedules and aggregated final PN;
- hourly forecast imbalance and net imbalance volume forecasts;
- daily interconnector net transfer capacity (NTC);
- 5-minute and 30-minute imbalance price/outcome reports; and
- hourly SSII/SIFF.

The parser supports both attribute-based and child-element XML rows. A 5-minute publication marked
`DefaultPriceUsage=Y` uses its `MarketBackupPrice` as the effective imbalance price. The raw value is
never treated as known before the report's effective publication time.

## Outputs

- `semo_market_signals.csv.gz`: normalized long-form signal values with interval, publication,
  retrieval, source, revision, units, and jurisdiction metadata.
- `semo_market_source_manifest.csv`: URL, local path, retrieval time, bytes, SHA-256, schema version,
  row count, and temporal coverage for every downloaded file.
- `semo_market_features_asof_30_60.csv`: one row per existing issue time and 30/60-minute horizon.
- `semo_market_feature_dictionary.csv`: role, unit, source, formula, availability rule, missingness
  handling, and dtype for every feature-table column.
- `semo_market_quality_report.json`: report coverage plus duplicate, alignment, future-information,
  missingness, availability, and staleness checks.
- `benchmarks/semo_market_signals/ablation_report.json`: the named feature-family decision record.

## Causal feature construction

For target-specific forecasts and schedules, a backward as-of join selects the newest revision whose
publication is at or before the model's issue time and whose target interval matches the requested
horizon. Actual imbalance values are only joined after both the interval end and report publication.
Publication-scoped values use the latest report available at issue time. Completed EirGrid state uses
the latest history row no later than the issue time.

Useful engineered signals include scheduled generation, interconnector schedule, average PN,
scheduled-minus-actual generation, forecast imbalance, a downward-margin proxy, lagged reserve
surplus, and interconnector export headroom. Each source feature carries availability and staleness
flags; missing values remain null rather than being silently invented.

The validator fails on duplicate issue/horizon keys, target/horizon misalignment, non-finite numeric
values, future publications, or future observations.

## Current ablation decision

The committed January 2026 labelled benchmark and the SEMO high-frequency archive retained in
September 2026 do not overlap. Therefore the feature-family experiment is recorded as
`rejected_no_temporal_overlap`, with null metrics and `final_test_accessed=false`. The features are
ready for a future overlapping labelled window, but they are deliberately excluded from the current
production model until performance can be measured honestly.

This is a data-availability decision, not evidence that the signals are unhelpful. Re-run the builder
regularly to accumulate publication vintages, then evaluate the family under the frozen benchmark
contract once labelled overlap exists.
