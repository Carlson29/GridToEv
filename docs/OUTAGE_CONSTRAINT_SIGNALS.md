# Outage and constraint signals

Issue #6 adds auditable capacity-pressure signals from three official EirGrid
publication snapshots. The build normalizes the source workbooks, joins them to
the existing 30- and 60-minute modelling keys without future information, and
records a named feature-family ablation.

## Why these variables matter

Dispatch-down can rise when available generation exceeds what the network can
carry. A generator outage changes the available supply stack; a transmission
outage can reduce transfer capacity; and ECP constraint studies identify parts
of the grid that are structurally exposed to congestion. These variables give a
model direct context about temporary capacity loss and longer-term network
pressure.

They are context, not proof of causation. Planned outage schedules can change,
forced outages may not appear in a historical public snapshot, and the ECP
workbook is a forward study rather than a half-hourly operational measurement.

## Public inputs

| Family | Workbook | Normalized meaning |
| --- | --- | --- |
| Generation | All-Island Generation Outage Plan | Unit-level planned, forced, and derated unavailable MW |
| Transmission | Transmission Outage Programme | Circuit outage interval, status, voltage class, and regional membership |
| Constraint groups | ECP 2.5 Constraint Forecast | Installed MW and forecast constraint ratio by ECP area |

Exact URLs, first-retrieval timestamps, publication timestamps, file sizes,
SHA-256 hashes, schema versions, and coverage are stored in
`data/processed/outage_constraint_source_manifest.csv`. Raw workbooks are kept
under `data/raw/eirgrid/` and are ignored by Git. Review EirGrid's terms before
redistributing those files.

The publication timestamps are the official HTTP `Last-Modified` values
captured for these immutable snapshots. Feature construction never treats the
local download time as the time at which a source became knowable.

## Normalization rules

### Generation

- `F` is a full forced outage.
- `S` is a full scheduled outage.
- A numeric `Value` is remaining capacity during a derating, so unavailable MW
  is `NPR - Value`, clipped at zero.
- Units whose identifier begins with `NI` are labelled Northern Ireland; other
  units are labelled Ireland. This is a coarse jurisdiction label, not an ECP
  network area.

### Transmission

- `Scheduled` and `Proposed` are planned categories; other published statuses
  are retained in normalized form.
- A workbook finish date is an inclusive calendar date. The parser adds one day
  and stores an exclusive midnight endpoint.
- Regional membership comes from the workbook's South West, South East, North
  West, North East, Dublin, and Cork tabs. An event may belong to more than one
  region.
- Voltage in kV is extracted from the published feeder label. The normalized
  dataset does not invent an unavailable-MW value for transmission circuits.

### ECP constraints

- Groups are `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H1`, `H2`, `I`, `J`, and `K`.
- `constraint_ratio` is the workbook's published constraint percentage stored
  as a ratio.
- Overall and group pressure use installed-MW weighting. Group installed MW is
  also retained so the model can distinguish a high ratio in a small area from
  the same ratio in a large area.

## Causal join contract

The natural key remains `issue_timestamp_utc` plus
`forecast_horizon_minutes`. For each model row the builder:

1. selects the newest source snapshot whose `published_at_utc` is no later than
   `issue_timestamp_utc`;
2. evaluates outage activity at `target_timestamp_utc` using the half-open
   interval `[start, end)`;
3. calculates counts, unavailable MW, largest affected unit/voltage, regional
   counts, hours to the next start and first active end, constraint pressure,
   source age, and source-availability flags;
4. leaves unknown source values null and sets a matching missing flag. Unknown
   is never silently converted to a real zero.

The quality gate rejects duplicate issue/horizon keys, horizon misalignment,
publication timestamps later than issue time, and infinite numeric values.
Tests cover overlapping outages and exact start/end boundaries.

## Current measured coverage

The committed build contains:

- 569 normalized outage events: 47 generation and 522 transmission;
- 3,482 normalized ECP study rows;
- 2,867 feature rows and 63 columns, matching every baseline modelling row;
- 100% generation-snapshot coverage;
- 27.52% transmission-snapshot coverage, because that snapshot was published
  on 23 January 2026;
- 0% ECP coverage in the January modelling window, because the ECP snapshot was
  not published until 11 March 2026.

That last point is deliberate. Backfilling March knowledge into January rows
would create leakage. The ECP values will become usable for issue times on or
after their publication timestamp.

## Ablation decision

The ablation uses identical histogram-gradient-boosting residual regressors
with and without the new numeric features. It scores three expanding windows in
the 70% training period plus the following 15% validation period. The final 15%
test period remains sealed.

The feature family is currently **excluded** from the production model. The
unweighted mean of the four fold MAEs changed from 12.547 MWh to 13.751 MWh (a
9.60% worsening); the mean of the four per-fold improvement rates was -13.88%,
and one fold degraded materially. The acceptance rule requires at least a 15%
mean per-fold improvement and no degrading fold. This result is evidence that
one generation snapshot and a late transmission snapshot are too sparse to
justify production inclusion yet; it is not evidence that outage data has no
value. More historical publication vintages should be collected and the same
ablation rerun.

## Rebuild and outputs

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_outage_constraint_data.py
```

The same documented workflow is in
`notebooks/06_build_outage_constraint_signals.ipynb`.

Outputs:

- `data/processed/outage_events.csv.gz`
- `data/processed/ecp_constraint_pressure.csv.gz`
- `data/processed/outage_constraint_features_30_60.csv`
- `data/processed/outage_constraint_feature_dictionary.csv`
- `data/processed/outage_constraint_source_manifest.csv`
- `data/processed/outage_constraint_quality_report.json`
- `benchmarks/outage_constraint_signals/ablation_report.json`
- `benchmarks/outage_constraint_signals/experiment_registry.csv`
