"""Build audited Wind/Solar curtailment labels from the catalogued EirGrid workbooks.

Reuses the extended-data catalog and checksum-verified raw downloads. Writes a
parallel label table; the core history and every model artifact are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridtoev.data_sources import download_source, load_source_catalog
from gridtoev.source_curtailment import (
    build_daily_source_labels,
    build_source_label_table,
    normalise_source_dispatch_frame,
    reconcile_daily_with_v2,
    reconcile_with_core,
    source_label_quality_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
CATALOG = REPO_ROOT / "config" / "extended_data_sources.v1.json"
HALF_HOUR_OUTPUT = PROCESSED_ROOT / "eirgrid_source_curtailment_30min.csv.gz"
DAILY_OUTPUT = PROCESSED_ROOT / "eirgrid_source_curtailment_daily.csv"
REPORT_OUTPUT = PROCESSED_ROOT / "eirgrid_source_curtailment_quality_report.json"
GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def main() -> None:
    frames = []
    sources = []
    for spec in load_source_catalog(CATALOG):
        if spec.report != "dispatch_down":
            continue
        record = download_source(spec, RAW_ROOT)
        raw = pd.read_excel(record["local_path"], sheet_name="DD HH")
        frames.append(normalise_source_dispatch_frame(raw, source_id=spec.source_id))
        sources.append({"source_id": spec.source_id, "url": spec.url, "sha256": record["sha256"]})

    labels = build_source_label_table(frames)
    core = pd.read_csv(
        PROCESSED_ROOT / "eirgrid_core_history_30min.csv.gz",
        usecols=["timestamp_utc", "curtailment_mwh"],
    )
    reconciliation = reconcile_with_core(labels, core)
    daily = build_daily_source_labels(labels)
    v2 = pd.read_csv(
        PROCESSED_ROOT / "daily_curtailment_forecast_v2.csv.gz",
        usecols=["issue_timestamp_utc", "curtailment_mwh"],
    )
    daily_reconciliation = reconcile_daily_with_v2(daily, v2)

    labels.to_csv(HALF_HOUR_OUTPUT, index=False, date_format="%Y-%m-%dT%H:%M:%SZ", compression=GZIP)
    daily.assign(target_date_utc=daily["target_date_utc"].dt.strftime("%Y-%m-%d")).to_csv(
        DAILY_OUTPUT, index=False, float_format="%.6f",
    )
    report = source_label_quality_report(labels, daily)
    report["sources"] = sources
    report["core_reconciliation"] = reconciliation
    report["v2_daily_reconciliation"] = daily_reconciliation
    report["outputs"] = [
        path.relative_to(REPO_ROOT).as_posix() for path in (HALF_HOUR_OUTPUT, DAILY_OUTPUT)
    ]
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "complete_half_hours", "daily", "core_reconciliation")}, indent=2))


if __name__ == "__main__":
    main()
