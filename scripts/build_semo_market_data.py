"""Download, normalize, and feature-engineer Issue #5 SEMO market reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from gridtoev.data_sources import SourceSpec, download_source, write_manifest
from gridtoev.semo_market import (
    REPORT_NAMES,
    SCHEMA_VERSION,
    build_semo_ablation_record,
    build_semo_feature_matrix,
    deduplicate_market_signals,
    parse_semo_market_xml,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "semo_market_signals"
SEMO_API = "https://reports.sem-o.com/api/v1/documents/static-reports"
SEMO_DOCUMENTS = "https://reports.sem-o.com/documents"
LICENCE_NOTE = "Review the SEMO open-data terms and disclaimer before redistribution."


def _json_url(url: str, timeout: int = 120) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "GridToEV/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def query_catalog(
    report_name: str,
    dataset_name: str,
    *,
    date_from: date,
    date_to: date,
) -> list[dict[str, object]]:
    query = urllib.parse.urlencode(
        {
            "ReportName": report_name,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "page_size": 5000,
            "sort_by": "PublishTime",
            "order_by": "ASC",
        }
    )
    document = _json_url(f"{SEMO_API}?{query}")
    pagination = document.get("pagination", {})
    if int(pagination.get("totalPages", 1)) > 1:  # type: ignore[union-attr]
        raise RuntimeError(
            f"{report_name} exceeds the 5,000-file API page; reduce --days"
        )
    prefix = f"{dataset_name}_"
    return [
        item
        for item in document.get("items", [])  # type: ignore[union-attr]
        if str(item.get("ResourceName", "")).startswith(prefix)
        and str(item.get("ResourceName", "")).endswith(".xml")
    ]


def _source_spec(item: dict[str, object], dataset_name: str) -> SourceSpec:
    resource_name = str(item["ResourceName"])
    published = pd.Timestamp(str(item["PublishTime"]))
    return SourceSpec(
        source_id=f"semo_{Path(resource_name).stem.lower()}",
        provider="SEMO",
        report=dataset_name,
        year=published.year,
        url=f"{SEMO_DOCUMENTS}/{resource_name}",
        relative_path=(
            f"semo/market_signals/{dataset_name}/{published.year}/"
            f"{published.month:02d}/{resource_name}"
        ),
        licence_note=LICENCE_NOTE,
        schema_version=SCHEMA_VERSION,
    )


def _download(
    item: dict[str, object],
    dataset_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    record = download_source(_source_spec(item, dataset_name), RAW_ROOT)
    return record, item


def _relative_record(record: dict[str, object]) -> dict[str, object]:
    result = record.copy()
    # Whether this run used the cache is operational detail, not source provenance.
    # Excluding it keeps the committed manifest stable across identical rebuilds.
    result.pop("cache_hit", None)
    result["local_path"] = Path(str(result["local_path"])).relative_to(
        REPO_ROOT
    ).as_posix()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    *,
    date_from: date,
    date_to: date,
    workers: int,
) -> dict[str, object]:
    catalog: list[tuple[dict[str, object], str]] = []
    for dataset_name, report_name in REPORT_NAMES.items():
        catalog.extend(
            (item, dataset_name)
            for item in query_catalog(
                report_name,
                dataset_name,
                date_from=date_from,
                date_to=date_to,
            )
        )
    if not catalog:
        raise RuntimeError("The SEMO catalog returned no Issue #5 reports")

    downloaded: list[tuple[dict[str, object], dict[str, object]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(_download, item, dataset_name)
            for item, dataset_name in catalog
        ]
        for future in as_completed(futures):
            downloaded.append(future.result())
    downloaded.sort(key=lambda pair: str(pair[0]["source_id"]))

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for record, catalog_item in downloaded:
        frame = parse_semo_market_xml(
            Path(str(record["local_path"])),
            source_id=str(record["source_id"]),
            source_url=str(record["url"]),
            downloaded_at=str(record["retrieved_at_utc"]),
            catalog_published_at=str(catalog_item["PublishTime"]),
        )
        record["first_interval_utc"] = (
            frame["target_timestamp_utc"].min().isoformat()
            if frame["target_timestamp_utc"].notna().any()
            else frame["published_at_utc"].min().isoformat()
        )
        record["last_interval_utc"] = (
            frame["target_end_timestamp_utc"].max().isoformat()
            if frame["target_end_timestamp_utc"].notna().any()
            else frame["published_at_utc"].max().isoformat()
        )
        record["row_count"] = int(len(frame))
        frames.append(frame)
        manifest.append(_relative_record(record))

    signals = deduplicate_market_signals(
        pd.concat(frames, ignore_index=True, sort=False)
    )
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    signal_path = PROCESSED_ROOT / "semo_market_signals.csv.gz"
    signals.to_csv(
        signal_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    write_manifest(manifest, PROCESSED_ROOT / "semo_market_source_manifest.csv")

    modelling_rows = pd.read_csv(
        PROCESSED_ROOT / "forecast_features_asof_30_60.csv",
        usecols=[
            "issue_timestamp_utc",
            "target_timestamp_utc",
            "forecast_horizon_minutes",
        ],
    )
    history = pd.read_csv(PROCESSED_ROOT / "eirgrid_core_history_30min.csv.gz")
    features, dictionary, quality = build_semo_feature_matrix(
        modelling_rows,
        signals,
        history,
    )
    feature_path = PROCESSED_ROOT / "semo_market_features_asof_30_60.csv"
    features.to_csv(
        feature_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    dictionary.to_csv(
        PROCESSED_ROOT / "semo_market_feature_dictionary.csv",
        index=False,
    )
    quality.update(
        {
            "requested_date_from": date_from.isoformat(),
            "requested_date_to": date_to.isoformat(),
            "source_file_count": len(downloaded),
            "signal_dataset": signal_path.relative_to(REPO_ROOT).as_posix(),
            "signal_dataset_sha256": _sha256(signal_path),
            "source_limitation": (
                "The SEMO API retains these high-frequency reports for a limited "
                "period; unavailable historic vintages are not reconstructed."
            ),
        }
    )
    (PROCESSED_ROOT / "semo_market_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    baseline = pd.read_csv(
        PROCESSED_ROOT / "gridtoev_model_ready.csv",
        usecols=["issue_timestamp_utc"],
    )
    ablation = build_semo_ablation_record(baseline, features, signal_path)
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "ablation_report.json").write_text(
        json.dumps(ablation, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "experiment_name": ablation["experiment_name"],
                "feature_group": ablation["feature_group"],
                "signal_dataset_sha256": ablation["signal_dataset_sha256"],
                "overlap_rows": ablation["overlap_rows"],
                "status": ablation["status"],
                "production_decision": ablation["production_decision"],
                "reason": ablation["reason"],
            }
        ]
    ).to_csv(BENCHMARK_ROOT / "experiment_registry.csv", index=False)

    return {
        "catalog_files": len(catalog),
        "signal_rows": len(signals),
        "feature_rows": len(features),
        "feature_columns": len(features.columns),
        "ablation_status": ablation["status"],
        "production_decision": ablation["production_decision"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of SEMO trade dates to request, ending today (default: 2)",
    )
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days < 1:
        raise ValueError("--days must be at least 1")
    today = datetime.now(UTC).date()
    summary = build(
        date_from=today - timedelta(days=args.days - 1),
        date_to=today,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
