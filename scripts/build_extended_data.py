"""Download and build the Issue #2/#3 history and forecast-vintage artifacts."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from gridtoev.data_sources import (
    SourceSpec,
    download_source,
    load_source_catalog,
    write_manifest,
)
from gridtoev.eirgrid_history import (
    build_core_history,
    read_dispatch_workbook,
    read_system_workbook,
)
from gridtoev.forecast_vintages import (
    asof_join_forecasts,
    deduplicate_vintages,
    forecast_quality_report,
    parse_semo_forecast_xml,
    parse_smartgrid_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
SEMO_API = "https://reports.sem-o.com/api/v1/documents/static-reports"
SEMO_DOCUMENTS = "https://reports.sem-o.com/documents"
SEMO_REPORTS = {
    "Aggregated Wind Forecast": "wind",
    "Daily Load Forecast Summary": "demand",
    "Daily Interconnector NTC": "interconnector_ntc",
}
LICENCE_NOTE = "Review the publisher's open-data terms and disclaimer before redistribution."


def _json_url(url: str, timeout: int = 120) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "GridToEV/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def query_semo_catalog(
    report_name: str,
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
    items = document.get("items", [])
    return [item for item in items if str(item.get("ResourceName", "")).endswith(".xml")]


def _relative_manifest_path(record: dict[str, object]) -> dict[str, object]:
    result = record.copy()
    result["local_path"] = Path(str(record["local_path"])).relative_to(REPO_ROOT).as_posix()
    return result


def build_history(catalog_path: Path) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    specs = load_source_catalog(catalog_path)
    manifest: list[dict[str, object]] = []
    system_frames: list[pd.DataFrame] = []
    dispatch_frames: list[pd.DataFrame] = []

    # Public workbook downloads are independent; fetching them concurrently is
    # much faster than serial transfer while parsing remains deterministic.
    downloaded: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(specs))) as executor:
        futures = {
            executor.submit(download_source, spec, RAW_ROOT): spec.source_id
            for spec in specs
        }
        for future in as_completed(futures):
            downloaded[futures[future]] = future.result()

    for spec in specs:
        record = downloaded[spec.source_id]
        path = Path(str(record["local_path"]))
        if spec.report == "system":
            frame = read_system_workbook(path, source_id=spec.source_id)
            # The grouped workbook contains 2020 solely because 2021 shares a file.
            frame = frame.loc[frame["timestamp_utc"] >= "2021-01-01T00:00:00Z"]
            system_frames.append(frame)
        elif spec.report == "dispatch_down":
            frame = read_dispatch_workbook(path, source_id=spec.source_id)
            dispatch_frames.append(frame)
        else:
            continue
        record["first_interval_utc"] = frame["timestamp_utc"].min().isoformat()
        record["last_interval_utc"] = frame["timestamp_utc"].max().isoformat()
        record["row_count"] = int(len(frame))
        manifest.append(_relative_manifest_path(record))

    core, quality = build_core_history(system_frames, dispatch_frames)
    output_path = PROCESSED_ROOT / "eirgrid_core_history_30min.csv.gz"
    core.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    quality["dataset"] = output_path.relative_to(REPO_ROOT).as_posix()
    quality["catalog"] = catalog_path.relative_to(REPO_ROOT).as_posix()
    (PROCESSED_ROOT / "eirgrid_history_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    return core, quality, manifest


def _semo_spec(item: dict[str, object], report_slug: str) -> SourceSpec:
    resource = str(item["ResourceName"])
    published = pd.Timestamp(str(item["PublishTime"]))
    return SourceSpec(
        source_id=f"semo_{Path(resource).stem.lower()}",
        provider="SEMO",
        report=report_slug,
        year=published.year,
        url=f"{SEMO_DOCUMENTS}/{resource}",
        relative_path=(
            f"semo/forecast_vintages/{report_slug}/{published.year}/"
            f"{published.month:02d}/{resource}"
        ),
        licence_note=LICENCE_NOTE,
    )


def _download_semo_item(item: dict[str, object], report_slug: str) -> dict[str, object]:
    return download_source(_semo_spec(item, report_slug), RAW_ROOT)


def _download_smartgrid_solar(today: date) -> dict[str, object]:
    start = today.strftime("%Y%m%d0000")
    end = (today + timedelta(days=1)).strftime("%Y%m%d2359")
    url = (
        "https://www.vargen.smartgriddashboard.com/api/export/"
        f"{start}/{end}/ALL/SOLAR?FORMAT=JSON"
    )
    spec = SourceSpec(
        source_id=f"smartgrid_solar_snapshot_{today.isoformat()}",
        provider="EirGrid Smart Grid Dashboard",
        report="solar_forecast_snapshot",
        year=today.year,
        url=url,
        relative_path=f"eirgrid/forecast_snapshots/solar/{today.year}/{today.isoformat()}.json",
        licence_note=LICENCE_NOTE,
    )
    return download_source(spec, RAW_ROOT)


def _parse_downloaded_forecasts(
    semo_records: list[dict[str, object]],
    solar_record: dict[str, object] | None,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for record in semo_records:
        frame = parse_semo_forecast_xml(
            Path(str(record["local_path"])),
            source_id=str(record["source_id"]),
            source_url=str(record["url"]),
            downloaded_at=str(record["retrieved_at_utc"]),
        )
        record["first_interval_utc"] = frame["target_timestamp_utc"].min().isoformat()
        record["last_interval_utc"] = frame["target_timestamp_utc"].max().isoformat()
        record["row_count"] = int(len(frame))
        frames.append(frame)
    if solar_record is not None:
        solar = parse_smartgrid_snapshot(
            Path(str(solar_record["local_path"])).read_bytes(),
            area="solar",
            source_id=str(solar_record["source_id"]),
            source_url=str(solar_record["url"]),
            downloaded_at=str(solar_record["retrieved_at_utc"]),
        )
        if not solar.empty:
            solar_record["first_interval_utc"] = solar[
                "target_timestamp_utc"
            ].min().isoformat()
            solar_record["last_interval_utc"] = solar[
                "target_timestamp_utc"
            ].max().isoformat()
            solar_record["row_count"] = int(len(solar))
            frames.append(solar)
    return frames


def build_forecasts(
    *,
    forecast_days: int,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    today = datetime.now(UTC).date()
    date_from = today - timedelta(days=forecast_days - 1)
    catalog_items: list[tuple[dict[str, object], str]] = []
    for report_name, report_slug in SEMO_REPORTS.items():
        catalog_items.extend(
            (item, report_slug)
            for item in query_semo_catalog(
                report_name, date_from=date_from, date_to=today
            )
        )

    semo_records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(_download_semo_item, item, report_slug)
            for item, report_slug in catalog_items
        ]
        for future in as_completed(futures):
            semo_records.append(future.result())
    semo_records.sort(key=lambda row: str(row["source_id"]))

    solar_record: dict[str, object] | None
    try:
        solar_record = _download_smartgrid_solar(today)
    except Exception as error:
        # Solar snapshots are optional because the endpoint has no historical
        # publication archive. The quality report makes that absence explicit.
        print(f"Solar snapshot unavailable: {error}")
        solar_record = None

    frames = _parse_downloaded_forecasts(semo_records, solar_record)
    if not frames:
        raise RuntimeError("No forecast vintages were downloaded")
    vintages = deduplicate_vintages(pd.concat(frames, ignore_index=True, sort=False))
    output_path = PROCESSED_ROOT / "forecast_vintages.csv.gz"
    vintages.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    quality = forecast_quality_report(vintages)
    quality.update(
        {
            "dataset": output_path.relative_to(REPO_ROOT).as_posix(),
            "requested_date_from": date_from.isoformat(),
            "requested_date_to": today.isoformat(),
            "source_limitations": {
                "semo_retention": (
                    "The API returns only each report's retained archive; coverage is "
                    "reported exactly rather than backfilled or inferred."
                ),
                "solar_publication_time": (
                    "The Smart Grid Dashboard does not expose historical solar "
                    "publication times, so retrieval time is used as the conservative "
                    "availability time and snapshots accumulate on future runs."
                ),
            },
        }
    )
    (PROCESSED_ROOT / "forecast_vintage_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    manifest = [_relative_manifest_path(record) for record in semo_records]
    if solar_record is not None:
        manifest.append(_relative_manifest_path(solar_record))
    return vintages, quality, manifest


def build_asof_feature_matrix(core: pd.DataFrame, vintages: pd.DataFrame) -> pd.DataFrame:
    # Build rows for forecast targets actually present in the vintage archive.
    # The latest EirGrid labels are published monthly and may legitimately lag
    # the live SEMO feed, so forcing the intersection would erase all new data.
    targets = pd.DatetimeIndex(
        pd.to_datetime(vintages["target_timestamp_utc"], utc=True).drop_duplicates()
    )
    targets = targets[
        (targets.minute.isin([0, 30]))
        & (targets.second == 0)
    ].sort_values()
    blocks: list[pd.DataFrame] = []
    for horizon in (30, 60):
        block = pd.DataFrame(
            {
                "target_timestamp_utc": targets,
                "issue_timestamp_utc": targets - pd.Timedelta(minutes=horizon),
            }
        )
        block["forecast_horizon_minutes"] = horizon
        blocks.append(block)
    modelling_rows = pd.concat(blocks, ignore_index=True).sort_values(
        ["issue_timestamp_utc", "forecast_horizon_minutes"]
    )
    result = asof_join_forecasts(modelling_rows, vintages)
    label_availability = core.set_index("timestamp_utc")[
        "dispatch_down_available_flag"
    ]
    result["history_label_available_flag"] = (
        result["target_timestamp_utc"].map(label_availability).fillna(0).astype("int8")
    )
    output_path = PROCESSED_ROOT / "forecast_features_asof_30_60.csv"
    result.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    availability_columns = [
        column for column in result if column.endswith("_available_flag")
    ]
    asof_quality = {
        "dataset": output_path.relative_to(REPO_ROOT).as_posix(),
        "rows": int(len(result)),
        "forecast_horizons_minutes": [30, 60],
        "first_issue_utc": result["issue_timestamp_utc"].min().isoformat(),
        "last_issue_utc": result["issue_timestamp_utc"].max().isoformat(),
        "future_publication_violations": int(
            sum(
                (
                    result[column].notna()
                    & (result[column] > result["issue_timestamp_utc"])
                ).sum()
                for column in result
                if column.endswith("_published_at_utc")
            )
        ),
        "availability_rate_by_feature": {
            column.removesuffix("_available_flag"): float(result[column].mean())
            for column in availability_columns
        },
    }
    (PROCESSED_ROOT / "forecast_asof_quality_report.json").write_text(
        json.dumps(asof_quality, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "config" / "extended_data_sources.v1.json",
    )
    parser.add_argument("--forecast-days", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forecast_days < 1:
        raise ValueError("--forecast-days must be at least 1")
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    core, history_quality, history_manifest = build_history(args.catalog.resolve())
    vintages, forecast_quality, forecast_manifest = build_forecasts(
        forecast_days=args.forecast_days,
        workers=args.workers,
    )
    feature_matrix = build_asof_feature_matrix(core, vintages)
    write_manifest(
        [*history_manifest, *forecast_manifest],
        PROCESSED_ROOT / "extended_source_manifest.csv",
    )
    print(
        json.dumps(
            {
                "history_rows": len(core),
                "history_meets_24_month_target": history_quality[
                    "meets_24_month_target"
                ],
                "forecast_vintage_rows": len(vintages),
                "forecast_features": sorted(
                    forecast_quality["feature_coverage"].keys()
                ),
                "asof_feature_rows": len(feature_matrix),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
