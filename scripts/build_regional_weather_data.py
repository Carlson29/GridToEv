"""Download and build Issue #7 capacity-weighted regional weather features."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from gridtoev.data_sources import SourceSpec, download_source, write_manifest
from gridtoev.regional_weather import (
    API_TO_COLUMN,
    MODEL_NAME,
    SCHEMA_VERSION,
    build_regional_weather_features,
    build_weather_ablation_report,
    parse_open_meteo_analysis,
    parse_open_meteo_single_run,
    validate_region_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "regional_weather_signals"
WEIGHTS_PATH = REPO_ROOT / "config" / "regional_weather_capacity_weights.csv"
SINGLE_RUN_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_ENDPOINT = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)
LICENCE_NOTE = (
    "Open-Meteo/ECMWF weather data: CC BY 4.0 with attribution; this project "
    "modifies it through interpolation, weighting, and feature engineering."
)


def _api_parameters(regions: pd.DataFrame) -> dict[str, object]:
    return {
        "latitude": ",".join(regions["latitude"].astype(str)),
        "longitude": ",".join(regions["longitude"].astype(str)),
        "hourly": ",".join(API_TO_COLUMN),
        "models": MODEL_NAME,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "cell_selection": "land",
    }


def _single_run_spec(run: pd.Timestamp, regions: pd.DataFrame) -> SourceSpec:
    parameters = {
        **_api_parameters(regions),
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        # Fourteen hourly values cover a run through +13h. That final hour is
        # required to interpolate a +12:30 target just before the next run is
        # conservatively considered published.
        "forecast_hours": 14,
    }
    url = f"{SINGLE_RUN_ENDPOINT}?{urllib.parse.urlencode(parameters)}"
    stamp = run.strftime("%Y%m%dT%H%MZ")
    return SourceSpec(
        source_id=f"open_meteo_ecmwf_run_{stamp.lower()}_fh14",
        provider="Open-Meteo / ECMWF",
        report="ecmwf_ifs_single_run",
        year=run.year,
        url=url,
        relative_path=(
            f"weather/open_meteo/single_runs/{run.year}/{run.month:02d}/"
            f"ecmwf_ifs_{stamp}_fh14.json"
        ),
        licence_note=LICENCE_NOTE,
        schema_version=SCHEMA_VERSION,
    )


def _analysis_spec(
    start: pd.Timestamp,
    end: pd.Timestamp,
    regions: pd.DataFrame,
) -> SourceSpec:
    parameters = {
        **_api_parameters(regions),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }
    url = f"{HISTORICAL_FORECAST_ENDPOINT}?{urllib.parse.urlencode(parameters)}"
    return SourceSpec(
        source_id=(
            f"open_meteo_analysis_proxy_{start:%Y%m%d}_{end:%Y%m%d}"
        ),
        provider="Open-Meteo / ECMWF",
        report="historical_forecast_analysis_proxy",
        year=start.year,
        url=url,
        relative_path=(
            "weather/open_meteo/analysis_proxy/"
            f"historical_forecast_{start:%Y%m%d}_{end:%Y%m%d}.json"
        ),
        licence_note=LICENCE_NOTE,
        schema_version=SCHEMA_VERSION,
    )


def _download(spec: SourceSpec) -> dict[str, object]:
    return download_source(spec, RAW_ROOT, timeout=180, retries=4)


def _relative_record(record: dict[str, object]) -> dict[str, object]:
    result = record.copy()
    result.pop("cache_hit", None)
    result["local_path"] = Path(str(result["local_path"])).relative_to(
        REPO_ROOT
    ).as_posix()
    return result


def _json_payload(path: Path) -> object:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Weather API returned an error: {payload.get('reason')}")
    return payload


def _write_gzip(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def build(*, workers: int = 4) -> dict[str, object]:
    model_path = PROCESSED_ROOT / "gridtoev_model_ready.csv"
    model_data = pd.read_csv(model_path)
    issue = pd.to_datetime(model_data["issue_timestamp_utc"], utc=True)
    target = pd.to_datetime(model_data["target_timestamp_utc"], utc=True)
    regions = validate_region_weights(pd.read_csv(WEIGHTS_PATH))

    first_run = (issue.min() - pd.Timedelta(hours=6)).floor("6h")
    last_run = (issue.max() - pd.Timedelta(hours=6)).floor("6h")
    runs = pd.date_range(first_run, last_run, freq="6h")
    run_specs = [_single_run_spec(run, regions) for run in runs]
    analysis_spec = _analysis_spec(
        first_run.floor("D"),
        target.max().ceil("D"),
        regions,
    )

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_download, spec): spec for spec in run_specs}
        for future in as_completed(futures):
            records.append(future.result())
    analysis_record = _download(analysis_spec)

    forecast_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    records.sort(key=lambda record: str(record["source_id"]))
    for record in records:
        source_id = str(record["source_id"])
        match = re.search(r"(\d{8}t\d{4}z)", source_id)
        if match is None:
            raise RuntimeError(f"Cannot recover model run from source id: {source_id}")
        stamp = match.group(1)
        run = pd.to_datetime(stamp, format="%Y%m%dt%H%Mz", utc=True)
        frame = parse_open_meteo_single_run(
            _json_payload(Path(str(record["local_path"]))),
            regions,
            run_initialized_at=run,
            downloaded_at=record["retrieved_at_utc"],
            source_url=str(record["url"]),
        )
        forecast_frames.append(frame)
        record["published_at_utc"] = frame["published_at_utc"].min().isoformat()
        record["first_interval_utc"] = frame["valid_timestamp_utc"].min().isoformat()
        record["last_interval_utc"] = frame["valid_timestamp_utc"].max().isoformat()
        record["row_count"] = int(len(frame))
        manifest.append(_relative_record(record))
    forecasts = pd.concat(forecast_frames, ignore_index=True, sort=False)

    analysis = parse_open_meteo_analysis(
        _json_payload(Path(str(analysis_record["local_path"]))),
        regions,
        downloaded_at=analysis_record["retrieved_at_utc"],
        source_url=str(analysis_record["url"]),
    )
    analysis_record["published_at_utc"] = "per-row valid time plus 6 hours"
    analysis_record["first_interval_utc"] = (
        analysis["valid_timestamp_utc"].min().isoformat()
    )
    analysis_record["last_interval_utc"] = (
        analysis["valid_timestamp_utc"].max().isoformat()
    )
    analysis_record["row_count"] = int(len(analysis))
    manifest.append(_relative_record(analysis_record))

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    forecast_path = PROCESSED_ROOT / "regional_weather_forecast_vintages.csv.gz"
    analysis_path = PROCESSED_ROOT / "regional_weather_analysis_proxy.csv.gz"
    _write_gzip(forecasts, forecast_path)
    _write_gzip(analysis, analysis_path)
    write_manifest(manifest, PROCESSED_ROOT / "regional_weather_source_manifest.csv")

    features, dictionary, quality = build_regional_weather_features(
        model_data,
        forecasts,
        analysis,
        regions,
    )
    feature_path = PROCESSED_ROOT / "regional_weather_features_30_60.csv"
    features.to_csv(
        feature_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    dictionary.to_csv(
        PROCESSED_ROOT / "regional_weather_feature_dictionary.csv",
        index=False,
    )
    quality.update(
        {
            "forecast_vintage_rows": int(len(forecasts)),
            "analysis_proxy_rows": int(len(analysis)),
            "model_run_count": int(forecasts["run_initialized_at_utc"].nunique()),
            "region_count": int(len(regions)),
            "regions": regions["region"].tolist(),
            "wind_capacity_total_mw": float(regions["wind_capacity_mw"].sum()),
            "solar_capacity_total_mw": float(regions["solar_capacity_mw"].sum()),
            "weight_version": str(regions["weight_version"].iloc[0]),
            "forecast_publication_lag_hours": 6,
            "analysis_availability_lag_hours": 6,
            "source_limitation": (
                "Open-Meteo ECMWF archives are the documented fallback because "
                "Met Eireann does not expose a convenient January 2026 point-forecast "
                "vintage archive. Regional capacity values are transparent engineering "
                "proxies and must be refreshed from an authoritative asset register."
            ),
            "licence": "Open-Meteo/ECMWF CC BY 4.0; attribution required.",
        }
    )
    (PROCESSED_ROOT / "regional_weather_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    ablation = build_weather_ablation_report(model_data, features, feature_path)
    ablation["feature_artifact"] = feature_path.relative_to(REPO_ROOT).as_posix()
    (BENCHMARK_ROOT / "ablation_report.json").write_text(
        json.dumps(ablation, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "experiment_name": ablation["experiment_name"],
                "feature_group": ablation["feature_group"],
                "feature_artifact_sha256": ablation["feature_artifact_sha256"],
                "overlap_rows": ablation["overlap_rows"],
                "development_rows": ablation["development_rows"],
                "candidate_feature_count": ablation["candidate_feature_count"],
                "mean_mae_improvement_fraction": ablation["metrics"][
                    "mean_mae_improvement_fraction"
                ],
                "status": ablation["status"],
                "production_decision": ablation["production_decision"],
                "reason": ablation["reason"],
                "final_test_accessed": ablation["final_test_accessed"],
            }
        ]
    ).to_csv(BENCHMARK_ROOT / "experiment_registry.csv", index=False)

    return {
        "model_runs": len(runs),
        "regions": len(regions),
        "forecast_rows": len(forecasts),
        "analysis_rows": len(analysis),
        "feature_rows": len(features),
        "feature_columns": len(features.columns),
        "ablation_status": ablation["status"],
        "production_decision": ablation["production_decision"],
        "mean_mae_improvement_fraction": ablation["metrics"][
            "mean_mae_improvement_fraction"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    print(json.dumps(build(workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
