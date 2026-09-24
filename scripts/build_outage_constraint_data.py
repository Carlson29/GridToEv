"""Build leakage-safe EirGrid outage and ECP constraint signals for Issue #6."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridtoev.data_sources import SourceSpec, download_source, write_manifest
from gridtoev.outage_constraints import (
    ECP_SCHEMA_VERSION,
    OUTAGE_SCHEMA_VERSION,
    build_outage_ablation_report,
    build_outage_constraint_features,
    deduplicate_outage_snapshots,
    parse_ecp_constraint_report,
    parse_generation_outage_plan,
    parse_transmission_outage_programme,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "outage_constraint_signals"
LICENCE_NOTE = "Review EirGrid data terms and website disclaimer before redistribution."

SOURCES = (
    {
        "spec": SourceSpec(
            source_id="eirgrid_generation_outage_plan_2026",
            provider="EirGrid",
            report="all_island_generation_outage_plan",
            year=2026,
            url=(
                "https://cms.eirgrid.ie/sites/default/files/publications/"
                "Outage-Week-52-2025-15-2026.xlsx"
            ),
            relative_path=(
                "eirgrid/outages/generation/2026/"
                "Outage-Week-52-2025-15-2026.xlsx"
            ),
            licence_note=LICENCE_NOTE,
            schema_version=OUTAGE_SCHEMA_VERSION,
        ),
        "published_at_utc": "2025-12-17T14:07:59Z",
        "family": "generation",
    },
    {
        "spec": SourceSpec(
            source_id="eirgrid_transmission_outage_programme_2026",
            provider="EirGrid",
            report="transmission_outage_programme",
            year=2026,
            url=(
                "https://cms.eirgrid.ie/sites/default/files/publications/"
                "2026-Transmission-Outage-Programme-20260123.xlsx"
            ),
            relative_path=(
                "eirgrid/outages/transmission/2026/"
                "2026-Transmission-Outage-Programme-20260123.xlsx"
            ),
            licence_note=LICENCE_NOTE,
            schema_version=OUTAGE_SCHEMA_VERSION,
        ),
        "published_at_utc": "2026-01-23T16:19:11Z",
        "family": "transmission",
    },
    {
        "spec": SourceSpec(
            source_id="eirgrid_ecp_2_5_constraint_forecast",
            provider="EirGrid",
            report="ecp_constraint_forecast",
            year=2026,
            url=(
                "https://cms.eirgrid.ie/sites/default/files/publications/"
                "ECP-2.5-Constraints-Analysis-Excel-For-Publication.xlsx"
            ),
            relative_path=(
                "eirgrid/constraints/"
                "ECP-2.5-Constraints-Analysis-Excel-For-Publication.xlsx"
            ),
            licence_note=LICENCE_NOTE,
            schema_version=ECP_SCHEMA_VERSION,
        ),
        "published_at_utc": "2026-03-11T16:30:04Z",
        "family": "ecp",
    },
)


def _relative_manifest_record(record: dict[str, object]) -> dict[str, object]:
    result = record.copy()
    result.pop("cache_hit", None)
    result["local_path"] = Path(str(result["local_path"])).relative_to(
        REPO_ROOT
    ).as_posix()
    return result


def _write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def build() -> dict[str, object]:
    """Download fixed public snapshots, normalize them, and build model features."""

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    outage_frames: list[pd.DataFrame] = []
    constraint_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []

    for source in SOURCES:
        spec = source["spec"]
        assert isinstance(spec, SourceSpec)
        record = download_source(spec, RAW_ROOT)
        path = Path(str(record["local_path"]))
        published = str(source["published_at_utc"])
        if source["family"] == "generation":
            parsed = parse_generation_outage_plan(
                path,
                published_at=published,
                source_url=spec.url,
            )
            outage_frames.append(parsed)
            record["first_interval_utc"] = parsed["start_timestamp_utc"].min().isoformat()
            record["last_interval_utc"] = parsed["end_timestamp_utc"].max().isoformat()
        elif source["family"] == "transmission":
            parsed = parse_transmission_outage_programme(
                path,
                published_at=published,
                source_url=spec.url,
            )
            outage_frames.append(parsed)
            record["first_interval_utc"] = parsed["start_timestamp_utc"].min().isoformat()
            record["last_interval_utc"] = parsed["end_timestamp_utc"].max().isoformat()
        else:
            parsed = parse_ecp_constraint_report(
                path,
                published_at=published,
                source_url=spec.url,
            )
            constraint_frames.append(parsed)
            record["first_interval_utc"] = published
            record["last_interval_utc"] = published
        record["published_at_utc"] = published
        record["row_count"] = int(len(parsed))
        manifest.append(_relative_manifest_record(record))

    outages = deduplicate_outage_snapshots(
        pd.concat(outage_frames, ignore_index=True, sort=False)
    )
    constraints = pd.concat(constraint_frames, ignore_index=True, sort=False)
    outage_path = PROCESSED_ROOT / "outage_events.csv.gz"
    constraint_path = PROCESSED_ROOT / "ecp_constraint_pressure.csv.gz"
    _write_csv_gzip(outages, outage_path)
    _write_csv_gzip(constraints, constraint_path)
    write_manifest(manifest, PROCESSED_ROOT / "outage_constraint_source_manifest.csv")

    model_path = PROCESSED_ROOT / "gridtoev_model_ready.csv"
    model_data = pd.read_csv(model_path)
    features, dictionary, quality = build_outage_constraint_features(
        model_data,
        outages,
        constraints,
    )
    feature_path = PROCESSED_ROOT / "outage_constraint_features_30_60.csv"
    features.to_csv(
        feature_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    dictionary.to_csv(
        PROCESSED_ROOT / "outage_constraint_feature_dictionary.csv",
        index=False,
    )
    quality.update(
        {
            "outage_event_rows": int(len(outages)),
            "generation_event_rows": int(
                outages["source_family"].eq("generation_plan").sum()
            ),
            "transmission_event_rows": int(
                outages["source_family"].eq("transmission_programme").sum()
            ),
            "ecp_constraint_rows": int(len(constraints)),
            "source_file_count": len(manifest),
            "source_families": sorted(
                {str(record["report"]) for record in manifest}
            ),
            "interval_semantics": "half-open [start, end)",
            "publication_policy": (
                "newest snapshot with published_at_utc <= issue_timestamp_utc"
            ),
        }
    )
    (PROCESSED_ROOT / "outage_constraint_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    ablation = build_outage_ablation_report(
        model_data,
        features,
        feature_path,
    )
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
        "source_files": len(manifest),
        "outage_events": len(outages),
        "constraint_rows": len(constraints),
        "feature_rows": len(features),
        "feature_columns": len(features.columns),
        "ablation_status": ablation["status"],
        "production_decision": ablation["production_decision"],
        "mean_mae_improvement_fraction": ablation["metrics"][
            "mean_mae_improvement_fraction"
        ],
    }


def main() -> None:
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
