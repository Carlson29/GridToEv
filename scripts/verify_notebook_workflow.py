"""Run the two production notebooks offline without touching committed artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = (
    "hacktheclimate/generation.csv",
    "hacktheclimate/load.csv",
    "hacktheclimate/prices.csv",
    "eirgrid/system_data_qtr_hourly_2026_v7.xlsx",
    "eirgrid/dd_half_hourly_2026_v9.xlsx",
)
NOTEBOOKS = (
    "01_build_model_ready_dataset.ipynb",
    "02_train_and_export_models.ipynb",
)


def run(raw_cache: Path, output: Path) -> dict:
    """Check raw inputs, execute both notebooks, and smoke-test both horizons."""
    raw_cache = raw_cache.resolve()
    missing = [name for name in RAW_FILES if not (raw_cache / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Offline notebook verification needs cached raw sources: " + ", ".join(missing)
        )

    # The notebooks use Path.cwd() for outputs. A disposable root prevents an
    # accidental overwrite of the versioned production dataset or model.
    with tempfile.TemporaryDirectory(prefix="gridtoev-notebook-smoke-") as directory:
        scratch = Path(directory)
        shutil.copytree(ROOT / "src", scratch / "src")
        (scratch / "notebooks").mkdir()
        for name in NOTEBOOKS:
            shutil.copy2(ROOT / "notebooks" / name, scratch / "notebooks" / name)
        for name in RAW_FILES:
            destination = scratch / "data" / "raw" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_cache / name, destination)

        # The default python3 kernelspec uses `python`, so put this process's
        # environment first on PATH to use its pinned notebook dependencies.
        previous_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + previous_path
        try:
            executed = []
            for name in NOTEBOOKS:
                notebook = nbformat.read(scratch / "notebooks" / name, as_version=4)
                NotebookClient(
                    notebook,
                    timeout=1200,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(scratch)}},
                    allow_errors=False,
                ).execute()
                executed.append(name)
        finally:
            os.environ["PATH"] = previous_path

        dataset_path = scratch / "data" / "processed" / "gridtoev_model_ready.csv"
        model_path = scratch / "models" / "gridtoev_model_bundle.joblib"
        metrics_path = scratch / "models" / "training_metrics.json"
        metadata_path = scratch / "models" / "model_metadata.json"
        for path in (dataset_path, model_path, metrics_path, metadata_path):
            if not path.is_file():
                raise AssertionError(f"Notebook output missing: {path.relative_to(scratch)}")

        sys.path.insert(0, str(ROOT / "src"))
        from gridtoev.inference import PredictionService
        from gridtoev.release_preflight import sha256_file

        service = PredictionService(model_path, dataset_path)
        service.load()
        predictions = service.predict_latest()
        horizons = [item["forecast_horizon_minutes"] for item in predictions]
        if horizons != [30, 60]:
            raise AssertionError(f"Unexpected notebook prediction horizons: {horizons}")
        for prediction in predictions:
            total = prediction["predicted_dispatch_down_mwh"]
            components = (
                prediction["predicted_curtailment_mwh"]
                + prediction["predicted_constraint_mwh"]
            )
            if abs(total - components) > 1e-9:
                raise AssertionError("Notebook model components do not reconcile")

        dataset = pd.read_csv(dataset_path)
        frozen_sha = sha256_file(ROOT / "data" / "processed" / "gridtoev_model_ready.csv")
        report = {
            "schema_version": 1,
            "notebooks_executed": executed,
            "source_files_preloaded": list(RAW_FILES),
            "dataset_rows": len(dataset),
            "dataset_columns": len(dataset.columns),
            "rebuilt_dataset_sha256": sha256_file(dataset_path),
            "frozen_dataset_sha256": frozen_sha,
            "rebuilt_dataset_matches_frozen": sha256_file(dataset_path) == frozen_sha,
            "trained_model_version": service.metadata["model_version"],
            "prediction_horizons_minutes": horizons,
            "prediction_components_reconciled": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "release_preflight" / "notebook_smoke_report.json",
    )
    args = parser.parse_args()
    report = run(args.raw_cache, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
