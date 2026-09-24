"""Build leakage-safe forecast-error and grid-headroom model features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gridtoev.forecast_features import build_forecast_feature_table


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


def build_and_write(
    asof_path: Path,
    history_path: Path,
    vintages_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Read normalized inputs and write the three Issue #4 artifacts."""

    asof = pd.read_csv(asof_path)
    history = pd.read_csv(history_path)
    vintages = pd.read_csv(vintages_path, low_memory=False)
    table, dictionary, quality = build_forecast_feature_table(asof, history, vintages)

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        output_dir / "forecast_model_features_30_60.csv",
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    dictionary.to_csv(
        output_dir / "forecast_model_feature_dictionary.csv",
        index=False,
    )
    (output_dir / "forecast_model_feature_quality_report.json").write_text(
        json.dumps(quality, indent=2),
        encoding="utf-8",
    )
    return table, dictionary, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asof",
        type=Path,
        default=PROCESSED_ROOT / "forecast_features_asof_30_60.csv",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=PROCESSED_ROOT / "eirgrid_core_history_30min.csv.gz",
    )
    parser.add_argument(
        "--vintages",
        type=Path,
        default=PROCESSED_ROOT / "forecast_vintages.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, dictionary, quality = build_and_write(
        args.asof.resolve(),
        args.history.resolve(),
        args.vintages.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "rows": len(table),
                "columns": len(table.columns),
                "dictionary_rows": len(dictionary),
                "future_publication_violations": quality[
                    "future_publication_violations"
                ],
                "future_observation_violations": quality[
                    "future_observation_violations"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
