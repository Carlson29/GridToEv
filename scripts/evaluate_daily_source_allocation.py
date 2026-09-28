"""Evaluate Wind/Solar allocation of V2 predicted daily curtailment (handoff steps 3-5).

Experiment 1 writes benchmarks/daily_source_allocation_v2/evaluation.json.
Experiment 2 (capacity-aware physics share) writes physics_evaluation.json and
the frozen serving parameters models/v2/source_allocation_physics.json.

For the fresh-data confirmation, build an extended V2-format dataset to a
SEPARATE file (the frozen V2 dataset is checksum-verified by the live model)
and pass it with --confirmation-dataset. Neither model is retrained: both are
fitted on 2024-2025 only, so the new days stay an unbiased test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gridtoev.daily_model import DEFAULT_DAILY_DATASET
from gridtoev.source_allocation import evaluate, evaluate_physics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-dataset", type=Path, default=DEFAULT_DAILY_DATASET,
        help="V2-format daily dataset that extends past 2026-08-30 (experiment 2 only).",
    )
    args = parser.parse_args()
    for report in (evaluate(), evaluate_physics(dataset_path=args.confirmation_dataset)):
        summary = {
            "experiment": report["experiment"],
            "status": report["status"],
            "selected_on_validation": report["selected_on_validation"],
            "test_combined_source_mae_mwh": {
                name: round(scores["combined_source_mae_mwh"], 1) for name, scores in report["test_end_to_end"].items()
            },
            "release_gate": report["release_gate"],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
