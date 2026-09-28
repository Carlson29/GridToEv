"""Evaluate Wind/Solar allocation of V2 predicted daily curtailment (handoff steps 3-5).

Experiment 1 writes benchmarks/daily_source_allocation_v2/evaluation.json.
Experiment 2 (capacity-aware physics share) writes physics_evaluation.json.
Neither saves a serving artifact; a source prediction route may be added only
after a report records ``release_gate.passed = true``.
"""

from __future__ import annotations

import json

from gridtoev.source_allocation import evaluate, evaluate_physics


def main() -> None:
    for report in (evaluate(), evaluate_physics()):
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
