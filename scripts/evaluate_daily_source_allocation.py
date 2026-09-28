"""Evaluate Wind/Solar allocation of V2 predicted daily curtailment (handoff steps 3-5).

Writes benchmarks/daily_source_allocation_v2/evaluation.json. It does not save a
serving artifact or add an API route unless the recorded release gate passes.
"""

from __future__ import annotations

import json

from gridtoev.source_allocation import evaluate


def main() -> None:
    report = evaluate()
    summary = {
        "selected_on_validation": report["selected_on_validation"],
        "test_combined_source_mae_mwh": {
            name: round(scores["combined_source_mae_mwh"], 1) for name, scores in report["test_end_to_end"].items()
        },
        "release_gate": report["release_gate"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
