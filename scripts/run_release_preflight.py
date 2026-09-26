"""Regenerate the v2 release report; exit nonzero if a new candidate is required."""

import argparse
import json
from pathlib import Path

from gridtoev.release_guard import DEFAULT_REPORT_PATH, build_release_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--require-candidate", action="store_true")
    args = parser.parse_args()
    report = build_release_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Rollback verified: {report['rollback']['verified']}")
    print(f"Candidate approved: {report['decision']['candidate_approved']}")
    if not report["rollback"]["verified"] or (args.require_candidate and not report["decision"]["candidate_approved"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
