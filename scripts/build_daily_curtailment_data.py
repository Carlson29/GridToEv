"""Build the optional daily model dataset from archived 24-hour-old forecasts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from gridtoev.daily_curtailment import DEFAULT_DAILY_DATASET, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 31))
    parser.add_argument("--output", type=Path, default=DEFAULT_DAILY_DATASET)
    parser.add_argument("--refresh", action="store_true", help="Re-download cached raw source JSON")
    args = parser.parse_args()
    frame = build_dataset(start=args.start, end=args.end, refresh=args.refresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, compression={"method": "gzip", "mtime": 0})
    print(json.dumps({
        "output": str(args.output),
        "rows": len(frame),
        "start_utc": frame["issue_timestamp_utc"].min().isoformat(),
        "end_utc": frame["issue_timestamp_utc"].max().isoformat(),
        "event_rate": float(frame["curtailment_event"].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
