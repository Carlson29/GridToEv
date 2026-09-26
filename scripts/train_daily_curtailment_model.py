"""Train and evaluate the opt-in daily curtailment model without touching v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gridtoev.daily_curtailment import DEFAULT_DAILY_DATASET
from gridtoev.daily_model import DEFAULT_ARTIFACT, DEFAULT_REPORT, train_daily_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DAILY_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(train_daily_model(args.dataset, args.artifact, args.report), indent=2))


if __name__ == "__main__":
    main()
