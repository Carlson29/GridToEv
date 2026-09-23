from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    path: str,
    api_key: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Request a GridToEV prediction")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--capacity-mw", type=float, default=100.0)
    parser.add_argument("--timestamp", help="Historical issue time, for example 2026-01-31T22:00:00Z")
    parser.add_argument("--horizon", type=int, choices=(30, 60), default=30)
    args = parser.parse_args()

    if args.timestamp:
        result = request_json(
            args.base_url,
            "/predict/from-dataset",
            args.api_key,
            {
                "issue_timestamp_utc": args.timestamp,
                "forecast_horizon_minutes": args.horizon,
                "flexible_load_capacity_mw": args.capacity_mw,
            },
        )
    else:
        query = urlencode({"flexible_load_capacity_mw": args.capacity_mw})
        result = request_json(
            args.base_url,
            f"/predict/latest?{query}",
            args.api_key,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
