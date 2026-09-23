"""Reproducible downloads and provenance records for public grid data."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


SCHEMA_VERSION = "1.0"
USER_AGENT = "GridToEV/1.0 (+https://github.com/Carlson29/GridToEv)"


@dataclass(frozen=True)
class SourceSpec:
    """One immutable public source file declared by the data catalog."""

    source_id: str
    provider: str
    report: str
    year: int | str
    url: str
    relative_path: str
    licence_note: str
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "SourceSpec":
        return cls(**values)  # type: ignore[arg-type]


def utc_iso(value: datetime | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _safe_destination(raw_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Raw source path must stay below raw_root: {relative_path}")
    # Keep this check lexical. Calling resolve() on many not-yet-created sibling
    # directories concurrently is unreliable on Windows network-backed folders.
    root = raw_root.absolute()
    destination = (root / relative).absolute()
    if not destination.is_relative_to(root):
        raise ValueError(f"Raw source path escapes raw_root: {relative_path}")
    return destination


def download_source(
    spec: SourceSpec,
    raw_root: Path,
    *,
    fetch_bytes: Callable[[str, int], bytes] | None = None,
    retrieved_at: datetime | pd.Timestamp | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> dict[str, object]:
    """Download once, atomically, and return a stable provenance record.

    A small sidecar stores the first retrieval time and checksum. A rerun never
    contacts the publisher when the raw file is already present, which makes a
    build resumable and leaves the original bytes untouched.
    """

    destination = _safe_destination(raw_root, spec.relative_path)
    metadata_path = destination.with_suffix(destination.suffix + ".source.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = destination.exists()

    if cache_hit:
        checksum = sha256_file(destination)
        if metadata_path.exists():
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached_metadata.get("source_id") not in {None, spec.source_id}:
                raise ValueError(
                    f"Cached source id does not match catalog: {destination}"
                )
            if cached_metadata.get("url") not in {None, spec.url}:
                raise ValueError(f"Cached source URL does not match catalog: {destination}")
            recorded_checksum = cached_metadata.get("sha256")
            if recorded_checksum and recorded_checksum != checksum:
                raise ValueError(
                    f"Cached source changed since download: {destination} "
                    f"({recorded_checksum} != {checksum})"
                )
            retrieval_time = cached_metadata.get("retrieved_at_utc")
        else:
            retrieval_time = utc_iso(
                datetime.fromtimestamp(destination.stat().st_mtime, tz=UTC)
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "source_id": spec.source_id,
                        "url": spec.url,
                        "retrieved_at_utc": retrieval_time,
                        "bytes": destination.stat().st_size,
                        "sha256": checksum,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    else:
        fetch = fetch_bytes or _default_fetch
        last_error: Exception | None = None
        payload: bytes | None = None
        for attempt in range(retries):
            try:
                payload = fetch(spec.url, timeout)
                break
            except Exception as error:  # pragma: no cover - exercised by integration use
                last_error = error
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        if payload is None:
            raise RuntimeError(f"Could not download {spec.url}") from last_error
        if not payload:
            raise ValueError(f"Publisher returned an empty file: {spec.url}")
        temporary_path = destination.with_suffix(destination.suffix + ".part")
        temporary_path.write_bytes(payload)
        temporary_path.replace(destination)
        checksum = sha256_file(destination)
        retrieval_time = utc_iso(retrieved_at or datetime.now(UTC))
        metadata_path.write_text(
            json.dumps(
                {
                    "source_id": spec.source_id,
                    "url": spec.url,
                    "retrieved_at_utc": retrieval_time,
                    "bytes": destination.stat().st_size,
                    "sha256": checksum,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        **asdict(spec),
        "local_path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": checksum,
        "retrieved_at_utc": retrieval_time,
        "cache_hit": cache_hit,
        "first_interval_utc": None,
        "last_interval_utc": None,
        "row_count": None,
    }


def load_source_catalog(path: Path) -> list[SourceSpec]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [SourceSpec.from_mapping(item) for item in document["sources"]]


def write_manifest(records: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values(["provider", "report", "source_id"])
    frame.to_csv(path, index=False)
