import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from gridtoev.data_sources import SourceSpec, download_source, sha256_file


class DataSourceTests(unittest.TestCase):
    def test_transient_download_failure_is_retried(self) -> None:
        calls = 0

        def flaky_fetch(url: str, timeout: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("temporary publisher timeout")
            return b"recovered"

        spec = SourceSpec(
            source_id="retry",
            provider="test",
            report="test",
            year=2024,
            url="https://example.test/retry.csv",
            relative_path="retry.csv",
            licence_note="test",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("gridtoev.data_sources.time.sleep"):
                record = download_source(
                    spec, Path(temporary_directory), fetch_bytes=flaky_fetch
                )
        self.assertEqual(calls, 3)
        self.assertFalse(record["cache_hit"])

    def test_download_rejects_path_traversal(self) -> None:
        spec = SourceSpec(
            source_id="unsafe",
            provider="test",
            report="test",
            year=2024,
            url="https://example.test/file",
            relative_path="../outside.csv",
            licence_note="test",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "stay below raw_root"):
                download_source(spec, Path(temporary_directory), fetch_bytes=lambda *_: b"x")

    def test_download_is_cached_and_hash_is_reproducible(self) -> None:
        calls: list[str] = []

        def fetch(url: str, timeout: int) -> bytes:
            calls.append(url)
            self.assertEqual(timeout, 120)
            return b"stable public data"

        spec = SourceSpec(
            source_id="test_system_2024",
            provider="EirGrid",
            report="system",
            year=2024,
            url="https://example.test/system.xlsx",
            relative_path="eirgrid/system/2024/system.xlsx",
            licence_note="Publisher open-data terms apply.",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_root = Path(temporary_directory)
            retrieved_at = datetime(2026, 9, 23, 12, tzinfo=UTC)
            first = download_source(
                spec,
                raw_root,
                fetch_bytes=fetch,
                retrieved_at=retrieved_at,
            )
            second = download_source(
                spec,
                raw_root,
                fetch_bytes=fetch,
                retrieved_at=retrieved_at,
            )

            self.assertEqual(calls, [spec.url])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["sha256"], sha256_file(raw_root / spec.relative_path))
            self.assertEqual(first["schema_version"], "1.0")
            self.assertEqual(first["retrieved_at_utc"], "2026-09-23T12:00:00Z")

            changed_url = SourceSpec(
                **{
                    **spec.__dict__,
                    "url": "https://example.test/different-system.xlsx",
                }
            )
            with self.assertRaisesRegex(ValueError, "URL does not match"):
                download_source(changed_url, raw_root, fetch_bytes=fetch)


if __name__ == "__main__":
    unittest.main()
