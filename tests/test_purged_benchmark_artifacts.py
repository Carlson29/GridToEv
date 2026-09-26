"""Audit the committed, reproducible v2 benchmark evidence."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridtoev.benchmarking import load_benchmark_contract


class PurgedBenchmarkArtifactTests(unittest.TestCase):
    def test_v2_report_and_registry_use_distinct_purged_contract(self) -> None:
        contract = load_benchmark_contract(ROOT / "config" / "benchmark_contract.v2.json")
        report = json.loads(
            (ROOT / "benchmarks" / "v2-purged" / "benchmark_report.json").read_text(
                encoding="utf-8"
            )
        )
        registry = pd.read_csv(ROOT / "benchmarks" / "v2-purged" / "experiment_registry.csv")
        self.assertEqual(contract.label_maturity_policy, "target_before_score_issue")
        self.assertEqual(report["contract"]["contract_id"], contract.contract_id)
        self.assertTrue(report["selection"]["label_maturity"]["passed"])
        self.assertTrue(report["rolling_origin"]["label_maturity_verified"])
        self.assertTrue(report["rolling_origin"]["score_label_maturity_verified"])
        self.assertTrue(
            all(fold["purged_fit_rows"] == 3 for fold in report["rolling_origin"]["folds"])
        )
        self.assertTrue(
            all(fold["purged_score_rows"] == 3 for fold in report["rolling_origin"]["folds"])
        )
        self.assertTrue(report["frozen_baseline_verification"]["passed"])
        self.assertEqual(registry["contract_id"].tolist(), [contract.contract_id])
        self.assertEqual(
            report["final_test"]["metrics"]["regression"]["dispatch_down_mwh"]["mae"],
            contract.frozen_baseline["metrics"]["dispatch_down_mae_mwh"],
        )
        self.assertEqual(
            report["rolling_origin"]["aggregate"]["overall"]["mae"],
            contract.frozen_baseline["rolling_mae_mwh"],
        )

    def test_rollback_model_bytes_remain_frozen(self) -> None:
        contract = load_benchmark_contract(ROOT / "config" / "benchmark_contract.v2.json")
        artifact = ROOT / contract.frozen_baseline["model_artifact_path"]
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.assertEqual(actual, contract.frozen_baseline["model_artifact_sha256"])
