"""Reproduce the v2 label-mature benchmark without modifying v1.1.0 artifacts."""

from gridtoev.benchmarking import run_baseline_benchmark
from gridtoev.constants import PROJECT_ROOT


if __name__ == "__main__":
    result = run_baseline_benchmark(
        contract_path=PROJECT_ROOT / "config" / "benchmark_contract.v2.json",
        output_dir=PROJECT_ROOT / "benchmarks" / "v2-purged",
    )
    metrics = result.report["final_test"]["metrics"]["regression"]["dispatch_down_mwh"]
    print(f"Purged boundary verified: {result.report['rolling_origin']['label_maturity_verified']}")
    print(f"Frozen v2 baseline verified: {result.report['frozen_baseline_verification']['passed']}")
    print(f"Final-test MAE: {metrics['mae']:.6f} MWh")
    print(f"Report: {result.report_path}")
