#!/usr/bin/env python3
"""
PyTorch Deep Learning Market Regime Benchmark Suite
Standalone E2E Test Runner & Verification Harness

Usage:
    python tests/e2e/run_tests.py [OPTIONS]

Options:
    --tier {1,2,3,4,all}   Run specific test tier (default: 'all')
    -v, --verbose          Show detailed test names and outputs
    --report               Generate markdown test readiness report
    --output-file PATH     Path to write report (default: stdout)
    --list                 List all discovered test cases without executing
"""

import sys
import os
import time
import argparse
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.test_helpers import ensure_headless_matplotlib
ensure_headless_matplotlib()

TIER_MODULES = {
    "1": ("Tier 1: Feature Coverage", "tests.e2e.test_tier1_features"),
    "2": ("Tier 2: Boundary & Corner Cases", "tests.e2e.test_tier2_boundaries"),
    "3": ("Tier 3: Cross-Feature Combinations", "tests.e2e.test_tier3_combinations"),
    "4": ("Tier 4: Real-World Scenarios", "tests.e2e.test_tier4_real_world"),
}


class TierTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tier_records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.tier_records.append((test.id(), "PASS", ""))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.tier_records.append((test.id(), "FAIL", str(err[1])))

    def addError(self, test, err):
        super().addError(test, err)
        self.tier_records.append((test.id(), "ERROR", str(err[1])))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.tier_records.append((test.id(), "SKIP", reason))


def run_tier(tier_key, verbose=False):
    """Executes a single test tier and returns summary statistics."""
    title, module_name = TIER_MODULES[tier_key]
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(module_name)
    except Exception as e:
        print(f"Error loading {module_name}: {e}")
        return {
            "tier": tier_key,
            "title": title,
            "total": 0,
            "passed": 0,
            "failed": 1,
            "skipped": 0,
            "errors": 1,
            "duration": 0.0,
            "records": [("load_tests", "ERROR", str(e))]
        }

    runner = unittest.TextTestRunner(
        resultclass=TierTestResult,
        verbosity=2 if verbose else 1,
        stream=sys.stdout if verbose else open(os.devnull, 'w')
    )

    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - errors - skipped

    return {
        "tier": tier_key,
        "title": title,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "duration": duration,
        "records": result.tier_records,
    }


def generate_markdown_report(results, total_duration):
    """Generates a structured markdown report summarizing test status and coverage."""
    grand_total = sum(r["total"] for r in results)
    grand_passed = sum(r["passed"] for r in results)
    grand_failed = sum(r["failed"] for r in results)
    grand_errors = sum(r["errors"] for r in results)
    grand_skipped = sum(r["skipped"] for r in results)

    status_badge = "✅ PASSED" if grand_failed == 0 and grand_errors == 0 else "❌ FAILED"

    lines = [
        "# E2E Test Execution & Readiness Report",
        "",
        f"**Status**: {status_badge}  ",
        f"**Timestamp**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ",
        f"**Total Duration**: {total_duration:.3f}s  ",
        "",
        "## Summary by Tier",
        "",
        "| Tier | Name | Total Tests | Passed | Failed | Errors | Skipped (Pending Milestone) | Duration |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| Tier {r['tier']} | {r['title']} | {r['total']} | {r['passed']} | {r['failed']} | {r['errors']} | {r['skipped']} | {r['duration']:.3f}s |"
        )

    lines.append(
        f"| **TOTAL** | **All 4 Tiers** | **{grand_total}** | **{grand_passed}** | **{grand_failed}** | **{grand_errors}** | **{grand_skipped}** | **{total_duration:.3f}s** |"
    )
    lines.append("")
    lines.append("## Coverage Checklist")
    lines.append("")
    lines.append("- [x] **Tier 1**: Feature Coverage (14 features, chronological split, 3D warmup sequences, PyTorch architectures, baseline)")
    lines.append("- [x] **Tier 2**: Boundary & Corner Cases (empty data, zero volatility, zero volume, single/multi-index, extreme thresholds, backtester bounds)")
    lines.append("- [x] **Tier 3**: Cross-Feature Combinations (pipeline -> sequence creation -> model training -> strategy -> backtester)")
    lines.append("- [x] **Tier 4**: Real-World Application Scenarios (SPY 5-year data, 240-day out-of-sample test, performance metrics, headless visualizations, notebook)")
    lines.append("")
    lines.append("## Skipped Tests Rationale (Progressive Testability)")
    lines.append("Skipped tests represent features scheduled for parallel implementation in upcoming milestones:")
    lines.append("- **Milestone 1**: Extended 14 features (`candle_body`, `intraday_range`, `volume_ratio`, `ret_1d`, `ret_10d`, `vol_10d`, `sma20_ratio`, `macd_norm`, `macd_hist_norm`, `vol_regime`), `train_val_test_split_scale`, and `create_sequences` with context warmup.")
    lines.append("- **Milestones 2–4**: PyTorch deep learning architectures in `torch_models.py` (`RegimeDataset`, `MLPRegimeNet`, `RecurrentRegimeNet`, `Conv1DRegimeNet`, `TransformerRegimeNet`, `DeepRegimeClassifier`).")
    lines.append("- **Milestone 5**: Part C Portfolio visualizations (`plot_regime_agreement_barcode`, `plot_risk_return_frontier`).")
    lines.append("")
    lines.append("When implementation agents complete each milestone, re-running this test suite will automatically execute and verify those components against the interface contracts.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Market Regime Benchmark Suite — E2E Test Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all", help="Target test tier")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose test execution output")
    parser.add_argument("--report", action="store_true", help="Generate markdown readiness report")
    parser.add_argument("--output-file", type=str, default=None, help="File to write report")
    parser.add_argument("--list", action="store_true", help="List all discovered tests without running")
    args = parser.parse_args()

    tiers_to_run = ["1", "2", "3", "4"] if args.tier == "all" else [args.tier]

    if args.list:
        loader = unittest.TestLoader()
        for t in tiers_to_run:
            title, mod = TIER_MODULES[t]
            print(f"\n--- {title} ({mod}) ---")
            suite = loader.loadTestsFromName(mod)
            for test in suite:
                if hasattr(test, '_tests'):
                    for subtest in test._tests:
                        print(f"  {subtest.id()}")
                else:
                    print(f"  {test.id()}")
        sys.exit(0)

    print("======================================================================")
    print("  PyTorch Deep Learning Market Regime Benchmark Suite — E2E Runner")
    print("======================================================================")

    start_all = time.time()
    results = []
    for t in tiers_to_run:
        title, _ = TIER_MODULES[t]
        print(f"\n>>> Running {title}...")
        res = run_tier(t, verbose=args.verbose)
        results.append(res)
        print(f"    Completed: {res['total']} tests | {res['passed']} passed | {res['skipped']} skipped | {res['failed']} failed | {res['errors']} errors ({res['duration']:.3f}s)")

    total_duration = time.time() - start_all

    grand_total = sum(r["total"] for r in results)
    grand_passed = sum(r["passed"] for r in results)
    grand_failed = sum(r["failed"] for r in results)
    grand_errors = sum(r["errors"] for r in results)
    grand_skipped = sum(r["skipped"] for r in results)

    print("\n======================================================================")
    print(f"  SUMMARY: {grand_total} Tests | {grand_passed} Passed | {grand_skipped} Skipped | {grand_failed} Failed | {grand_errors} Errors")
    print(f"  Total Duration: {total_duration:.3f} seconds")
    print("======================================================================")

    if args.report or args.output_file:
        report_md = generate_markdown_report(results, total_duration)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(report_md)
            print(f"\nReport written to: {args.output_file}")
        else:
            print("\n" + report_md)

    if grand_failed > 0 or grand_errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
