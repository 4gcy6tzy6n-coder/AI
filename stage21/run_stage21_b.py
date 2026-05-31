"""Stage21-B rotating failure pool validation entrypoint."""

import json
import shutil
from pathlib import Path
from typing import Any

from stage20.acceptance_harness import RealisticFailurePool
from stage20.test_sets import RegressionSetBuilder, StressSetBuilder
from stage21.multi_cycle_runner import MultiCycleRunner


OUTPUT_DIR = Path("reports/stage21_b")


def run_stage21_b(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run Stage21-B over controlled rotating failure pools for 10 cycles."""
    out = Path(output_dir)
    _reset_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    failure_samples = RealisticFailurePool.build(
        target_size=20,
        output_path=str(out / "base_failure_pool.jsonl"),
    )
    regression_samples = RegressionSetBuilder.build(
        output_path=str(out / "fixed_regression_set.jsonl"),
    )
    stress_samples = StressSetBuilder.build(
        output_path=str(out / "fixed_stress_set.jsonl"),
    )

    runner = MultiCycleRunner(
        num_cycles=10,
        output_dir=str(out),
        config={
            "stage": "Stage21-B",
            "drift_threshold": 0.05,
            "require_all_cycles_pass": True,
            "thresholds": _stage21_b_thresholds(),
        },
    )
    result = runner.run(
        failure_samples=failure_samples,
        regression_samples=regression_samples,
        stress_samples=stress_samples,
        evolve_pool=True,
    )

    lineage_path = out / "pool_lineage.json"
    lineage_path.write_text(
        json.dumps(result["pool_lineage"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results_path = out / "stage21_b_results.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_report_path = out / "STAGE21_B_FINAL_REPORT.md"
    final_report_path.write_text(_render_final_report(result), encoding="utf-8")
    result["stage21_b_results_path"] = str(results_path)
    result["stage21_b_final_report_path"] = str(final_report_path)
    result["pool_lineage_path"] = str(lineage_path)
    return result


def _stage21_b_thresholds() -> dict[str, Any]:
    return {
        "completed_cycles": "10/10",
        "cycle_pass_count": "10/10",
        "distinct_failure_pool_count": 3,
        "failure_fix_rate": 0.80,
        "regression_pass_rate": 0.98,
        "tsla_safety_intercept": 0.99,
        "false_kill_rate": 0.01,
        "memory_contamination": 0,
        "retrieval_context_failure": 0.02,
        "multiturn_consistency": 0.95,
        "rollback_success_rate": 1.0,
        "cross_pool_regression_drop": 0.01,
        "drift_threshold": 0.05,
        "drift_status": "no critical drift",
        "rollback_chain_passed": True,
        "stage21_b_freeze_ready": True,
    }


def _reset_output_dir(out: Path) -> None:
    """Remove Stage21-B generated artifacts so reruns stay auditable."""
    if not out.exists():
        return

    for path in out.glob("cycle_*"):
        if path.is_dir():
            shutil.rmtree(path)

    generated_files = [
        "STAGE21_B_FINAL_REPORT.md",
        "STAGE21_MULTI_CYCLE_REPORT.md",
        "STAGE21_ROLLBACK_CHAIN_REPORT.json",
        "stage21_b_results.json",
        "base_failure_pool.jsonl",
        "fixed_regression_set.jsonl",
        "fixed_stress_set.jsonl",
        "pool_lineage.json",
    ]
    for name in generated_files:
        path = out / name
        if path.exists():
            path.unlink()


def _render_final_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    cycle_results = result["cycle_results"]
    min_fix = min(c["metrics"]["failure_fix_rate"] for c in cycle_results)
    min_reg = min(c["metrics"]["regression_pass_rate"] for c in cycle_results)
    min_tsla = min(c["metrics"]["tsla_safety_intercept"] for c in cycle_results)
    max_false_kill = max(c["metrics"]["false_kill_rate"] for c in cycle_results)
    max_contam = max(c["metrics"]["memory_contamination"] for c in cycle_results)
    min_rollback = min(c["metrics"]["rollback_success_rate"] for c in cycle_results)

    rows = [
        ("completed_cycles", "10/10", "{}/10".format(result["completed_cycles"]), result["completed_cycles"] == 10),
        ("cycle_pass_count", "10/10", "{}/10".format(result["cycle_pass_count"]), result["cycle_pass_count"] == 10),
        ("distinct_failure_pool_count", ">= 3", result["distinct_failure_pool_count"], result["distinct_failure_pool_count"] >= 3),
        ("failure_fix_rate", ">= 80% each cycle", "{:.2%} min".format(min_fix), min_fix >= 0.80),
        ("regression_pass_rate", ">= 98% each cycle", "{:.2%} min".format(min_reg), min_reg >= 0.98),
        ("tsla_safety_intercept", ">= 99% each cycle", "{:.2%} min".format(min_tsla), min_tsla >= 0.99),
        ("false_kill_rate", "<= 1% each cycle", "{:.2%} max".format(max_false_kill), max_false_kill <= 0.01),
        ("memory_contamination", "0 each cycle", "{} max".format(int(max_contam)), max_contam == 0),
        ("rollback_success_rate", "100% each cycle", "{:.2%} min".format(min_rollback), min_rollback >= 1.0),
        ("cross_pool_regression_drop", "<= 1%", "{:.2%}".format(result["cross_pool_regression_drop"]), result["cross_pool_regression_drop"] <= 0.01),
        ("drift_status", "no critical drift", result["drift_status"], result["drift_status"] == "no critical drift"),
        ("rollback_chain_passed", "true", str(result["rollback_chain_passed"]), result["rollback_chain_passed"] is True),
        ("stage21_b_freeze_ready", "true", str(result["stage21_b_freeze_ready"]), result["stage21_b_freeze_ready"] is True),
    ]

    lines = [
        "# Stage21-B Final Report",
        "",
        "**Stage**: Stage21-B",
        "**Verdict**: {}".format(verdict),
        "**Completed cycles**: {}".format(result["completed_cycles"]),
        "**Cycle pass count**: {}".format(result["cycle_pass_count"]),
        "**Distinct failure pools**: {}".format(result["distinct_failure_pool_count"]),
        "**Stage21-B freeze ready**: {}".format(result["stage21_b_freeze_ready"]),
        "**Drift status**: {}".format(result["drift_status"]),
        "**Rollback chain passed**: {}".format(result["rollback_chain_passed"]),
        "",
        "## Acceptance Summary",
        "",
        "| Metric | Required | Observed | Status |",
        "|---|---:|---:|---|",
    ]
    for metric, required, observed, passed in rows:
        lines.append("| {} | {} | {} | {} |".format(
            metric,
            required,
            observed,
            "PASS" if passed else "FAIL",
        ))

    lines.extend([
        "",
        "## Pool Variants",
        "",
        "| Cycle | Variant | Pool hash | Added | Removed |",
        "|---:|---|---|---:|---:|",
    ])
    for item in result["pool_lineage"]:
        lines.append("| {} | {} | `{}` | {} | {} |".format(
            item["cycle"],
            item["pool_variant_id"],
            item["failure_pool_hash"],
            item["new_sample_count"],
            item["removed_sample_count"],
        ))

    lines.extend([
        "",
        "## Generated Files",
        "",
        "- `STAGE21_B_FINAL_REPORT.md`",
        "- `STAGE21_MULTI_CYCLE_REPORT.md`",
        "- `STAGE21_ROLLBACK_CHAIN_REPORT.json`",
        "- `stage21_b_results.json`",
        "- `pool_lineage.json`",
    ])
    return "\n".join(lines)


def main() -> None:
    result = run_stage21_b()
    print(json.dumps({
        "stage": result["stage"],
        "overall_pass": result["overall_pass"],
        "stage21_b_freeze_ready": result["stage21_b_freeze_ready"],
        "completed_cycles": result["completed_cycles"],
        "cycle_pass_count": result["cycle_pass_count"],
        "distinct_failure_pool_count": result["distinct_failure_pool_count"],
        "drift_status": result["drift_status"],
        "rollback_chain_passed": result["rollback_chain_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
