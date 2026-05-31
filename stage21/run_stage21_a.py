"""Stage21-A fixed failure pool validation entrypoint."""

import json
import shutil
from pathlib import Path
from typing import Any

from stage20.acceptance_harness import RealisticFailurePool
from stage20.test_sets import RegressionSetBuilder, StressSetBuilder
from stage21.multi_cycle_runner import MultiCycleRunner


OUTPUT_DIR = Path("reports/stage21_a")


def run_stage21_a(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run Stage21-A over one frozen failure pool for exactly 5 cycles."""
    out = Path(output_dir)
    _reset_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    failure_samples = RealisticFailurePool.build(
        target_size=20,
        output_path=str(out / "fixed_failure_pool.jsonl"),
    )
    regression_samples = RegressionSetBuilder.build(
        output_path=str(out / "fixed_regression_set.jsonl"),
    )
    stress_samples = StressSetBuilder.build(
        output_path=str(out / "fixed_stress_set.jsonl"),
    )

    runner = MultiCycleRunner(
        num_cycles=5,
        output_dir=str(out),
        config={
            "stage": "Stage21-A",
            "drift_threshold": 0.05,
            "require_all_cycles_pass": True,
        },
    )
    result = runner.run(
        failure_samples=failure_samples,
        regression_samples=regression_samples,
        stress_samples=stress_samples,
        evolve_pool=False,
    )

    results_path = out / "stage21_a_results.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_report_path = out / "STAGE21_A_FINAL_REPORT.md"
    final_report_path.write_text(_render_final_report(result), encoding="utf-8")
    result["stage21_a_results_path"] = str(results_path)
    result["stage21_a_final_report_path"] = str(final_report_path)
    return result


def _reset_output_dir(out: Path) -> None:
    """Remove Stage21-A generated artifacts so reruns stay auditable."""
    if not out.exists():
        return

    for path in out.glob("cycle_*"):
        if path.is_dir():
            shutil.rmtree(path)

    generated_files = [
        "STAGE21_A_FINAL_REPORT.md",
        "STAGE21_MULTI_CYCLE_REPORT.md",
        "STAGE21_ROLLBACK_CHAIN_REPORT.json",
        "stage21_a_results.json",
        "fixed_failure_pool.jsonl",
        "fixed_regression_set.jsonl",
        "fixed_stress_set.jsonl",
    ]
    for name in generated_files:
        path = out / name
        if path.exists():
            path.unlink()


def _render_final_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    lines = [
        "# Stage21-A Final Report",
        "",
        "**Stage**: Stage21-A",
        "**Verdict**: {}".format(verdict),
        "**Completed cycles**: {}".format(result["completed_cycles"]),
        "**Cycle pass count**: {}".format(result["cycle_pass_count"]),
        "**Stage21-A freeze ready**: {}".format(result["stage21_a_freeze_ready"]),
        "**Drift status**: {}".format(result["drift_status"]),
        "**Rollback chain passed**: {}".format(result["rollback_chain_passed"]),
        "**Fixed failure pool hash**: `{}`".format(result["fixed_failure_pool_hash"]),
        "",
        "## Acceptance Summary",
        "",
        "| Metric | Required | Observed | Status |",
        "|---|---:|---:|---|",
    ]

    cycle_results = result["cycle_results"]
    min_fix = min(c["metrics"]["failure_fix_rate"] for c in cycle_results)
    min_reg = min(c["metrics"]["regression_pass_rate"] for c in cycle_results)
    min_tsla = min(c["metrics"]["tsla_safety_intercept"] for c in cycle_results)
    max_false_kill = max(c["metrics"]["false_kill_rate"] for c in cycle_results)
    max_contam = max(c["metrics"]["memory_contamination"] for c in cycle_results)
    min_rollback = min(c["metrics"]["rollback_success_rate"] for c in cycle_results)

    rows = [
        ("completed_cycles", "5/5", "{}/5".format(result["completed_cycles"]), result["completed_cycles"] == 5),
        ("cycle_pass_count", "5/5", "{}/5".format(result["cycle_pass_count"]), result["cycle_pass_count"] == 5),
        ("failure_fix_rate", ">= 80% each cycle", "{:.2%} min".format(min_fix), min_fix >= 0.80),
        ("regression_pass_rate", ">= 98% each cycle", "{:.2%} min".format(min_reg), min_reg >= 0.98),
        ("tsla_safety_intercept", ">= 99% each cycle", "{:.2%} min".format(min_tsla), min_tsla >= 0.99),
        ("false_kill_rate", "<= 1% each cycle", "{:.2%} max".format(max_false_kill), max_false_kill <= 0.01),
        ("memory_contamination", "0 each cycle", "{} max".format(int(max_contam)), max_contam == 0),
        ("rollback_success_rate", "100% each cycle", "{:.2%} min".format(min_rollback), min_rollback >= 1.0),
        ("drift_status", "no critical drift", result["drift_status"], result["drift_status"] == "no critical drift"),
        ("rollback_chain_passed", "true", str(result["rollback_chain_passed"]), result["rollback_chain_passed"] is True),
        ("stage21_a_freeze_ready", "true", str(result["stage21_a_freeze_ready"]), result["stage21_a_freeze_ready"] is True),
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
        "## Per-Cycle Artifacts",
        "",
    ])
    for cycle in range(1, result["completed_cycles"] + 1):
        lines.append("- `cycle_{:02d}/`".format(cycle))

    lines.extend([
        "",
        "## Generated Files",
        "",
        "- `STAGE21_A_FINAL_REPORT.md`",
        "- `STAGE21_MULTI_CYCLE_REPORT.md`",
        "- `STAGE21_ROLLBACK_CHAIN_REPORT.json`",
        "- `stage21_a_results.json`",
    ])
    return "\n".join(lines)


def main() -> None:
    result = run_stage21_a()
    print(json.dumps({
        "stage": result["stage"],
        "overall_pass": result["overall_pass"],
        "stage21_a_freeze_ready": result["stage21_a_freeze_ready"],
        "completed_cycles": result["completed_cycles"],
        "cycle_pass_count": result["cycle_pass_count"],
        "drift_status": result["drift_status"],
        "rollback_chain_passed": result["rollback_chain_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
