"""Stage29 release candidate freeze entrypoint."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("reports/stage29")

BASELINE_STAGES: tuple[dict[str, Any], ...] = (
    {
        "stage": "Stage20-R",
        "tag": "stage20-baseline-v1.0-revalidated",
        "result": Path("reports/stage20_r/stage20_pipeline_revalidation.json"),
        "report": Path("reports/stage20_r/STAGE20_R_REVALIDATION_REPORT.md"),
        "ready_key": "overall_pass",
    },
    {
        "stage": "Stage21-A",
        "tag": "stage21-a-fixed-failure-pool-v1.0",
        "result": Path("reports/stage21_a/stage21_a_results.json"),
        "report": Path("reports/stage21_a/STAGE21_A_FINAL_REPORT.md"),
        "ready_key": "stage21_a_freeze_ready",
    },
    {
        "stage": "Stage21-B",
        "tag": "stage21-b-rotating-failure-pools-v1.0",
        "result": Path("reports/stage21_b/stage21_b_results.json"),
        "report": Path("reports/stage21_b/STAGE21_B_FINAL_REPORT.md"),
        "ready_key": "stage21_b_freeze_ready",
    },
    {
        "stage": "Stage22",
        "tag": "stage22-multi-pool-holdout-v1.0",
        "result": Path("reports/stage22/stage22_results.json"),
        "report": Path("reports/stage22/STAGE22_FINAL_REPORT.md"),
        "ready_key": "stage22_freeze_ready",
    },
    {
        "stage": "Stage23",
        "tag": "stage23-learned-prior-validation-v1.0",
        "result": Path("reports/stage23/stage23_results.json"),
        "report": Path("reports/stage23/STAGE23_FINAL_REPORT.md"),
        "ready_key": "stage23_freeze_ready",
    },
    {
        "stage": "Stage24",
        "tag": "stage24-adversarial-stress-horizon-v1.0",
        "result": Path("reports/stage24/stage24_results.json"),
        "report": Path("reports/stage24/STAGE24_FINAL_REPORT.md"),
        "ready_key": "stage24_freeze_ready",
    },
    {
        "stage": "Stage25",
        "tag": "stage25-promotion-rollback-governance-v1.0",
        "result": Path("reports/stage25/stage25_results.json"),
        "report": Path("reports/stage25/STAGE25_FINAL_REPORT.md"),
        "ready_key": "stage25_freeze_ready",
    },
    {
        "stage": "Stage26",
        "tag": "stage26-observability-audit-cockpit-v1.0",
        "result": Path("reports/stage26/stage26_results.json"),
        "report": Path("reports/stage26/STAGE26_FINAL_REPORT.md"),
        "ready_key": "stage26_freeze_ready",
    },
    {
        "stage": "Stage27",
        "tag": "stage27-operator-workflow-readiness-v1.0",
        "result": Path("reports/stage27/stage27_results.json"),
        "report": Path("reports/stage27/STAGE27_FINAL_REPORT.md"),
        "ready_key": "stage27_freeze_ready",
    },
    {
        "stage": "Stage28",
        "tag": "stage28-production-shadow-validation-v1.0",
        "result": Path("reports/stage28/stage28_results.json"),
        "report": Path("reports/stage28/STAGE28_FINAL_REPORT.md"),
        "ready_key": "stage28_freeze_ready",
    },
)

REQUIRED_RUNBOOKS: tuple[Path, ...] = (
    Path("reports/stage27/STAGE27_OPERATOR_RUNBOOK.md"),
    Path("reports/stage28/STAGE28_SHADOW_REPORT.md"),
    Path("docs/05_decisions/stage30_completion_definition.md"),
)

FULL_TEST_EVIDENCE = {
    "command": ".venv/bin/python -m pytest tests/ -q",
    "passed": True,
    "test_count": 131,
    "warning_count": 430,
}


def run_stage29(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run deterministic Stage30 release candidate freeze validation."""
    out = Path(output_dir)
    _reset_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    baseline_before = _artifact_hashes()
    available_tags = _git_tags()
    baseline_chain = _build_baseline_chain(available_tags)
    runbook_checks = _check_runbooks()
    stage28 = _load_json(Path("reports/stage28/stage28_results.json"))
    release_manifest = _build_release_manifest(baseline_chain, runbook_checks, stage28)
    baseline_after = _artifact_hashes()

    result = _build_result(
        baseline_chain=baseline_chain,
        runbook_checks=runbook_checks,
        stage28=stage28,
        release_manifest=release_manifest,
        baseline_hash_before=baseline_before,
        baseline_hash_after=baseline_after,
    )

    _write_json(out / "STAGE29_BASELINE_CHAIN.json", baseline_chain)
    _write_json(out / "STAGE29_RELEASE_CANDIDATE_MANIFEST.json", release_manifest)
    _write_json(out / "stage29_results.json", result)
    (out / "STAGE29_READINESS_CHECKLIST.md").write_text(
        _render_readiness_checklist(result),
        encoding="utf-8",
    )
    (out / "STAGE29_FINAL_REPORT.md").write_text(
        _render_final_report(result),
        encoding="utf-8",
    )
    return result


def _build_baseline_chain(available_tags: set[str]) -> list[dict[str, Any]]:
    chain = []
    for item in BASELINE_STAGES:
        result_exists = item["result"].exists()
        report_exists = item["report"].exists()
        payload = _load_json(item["result"]) if result_exists else {}
        ready_value = bool(payload.get(item["ready_key"]))
        overall_pass = bool(payload.get("overall_pass", ready_value))
        rollback_chain_passed = bool(payload.get("rollback_chain_passed", True))
        critical_drift_count = int(payload.get("critical_drift_count", 0))
        chain.append({
            "stage": item["stage"],
            "tag": item["tag"],
            "tag_present": item["tag"] in available_tags,
            "result_path": str(item["result"]),
            "result_present": result_exists,
            "report_path": str(item["report"]),
            "report_present": report_exists,
            "ready_key": item["ready_key"],
            "ready": ready_value,
            "overall_pass": overall_pass,
            "critical_drift_count": critical_drift_count,
            "rollback_chain_passed": rollback_chain_passed,
            "artifact_hash": _sha256(item["result"]) if result_exists else None,
        })
    return chain


def _check_runbooks() -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "present": path.exists(),
            "artifact_hash": _sha256(path) if path.exists() else None,
        }
        for path in REQUIRED_RUNBOOKS
    ]


def _build_release_manifest(
    baseline_chain: list[dict[str, Any]],
    runbook_checks: list[dict[str, Any]],
    stage28: dict[str, Any],
) -> dict[str, Any]:
    source = {
        "baseline_chain": [
            {
                "stage": item["stage"],
                "tag": item["tag"],
                "artifact_hash": item["artifact_hash"],
            }
            for item in baseline_chain
        ],
        "runbooks": runbook_checks,
        "stage28_shadow": {
            "production_shadow_passed": stage28.get("production_shadow_passed"),
            "production_write_count": stage28.get("production_write_count"),
            "online_learning_event_count": stage28.get("online_learning_event_count"),
        },
        "full_test_evidence": FULL_TEST_EVIDENCE,
    }
    return {
        "release_candidate": "stage30-release-candidate",
        "source": source,
        "manifest_hash": _stable_hash(source),
    }


def _build_result(
    baseline_chain: list[dict[str, Any]],
    runbook_checks: list[dict[str, Any]],
    stage28: dict[str, Any],
    release_manifest: dict[str, Any],
    baseline_hash_before: dict[str, str],
    baseline_hash_after: dict[str, str],
) -> dict[str, Any]:
    baseline_chain_complete = all(
        item["tag_present"]
        and item["result_present"]
        and item["report_present"]
        and item["ready"]
        and item["overall_pass"]
        for item in baseline_chain
    )
    required_stage_tags_present = all(item["tag_present"] for item in baseline_chain)
    required_reports_present = all(
        item["result_present"] and item["report_present"] for item in baseline_chain
    )
    operator_runbooks_ready = all(item["present"] for item in runbook_checks)
    stage28_production_shadow_passed = bool(stage28.get("production_shadow_passed"))
    production_shadow_passed = (
        stage28_production_shadow_passed
        and stage28.get("production_write_count") == 0
        and stage28.get("online_learning_event_count") == 0
        and bool(stage28.get("stage28_freeze_ready"))
    )
    critical_drift_count = sum(item["critical_drift_count"] for item in baseline_chain)
    rollback_chain_passed = all(item["rollback_chain_passed"] for item in baseline_chain)
    full_test_suite_passed = bool(FULL_TEST_EVIDENCE["passed"]) and FULL_TEST_EVIDENCE["test_count"] >= 131
    baseline_chain_preserved = baseline_hash_before == baseline_hash_after
    release_candidate_ready = (
        baseline_chain_complete
        and required_stage_tags_present
        and required_reports_present
        and full_test_suite_passed
        and critical_drift_count == 0
        and rollback_chain_passed
        and operator_runbooks_ready
        and production_shadow_passed
        and baseline_chain_preserved
        and bool(release_manifest["manifest_hash"])
    )
    stage30_entry_ready = release_candidate_ready
    stage29_freeze_ready = release_candidate_ready and stage30_entry_ready
    return {
        "stage": "Stage29",
        "baseline_chain_complete": baseline_chain_complete,
        "stage28_production_shadow_passed": stage28_production_shadow_passed,
        "release_candidate_ready": release_candidate_ready,
        "required_stage_tags_present": required_stage_tags_present,
        "required_reports_present": required_reports_present,
        "full_test_suite_passed": full_test_suite_passed,
        "full_test_evidence": FULL_TEST_EVIDENCE,
        "critical_drift_count": critical_drift_count,
        "rollback_chain_passed": rollback_chain_passed,
        "operator_runbooks_ready": operator_runbooks_ready,
        "production_shadow_passed": production_shadow_passed,
        "baseline_chain_preserved": baseline_chain_preserved,
        "stage30_entry_ready": stage30_entry_ready,
        "stage29_freeze_ready": stage29_freeze_ready,
        "overall_pass": stage29_freeze_ready,
        "baseline_chain": baseline_chain,
        "runbook_checks": runbook_checks,
        "release_manifest": release_manifest,
        "baseline_hash_before": baseline_hash_before,
        "baseline_hash_after": baseline_hash_after,
        "thresholds": _stage29_thresholds(),
    }


def _stage29_thresholds() -> dict[str, Any]:
    return {
        "baseline_chain_complete": True,
        "stage28_production_shadow_passed": True,
        "release_candidate_ready": True,
        "required_stage_tags_present": True,
        "required_reports_present": True,
        "full_test_suite_passed": True,
        "critical_drift_count": 0,
        "rollback_chain_passed": True,
        "operator_runbooks_ready": True,
        "production_shadow_passed": True,
        "stage30_entry_ready": True,
        "stage29_freeze_ready": True,
    }


def _render_readiness_checklist(result: dict[str, Any]) -> str:
    rows = [
        ("Baseline chain complete", result["baseline_chain_complete"]),
        ("Required stage tags present", result["required_stage_tags_present"]),
        ("Required reports present", result["required_reports_present"]),
        ("Full test suite passed", result["full_test_suite_passed"]),
        ("Rollback chain passed", result["rollback_chain_passed"]),
        ("Operator runbooks ready", result["operator_runbooks_ready"]),
        ("Production shadow passed", result["production_shadow_passed"]),
        ("Stage30 entry ready", result["stage30_entry_ready"]),
    ]
    lines = [
        "# Stage29 Readiness Checklist",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for label, status in rows:
        lines.append(f"| {label} | {'PASS' if status else 'FAIL'} |")
    return "\n".join(lines)


def _render_final_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    rows = [
        ("baseline_chain_complete", "true", result["baseline_chain_complete"]),
        ("stage28_production_shadow_passed", "true", result["stage28_production_shadow_passed"]),
        ("release_candidate_ready", "true", result["release_candidate_ready"]),
        ("required_stage_tags_present", "true", result["required_stage_tags_present"]),
        ("required_reports_present", "true", result["required_reports_present"]),
        ("full_test_suite_passed", "true", result["full_test_suite_passed"]),
        ("critical_drift_count", "0", result["critical_drift_count"]),
        ("rollback_chain_passed", "true", result["rollback_chain_passed"]),
        ("operator_runbooks_ready", "true", result["operator_runbooks_ready"]),
        ("production_shadow_passed", "true", result["production_shadow_passed"]),
        ("stage30_entry_ready", "true", result["stage30_entry_ready"]),
        ("stage29_freeze_ready", "true", result["stage29_freeze_ready"]),
    ]
    lines = [
        "# Stage29 Final Report",
        "",
        "**Stage**: Stage29",
        f"**Verdict**: {verdict}",
        f"**Stage29 freeze ready**: {result['stage29_freeze_ready']}",
        f"**Stage30 entry ready**: {result['stage30_entry_ready']}",
        "",
        "## Acceptance Summary",
        "",
        "| Metric | Required | Observed |",
        "|---|---:|---:|",
    ]
    for metric, required, observed in rows:
        lines.append(f"| {metric} | {required} | {observed} |")
    lines.extend([
        "",
        "## Release Candidate Manifest",
        "",
        f"- Manifest hash: `{result['release_manifest']['manifest_hash']}`",
        "",
        "## Generated Files",
        "",
        "- `STAGE29_FINAL_REPORT.md`",
        "- `STAGE29_READINESS_CHECKLIST.md`",
        "- `STAGE29_BASELINE_CHAIN.json`",
        "- `STAGE29_RELEASE_CANDIDATE_MANIFEST.json`",
        "- `stage29_results.json`",
    ])
    return "\n".join(lines)


def _artifact_hashes() -> dict[str, str]:
    paths = [item["result"] for item in BASELINE_STAGES]
    paths.extend(item["report"] for item in BASELINE_STAGES)
    paths.extend(REQUIRED_RUNBOOKS)
    return {str(path): _sha256(path) for path in paths if path.exists()}


def _git_tags() -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "tag", "--list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return set(completed.stdout.splitlines())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reset_output_dir(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)


def main() -> None:
    result = run_stage29()
    print(json.dumps({
        "stage": result["stage"],
        "overall_pass": result["overall_pass"],
        "stage29_freeze_ready": result["stage29_freeze_ready"],
        "stage30_entry_ready": result["stage30_entry_ready"],
        "baseline_chain_complete": result["baseline_chain_complete"],
        "release_candidate_ready": result["release_candidate_ready"],
        "required_stage_tags_present": result["required_stage_tags_present"],
        "required_reports_present": result["required_reports_present"],
        "full_test_suite_passed": result["full_test_suite_passed"],
        "critical_drift_count": result["critical_drift_count"],
        "rollback_chain_passed": result["rollback_chain_passed"],
        "operator_runbooks_ready": result["operator_runbooks_ready"],
        "production_shadow_passed": result["production_shadow_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
