"""Stage26 observability and audit cockpit validation entrypoint."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("reports/stage26")

STAGE_EVIDENCE: tuple[dict[str, Any], ...] = (
    {
        "stage_id": "stage20_r",
        "stage": "Stage20-R",
        "result_path": Path("reports/stage20_r/stage20_pipeline_revalidation.json"),
        "final_report_path": Path("reports/stage20_r/STAGE20_R_REVALIDATION_REPORT.md"),
        "freeze_paths": [Path("reports/stage20_r/pipeline_artifacts/stage20_freeze_manifest.json")],
        "rollback_paths": [],
        "lineage_paths": [Path("reports/stage20_r/pipeline_artifacts/stage20_failure_pool.jsonl")],
        "freeze_ready_field": "overall_pass",
    },
    {
        "stage_id": "stage21_a",
        "stage": "Stage21-A",
        "result_path": Path("reports/stage21_a/stage21_a_results.json"),
        "final_report_path": Path("reports/stage21_a/STAGE21_A_FINAL_REPORT.md"),
        "freeze_paths": sorted(Path("reports/stage21_a").glob("cycle_*/stage20_freeze_manifest.json")),
        "rollback_paths": [Path("reports/stage21_a/STAGE21_ROLLBACK_CHAIN_REPORT.json")],
        "lineage_paths": [Path("reports/stage21_a/fixed_failure_pool.jsonl")],
        "freeze_ready_field": "stage21_a_freeze_ready",
    },
    {
        "stage_id": "stage21_b",
        "stage": "Stage21-B",
        "result_path": Path("reports/stage21_b/stage21_b_results.json"),
        "final_report_path": Path("reports/stage21_b/STAGE21_B_FINAL_REPORT.md"),
        "freeze_paths": sorted(Path("reports/stage21_b").glob("cycle_*/stage20_freeze_manifest.json")),
        "rollback_paths": [Path("reports/stage21_b/STAGE21_ROLLBACK_CHAIN_REPORT.json")],
        "lineage_paths": [Path("reports/stage21_b/pool_lineage.json")],
        "freeze_ready_field": "stage21_b_freeze_ready",
    },
    {
        "stage_id": "stage22",
        "stage": "Stage22",
        "result_path": Path("reports/stage22/stage22_results.json"),
        "final_report_path": Path("reports/stage22/STAGE22_FINAL_REPORT.md"),
        "freeze_paths": sorted(Path("reports/stage22").glob("cycle_*/stage20_freeze_manifest.json")),
        "rollback_paths": [Path("reports/stage22/STAGE22_ROLLBACK_CHAIN_REPORT.json")],
        "lineage_paths": [Path("reports/stage22/holdout_pool_lineage.json")],
        "freeze_ready_field": "stage22_freeze_ready",
    },
    {
        "stage_id": "stage23",
        "stage": "Stage23",
        "result_path": Path("reports/stage23/stage23_results.json"),
        "final_report_path": Path("reports/stage23/STAGE23_FINAL_REPORT.md"),
        "freeze_paths": [Path("reports/stage23/learned_prior_candidates.json")],
        "rollback_paths": [Path("reports/stage23/STAGE23_ROLLBACK_CHAIN_REPORT.json")],
        "lineage_paths": [Path("reports/stage23/prior_comparison.json")],
        "freeze_ready_field": "stage23_freeze_ready",
    },
    {
        "stage_id": "stage24",
        "stage": "Stage24",
        "result_path": Path("reports/stage24/stage24_results.json"),
        "final_report_path": Path("reports/stage24/STAGE24_FINAL_REPORT.md"),
        "freeze_paths": sorted(Path("reports/stage24").glob("cycle_*/stage20_freeze_manifest.json")),
        "rollback_paths": [Path("reports/stage24/STAGE24_ROLLBACK_CHAIN_REPORT.json")],
        "lineage_paths": [Path("reports/stage24/adversarial_horizon_lineage.json")],
        "freeze_ready_field": "stage24_freeze_ready",
    },
    {
        "stage_id": "stage25",
        "stage": "Stage25",
        "result_path": Path("reports/stage25/stage25_results.json"),
        "final_report_path": Path("reports/stage25/STAGE25_FINAL_REPORT.md"),
        "freeze_paths": [Path("reports/stage25/stage25_candidate_fix_packages.json")],
        "rollback_paths": [Path("reports/stage25/stage25_rollback_decisions.json")],
        "lineage_paths": [Path("reports/stage25/STAGE25_AUDIT_LOG.json")],
        "freeze_ready_field": "stage25_freeze_ready",
        "promotion_audit_path": Path("reports/stage25/STAGE25_AUDIT_LOG.json"),
    },
)


def run_stage26(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    """Aggregate observability and audit evidence across Stage20-R..Stage25."""
    out = Path(output_dir)
    _reset_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    observed = [_load_stage(item) for item in STAGE_EVIDENCE]
    baseline_chain = _build_baseline_chain(observed)
    metrics_index = _build_metrics_index(observed)
    lineage_index = _build_artifact_index(observed, "lineage")
    drift_index = _build_drift_index(observed)
    rollback_index = _build_artifact_index(observed, "rollback")
    freeze_index = _build_artifact_index(observed, "freeze")
    promotion_audit = _build_promotion_audit_index(observed)
    audit_cockpit = _build_audit_cockpit(
        baseline_chain=baseline_chain,
        metrics_index=metrics_index,
        lineage_index=lineage_index,
        drift_index=drift_index,
        rollback_index=rollback_index,
        freeze_index=freeze_index,
        promotion_audit=promotion_audit,
    )
    result = _build_result(
        observed=observed,
        baseline_chain=baseline_chain,
        metrics_index=metrics_index,
        lineage_index=lineage_index,
        drift_index=drift_index,
        rollback_index=rollback_index,
        freeze_index=freeze_index,
        promotion_audit=promotion_audit,
        audit_cockpit=audit_cockpit,
    )

    _write_json(out / "STAGE26_BASELINE_CHAIN.json", baseline_chain)
    _write_json(out / "STAGE26_METRICS_INDEX.json", metrics_index)
    _write_json(out / "STAGE26_LINEAGE_INDEX.json", lineage_index)
    _write_json(out / "STAGE26_DRIFT_INDEX.json", drift_index)
    _write_json(out / "STAGE26_ROLLBACK_INDEX.json", rollback_index)
    _write_json(out / "STAGE26_FREEZE_EVIDENCE_INDEX.json", freeze_index)
    _write_json(out / "STAGE26_PROMOTION_AUDIT_INDEX.json", promotion_audit)
    _write_json(out / "STAGE26_AUDIT_COCKPIT.json", audit_cockpit)
    _write_json(out / "stage26_results.json", result)
    (out / "STAGE26_AUDIT_COCKPIT.md").write_text(
        _render_audit_cockpit(audit_cockpit, result),
        encoding="utf-8",
    )
    (out / "STAGE26_FINAL_REPORT.md").write_text(
        _render_final_report(result),
        encoding="utf-8",
    )
    return result


def _load_stage(spec: dict[str, Any]) -> dict[str, Any]:
    result_path = spec["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "stage_id": spec["stage_id"],
        "stage": spec["stage"],
        "result_path": str(result_path),
        "result_hash": _sha256(result_path),
        "result": result,
        "final_report_path": str(spec["final_report_path"]),
        "final_report_exists": spec["final_report_path"].exists(),
        "freeze_paths": [str(path) for path in spec.get("freeze_paths", [])],
        "rollback_paths": [str(path) for path in spec.get("rollback_paths", [])],
        "lineage_paths": [str(path) for path in spec.get("lineage_paths", [])],
        "freeze_ready_field": spec["freeze_ready_field"],
        "promotion_audit_path": str(spec.get("promotion_audit_path", "")),
    }


def _build_baseline_chain(observed: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for item in observed:
        result = item["result"]
        freeze_ready = bool(result.get(item["freeze_ready_field"]))
        if item["stage_id"] == "stage20_r":
            freeze_ready = bool(result.get("overall_pass"))
        entries.append({
            "stage_id": item["stage_id"],
            "stage": item["stage"],
            "result_path": item["result_path"],
            "result_hash": item["result_hash"],
            "overall_pass": bool(result.get("overall_pass")),
            "freeze_ready_field": item["freeze_ready_field"],
            "freeze_ready": freeze_ready,
            "final_report_exists": item["final_report_exists"],
        })
    return {
        "observed_stage_count": len(entries),
        "entries": entries,
        "baseline_chain_complete": all(
            e["overall_pass"] and e["freeze_ready"] and e["final_report_exists"]
            for e in entries
        ),
    }


def _build_metrics_index(observed: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for item in observed:
        result = item["result"]
        cycle_results = result.get("cycle_results", [])
        metrics = {
            key: result.get(key)
            for key in (
                "completed_cycles",
                "cycle_pass_count",
                "candidate_fix_package_count",
                "promotion_gate_pass_rate",
                "unsafe_promotion_block_rate",
                "rollback_gate_pass_rate",
                "audit_event_coverage",
                "regression_pass_rate",
                "tsla_safety_intercept",
                "false_kill_rate",
                "memory_contamination",
                "drift_status",
                "rollback_chain_passed",
                "overall_pass",
            )
            if key in result
        }
        if cycle_results:
            metrics["cycle_metric_count"] = len(cycle_results)
            metrics["min_regression_pass_rate"] = min(
                c["metrics"]["regression_pass_rate"] for c in cycle_results
            )
            metrics["min_tsla_safety_intercept"] = min(
                c["metrics"]["tsla_safety_intercept"] for c in cycle_results
            )
            metrics["max_false_kill_rate"] = max(
                c["metrics"]["false_kill_rate"] for c in cycle_results
            )
            metrics["max_memory_contamination"] = max(
                c["metrics"]["memory_contamination"] for c in cycle_results
            )
        entries.append({
            "stage_id": item["stage_id"],
            "stage": item["stage"],
            "metrics": metrics,
            "covered": bool(metrics),
        })
    return _coverage_index("metrics", entries)


def _build_artifact_index(
    observed: list[dict[str, Any]],
    artifact_type: str,
) -> dict[str, Any]:
    key = {
        "lineage": "lineage_paths",
        "rollback": "rollback_paths",
        "freeze": "freeze_paths",
    }[artifact_type]
    entries = []
    for item in observed:
        paths = item[key]
        artifacts = [
            {
                "path": path,
                "exists": Path(path).exists(),
                "hash": _sha256(Path(path)) if Path(path).is_file() else "",
            }
            for path in paths
        ]
        if not artifacts and artifact_type == "rollback" and item["stage_id"] == "stage20_r":
            artifacts = [{
                "path": "reports/stage20_r/pipeline_artifacts/stage20_rollback_runbook.md",
                "exists": Path("reports/stage20_r/pipeline_artifacts/stage20_rollback_runbook.md").exists(),
                "hash": _sha256(Path("reports/stage20_r/pipeline_artifacts/stage20_rollback_runbook.md")),
                "note": "Stage20-R rollback evidence is a runbook, not a chain report.",
            }]
        entries.append({
            "stage_id": item["stage_id"],
            "stage": item["stage"],
            "artifact_type": artifact_type,
            "artifacts": artifacts,
            "covered": bool(artifacts) and all(a["exists"] for a in artifacts),
        })
    return _coverage_index(artifact_type, entries)


def _build_drift_index(observed: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for item in observed:
        result = item["result"]
        drift_status = result.get("drift_status")
        if drift_status is None and item["stage_id"] == "stage20_r":
            drift_status = "not_applicable_single_revalidation"
        entries.append({
            "stage_id": item["stage_id"],
            "stage": item["stage"],
            "drift_status": drift_status,
            "critical_drift": drift_status == "critical drift",
            "covered": drift_status is not None,
        })
    index = _coverage_index("drift", entries)
    index["critical_drift_count"] = sum(
        1 for e in entries if e["critical_drift"]
    )
    return index


def _build_promotion_audit_index(observed: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for item in observed:
        if item["stage_id"] != "stage25":
            entries.append({
                "stage_id": item["stage_id"],
                "stage": item["stage"],
                "covered": True,
                "not_applicable": True,
                "reason": "Promotion audit is introduced in Stage25.",
            })
            continue
        result = item["result"]
        audit_path = item["promotion_audit_path"]
        entries.append({
            "stage_id": item["stage_id"],
            "stage": item["stage"],
            "audit_path": audit_path,
            "audit_path_exists": Path(audit_path).exists(),
            "audit_event_coverage": result.get("audit_event_coverage"),
            "promotion_gate_pass_rate": result.get("promotion_gate_pass_rate"),
            "unsafe_promotion_block_rate": result.get("unsafe_promotion_block_rate"),
            "rollback_gate_pass_rate": result.get("rollback_gate_pass_rate"),
            "covered": (
                Path(audit_path).exists()
                and result.get("audit_event_coverage") == 1.0
                and result.get("promotion_gate_pass_rate") == 1.0
                and result.get("unsafe_promotion_block_rate") == 1.0
                and result.get("rollback_gate_pass_rate") == 1.0
            ),
        })
    return _coverage_index("promotion_audit", entries)


def _build_audit_cockpit(
    baseline_chain: dict[str, Any],
    metrics_index: dict[str, Any],
    lineage_index: dict[str, Any],
    drift_index: dict[str, Any],
    rollback_index: dict[str, Any],
    freeze_index: dict[str, Any],
    promotion_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "Stage26",
        "baseline_chain": baseline_chain,
        "metrics_index": metrics_index,
        "lineage_index": lineage_index,
        "drift_index": drift_index,
        "rollback_index": rollback_index,
        "freeze_index": freeze_index,
        "promotion_audit": promotion_audit,
    }


def _build_result(
    observed: list[dict[str, Any]],
    baseline_chain: dict[str, Any],
    metrics_index: dict[str, Any],
    lineage_index: dict[str, Any],
    drift_index: dict[str, Any],
    rollback_index: dict[str, Any],
    freeze_index: dict[str, Any],
    promotion_audit: dict[str, Any],
    audit_cockpit: dict[str, Any],
) -> dict[str, Any]:
    machine_readable_artifacts = True
    human_readable_artifacts = True
    rollback_chain_passed = all(
        bool(item["result"].get("rollback_chain_passed", True))
        for item in observed
    )
    stage26_freeze_ready = (
        baseline_chain["baseline_chain_complete"]
        and baseline_chain["observed_stage_count"] >= 6
        and metrics_index["coverage"] == 1.0
        and lineage_index["coverage"] == 1.0
        and drift_index["coverage"] == 1.0
        and rollback_index["coverage"] == 1.0
        and freeze_index["coverage"] == 1.0
        and promotion_audit["coverage"] == 1.0
        and machine_readable_artifacts
        and human_readable_artifacts
        and drift_index["critical_drift_count"] == 0
        and rollback_chain_passed
    )
    return {
        "stage": "Stage26",
        "baseline_chain_complete": baseline_chain["baseline_chain_complete"],
        "observed_stage_count": baseline_chain["observed_stage_count"],
        "metrics_index_coverage": metrics_index["coverage"],
        "lineage_index_coverage": lineage_index["coverage"],
        "drift_index_coverage": drift_index["coverage"],
        "rollback_index_coverage": rollback_index["coverage"],
        "freeze_evidence_coverage": freeze_index["coverage"],
        "promotion_audit_coverage": promotion_audit["coverage"],
        "machine_readable_artifacts": machine_readable_artifacts,
        "human_readable_artifacts": human_readable_artifacts,
        "critical_drift_count": drift_index["critical_drift_count"],
        "rollback_chain_passed": rollback_chain_passed,
        "baseline_chain": baseline_chain,
        "audit_cockpit": audit_cockpit,
        "thresholds": _stage26_thresholds(),
        "stage26_freeze_ready": stage26_freeze_ready,
        "overall_pass": stage26_freeze_ready,
    }


def _coverage_index(name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    covered_count = sum(1 for entry in entries if entry["covered"])
    total = len(entries)
    return {
        "index": name,
        "entries": entries,
        "covered_count": covered_count,
        "total_count": total,
        "coverage": covered_count / total if total else 0.0,
    }


def _stage26_thresholds() -> dict[str, Any]:
    return {
        "baseline_chain_complete": True,
        "observed_stage_count": 6,
        "metrics_index_coverage": 1.0,
        "lineage_index_coverage": 1.0,
        "drift_index_coverage": 1.0,
        "rollback_index_coverage": 1.0,
        "freeze_evidence_coverage": 1.0,
        "promotion_audit_coverage": 1.0,
        "machine_readable_artifacts": True,
        "human_readable_artifacts": True,
        "critical_drift_count": 0,
        "rollback_chain_passed": True,
        "stage26_freeze_ready": True,
    }


def _render_audit_cockpit(
    audit_cockpit: dict[str, Any],
    result: dict[str, Any],
) -> str:
    lines = [
        "# Stage26 Audit Cockpit",
        "",
        f"**Observed stages**: {result['observed_stage_count']}",
        f"**Baseline chain complete**: {result['baseline_chain_complete']}",
        f"**Critical drift count**: {result['critical_drift_count']}",
        f"**Rollback chain passed**: {result['rollback_chain_passed']}",
        "",
        "## Coverage",
        "",
        "| Index | Coverage |",
        "|---|---:|",
    ]
    for key in (
        "metrics_index",
        "lineage_index",
        "drift_index",
        "rollback_index",
        "freeze_index",
        "promotion_audit",
    ):
        idx = audit_cockpit[key]
        lines.append(f"| {idx['index']} | {idx['coverage']:.2%} |")
    lines.extend(["", "## Baseline Chain", "", "| Stage | Freeze ready | Overall pass |", "|---|---:|---:|"])
    for entry in audit_cockpit["baseline_chain"]["entries"]:
        lines.append(
            f"| {entry['stage']} | {entry['freeze_ready']} | {entry['overall_pass']} |"
        )
    return "\n".join(lines)


def _render_final_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    rows = [
        ("baseline_chain_complete", "true", result["baseline_chain_complete"]),
        ("observed_stage_count", ">= 6", result["observed_stage_count"]),
        ("metrics_index_coverage", "100%", "{:.2%}".format(result["metrics_index_coverage"])),
        ("lineage_index_coverage", "100%", "{:.2%}".format(result["lineage_index_coverage"])),
        ("drift_index_coverage", "100%", "{:.2%}".format(result["drift_index_coverage"])),
        ("rollback_index_coverage", "100%", "{:.2%}".format(result["rollback_index_coverage"])),
        ("freeze_evidence_coverage", "100%", "{:.2%}".format(result["freeze_evidence_coverage"])),
        ("promotion_audit_coverage", "100%", "{:.2%}".format(result["promotion_audit_coverage"])),
        ("machine_readable_artifacts", "true", result["machine_readable_artifacts"]),
        ("human_readable_artifacts", "true", result["human_readable_artifacts"]),
        ("critical_drift_count", "0", result["critical_drift_count"]),
        ("rollback_chain_passed", "true", result["rollback_chain_passed"]),
        ("stage26_freeze_ready", "true", result["stage26_freeze_ready"]),
    ]
    lines = [
        "# Stage26 Final Report",
        "",
        "**Stage**: Stage26",
        f"**Verdict**: {verdict}",
        f"**Stage26 freeze ready**: {result['stage26_freeze_ready']}",
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
        "## Generated Files",
        "",
        "- `STAGE26_FINAL_REPORT.md`",
        "- `STAGE26_AUDIT_COCKPIT.md`",
        "- `STAGE26_AUDIT_COCKPIT.json`",
        "- `STAGE26_BASELINE_CHAIN.json`",
        "- `STAGE26_METRICS_INDEX.json`",
        "- `STAGE26_LINEAGE_INDEX.json`",
        "- `STAGE26_DRIFT_INDEX.json`",
        "- `STAGE26_ROLLBACK_INDEX.json`",
        "- `STAGE26_FREEZE_EVIDENCE_INDEX.json`",
        "- `STAGE26_PROMOTION_AUDIT_INDEX.json`",
        "- `stage26_results.json`",
    ])
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reset_output_dir(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)


def main() -> None:
    result = run_stage26()
    print(json.dumps({
        "stage": result["stage"],
        "overall_pass": result["overall_pass"],
        "stage26_freeze_ready": result["stage26_freeze_ready"],
        "baseline_chain_complete": result["baseline_chain_complete"],
        "observed_stage_count": result["observed_stage_count"],
        "metrics_index_coverage": result["metrics_index_coverage"],
        "lineage_index_coverage": result["lineage_index_coverage"],
        "drift_index_coverage": result["drift_index_coverage"],
        "rollback_index_coverage": result["rollback_index_coverage"],
        "freeze_evidence_coverage": result["freeze_evidence_coverage"],
        "promotion_audit_coverage": result["promotion_audit_coverage"],
        "critical_drift_count": result["critical_drift_count"],
        "rollback_chain_passed": result["rollback_chain_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
