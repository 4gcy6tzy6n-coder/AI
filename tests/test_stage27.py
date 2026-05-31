"""Tests for Stage27 integration and operator workflow validation."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


FROZEN_ARTIFACTS = [
    Path("reports/stage21_a/stage21_a_results.json"),
    Path("reports/stage21_b/stage21_b_results.json"),
    Path("reports/stage22/stage22_results.json"),
    Path("reports/stage23/stage23_results.json"),
    Path("reports/stage24/stage24_results.json"),
    Path("reports/stage25/stage25_results.json"),
    Path("reports/stage26/stage26_results.json"),
]


class TestStage27:
    def test_stage27_runs_operator_workflow_validation(self):
        from stage27.run_stage27 import run_stage27

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage27(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage27"
            assert result["operator_workflow_count"] >= 5
            assert result["inspect_workflow_passed"] is True
            assert result["validate_workflow_passed"] is True
            assert result["promote_workflow_guarded"] is True
            assert result["rollback_workflow_passed"] is True
            assert result["unsafe_operation_block_rate"] == 1.0
            assert result["runbook_coverage"] == 1.0
            assert result["safe_default_mode"] is True
            assert result["baseline_chain_preserved"] is True
            assert result["critical_drift_count"] == 0
            assert result["rollback_chain_passed"] is True
            assert result["overall_pass"] is True
            assert result["stage27_freeze_ready"] is True

            for name in [
                "STAGE27_FINAL_REPORT.md",
                "STAGE27_OPERATOR_RUNBOOK.md",
                "stage27_results.json",
                "stage27_workflow_results.json",
                "stage27_unsafe_operation_results.json",
            ]:
                assert (out / name).exists()

            persisted = json.loads((out / "stage27_results.json").read_text())
            assert persisted["stage27_freeze_ready"] is True

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_synthetic_missing_runbook_section_fails_stage27(self):
        from stage27.run_stage27 import _build_result

        workflows = _workflow_fixtures()
        unsafe = [{"blocked": True, "passed": True}]
        result = _build_result(
            stage26=_stage26_fixture(),
            workflow_results=workflows,
            unsafe_results=unsafe,
            runbook="Inspect Baseline Chain only",
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )

        assert result["runbook_coverage"] < 1.0
        assert result["overall_pass"] is False
        assert result["stage27_freeze_ready"] is False

    def test_synthetic_baseline_mutation_fails_stage27(self):
        from stage27.run_stage27 import _build_result

        result = _build_result(
            stage26=_stage26_fixture(),
            workflow_results=_workflow_fixtures(),
            unsafe_results=[{"blocked": True, "passed": True}],
            runbook="Inspect Baseline Chain\nValidate Audit Cockpit\nPromote Candidate Guarded\nRollback Candidate\nBlock Unsafe Operation",
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "2"},
        )

        assert result["baseline_chain_preserved"] is False
        assert result["overall_pass"] is False
        assert result["stage27_freeze_ready"] is False


def _stage26_fixture() -> dict:
    return {
        "overall_pass": True,
        "critical_drift_count": 0,
        "rollback_chain_passed": True,
    }


def _workflow_fixtures() -> list[dict]:
    return [
        {"workflow_id": "inspect_baseline_chain", "passed": True, "guarded": False, "mode": "read_only", "safe_default": True, "runbook_section": "Inspect Baseline Chain"},
        {"workflow_id": "validate_audit_cockpit", "passed": True, "guarded": False, "mode": "read_only", "safe_default": True, "runbook_section": "Validate Audit Cockpit"},
        {"workflow_id": "promote_candidate_guarded", "passed": True, "guarded": True, "mode": "dry_run", "safe_default": True, "runbook_section": "Promote Candidate Guarded"},
        {"workflow_id": "rollback_candidate", "passed": True, "guarded": True, "mode": "dry_run", "safe_default": True, "runbook_section": "Rollback Candidate"},
        {"workflow_id": "block_unsafe_operation", "passed": True, "guarded": False, "mode": "apply", "safe_default": False, "runbook_section": "Block Unsafe Operation"},
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
