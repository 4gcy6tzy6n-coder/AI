"""Tests for Stage25 promotion and rollback governance validation."""

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
]


class TestStage25:
    def test_stage25_runs_promotion_and_rollback_governance(self):
        from stage25.run_stage25 import run_stage25

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage25(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage25"
            assert result["baseline_chain_complete"] is True
            assert result["candidate_fix_package_count"] >= 16
            assert result["safe_candidate_count"] >= 16
            assert result["unsafe_candidate_count"] >= 8
            assert result["promotion_gate_pass_rate"] == 1.0
            assert result["unsafe_promotion_block_rate"] == 1.0
            assert result["rollback_gate_pass_rate"] == 1.0
            assert result["audit_event_coverage"] == 1.0
            assert result["regression_pass_rate"] >= 0.98
            assert result["tsla_safety_intercept"] >= 0.99
            assert result["false_kill_rate"] <= 0.01
            assert result["memory_contamination"] == 0
            assert result["drift_status"] == "no critical drift"
            assert result["rollback_chain_passed"] is True
            assert result["overall_pass"] is True
            assert result["stage25_freeze_ready"] is True

            assert (out / "STAGE25_FINAL_REPORT.md").exists()
            assert (out / "stage25_results.json").exists()
            assert (out / "stage25_candidate_fix_packages.json").exists()
            assert (out / "stage25_promotion_decisions.json").exists()
            assert (out / "stage25_rollback_decisions.json").exists()
            assert (out / "STAGE25_AUDIT_LOG.json").exists()

            persisted = json.loads((out / "stage25_results.json").read_text())
            assert persisted["stage25_freeze_ready"] is True
            assert len(persisted["audit_events"]) == (
                len(persisted["promotion_decisions"])
                + len(persisted["unsafe_block_decisions"])
                + len(persisted["rollback_decisions"])
            )

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_synthetic_unsafe_promotion_leak_fails_stage25(self):
        from stage25.run_stage25 import _build_result

        stage24 = _stage24_fixture()
        promotion_results = [{
            "candidate_id": "safe_001",
            "decision": "promote_to_normal",
            "promotion_passed": True,
        }]
        block_results = [{
            "candidate_id": "unsafe_001",
            "unsafe_mode": "hard_veto",
            "decision": "promote_to_normal",
            "blocked": False,
        }]
        rollback_results = [{
            "candidate_id": "unsafe_001",
            "rollback": True,
            "passed": True,
        }]
        audit_events = [
            {"event_id": "1", "candidate_id": "safe_001", "passed": True},
            {"event_id": "2", "candidate_id": "unsafe_001", "passed": False},
            {"event_id": "3", "candidate_id": "unsafe_001", "passed": True},
        ]

        result = _build_result(
            stage24=stage24,
            candidates=[{"candidate_id": f"candidate_{i}"} for i in range(16)],
            safe_candidates=[{"candidate_id": f"safe_{i}"} for i in range(16)],
            unsafe_candidates=[{"candidate_id": f"unsafe_{i}"} for i in range(8)],
            promotion_results=promotion_results,
            block_results=block_results,
            rollback_results=rollback_results,
            audit_events=audit_events,
        )

        assert result["unsafe_promotion_block_rate"] == 0.0
        assert result["overall_pass"] is False
        assert result["stage25_freeze_ready"] is False

    def test_synthetic_missing_audit_event_fails_stage25(self):
        from stage25.run_stage25 import _build_result

        result = _build_result(
            stage24=_stage24_fixture(),
            candidates=[{"candidate_id": f"candidate_{i}"} for i in range(16)],
            safe_candidates=[{"candidate_id": f"safe_{i}"} for i in range(16)],
            unsafe_candidates=[{"candidate_id": f"unsafe_{i}"} for i in range(8)],
            promotion_results=[
                {"candidate_id": "safe_001", "promotion_passed": True}
            ],
            block_results=[
                {"candidate_id": "unsafe_001", "blocked": True}
            ],
            rollback_results=[
                {"candidate_id": "unsafe_001", "passed": True}
            ],
            audit_events=[
                {"event_id": "1", "candidate_id": "safe_001", "passed": True}
            ],
        )

        assert result["audit_event_coverage"] < 1.0
        assert result["overall_pass"] is False
        assert result["stage25_freeze_ready"] is False


def _stage24_fixture() -> dict:
    return {
        "baseline_chain_complete": True,
        "operational_metrics": {
            "regression_pass_rate": 1.0,
            "tsla_safety_intercept": 1.0,
            "false_kill_rate": 0.0,
            "memory_contamination": 0,
            "drift_status": "no critical drift",
            "rollback_chain_passed": True,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
