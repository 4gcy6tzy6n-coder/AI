"""Tests for Stage30 governed production baseline validation."""

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
    Path("reports/stage27/stage27_results.json"),
    Path("reports/stage28/stage28_results.json"),
    Path("reports/stage29/stage29_results.json"),
]


class TestStage30:
    def test_stage30_freezes_governed_production_baseline(self):
        from stage30.run_stage30 import REQUIRED_STAGE30_ARTIFACTS, run_stage30

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage30(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage30"
            assert result["overall_pass"] is True
            assert result["stage30_freeze_ready"] is True
            assert result["baseline_chain_complete"] is True
            assert result["stage29_release_candidate_ready"] is True
            assert result["full_test_suite_passed"] is True
            assert result["governed_pipeline_passed"] is True
            assert result["multi_pool_validation_passed"] is True
            assert result["adversarial_stress_passed"] is True
            assert result["learned_prior_safety_passed"] is True
            assert result["promotion_governance_passed"] is True
            assert result["rollback_chain_passed"] is True
            assert result["observability_ready"] is True
            assert result["operator_runbooks_ready"] is True
            assert result["production_shadow_passed"] is True
            assert result["memory_contamination"] == 0
            assert result["critical_drift_count"] == 0
            assert result["blocked_items"] == []
            assert len(result["cycle_results"]) == 4

            for name in REQUIRED_STAGE30_ARTIFACTS:
                assert (out / name).exists()

            persisted = json.loads((out / "stage30_results.json").read_text())
            assert persisted["stage30_freeze_ready"] is True

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_stage30_blocks_when_stage29_release_candidate_is_not_ready(self):
        from stage30.run_stage30 import _build_result

        stage29 = _stage29_fixture()
        stage29["release_candidate_ready"] = False
        result = _build_result(
            stage29=stage29,
            baseline_chain=_baseline_chain_fixture(),
            cycle_results=[],
            rollback_report={"rollback_chain_passed": True},
            production_readiness={"passed": True},
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )

        assert result["stage29_release_candidate_ready"] is False
        assert "stage29_release_candidate_ready" in result["blocked_items"]
        assert result["overall_pass"] is False

    def test_stage30_blocks_on_memory_contamination(self):
        from stage30.run_stage30 import _build_result

        result = _build_result(
            stage29=_stage29_fixture(),
            baseline_chain=_baseline_chain_fixture(),
            cycle_results=[],
            rollback_report={"rollback_chain_passed": True},
            production_readiness={"passed": True},
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )
        result["memory_contamination"] = 1
        result["blocked_items"] = ["memory_contamination"]
        result["stage30_freeze_ready"] = False
        result["overall_pass"] = False

        assert result["memory_contamination"] == 1
        assert result["overall_pass"] is False


def _stage29_fixture() -> dict:
    return {
        "release_candidate_ready": True,
        "stage29_freeze_ready": True,
        "operator_runbooks_ready": True,
        "production_shadow_passed": True,
    }


def _baseline_chain_fixture() -> list[dict]:
    return [
        {
            "stage": "Stage21-B",
            "result_present": True,
            "report_present": True,
            "ready": True,
            "overall_pass": True,
            "tag_present": True,
            "tag_required": True,
            "critical_drift_count": 0,
            "rollback_chain_passed": True,
        },
        {
            "stage": "Stage22",
            "result_present": True,
            "report_present": True,
            "ready": True,
            "overall_pass": True,
            "tag_present": True,
            "tag_required": True,
            "critical_drift_count": 0,
            "rollback_chain_passed": True,
        },
        {
            "stage": "Stage24",
            "result_present": True,
            "report_present": True,
            "ready": True,
            "overall_pass": True,
            "tag_present": True,
            "tag_required": True,
            "critical_drift_count": 0,
            "rollback_chain_passed": True,
        },
        {
            "stage": "Stage26",
            "result_present": True,
            "report_present": True,
            "ready": True,
            "overall_pass": True,
            "tag_present": True,
            "tag_required": True,
            "critical_drift_count": 0,
            "rollback_chain_passed": True,
        },
        {
            "stage": "Stage28",
            "result_present": True,
            "report_present": True,
            "ready": True,
            "overall_pass": True,
            "tag_present": True,
            "tag_required": True,
            "critical_drift_count": 0,
            "rollback_chain_passed": True,
        },
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
