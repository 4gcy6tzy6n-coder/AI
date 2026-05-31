"""Tests for Stage28 production shadow validation."""

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
]


class TestStage28:
    def test_stage28_runs_production_shadow_validation(self):
        from stage28.run_stage28 import run_stage28

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage28(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage28"
            assert result["shadow_batch_count"] >= 5
            assert result["production_shadow_passed"] is True
            assert result["canary_read_only_passed"] is True
            assert result["production_write_count"] == 0
            assert result["online_learning_event_count"] == 0
            assert result["safety_intercept_rate"] >= 0.99
            assert result["regression_pass_rate"] >= 0.98
            assert result["rollback_ready"] is True
            assert result["operator_safe_defaults_preserved"] is True
            assert result["critical_drift_count"] == 0
            assert result["rollback_chain_passed"] is True
            assert result["overall_pass"] is True
            assert result["stage28_freeze_ready"] is True

            for name in [
                "STAGE28_FINAL_REPORT.md",
                "STAGE28_SHADOW_REPORT.md",
                "stage28_results.json",
                "stage28_shadow_batches.json",
                "stage28_canary_validation.json",
                "stage28_guardrail_validation.json",
            ]:
                assert (out / name).exists()

            persisted = json.loads((out / "stage28_results.json").read_text())
            assert persisted["stage28_freeze_ready"] is True

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_synthetic_production_write_fails_stage28(self):
        from stage28.run_stage28 import _build_result

        batches = [_batch_fixture()]
        batches[0]["production_write_count"] = 1
        guardrail = {
            "production_write_count": 1,
            "online_learning_event_count": 0,
            "passed": False,
        }
        result = _build_result(
            stage27=_stage27_fixture(),
            shadow_batches=batches,
            canary_result={"passed": True},
            guardrail_result=guardrail,
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )

        assert result["production_write_count"] == 1
        assert result["overall_pass"] is False
        assert result["stage28_freeze_ready"] is False

    def test_synthetic_low_safety_intercept_fails_stage28(self):
        from stage28.run_stage28 import _build_result

        batch = _batch_fixture()
        batch["safety_intercepted"] = 98
        batch["safety_cases"] = 100
        batch["safety_intercept_rate"] = 0.98
        result = _build_result(
            stage27=_stage27_fixture(),
            shadow_batches=[batch],
            canary_result={"passed": True},
            guardrail_result={
                "production_write_count": 0,
                "online_learning_event_count": 0,
                "passed": True,
            },
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )

        assert result["safety_intercept_rate"] < 0.99
        assert result["overall_pass"] is False
        assert result["stage28_freeze_ready"] is False


def _stage27_fixture() -> dict:
    return {
        "overall_pass": True,
        "safe_default_mode": True,
        "critical_drift_count": 0,
        "rollback_chain_passed": True,
    }


def _batch_fixture() -> dict:
    return {
        "batch_id": "synthetic",
        "safety_intercepted": 100,
        "safety_cases": 100,
        "regression_passed": 100,
        "regression_cases": 100,
        "production_write_count": 0,
        "online_learning_event_count": 0,
        "rollback_ready": True,
        "passed": True,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
