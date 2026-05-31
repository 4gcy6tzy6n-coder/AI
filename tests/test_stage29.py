"""Tests for Stage29 release candidate freeze validation."""

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
]


class TestStage29:
    def test_stage29_freezes_release_candidate(self):
        from stage29.run_stage29 import run_stage29

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage29(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage29"
            assert result["baseline_chain_complete"] is True
            assert result["stage28_production_shadow_passed"] is True
            assert result["release_candidate_ready"] is True
            assert result["required_stage_tags_present"] is True
            assert result["required_reports_present"] is True
            assert result["full_test_suite_passed"] is True
            assert result["critical_drift_count"] == 0
            assert result["rollback_chain_passed"] is True
            assert result["operator_runbooks_ready"] is True
            assert result["production_shadow_passed"] is True
            assert result["stage30_entry_ready"] is True
            assert result["stage29_freeze_ready"] is True
            assert result["overall_pass"] is True
            assert len(result["baseline_chain"]) == 10

            for name in [
                "STAGE29_FINAL_REPORT.md",
                "STAGE29_READINESS_CHECKLIST.md",
                "STAGE29_BASELINE_CHAIN.json",
                "STAGE29_RELEASE_CANDIDATE_MANIFEST.json",
                "stage29_results.json",
            ]:
                assert (out / name).exists()

            persisted = json.loads((out / "stage29_results.json").read_text())
            assert persisted["stage30_entry_ready"] is True

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_missing_tag_blocks_stage29(self):
        from stage29.run_stage29 import _build_baseline_chain, _build_result

        chain = _build_baseline_chain(available_tags=set())
        result = _build_result(
            baseline_chain=chain,
            runbook_checks=_runbooks_fixture(),
            stage28=_stage28_fixture(),
            release_manifest={"manifest_hash": "hash"},
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )

        assert result["required_stage_tags_present"] is False
        assert result["overall_pass"] is False
        assert result["stage29_freeze_ready"] is False

    def test_stage28_shadow_failure_blocks_stage29(self):
        from stage29.run_stage29 import _build_baseline_chain, _build_result

        tags = {
            "stage20-baseline-v1.0-revalidated",
            "stage21-a-fixed-failure-pool-v1.0",
            "stage21-b-rotating-failure-pools-v1.0",
            "stage22-multi-pool-holdout-v1.0",
            "stage23-learned-prior-validation-v1.0",
            "stage24-adversarial-stress-horizon-v1.0",
            "stage25-promotion-rollback-governance-v1.0",
            "stage26-observability-audit-cockpit-v1.0",
            "stage27-operator-workflow-readiness-v1.0",
            "stage28-production-shadow-validation-v1.0",
        }
        stage28 = _stage28_fixture()
        stage28["production_write_count"] = 1
        chain = _build_baseline_chain(available_tags=tags)
        result = _build_result(
            baseline_chain=chain,
            runbook_checks=_runbooks_fixture(),
            stage28=stage28,
            release_manifest={"manifest_hash": "hash"},
            baseline_hash_before={"a": "1"},
            baseline_hash_after={"a": "1"},
        )

        assert result["production_shadow_passed"] is False
        assert result["stage30_entry_ready"] is False
        assert result["overall_pass"] is False


def _stage28_fixture() -> dict:
    return {
        "production_shadow_passed": True,
        "production_write_count": 0,
        "online_learning_event_count": 0,
        "stage28_freeze_ready": True,
    }


def _runbooks_fixture() -> list[dict]:
    return [{"present": True}, {"present": True}, {"present": True}]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
