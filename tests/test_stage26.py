"""Tests for Stage26 observability and audit cockpit validation."""

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
]


class TestStage26:
    def test_stage26_runs_observability_audit_aggregation(self):
        from stage26.run_stage26 import run_stage26

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage26(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage26"
            assert result["baseline_chain_complete"] is True
            assert result["observed_stage_count"] >= 6
            assert result["metrics_index_coverage"] == 1.0
            assert result["lineage_index_coverage"] == 1.0
            assert result["drift_index_coverage"] == 1.0
            assert result["rollback_index_coverage"] == 1.0
            assert result["freeze_evidence_coverage"] == 1.0
            assert result["promotion_audit_coverage"] == 1.0
            assert result["machine_readable_artifacts"] is True
            assert result["human_readable_artifacts"] is True
            assert result["critical_drift_count"] == 0
            assert result["rollback_chain_passed"] is True
            assert result["overall_pass"] is True
            assert result["stage26_freeze_ready"] is True

            expected_files = [
                "STAGE26_FINAL_REPORT.md",
                "STAGE26_AUDIT_COCKPIT.md",
                "STAGE26_AUDIT_COCKPIT.json",
                "STAGE26_BASELINE_CHAIN.json",
                "STAGE26_METRICS_INDEX.json",
                "STAGE26_LINEAGE_INDEX.json",
                "STAGE26_DRIFT_INDEX.json",
                "STAGE26_ROLLBACK_INDEX.json",
                "STAGE26_FREEZE_EVIDENCE_INDEX.json",
                "STAGE26_PROMOTION_AUDIT_INDEX.json",
                "stage26_results.json",
            ]
            for name in expected_files:
                assert (out / name).exists()

            persisted = json.loads((out / "stage26_results.json").read_text())
            assert persisted["stage26_freeze_ready"] is True
            cockpit = json.loads((out / "STAGE26_AUDIT_COCKPIT.json").read_text())
            assert cockpit["baseline_chain"]["baseline_chain_complete"] is True

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_synthetic_missing_metrics_coverage_fails_stage26(self):
        from stage26.run_stage26 import _build_result, _coverage_index

        observed = [_observed_stage("stage25", rollback=True)]
        baseline_chain = {
            "baseline_chain_complete": True,
            "observed_stage_count": 6,
            "entries": [],
        }
        metrics_index = _coverage_index("metrics", [
            {"stage_id": "stage25", "covered": False}
        ])
        covered = _coverage_index("covered", [
            {"stage_id": "stage25", "covered": True}
        ])
        drift = _coverage_index("drift", [
            {"stage_id": "stage25", "covered": True, "critical_drift": False}
        ])
        drift["critical_drift_count"] = 0

        result = _build_result(
            observed=observed,
            baseline_chain=baseline_chain,
            metrics_index=metrics_index,
            lineage_index=covered,
            drift_index=drift,
            rollback_index=covered,
            freeze_index=covered,
            promotion_audit=covered,
            audit_cockpit={},
        )

        assert result["metrics_index_coverage"] == 0.0
        assert result["overall_pass"] is False
        assert result["stage26_freeze_ready"] is False

    def test_synthetic_critical_drift_fails_stage26(self):
        from stage26.run_stage26 import _build_result, _coverage_index

        observed = [_observed_stage("stage24", rollback=True)]
        baseline_chain = {
            "baseline_chain_complete": True,
            "observed_stage_count": 6,
            "entries": [],
        }
        covered = _coverage_index("covered", [
            {"stage_id": "stage24", "covered": True}
        ])
        drift = _coverage_index("drift", [
            {"stage_id": "stage24", "covered": True, "critical_drift": True}
        ])
        drift["critical_drift_count"] = 1

        result = _build_result(
            observed=observed,
            baseline_chain=baseline_chain,
            metrics_index=covered,
            lineage_index=covered,
            drift_index=drift,
            rollback_index=covered,
            freeze_index=covered,
            promotion_audit=covered,
            audit_cockpit={},
        )

        assert result["critical_drift_count"] == 1
        assert result["overall_pass"] is False
        assert result["stage26_freeze_ready"] is False


def _observed_stage(stage_id: str, rollback: bool) -> dict:
    return {
        "stage_id": stage_id,
        "result": {"rollback_chain_passed": rollback},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
