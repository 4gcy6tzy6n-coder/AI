"""Tests for Stage23 bounded learned-prior validation."""

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


FROZEN_ARTIFACTS = [
    Path("reports/stage21_a/stage21_a_results.json"),
    Path("reports/stage21_b/stage21_b_results.json"),
    Path("reports/stage22/stage22_results.json"),
]


class TestStage23:
    def test_stage23_runs_bounded_learned_prior_validation(self):
        from stage23.run_stage23 import run_stage23

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage23(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage23"
            assert result["baseline_chain_complete"] is True
            assert result["learned_prior_candidate_count"] >= 1
            assert result["prior_comparison_cycles"] >= 12
            assert result["learned_prior_safety_regression"] == 0
            assert result["rollback_chain_passed"] is True
            assert result["drift_status"] == "no critical drift"
            assert result["overall_pass"] is True
            assert result["stage23_freeze_ready"] is True
            assert (
                result["prior_comparison"]["learned_accuracy"]
                >= result["prior_comparison"]["heuristic_accuracy"]
            )
            assert (
                result["prior_comparison"]["learned_expected_cause_probability"]
                >= result["prior_comparison"]["heuristic_expected_cause_probability"]
            )

            assert (out / "STAGE23_FINAL_REPORT.md").exists()
            assert (out / "stage23_results.json").exists()
            assert (out / "learned_prior_candidates.json").exists()
            assert (out / "prior_comparison.json").exists()
            assert (out / "STAGE23_ROLLBACK_CHAIN_REPORT.json").exists()

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_synthetic_safety_regression_fails_stage23(self):
        import stage23.run_stage23 as runner

        operational = {
            "completed_cycles": 12,
            "drift_status": "no critical drift",
            "frozen_regression_pass_rate": 1.0,
            "frozen_tsla_safety_intercept": 1.0,
            "frozen_false_kill_rate": 0.0,
            "frozen_memory_contamination": 0,
            "learned_regression_pass_rate": 0.97,
            "learned_tsla_safety_intercept": 1.0,
            "learned_false_kill_rate": 0.0,
            "learned_memory_contamination": 0,
        }

        assert runner._safety_regression_count(operational) == 1

    def test_prior_rollback_requires_distinct_candidate(self):
        from src.core.bayes.bayesian_inferencer import DEFAULT_PRIORS
        from stage23.run_stage23 import _verify_prior_rollback

        result = _verify_prior_rollback(DEFAULT_PRIORS)

        assert result["all_passed"] is False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
