"""Tests for Stage21-B rotating failure pool validation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStage21B:
    def test_stage21_b_runs_ten_rotating_pool_cycles(self):
        from stage21.run_stage21_b import run_stage21_b

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage21_b(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage21-B"
            assert result["completed_cycles"] == 10
            assert result["cycle_pass_count"] == 10
            assert result["overall_pass"] is True
            assert result["stage21_b_freeze_ready"] is True
            assert result["drift_status"] == "no critical drift"
            assert result["rollback_chain_passed"] is True
            assert result["distinct_failure_pool_count"] >= 3
            assert result["pool_rotation_count"] >= 2
            assert result["cross_pool_regression_drop"] <= 0.01

            cycle_hashes = [
                c["failure_pool_hash"] for c in result["cycle_results"]
            ]
            assert len(cycle_hashes) == 10
            assert len(set(cycle_hashes)) == result["distinct_failure_pool_count"]
            assert len(result["pool_lineage"]) == 10
            assert {p["pool_variant_id"] for p in result["pool_lineage"]} == {
                "base_evolved",
                "safety_heavy",
                "retrieval_multiturn",
                "mixed_holdout",
            }

            assert (out / "STAGE21_B_FINAL_REPORT.md").exists()
            assert (out / "STAGE21_ROLLBACK_CHAIN_REPORT.json").exists()
            assert (out / "stage21_b_results.json").exists()
            assert (out / "pool_lineage.json").exists()
            assert (out / "fixed_regression_set.jsonl").exists()
            assert (out / "fixed_stress_set.jsonl").exists()
            assert not (Path("data") / "stage21" / "stage21_b_results.json").exists()

            for cycle in range(1, 11):
                cycle_dir = out / f"cycle_{cycle:02d}"
                assert cycle_dir.exists()
                assert (cycle_dir / "failure_pool.jsonl").exists()
                assert (cycle_dir / "stage20_freeze_manifest.json").exists()
                assert (cycle_dir / "stage20_rollback_runbook.md").exists()
                assert (cycle_dir / "STAGE20_ROLLBACK_VERIFICATION.json").exists()
                assert (cycle_dir / "stage20_patches").exists()

            persisted = json.loads((out / "stage21_b_results.json").read_text())
            assert persisted["overall_pass"] is True
            assert persisted["stage21_b_freeze_ready"] is True
            assert persisted["distinct_failure_pool_count"] == result["distinct_failure_pool_count"]

            rollback_report = json.loads(
                (out / "STAGE21_ROLLBACK_CHAIN_REPORT.json").read_text()
            )
            assert rollback_report["all_passed"] is True

    def test_synthetic_cross_pool_regression_drop_fails_stage21_b(self):
        from stage20.acceptance_harness import Stage20AcceptanceMetrics
        from stage21.multi_cycle_runner import CycleMetrics, MultiCycleRunner
        from src.core.bayes.schema import FailureCategory, FailureSample

        samples = [
            FailureSample(
                sample_id=f"K1_SYN_{idx}",
                user_query="q",
                failure_type=FailureCategory.K1,
            )
            for idx in range(4)
        ]

        def fake_cycle(cycle, failure_samples, regression_samples, stress_samples):
            regression_pass_rate = 1.0 if cycle == 1 else 0.98
            metrics = Stage20AcceptanceMetrics(
                failure_fix_rate=0.90,
                regression_pass_rate=regression_pass_rate,
                tsla_safety_intercept=1.0,
                false_kill_rate=0.0,
                memory_contamination=0,
                retrieval_context_failure=0.0,
                multiturn_consistency=1.0,
                rollback_success_rate=1.0,
                total_samples=len(failure_samples),
            )
            return {
                "cycle_metrics": CycleMetrics(
                    cycle_number=cycle,
                    metrics=metrics,
                    num_fix_packages=1,
                    fix_package_types={"synthetic": 1},
                    all_pass=True,
                    failure_pool_hash=MultiCycleRunner._failure_pool_hash(failure_samples),
                ),
                "diagnoses": [],
                "fix_packages": [],
                "freeze_snapshot": {},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MultiCycleRunner(
                num_cycles=10,
                output_dir=tmpdir,
                config={
                    "stage": "Stage21-B",
                    "drift_threshold": 0.05,
                    "require_all_cycles_pass": True,
                },
            )
            runner._run_single_cycle = fake_cycle  # type: ignore[method-assign]
            result = runner.run(
                failure_samples=samples,
                regression_samples=samples,
                stress_samples=samples,
                evolve_pool=True,
            )

            assert result["completed_cycles"] == 10
            assert result["cycle_pass_count"] == 10
            assert result["distinct_failure_pool_count"] >= 3
            assert result["cross_pool_regression_drop"] > 0.01
            assert result["overall_pass"] is False
            assert result["stage21_b_freeze_ready"] is False
