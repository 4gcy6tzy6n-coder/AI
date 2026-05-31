"""Tests for Stage21-A fixed failure pool validation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStage21A:
    def test_stage21_a_runs_five_fixed_pool_cycles(self):
        from stage21.run_stage21_a import run_stage21_a

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage21_a(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage21-A"
            assert result["completed_cycles"] == 5
            assert result["cycle_pass_count"] == 5
            assert result["overall_pass"] is True
            assert result["stage21_a_freeze_ready"] is True
            assert result["drift_status"] == "no critical drift"
            assert result["rollback_chain_passed"] is True
            assert result["fixed_failure_pool_hash"]

            cycle_hashes = {
                c["failure_pool_hash"] for c in result["cycle_results"]
            }
            assert cycle_hashes == {result["fixed_failure_pool_hash"]}

            assert (out / "STAGE21_A_FINAL_REPORT.md").exists()
            assert (out / "STAGE21_ROLLBACK_CHAIN_REPORT.json").exists()
            assert (out / "stage21_a_results.json").exists()
            for cycle in range(1, 6):
                cycle_dir = out / f"cycle_{cycle:02d}"
                assert cycle_dir.exists()
                assert (cycle_dir / "failure_pool.jsonl").exists()
                assert (cycle_dir / "stage20_freeze_manifest.json").exists()
                assert (cycle_dir / "stage20_rollback_runbook.md").exists()
                assert (cycle_dir / "STAGE20_ROLLBACK_VERIFICATION.json").exists()
                assert (cycle_dir / "stage20_patches").exists()

            persisted = json.loads((out / "stage21_a_results.json").read_text())
            assert persisted["overall_pass"] is True
            assert persisted["fixed_failure_pool_hash"] == result["fixed_failure_pool_hash"]

    def test_default_runner_writes_under_configured_output_dir(self):
        from stage21.multi_cycle_runner import MultiCycleRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            result = MultiCycleRunner(
                num_cycles=1,
                output_dir=tmpdir,
                config={"stage": "Stage21-A", "require_all_cycles_pass": True},
            ).run()

            out = Path(tmpdir)
            assert result["completed_cycles"] == 1
            assert (out / "fixed_failure_pool.jsonl").exists()
            assert (out / "fixed_regression_set.jsonl").exists()
            assert (out / "fixed_stress_set.jsonl").exists()
            assert (out / "cycle_01" / "failure_pool.jsonl").exists()
            assert str(out) in result["report_path"]

    def test_synthetic_degraded_metric_fails_overall_pass(self):
        from stage20.acceptance_harness import Stage20AcceptanceMetrics
        from stage21.multi_cycle_runner import CycleMetrics, MultiCycleRunner
        from src.core.bayes.schema import FailureCategory, FailureSample

        samples = [
            FailureSample(
                sample_id="K1_SYN",
                user_query="q",
                failure_type=FailureCategory.K1,
            )
        ]

        def fake_cycle(cycle, failure_samples, regression_samples, stress_samples):
            false_kill = 0.0 if cycle == 1 else 0.07
            metrics = Stage20AcceptanceMetrics(
                failure_fix_rate=0.90,
                regression_pass_rate=1.0,
                tsla_safety_intercept=1.0,
                false_kill_rate=false_kill,
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
                num_cycles=2,
                output_dir=tmpdir,
                config={
                    "stage": "Stage21-A",
                    "drift_threshold": 0.05,
                    "require_all_cycles_pass": True,
                },
            )
            runner._run_single_cycle = fake_cycle  # type: ignore[method-assign]
            result = runner.run(
                failure_samples=samples,
                regression_samples=samples,
                stress_samples=samples,
                evolve_pool=False,
            )

            assert result["cycle_pass_count"] == 2
            assert result["drift_status"] == "critical drift"
            assert result["overall_pass"] is False
            assert result["stage21_a_freeze_ready"] is False
