"""Tests for Stage22 independent multi-pool holdout validation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStage22:
    def test_stage22_runs_twelve_independent_holdout_cycles(self):
        from stage22.run_stage22 import run_stage22

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage22(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage22"
            assert result["completed_cycles"] == 12
            assert result["cycle_pass_count"] == 12
            assert result["overall_pass"] is True
            assert result["stage22_freeze_ready"] is True
            assert result["drift_status"] == "no critical drift"
            assert result["rollback_chain_passed"] is True
            assert result["independent_holdout_pool_count"] >= 4
            assert all(v >= 3 for v in result["cycles_per_holdout_pool"].values())
            assert result["cross_pool_regression_drop"] <= 0.01

            holdout_hashes = result["holdout_pool_hashes"]
            assert len(holdout_hashes) == 12
            assert len(set(holdout_hashes)) == 4
            assert len(result["holdout_pool_lineage"]) == 12

            assert (out / "STAGE22_FINAL_REPORT.md").exists()
            assert (out / "STAGE22_MULTI_POOL_REPORT.md").exists()
            assert (out / "STAGE22_ROLLBACK_CHAIN_REPORT.json").exists()
            assert (out / "stage22_results.json").exists()
            assert (out / "holdout_pool_lineage.json").exists()
            assert (out / "fixed_regression_set.jsonl").exists()
            assert (out / "fixed_stress_set.jsonl").exists()
            assert not (Path("data") / "stage21" / "stage22_results.json").exists()

            for pool_idx in range(1, 5):
                assert (out / "holdout_pools" / f"holdout_{pool_idx:02d}.jsonl").exists()

            for cycle in range(1, 13):
                cycle_dir = out / f"cycle_{cycle:02d}"
                assert cycle_dir.exists()
                assert (cycle_dir / "failure_pool.jsonl").exists()
                assert (cycle_dir / "stage20_freeze_manifest.json").exists()
                assert (cycle_dir / "stage20_rollback_runbook.md").exists()
                assert (cycle_dir / "STAGE20_ROLLBACK_VERIFICATION.json").exists()
                assert (cycle_dir / "stage20_patches").exists()

            persisted = json.loads((out / "stage22_results.json").read_text())
            assert persisted["overall_pass"] is True
            assert persisted["stage22_freeze_ready"] is True
            assert persisted["independent_holdout_pool_count"] == 4

            rollback_report = json.loads(
                (out / "STAGE22_ROLLBACK_CHAIN_REPORT.json").read_text()
            )
            assert rollback_report["all_passed"] is True

    def test_synthetic_cross_pool_regression_drop_fails_stage22(self):
        from stage20.acceptance_harness import Stage20AcceptanceMetrics
        from stage21.multi_cycle_runner import CycleMetrics, MultiCycleRunner
        from stage22.run_stage22 import _build_result

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MultiCycleRunner(
                num_cycles=12,
                output_dir=tmpdir,
                config={"stage": "Stage22", "require_all_cycles_pass": True},
            )
            holdout_lineage = []
            for cycle in range(1, 13):
                pool_id = f"holdout_{((cycle - 1) // 3) + 1:02d}"
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
                    total_samples=20,
                )
                runner.cycle_results.append(
                    CycleMetrics(
                        cycle_number=cycle,
                        metrics=metrics,
                        num_fix_packages=1,
                        fix_package_types={"synthetic": 1},
                        all_pass=True,
                        failure_pool_hash=f"hash_{pool_id}",
                        pool_variant_id=pool_id,
                        pool_lineage={
                            "holdout_pool_id": pool_id,
                            "holdout_pool_hash": f"hash_{pool_id}",
                        },
                    )
                )
                holdout_lineage.append({
                    "cycle": cycle,
                    "holdout_pool_id": pool_id,
                    "holdout_pool_hash": f"hash_{pool_id}",
                    "sample_count": 20,
                    "source": "synthetic",
                    "sample_signature": [],
                })

            result = _build_result(runner, holdout_lineage)

            assert result["completed_cycles"] == 12
            assert result["cycle_pass_count"] == 12
            assert result["independent_holdout_pool_count"] == 4
            assert result["cross_pool_regression_drop"] > 0.01
            assert result["overall_pass"] is False
            assert result["stage22_freeze_ready"] is False

    def test_synthetic_memory_contamination_fails_stage22(self):
        from stage20.acceptance_harness import Stage20AcceptanceMetrics
        from stage21.multi_cycle_runner import CycleMetrics, MultiCycleRunner
        from stage22.run_stage22 import _build_result

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MultiCycleRunner(
                num_cycles=12,
                output_dir=tmpdir,
                config={"stage": "Stage22", "require_all_cycles_pass": True},
            )
            holdout_lineage = []
            for cycle in range(1, 13):
                pool_id = f"holdout_{((cycle - 1) // 3) + 1:02d}"
                metrics = Stage20AcceptanceMetrics(
                    failure_fix_rate=0.90,
                    regression_pass_rate=1.0,
                    tsla_safety_intercept=1.0,
                    false_kill_rate=0.0,
                    memory_contamination=1 if cycle == 5 else 0,
                    retrieval_context_failure=0.0,
                    multiturn_consistency=1.0,
                    rollback_success_rate=1.0,
                    total_samples=20,
                )
                runner.cycle_results.append(
                    CycleMetrics(
                        cycle_number=cycle,
                        metrics=metrics,
                        num_fix_packages=1,
                        fix_package_types={"synthetic": 1},
                        all_pass=cycle != 5,
                        failure_pool_hash=f"hash_{pool_id}",
                        pool_variant_id=pool_id,
                        pool_lineage={
                            "holdout_pool_id": pool_id,
                            "holdout_pool_hash": f"hash_{pool_id}",
                        },
                    )
                )
                holdout_lineage.append({
                    "cycle": cycle,
                    "holdout_pool_id": pool_id,
                    "holdout_pool_hash": f"hash_{pool_id}",
                    "sample_count": 20,
                    "source": "synthetic",
                    "sample_signature": [],
                })

            result = _build_result(runner, holdout_lineage)

            assert result["drift_status"] == "critical drift"
            assert result["cycle_pass_count"] == 11
            assert result["overall_pass"] is False
            assert result["stage22_freeze_ready"] is False
