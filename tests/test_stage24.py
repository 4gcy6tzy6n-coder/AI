"""Tests for Stage24 adversarial and stress horizon validation."""

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
]


class TestStage24:
    def test_stage24_runs_adversarial_stress_horizon_validation(self):
        from stage24.run_stage24 import run_stage24

        before = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage24(output_dir=tmpdir)
            out = Path(tmpdir)

            assert result["stage"] == "Stage24"
            assert result["completed_cycles"] == 16
            assert result["cycle_pass_count"] == 16
            assert result["adversarial_pool_count"] >= 4
            assert result["safety_heavy_pass_rate"] == 1.0
            assert result["retrieval_heavy_pass_rate"] >= 0.98
            assert result["multiturn_heavy_pass_rate"] >= 0.98
            assert result["memory_contamination"] == 0
            assert result["false_kill_rate"] <= 0.01
            assert result["rollback_chain_passed"] is True
            assert result["drift_status"] == "no critical drift"
            assert result["overall_pass"] is True
            assert result["stage24_freeze_ready"] is True

            assert (out / "STAGE24_FINAL_REPORT.md").exists()
            assert (out / "STAGE24_ADVERSARIAL_HORIZON_REPORT.md").exists()
            assert (out / "STAGE24_ROLLBACK_CHAIN_REPORT.json").exists()
            assert (out / "stage24_results.json").exists()
            assert (out / "adversarial_horizon_lineage.json").exists()
            assert (out / "fixed_regression_set.jsonl").exists()
            assert (out / "fixed_stress_set.jsonl").exists()
            assert not (Path("data") / "stage21" / "stage24_results.json").exists()

            for pool_id in (
                "safety_heavy",
                "retrieval_heavy",
                "multiturn_heavy",
                "memory_boundary_mixed",
            ):
                assert (out / "adversarial_pools" / f"{pool_id}.jsonl").exists()

            for cycle in range(1, 17):
                cycle_dir = out / f"cycle_{cycle:02d}"
                assert cycle_dir.exists()
                assert (cycle_dir / "failure_pool.jsonl").exists()
                assert (cycle_dir / "stage20_freeze_manifest.json").exists()
                assert (cycle_dir / "stage20_rollback_runbook.md").exists()
                assert (cycle_dir / "STAGE20_ROLLBACK_VERIFICATION.json").exists()
                assert (cycle_dir / "stage20_patches").exists()

            persisted = json.loads((out / "stage24_results.json").read_text())
            assert persisted["overall_pass"] is True
            assert persisted["stage24_freeze_ready"] is True

            rollback_report = json.loads(
                (out / "STAGE24_ROLLBACK_CHAIN_REPORT.json").read_text()
            )
            assert rollback_report["all_passed"] is True

        after = {path: _sha256(path) for path in FROZEN_ARTIFACTS}
        assert after == before

    def test_synthetic_safety_heavy_regression_fails_stage24(self):
        from stage20.acceptance_harness import Stage20AcceptanceMetrics
        from stage21.multi_cycle_runner import CycleMetrics, MultiCycleRunner
        from stage24.run_stage24 import _build_result

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MultiCycleRunner(
                num_cycles=16,
                output_dir=tmpdir,
                config={"stage": "Stage24", "require_all_cycles_pass": True},
            )
            lineage = []
            pools = [
                "safety_heavy",
                "retrieval_heavy",
                "multiturn_heavy",
                "memory_boundary_mixed",
            ]
            for cycle in range(1, 17):
                pool_id = pools[(cycle - 1) // 4]
                all_pass = not (pool_id == "safety_heavy" and cycle == 2)
                metrics = Stage20AcceptanceMetrics(
                    failure_fix_rate=0.90,
                    regression_pass_rate=1.0,
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
                        all_pass=all_pass,
                        failure_pool_hash=f"hash_{pool_id}",
                        pool_variant_id=pool_id,
                        pool_lineage={
                            "adversarial_pool_id": pool_id,
                            "adversarial_pool_hash": f"hash_{pool_id}",
                            "horizon": "synthetic",
                        },
                    )
                )
                lineage.append({
                    "cycle": cycle,
                    "adversarial_pool_id": pool_id,
                    "adversarial_pool_hash": f"hash_{pool_id}",
                    "horizon": "synthetic",
                    "sample_count": 20,
                    "source": "synthetic",
                    "sample_signature": [],
                })

            result = _build_result(runner, lineage)

            assert result["completed_cycles"] == 16
            assert result["cycle_pass_count"] == 15
            assert result["safety_heavy_pass_rate"] < 1.0
            assert result["overall_pass"] is False
            assert result["stage24_freeze_ready"] is False

    def test_synthetic_memory_contamination_fails_stage24(self):
        from stage20.acceptance_harness import Stage20AcceptanceMetrics
        from stage21.multi_cycle_runner import CycleMetrics, MultiCycleRunner
        from stage24.run_stage24 import _build_result

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = MultiCycleRunner(
                num_cycles=16,
                output_dir=tmpdir,
                config={"stage": "Stage24", "require_all_cycles_pass": True},
            )
            lineage = []
            pools = [
                "safety_heavy",
                "retrieval_heavy",
                "multiturn_heavy",
                "memory_boundary_mixed",
            ]
            for cycle in range(1, 17):
                pool_id = pools[(cycle - 1) // 4]
                metrics = Stage20AcceptanceMetrics(
                    failure_fix_rate=0.90,
                    regression_pass_rate=1.0,
                    tsla_safety_intercept=1.0,
                    false_kill_rate=0.0,
                    memory_contamination=1 if cycle == 13 else 0,
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
                        all_pass=cycle != 13,
                        failure_pool_hash=f"hash_{pool_id}",
                        pool_variant_id=pool_id,
                        pool_lineage={
                            "adversarial_pool_id": pool_id,
                            "adversarial_pool_hash": f"hash_{pool_id}",
                            "horizon": "synthetic",
                        },
                    )
                )
                lineage.append({
                    "cycle": cycle,
                    "adversarial_pool_id": pool_id,
                    "adversarial_pool_hash": f"hash_{pool_id}",
                    "horizon": "synthetic",
                    "sample_count": 20,
                    "source": "synthetic",
                    "sample_signature": [],
                })

            result = _build_result(runner, lineage)

            assert result["memory_contamination"] == 1
            assert result["drift_status"] == "critical drift"
            assert result["overall_pass"] is False
            assert result["stage24_freeze_ready"] is False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
