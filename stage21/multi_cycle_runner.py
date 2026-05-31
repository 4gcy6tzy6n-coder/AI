"""Stage 21: Multi-Cycle Evolution Engine.

Runs the Stage 20 pipeline across multiple cycles, tracking metrics,
accumulating fix packages, detecting drift, and verifying multi-version
rollback capability.

Sub-stages:
  21-A: 5-cycle with fixed failure pool
  21-B: 10-cycle with evolving failure pool
  21-C: 20-cycle with learned Bayesian priors (future)
  21-D: Multi-version rollback chain verification
  21-E: Production readiness assessment (future)
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.bayes.schema import FailureSample, FixPackage

from stage20.acceptance_harness import (
    AcceptanceMetricsCollector,
    RealisticFailurePool,
    RollbackVerificationHarness,
    Stage20AcceptanceMetrics,
)
from stage20.failure_pool import TriggerCondition
from stage20.offline_replay import OfflineReplayVerifier
from stage20.stage20_pipeline import Stage20Pipeline
from stage20.test_sets import RegressionSetBuilder, StressSetBuilder
from stage21.rollback_chain import RollbackChainVerifier
from src.core.bayes.schema import ReplayResult


# ─── Cycle Metrics ────────────────────────────────────────────────

@dataclass
class CycleMetrics:
    """Metrics for a single evolution cycle."""
    cycle_number: int
    metrics: Stage20AcceptanceMetrics
    num_fix_packages: int
    fix_package_types: dict[str, int] = field(default_factory=dict)
    all_pass: bool = False
    timestamp: str = ""
    failure_pool_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle_number,
            "metrics": self.metrics.to_dict(),
            "num_fix_packages": self.num_fix_packages,
            "fix_package_types": self.fix_package_types,
            "all_pass": self.all_pass,
            "timestamp": self.timestamp,
            "failure_pool_hash": self.failure_pool_hash,
        }


@dataclass
class CycleDriftReport:
    """Drift detection across cycles."""
    metric_drifts: dict[str, list[float]] = field(default_factory=dict)
    max_drifts: dict[str, float] = field(default_factory=dict)
    degraded_cycles: list[int] = field(default_factory=list)
    stable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_drifts": self.metric_drifts,
            "max_drifts": self.max_drifts,
            "degraded_cycles": self.degraded_cycles,
            "stable": self.stable,
        }


# ─── Multi-Cycle Runner ───────────────────────────────────────────

class MultiCycleRunner:
    """Runs the Stage 20 pipeline for N cycles.

    Each cycle:
    1. Runs the full Stage 20 pipeline (trigger→diagnose→fix→replay→freeze)
    2. Collects all 8 acceptance metrics
    3. Accumulates fix packages
    4. Detects drift from previous cycles
    5. Freezes a per-cycle baseline snapshot
    """

    def __init__(
        self,
        num_cycles: int = 5,
        output_dir: str = "data/stage21/",
        config: dict[str, Any] | None = None,
    ):
        self.num_cycles = num_cycles
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cycle_results: list[CycleMetrics] = []
        self.all_fix_packages: list[FixPackage] = []
        self.all_diagnoses: list[dict[str, Any]] = []
        self.rollback_snapshots: list[dict[str, Any]] = []

        cfg = config or {}
        self.drift_threshold = cfg.get("drift_threshold", 0.05)  # 5% max drift
        self.min_pass_rate = cfg.get("min_pass_rate", 0.80)
        self.require_all_cycles_pass = cfg.get("require_all_cycles_pass", False)
        self.stage_name = cfg.get("stage", "Stage21")
        self.thresholds = cfg.get("thresholds", self._default_thresholds())
        self.fixed_failure_pool_hash = ""
        self.fixed_failure_pool_signature: list[dict[str, str]] = []

    @staticmethod
    def _default_thresholds() -> dict[str, Any]:
        return {
            "completed_cycles": "5/5",
            "cycle_pass_count": "5/5",
            "failure_fix_rate": 0.80,
            "regression_pass_rate": 0.98,
            "tsla_safety_intercept": 0.99,
            "false_kill_rate": 0.01,
            "memory_contamination": 0,
            "retrieval_context_failure": 0.02,
            "multiturn_consistency": 0.95,
            "rollback_success_rate": 1.0,
            "drift_threshold": 0.05,
            "drift_status": "no critical drift",
            "rollback_chain_passed": True,
            "stage21_a_freeze_ready": True,
        }

    def run(
        self,
        failure_samples: list[FailureSample] | None = None,
        regression_samples: list[FailureSample] | None = None,
        stress_samples: list[FailureSample] | None = None,
        evolve_pool: bool = False,
    ) -> dict[str, Any]:
        """Run N cycles of Stage 20 pipeline.

        Args:
            failure_samples: Base failure pool (used in all cycles if not evolving)
            regression_samples: Regression set
            stress_samples: Stress set
            evolve_pool: If True, inject new failure types each cycle (21-B mode)

        Returns dict with all cycle results, drift report, and overall pass/fail.
        """
        if failure_samples is None:
            failure_samples = RealisticFailurePool.build(
                target_size=20,
                output_path=str(self.output_dir / "fixed_failure_pool.jsonl"),
            )
        if regression_samples is None:
            regression_samples = RegressionSetBuilder.build(
                output_path=str(self.output_dir / "fixed_regression_set.jsonl"),
            )
        if stress_samples is None:
            stress_samples = StressSetBuilder.build(
                output_path=str(self.output_dir / "fixed_stress_set.jsonl"),
            )

        current_failures = list(failure_samples)
        self.fixed_failure_pool_signature = self._failure_pool_signature(current_failures)
        self.fixed_failure_pool_hash = self._failure_pool_hash(current_failures)
        timestamp = datetime.now(timezone.utc).isoformat()

        for cycle in range(1, self.num_cycles + 1):
            print("  Cycle {}/{}: ".format(cycle, self.num_cycles), end="")
            if not evolve_pool:
                self._assert_fixed_failure_pool(current_failures)

            # Run pipeline
            result = self._run_single_cycle(
                cycle=cycle,
                failure_samples=current_failures,
                regression_samples=regression_samples,
                stress_samples=stress_samples,
            )

            # Collect cycle metrics
            cycle_metrics = result["cycle_metrics"]
            self.cycle_results.append(cycle_metrics)

            # Accumulate
            self.all_fix_packages.extend(result["fix_packages"])
            self.all_diagnoses.extend(result["diagnoses"])
            self.rollback_snapshots.append(result["freeze_snapshot"])

            status = "PASS" if cycle_metrics.all_pass else "FAIL"
            print("{} ({} fix pkgs, {})".format(
                status, cycle_metrics.num_fix_packages,
                "/".join("{}={}".format(k, v) for k, v in
                         sorted(cycle_metrics.fix_package_types.items()))))

            # Evolve pool for 21-B
            if evolve_pool:
                current_failures = self._evolve_failure_pool(
                    current_failures, cycle, result["diagnoses"])

        # Drift analysis
        drift = self._analyze_drift()
        drift_status = "no critical drift" if drift.stable else "critical drift"

        # Multi-version rollback chain verification
        rollback_chain = RollbackChainVerifier(output_dir=str(self.output_dir))
        rollback_result = rollback_chain.verify_chain(
            [c.to_dict() for c in self.cycle_results]
        )
        rollback_chain_passed = rollback_result.all_passed

        # Overall assessment
        cycle_pass_count = sum(1 for c in self.cycle_results if c.all_pass)
        cycles_requirement_met = (
            cycle_pass_count == self.num_cycles
            if self.require_all_cycles_pass
            else cycle_pass_count >= self.num_cycles * self.min_pass_rate
        )
        stage_cycle_count_valid = (
            self.num_cycles == 5 if self.stage_name == "Stage21-A" else True
        )
        overall_pass = (
            cycles_requirement_met
            and stage_cycle_count_valid
            and drift.stable
            and rollback_chain_passed
        )
        stage21_a_freeze_ready = (
            self.stage_name == "Stage21-A"
            and self.num_cycles == 5
            and overall_pass
            and self.fixed_failure_pool_hash != ""
            and cycle_pass_count == 5
            and rollback_chain_passed
        )

        # Generate report
        report = self._generate_report(
            timestamp,
            drift,
            overall_pass,
            rollback_chain_passed=rollback_chain_passed,
            stage21_a_freeze_ready=stage21_a_freeze_ready,
        )

        return {
            "stage": self.stage_name,
            "num_cycles": self.num_cycles,
            "cycles_completed": self.num_cycles,
            "completed_cycles": self.num_cycles,
            "cycle_pass_count": cycle_pass_count,
            "stage21_a_freeze_ready": stage21_a_freeze_ready,
            "cycle_results": [c.to_dict() for c in self.cycle_results],
            "drift_report": drift.to_dict(),
            "drift_status": drift_status,
            "total_fix_packages": len(self.all_fix_packages),
            "fixed_failure_pool_hash": self.fixed_failure_pool_hash,
            "fixed_failure_pool_signature": self.fixed_failure_pool_signature,
            "rollback_chain_passed": rollback_chain_passed,
            "rollback_chain_report": rollback_result.to_dict(),
            "thresholds": self.thresholds,
            "overall_pass": overall_pass,
            "report_path": report,
            "timestamp": timestamp,
        }

    def _run_single_cycle(
        self,
        cycle: int,
        failure_samples: list[FailureSample],
        regression_samples: list[FailureSample],
        stress_samples: list[FailureSample],
    ) -> dict[str, Any]:
        """Execute one full Stage 20 pipeline cycle."""
        cycle_dir = self.output_dir / "cycle_{:02d}".format(cycle)
        cycle_dir.mkdir(parents=True, exist_ok=True)
        failure_pool_path = cycle_dir / "failure_pool.jsonl"
        self._write_failure_pool(failure_pool_path, failure_samples)

        trigger = TriggerCondition(
            knowledge_miss_count=25,
            unnatural_generation_count=12,
            multiturn_anomaly_count=5,
        )

        pipeline = Stage20Pipeline(config={
            "failure_pool_path": str(failure_pool_path),
            "output_dir": str(cycle_dir),
            "patch_output_dir": str(cycle_dir / "stage20_patches"),
        })
        result = pipeline.run(
            trigger_condition=trigger,
            failure_samples=failure_samples,
            regression_samples=regression_samples,
            stress_samples=stress_samples,
        )

        # Collect metrics
        collector = AcceptanceMetricsCollector()
        diagnoses = result["steps"].get("20-2_5_btsta_diagnoses", {}).get("diagnoses", [])
        collector.feed_diagnoses(diagnoses)

        all_replays = []
        for r_dict in result["steps"].get("20-6_replay", {}).get("results", []):
            rr = ReplayResult(
                test_set_name=r_dict["test_set_name"],
                total_cases=r_dict["total_cases"],
                passed_cases=r_dict["passed_cases"],
                failure_fix_rate=r_dict.get("failure_fix_rate", 0),
                regression_pass_rate=r_dict.get("regression_pass_rate", 0),
                tsla_safety_intercept=r_dict.get("tsla_safety_intercept", 0),
                false_kill_rate=r_dict.get("false_kill_rate", 0),
                memory_contamination_count=r_dict.get("memory_contamination_count", 0),
            )
            all_replays.append(rr)
        collector.feed_replay(all_replays)

        # Rollback
        rollback = RollbackVerificationHarness(output_dir=str(cycle_dir))
        rb_passed, _ = rollback.run_harness()
        collector.feed_rollback_test(rb_passed)

        metrics = collector.compute(failure_samples)

        # Fix package type distribution
        fix_packages = result["steps"].get("_fix_packages", [])
        pkg_types: dict[str, int] = {}
        for p in fix_packages:
            t = p.get("package_type", "unknown")
            pkg_types[t] = pkg_types.get(t, 0) + 1

        passed, _ = metrics.all_passing(self.thresholds)
        passed = passed and rb_passed

        cm = CycleMetrics(
            cycle_number=cycle,
            metrics=metrics,
            num_fix_packages=len(fix_packages),
            fix_package_types=pkg_types,
            all_pass=passed,
            timestamp=metrics.timestamp,
            failure_pool_hash=self._failure_pool_hash(failure_samples),
        )

        # Extract FixPackage objects
        step25 = result["steps"].get("20-2_5_btsta_diagnoses", {})
        actual_packages = step25.get("fix_packages", [])

        freeze_data = result["steps"].get("20-9_freeze", {}).get("freeze_point", {})

        return {
            "cycle_metrics": cm,
            "diagnoses": diagnoses,
            "fix_packages": actual_packages if actual_packages else [],
            "freeze_snapshot": freeze_data,
        }

    @staticmethod
    def _failure_pool_signature(samples: list[FailureSample]) -> list[dict[str, str]]:
        """Stable fixed-pool identity, intentionally excluding timestamps."""
        return [
            {
                "sample_id": sample.sample_id,
                "failure_type": sample.failure_type.value,
            }
            for sample in samples
        ]

    @classmethod
    def _failure_pool_hash(cls, samples: list[FailureSample]) -> str:
        payload = json.dumps(
            cls._failure_pool_signature(samples),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _assert_fixed_failure_pool(self, samples: list[FailureSample]) -> None:
        current_signature = self._failure_pool_signature(samples)
        if current_signature != self.fixed_failure_pool_signature:
            raise ValueError("Stage21-A fixed failure pool changed between cycles")

    @staticmethod
    def _write_failure_pool(path: Path, samples: list[FailureSample]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    def _evolve_failure_pool(
        self,
        current: list[FailureSample],
        cycle: int,
        diagnoses: list[dict[str, Any]],
    ) -> list[FailureSample]:
        """Evolve the failure pool for 21-B mode.

        Each cycle: keep all existing failures, add 2-3 new ones
        with types that were under-diagnosed in previous cycles.
        """
        from src.core.bayes.schema import FailureCategory, RiskLevel

        evolved = list(current)

        # Add 2 new failure samples per cycle, rotating through types
        new_types = [
            FailureCategory.K1, FailureCategory.M1, FailureCategory.R1,
            FailureCategory.G1, FailureCategory.T1, FailureCategory.S1,
            FailureCategory.T2, FailureCategory.W1,
        ]
        type_idx = (cycle - 1) % len(new_types)

        for i in range(2):
            ft = new_types[(type_idx + i) % len(new_types)]
            new_sample = FailureSample(
                sample_id="EVO_C{}_N{}".format(cycle, i + 1),
                user_query="[Evolved cycle {}] {} test query".format(cycle, ft.value),
                system_response="[Evolved response for cycle {}]".format(cycle),
                failure_type=ft,
                risk_level=RiskLevel.MEDIUM,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"evolved": True, "cycle": cycle},
            )
            evolved.append(new_sample)

        return evolved

    def _analyze_drift(self) -> CycleDriftReport:
        """Detect metric drift across cycles."""
        if len(self.cycle_results) < 2:
            return CycleDriftReport(stable=True)

        metric_names = [
            "failure_fix_rate", "regression_pass_rate",
            "tsla_safety_intercept", "false_kill_rate",
            "memory_contamination", "retrieval_context_failure",
            "multiturn_consistency", "rollback_success_rate",
        ]

        drifts: dict[str, list[float]] = {n: [] for n in metric_names}
        max_drifts: dict[str, float] = {}
        degraded_cycles: set[int] = set()

        baseline_values: dict[str, float] = {}
        for name in metric_names:
            baseline_values[name] = getattr(
                self.cycle_results[0].metrics, name, 0.0)

        for cycle_result in self.cycle_results[1:]:
            cycle = cycle_result.cycle_number
            for name in metric_names:
                current = getattr(cycle_result.metrics, name, 0.0)
                baseline = baseline_values[name]
                delta = abs(current - baseline)
                drifts[name].append(delta)

                if name == "memory_contamination" and current > 0:
                    max_drifts[name] = max(max_drifts.get(name, 0.0), delta)
                    degraded_cycles.add(cycle)
                    continue

                if delta > self.drift_threshold:
                    max_drifts[name] = max(
                        max_drifts.get(name, 0.0), delta)
                    # Only flag as degraded if drift is negative
                    # (metric got worse, not better)
                    if (name in ("false_kill_rate", "memory_contamination",
                                  "retrieval_context_failure")
                            and current > baseline):
                        degraded_cycles.add(cycle)
                    elif (name not in ("false_kill_rate", "memory_contamination",
                                        "retrieval_context_failure")
                          and current < baseline):
                        degraded_cycles.add(cycle)

        first_cycle_contaminated = (
            self.cycle_results[0].metrics.memory_contamination > 0
        )
        if first_cycle_contaminated:
            degraded_cycles.add(self.cycle_results[0].cycle_number)

        stable = len(degraded_cycles) == 0

        return CycleDriftReport(
            metric_drifts=drifts,
            max_drifts=max_drifts,
            degraded_cycles=sorted(degraded_cycles),
            stable=stable,
        )

    def _generate_report(
        self,
        timestamp: str,
        drift: CycleDriftReport,
        overall_pass: bool,
        rollback_chain_passed: bool = False,
        stage21_a_freeze_ready: bool = False,
    ) -> str:
        """Generate multi-cycle evolution report."""
        lines: list[str] = []
        title = (
            "Stage 21-A Fixed Failure Pool Validation Report"
            if self.stage_name == "Stage21-A"
            else "Stage 21 Multi-Cycle Evolution Report"
        )
        lines.append("# {}".format(title))
        lines.append("")
        lines.append("**Generated**: {}".format(timestamp))
        lines.append("**Cycles**: {}".format(self.num_cycles))
        if self.fixed_failure_pool_hash:
            lines.append("**Fixed failure pool hash**: `{}`".format(
                self.fixed_failure_pool_hash
            ))
        lines.append("")
        lines.append("---")
        lines.append("")

        verdict = "PASS" if overall_pass else "FAIL"
        lines.append("## Overall Verdict: **{}**".format(verdict))
        lines.append("")

        cycle_pass = sum(1 for c in self.cycle_results if c.all_pass)
        lines.append("- Cycles passed: {}/{}".format(cycle_pass, self.num_cycles))
        lines.append("- Total fix packages: {}".format(len(self.all_fix_packages)))
        lines.append("- Drift stable: {}".format(drift.stable))
        lines.append("- Rollback chain passed: {}".format(rollback_chain_passed))
        lines.append("- Stage21-A freeze ready: {}".format(stage21_a_freeze_ready))
        if drift.degraded_cycles:
            lines.append("- Degraded cycles: {}".format(drift.degraded_cycles))
        lines.append("")

        # Per-cycle table
        lines.append("## Per-Cycle Metrics")
        lines.append("")
        lines.append("| Cycle | Fix Rate | Reg Pass | TSLA Int | False Kill | Contam | R1 Fail | MT Consist | Rollback | Status |")
        lines.append("|-------|----------|----------|----------|------------|--------|---------|------------|----------|--------|")
        for cm in self.cycle_results:
            m = cm.metrics
            status = "PASS" if cm.all_pass else "FAIL"
            lines.append(
                "| {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {} | {:.0%} | {:.0%} | {:.0%} | {} |".format(
                    cm.cycle_number, m.failure_fix_rate, m.regression_pass_rate,
                    m.tsla_safety_intercept, m.false_kill_rate,
                    int(m.memory_contamination), m.retrieval_context_failure,
                    m.multiturn_consistency, m.rollback_success_rate, status))
        lines.append("")

        # Drift section
        lines.append("## Drift Analysis")
        lines.append("")
        if drift.stable:
            lines.append("No significant drift detected across {} cycles.".format(
                self.num_cycles))
            lines.append("")
        else:
            lines.append("Drift detected in the following metrics:")
            for name, max_d in drift.max_drifts.items():
                lines.append("- **{}**: max drift {:.2%}".format(name, max_d))
            lines.append("")
            lines.append("Degraded cycles: {}".format(drift.degraded_cycles))
            lines.append("")

        # Package accumulation
        lines.append("## Fix Package Accumulation")
        lines.append("")
        all_types: dict[str, int] = {}
        for cm in self.cycle_results:
            for ptype, count in cm.fix_package_types.items():
                all_types[ptype] = all_types.get(ptype, 0) + count
        lines.append("Total packages by type across {} cycles:".format(
            self.num_cycles))
        for ptype, count in sorted(all_types.items()):
            lines.append("- **{}**: {}".format(ptype, count))
        lines.append("")

        content = "\n".join(lines)
        path = self.output_dir / "STAGE21_MULTI_CYCLE_REPORT.md"
        path.write_text(content, encoding="utf-8")
        return str(path)
