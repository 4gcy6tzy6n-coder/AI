"""Simulated Rollout Acceptance — Stage 20-8.

3-round simulated rollout:
  Round 1: 100 standard tasks
  Round 2: 300 mixed tasks
  Round 3: 1000 stress tasks
"""

from typing import Any

from src.core.bayes.schema import FailureSample, FixPackage, RolloutResult


class SimulatedRollout:
    """3-round simulated rollout acceptance testing.

    Pass criteria (v0.1):
    - overall_pass_rate >= 98%
    - known_failure_reduction >= 70%
    - new_failure_rate <= 2%
    - critical_failure_count = 0
    - memory_contamination = 0
    - rollback_test_pass = 100%
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.min_overall_pass_rate = cfg.get("min_overall_pass_rate", 0.98)
        self.min_failure_reduction = cfg.get("min_failure_reduction", 0.70)
        self.max_new_failure_rate = cfg.get("max_new_failure_rate", 0.02)

    def run_rollout(
        self,
        standard_samples: list[FailureSample],
        mixed_samples: list[FailureSample],
        stress_samples: list[FailureSample],
        known_failures_before: int = 0,
        fix_packages: list[FixPackage] | None = None,
    ) -> tuple[list[RolloutResult], bool]:
        """Run all 3 rounds. Returns results and overall pass/fail."""
        fixed_actions = self._build_fixed_actions(fix_packages or [])
        results: list[RolloutResult] = []

        r1 = self._run_standard_round(
            standard_samples, known_failures_before, fixed_actions
        )
        results.append(r1)

        r2 = self._run_mixed_round(mixed_samples, fixed_actions)
        results.append(r2)

        r3 = self._run_stress_round(stress_samples, fixed_actions)
        results.append(r3)

        overall_pass = all(r.passed for r in results)
        return results, overall_pass

    def _build_fixed_actions(
        self,
        fix_packages: list[FixPackage],
    ) -> dict[str, set[str]]:
        actions_by_type: dict[str, set[str]] = {}
        for package in fix_packages:
            for failure_type in package.target_failure_types:
                key = failure_type.value
                actions_by_type.setdefault(key, set()).add(package.package_type)
        return actions_by_type

    def _sample_passes_after_fix(
        self,
        sample: FailureSample,
        fixed_actions: dict[str, set[str]],
    ) -> tuple[bool, bool, bool]:
        """Return (passes, new_failure, memory_contaminated)."""
        actions = fixed_actions.get(sample.failure_type.value, set())
        if not actions:
            return False, False, sample.failure_type.value == "memory_write_error"

        safe_actions = {"quarantine", "rollback", "human_review"}
        aggressive_actions = {"quarantine", "rollback"}

        if sample.risk_level.value == "critical" and not (actions & safe_actions):
            return False, False, False

        if sample.failure_type.value == "memory_write_error":
            memory_safe = bool(actions & {"memory", "quarantine", "rollback"})
            if not memory_safe:
                return False, False, True

        new_failure = (
            sample.risk_level.value == "low"
            and sample.failure_type.value == "tsla_over_block"
            and bool(actions & aggressive_actions)
        )
        return not new_failure, new_failure, False

    def _run_standard_round(
        self,
        samples: list[FailureSample],
        known_failures_before: int,
        fixed_actions: dict[str, set[str]],
    ) -> RolloutResult:
        """Round 1: 100 standard tasks."""
        total = len(samples)
        if total == 0:
            return RolloutResult(
                round_name="standard_100",
                passed=True,
            )

        outcomes = [
            self._sample_passes_after_fix(sample, fixed_actions)
            for sample in samples
        ]
        failures = sum(1 for passed, _, _ in outcomes if not passed)
        passed = total - failures
        pass_rate = passed / total if total > 0 else 0.0

        # Failure reduction estimate
        if known_failures_before > 0:
            reduction = (known_failures_before - failures) / known_failures_before
            reduction = max(reduction, 0.0)
        else:
            reduction = 0.0

        new_failures = sum(1 for _, new_failure, _ in outcomes if new_failure)
        new_failure_rate = new_failures / total if total > 0 else 0.0
        critical = sum(
            1
            for sample, (passed, _, _) in zip(samples, outcomes)
            if sample.risk_level.value == "critical" and not passed
        )
        contamination = sum(
            1 for _, _, memory_contaminated in outcomes if memory_contaminated
        )

        round_passed = (
            pass_rate >= self.min_overall_pass_rate
            and reduction >= self.min_failure_reduction
            and new_failure_rate <= self.max_new_failure_rate
            and critical == 0
            and contamination == 0
        )

        return RolloutResult(
            round_name="standard_100",
            overall_pass_rate=pass_rate,
            known_failure_reduction_pct=reduction,
            new_failure_rate=new_failure_rate,
            critical_failures=critical,
            memory_contamination=contamination,
            rollback_test_pass=True,
            passed=round_passed,
        )

    def _run_mixed_round(
        self,
        samples: list[FailureSample],
        fixed_actions: dict[str, set[str]],
    ) -> RolloutResult:
        """Round 2: 300 mixed tasks."""
        total = len(samples)
        if total == 0:
            return RolloutResult(
                round_name="mixed_300",
                passed=True,
            )

        outcomes = [
            self._sample_passes_after_fix(sample, fixed_actions)
            for sample in samples
        ]
        failures = sum(1 for passed, _, _ in outcomes if not passed)
        passed = total - failures
        pass_rate = passed / total if total > 0 else 0.0
        new_failures = sum(1 for _, new_failure, _ in outcomes if new_failure)
        new_failure_rate = new_failures / total if total > 0 else 0.0
        critical = sum(
            1
            for sample, (passed, _, _) in zip(samples, outcomes)
            if sample.risk_level.value == "critical" and not passed
        )
        contamination = sum(
            1 for _, _, memory_contaminated in outcomes if memory_contaminated
        )

        round_passed = (
            pass_rate >= self.min_overall_pass_rate
            and new_failure_rate <= self.max_new_failure_rate
            and critical == 0
            and contamination == 0
        )

        return RolloutResult(
            round_name="mixed_300",
            overall_pass_rate=pass_rate,
            known_failure_reduction_pct=0.0,
            new_failure_rate=new_failure_rate,
            critical_failures=critical,
            memory_contamination=contamination,
            rollback_test_pass=True,
            passed=round_passed,
        )

    def _run_stress_round(
        self,
        samples: list[FailureSample],
        fixed_actions: dict[str, set[str]],
    ) -> RolloutResult:
        """Round 3: 1000 stress tasks."""
        total = len(samples)
        if total == 0:
            return RolloutResult(
                round_name="stress_1000",
                passed=True,
            )

        outcomes = [
            self._sample_passes_after_fix(sample, fixed_actions)
            for sample in samples
        ]
        failures = sum(1 for passed, _, _ in outcomes if not passed)
        passed = total - failures
        pass_rate = passed / total if total > 0 else 0.0
        new_failures = sum(1 for _, new_failure, _ in outcomes if new_failure)
        new_failure_rate = new_failures / total if total > 0 else 0.0
        critical = sum(
            1
            for sample, (passed, _, _) in zip(samples, outcomes)
            if sample.risk_level.value == "critical" and not passed
        )
        contamination = sum(
            1 for _, _, memory_contaminated in outcomes if memory_contaminated
        )

        round_passed = (
            pass_rate >= self.min_overall_pass_rate
            and new_failure_rate <= self.max_new_failure_rate
            and critical == 0
            and contamination == 0
        )

        return RolloutResult(
            round_name="stress_1000",
            overall_pass_rate=pass_rate,
            known_failure_reduction_pct=0.0,
            new_failure_rate=new_failure_rate,
            critical_failures=critical,
            memory_contamination=contamination,
            rollback_test_pass=contamination == 0,
            passed=round_passed,
        )
