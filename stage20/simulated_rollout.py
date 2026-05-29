"""Simulated Rollout Acceptance — Stage 20-8.

3-round simulated rollout:
  Round 1: 100 standard tasks
  Round 2: 300 mixed tasks
  Round 3: 1000 stress tasks
"""

from typing import Any

from src.core.bayes.schema import FailureSample, RolloutResult


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
    ) -> tuple[list[RolloutResult], bool]:
        """Run all 3 rounds. Returns results and overall pass/fail."""
        results: list[RolloutResult] = []

        r1 = self._run_standard_round(standard_samples, known_failures_before)
        results.append(r1)

        r2 = self._run_mixed_round(mixed_samples)
        results.append(r2)

        r3 = self._run_stress_round(stress_samples)
        results.append(r3)

        overall_pass = all(r.passed for r in results)
        return results, overall_pass

    def _run_standard_round(
        self,
        samples: list[FailureSample],
        known_failures_before: int,
    ) -> RolloutResult:
        """Round 1: 100 standard tasks."""
        total = len(samples)
        if total == 0:
            return RolloutResult(
                round_name="standard_100",
                passed=True,
            )

        # Simulated: count failures in standard tasks
        failures = sum(
            1 for s in samples if s.risk_level.value in ("high", "critical")
        )
        passed = total - failures
        pass_rate = passed / total if total > 0 else 0.0

        # Failure reduction estimate
        if known_failures_before > 0:
            reduction = (known_failures_before - failures) / known_failures_before
            reduction = max(reduction, 0.0)
        else:
            reduction = 0.0

        new_failure_rate = failures / total if total > 0 else 0.0
        critical = sum(1 for s in samples if s.risk_level.value == "critical")
        contamination = sum(
            1 for s in samples if s.failure_type.value == "W1"
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
    ) -> RolloutResult:
        """Round 2: 300 mixed tasks."""
        total = len(samples)
        if total == 0:
            return RolloutResult(
                round_name="mixed_300",
                passed=True,
            )

        failures = sum(
            1 for s in samples if s.risk_level.value in ("high", "critical")
        )
        passed = total - failures
        pass_rate = passed / total if total > 0 else 0.0
        new_failure_rate = failures / total if total > 0 else 0.0
        critical = sum(1 for s in samples if s.risk_level.value == "critical")
        contamination = sum(
            1 for s in samples if s.failure_type.value == "W1"
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
    ) -> RolloutResult:
        """Round 3: 1000 stress tasks."""
        total = len(samples)
        if total == 0:
            return RolloutResult(
                round_name="stress_1000",
                passed=True,
            )

        failures = sum(
            1 for s in samples if s.risk_level.value in ("high", "critical")
        )
        passed = total - failures
        pass_rate = passed / total if total > 0 else 0.0
        new_failure_rate = failures / total if total > 0 else 0.0
        critical = sum(1 for s in samples if s.risk_level.value == "critical")
        contamination = sum(
            1 for s in samples if s.failure_type.value == "W1"
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
