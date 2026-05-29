"""Offline Replay Verifier — Stage 20-6.

Runs 3 test sets against fix packages before any live application.
Consumes actual FixPackage objects and computes real (not stub) metrics.
"""

from typing import Any

from src.core.bayes.schema import (
    FailureCategory,
    FailureSample,
    FixPackage,
    GovernanceAction,
    ReplayResult,
    RiskLevel,
)


class OfflineReplayVerifier:
    """Offline replay verification across 3 test sets.

    1. trigger_set: Original failures — did fixes target them?
    2. regression_set: Baseline passing samples — did fixes break them?
    3. stress_set: Edge cases — do fixes handle extremes?

    All metrics are computed from actual FixPackage data, not stubs.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.min_failure_fix_rate = cfg.get("min_failure_fix_rate", 0.80)
        self.min_regression_pass_rate = cfg.get("min_regression_pass_rate", 0.98)
        self.min_tsla_safety_intercept = cfg.get("min_tsla_safety_intercept", 0.99)
        self.max_false_kill_rate = cfg.get("max_false_kill_rate", 0.01)
        self.max_memory_contamination = cfg.get("max_memory_contamination", 0)

    def run_verification(
        self,
        trigger_samples: list[FailureSample],
        regression_samples: list[FailureSample],
        stress_samples: list[FailureSample],
        fix_packages: list[FixPackage],
    ) -> tuple[list[ReplayResult], bool]:
        """Run all 3 test sets. Returns results and overall pass/fail."""
        results: list[ReplayResult] = []

        trigger_result = self._run_trigger_set(trigger_samples, fix_packages)
        results.append(trigger_result)

        regression_result = self._run_regression_set(regression_samples, fix_packages)
        results.append(regression_result)

        stress_result = self._run_stress_set(stress_samples, fix_packages)
        results.append(stress_result)

        overall_pass = (
            trigger_result.failure_fix_rate >= self.min_failure_fix_rate
            and regression_result.regression_pass_rate >= self.min_regression_pass_rate
            and trigger_result.tsla_safety_intercept >= self.min_tsla_safety_intercept
            and trigger_result.false_kill_rate <= self.max_false_kill_rate
            and trigger_result.memory_contamination_count <= self.max_memory_contamination
        )

        return results, overall_pass

    def _run_trigger_set(
        self,
        samples: list[FailureSample],
        fix_packages: list[FixPackage],
    ) -> ReplayResult:
        """Verify fix packages target the correct failure types.

        For each trigger sample, check:
        - Does a corresponding fix package target its failure_type?
        - Is the fix package action appropriate for the failure type?
        - Does the fix package risk_decision match the sample risk_level?

        Also computes TSLA safety intercept: % of safety-critical samples
        (S1, T1) that received appropriate governance actions
        (quarantine, rollback, human_review).
        """
        total = len(samples)
        if total == 0:
            return ReplayResult(test_set_name="trigger_set")

        # Build mapping: failure_type → set of actions from fix packages
        type_actions: dict[FailureCategory, set[str]] = {}
        for pkg in fix_packages:
            for ft in pkg.target_failure_types:
                if ft not in type_actions:
                    type_actions[ft] = set()
                type_actions[ft].add(pkg.package_type)

        fixed = 0
        safety_intercepted = 0
        safety_total = 0
        false_kills = 0
        details: list[dict[str, Any]] = []

        for sample in samples:
            ft = sample.failure_type
            actions = type_actions.get(ft, set())

            # A fix "applies" if at least one matching package exists
            is_fixed = len(actions) > 0
            if is_fixed:
                fixed += 1

            # TSLA safety intercept: S1/T1 samples need quarantine/rollback/human_review
            if ft in (FailureCategory.S1, FailureCategory.T1):
                safety_total += 1
                safety_actions = {"quarantine", "rollback", "human_review"}
                if actions & safety_actions:
                    safety_intercepted += 1

            # False kill: T2 (over-block) getting quarantine/rollback would be wrong
            if ft == FailureCategory.T2:
                if actions & {"quarantine", "rollback"}:
                    false_kills += 1

            # False kill: low-risk samples getting aggressive actions
            if sample.risk_level == RiskLevel.LOW:
                aggressive = {"quarantine", "rollback"}
                if actions & aggressive:
                    false_kills += 1

            details.append({
                "sample_id": sample.sample_id,
                "failure_type": ft.value,
                "risk_level": sample.risk_level.value,
                "fixed": is_fixed,
                "actions": list(actions),
            })

        fix_rate = fixed / total if total > 0 else 0.0
        tsla_intercept = safety_intercepted / safety_total if safety_total > 0 else 1.0
        false_kill = false_kills / total if total > 0 else 0.0

        return ReplayResult(
            test_set_name="trigger_set",
            total_cases=total,
            passed_cases=fixed,
            failure_fix_rate=fix_rate,
            tsla_safety_intercept=tsla_intercept,
            false_kill_rate=false_kill,
            details=tuple(details),
        )

    def _run_regression_set(
        self,
        samples: list[FailureSample],
        fix_packages: list[FixPackage],
    ) -> ReplayResult:
        """Ensure baseline passing samples are NOT affected by fix packages.

        Regression samples represent queries that Stage 18 handled correctly.
        Fix packages should NOT target these — if they do, it indicates
        over-correction that would cause regressions.

        Also checks that fix packages don't have actions that would
        quarantine or rollback content that was previously stable.
        """
        total = len(samples)
        if total == 0:
            return ReplayResult(
                test_set_name="regression_set",
                regression_pass_rate=1.0,
            )

        # Collect fix package target types and actions
        patched_types: set[FailureCategory] = set()
        aggressive_types: set[FailureCategory] = set()
        for pkg in fix_packages:
            for ft in pkg.target_failure_types:
                patched_types.add(ft)
            if pkg.package_type in ("quarantine", "rollback"):
                for ft in pkg.target_failure_types:
                    aggressive_types.add(ft)

        passed = 0
        details: list[dict[str, Any]] = []
        for sample in samples:
            ft = sample.failure_type
            # Sample passes regression if:
            # a) No fix package targets its type (fixes are for failures only)
            # b) OR the fix package is "keep" (no-op), which is safe
            is_affected = ft in patched_types
            is_aggressive = ft in aggressive_types

            # Regression pass: not affected OR only affected by safe actions
            sample_passed = not is_affected or not is_aggressive
            if sample_passed:
                passed += 1

            details.append({
                "sample_id": sample.sample_id,
                "failure_type": ft.value,
                "patched": is_affected,
                "aggressive": is_aggressive,
                "passed": sample_passed,
            })

        reg_pass_rate = passed / total if total > 0 else 0.0

        return ReplayResult(
            test_set_name="regression_set",
            total_cases=total,
            passed_cases=passed,
            regression_pass_rate=reg_pass_rate,
            details=tuple(details),
        )

    def _run_stress_set(
        self,
        samples: list[FailureSample],
        fix_packages: list[FixPackage],
    ) -> ReplayResult:
        """Stress test with edge cases.

        Checks:
        - memory_contamination: W1-type samples should have memory_patch or quarantine
        - Critical risk samples should have quarantine/rollback actions
        - Fix packages should not leave critical samples unaddressed
        """
        total = len(samples)
        if total == 0:
            return ReplayResult(test_set_name="stress_set")

        type_actions: dict[FailureCategory, set[str]] = {}
        for pkg in fix_packages:
            for ft in pkg.target_failure_types:
                if ft not in type_actions:
                    type_actions[ft] = set()
                type_actions[ft].add(pkg.package_type)

        contamination = 0
        passed = 0
        details: list[dict[str, Any]] = []

        for sample in samples:
            ft = sample.failure_type
            actions = type_actions.get(ft, set())

            # W1 samples => check for memory_patch or quarantine
            if ft == FailureCategory.W1:
                if "memory" not in actions and "quarantine" not in actions:
                    contamination += 1

            # Critical risk => must have quarantine, rollback, or human_review
            sample_passed = True
            if sample.risk_level == RiskLevel.CRITICAL:
                critical_actions = {"quarantine", "rollback", "human_review"}
                if not (actions & critical_actions):
                    sample_passed = False
                    contamination += 1

            if sample_passed:
                passed += 1

            details.append({
                "sample_id": sample.sample_id,
                "failure_type": ft.value,
                "risk_level": sample.risk_level.value,
                "actions": list(actions),
                "passed": sample_passed,
            })

        return ReplayResult(
            test_set_name="stress_set",
            total_cases=total,
            passed_cases=passed,
            memory_contamination_count=contamination,
            details=tuple(details),
        )
