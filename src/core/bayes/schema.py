"""B-TSLA core data models — Stage 20.

All dataclasses are immutable. No logic, only data containers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class FailureCategory(str, Enum):
    """8 failure types extending existing H1/H2/H4/H5 taxonomy."""
    K1 = "knowledge_miss"
    G1 = "unnatural_generation"
    M1 = "multiturn_anomaly"
    R1 = "retrieval_mismatch"
    T1 = "tsla_false_pass"
    T2 = "tsla_over_block"
    S1 = "safety_boundary_error"
    W1 = "memory_write_error"


class GovernanceAction(str, Enum):
    """8 governance actions for risk policy decisions."""
    KEEP = "keep"
    RETRIEVAL_PATCH = "retrieval_patch"
    MEMORY_PATCH = "memory_patch"
    GENERATION_PATCH = "generation_patch"
    ROUTE_PATCH = "route_patch"
    TSLA_THRESHOLD_TUNE = "tsla_threshold_tune"
    QUARANTINE = "quarantine"
    ROLLBACK = "rollback"
    HUMAN_REVIEW = "human_review"


class PosteriorState(str, Enum):
    """6 posterior states for Bayesian root-cause attribution."""
    KNOWLEDGE_GAP = "knowledge_gap"
    RETRIEVAL_FAILURE = "retrieval_failure"
    GOVERNANCE_FAILURE = "governance_failure"
    GENERATION_FAILURE = "generation_failure"
    MEMORY_FAILURE = "memory_failure"
    ROUTE_FAILURE = "route_failure"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─── 20-1: Failure Pool ───────────────────────────────────────────

@dataclass(frozen=True)
class FailureSample:
    """Single failure case frozen in the stage20 failure pool."""
    sample_id: str
    user_query: str
    system_response: str = ""
    retrieved_context: tuple[str, ...] = ()
    memory_used: tuple[str, ...] = ()
    strategy_route: str = ""
    failure_type: FailureCategory = FailureCategory.W1
    human_or_teacher_label: str = ""
    expected_behavior: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "user_query": self.user_query,
            "system_response": self.system_response,
            "retrieved_context": list(self.retrieved_context),
            "memory_used": list(self.memory_used),
            "strategy_route": self.strategy_route,
            "failure_type": self.failure_type.value,
            "human_or_teacher_label": self.human_or_teacher_label,
            "expected_behavior": self.expected_behavior,
            "risk_level": self.risk_level.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureSample":
        return cls(
            sample_id=data.get("sample_id", ""),
            user_query=data.get("user_query", ""),
            system_response=data.get("system_response", ""),
            retrieved_context=tuple(data.get("retrieved_context", [])),
            memory_used=tuple(data.get("memory_used", [])),
            strategy_route=data.get("strategy_route", ""),
            failure_type=FailureCategory(data.get("failure_type", "W1")),
            human_or_teacher_label=data.get("human_or_teacher_label", ""),
            expected_behavior=data.get("expected_behavior", ""),
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
        )


# ─── 20-2: Evidence Vector ────────────────────────────────────────

@dataclass(frozen=True)
class EvidenceVector:
    """8-dimension evidence vector extracted from TSLA scoring data.

    All values clamped to [0.0, 1.0].
    """
    retrieval_gap_score: float = 0.0
    context_conflict_score: float = 0.0
    generation_fluency_score: float = 0.5
    answer_faithfulness_score: float = 0.5
    memory_contamination_score: float = 0.0
    strategy_mismatch_score: float = 0.0
    tsla_confidence_score: float = 0.5
    user_intent_clarity_score: float = 0.5

    def to_list(self) -> list[float]:
        return [
            self.retrieval_gap_score,
            self.context_conflict_score,
            self.generation_fluency_score,
            self.answer_faithfulness_score,
            self.memory_contamination_score,
            self.strategy_mismatch_score,
            self.tsla_confidence_score,
            self.user_intent_clarity_score,
        ]

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_gap": self.retrieval_gap_score,
            "context_conflict": self.context_conflict_score,
            "generation_fluency": self.generation_fluency_score,
            "answer_faithfulness": self.answer_faithfulness_score,
            "memory_contamination": self.memory_contamination_score,
            "strategy_mismatch": self.strategy_mismatch_score,
            "tsla_confidence": self.tsla_confidence_score,
            "user_intent_clarity": self.user_intent_clarity_score,
        }


# ─── 20-3: Bayesian Posterior ─────────────────────────────────────

@dataclass(frozen=True)
class PosteriorDistribution:
    """6 posterior probabilities from Bayesian inference."""
    p_knowledge_gap: float = 0.0
    p_retrieval_failure: float = 0.0
    p_governance_failure: float = 0.0
    p_generation_failure: float = 0.0
    p_memory_failure: float = 0.0
    p_route_failure: float = 0.0
    dominant_cause: PosteriorState = PosteriorState.KNOWLEDGE_GAP
    confidence: float = 0.0
    entropy: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "knowledge_gap": self.p_knowledge_gap,
            "retrieval_failure": self.p_retrieval_failure,
            "governance_failure": self.p_governance_failure,
            "generation_failure": self.p_generation_failure,
            "memory_failure": self.p_memory_failure,
            "route_failure": self.p_route_failure,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "posteriors": self.as_dict(),
            "dominant_cause": self.dominant_cause.value,
            "confidence": self.confidence,
            "entropy": self.entropy,
        }


# ─── 20-4: Risk Decision ──────────────────────────────────────────

@dataclass(frozen=True)
class RiskDecision:
    """Governance action output from risk policy decider."""
    recommended_action: GovernanceAction = GovernanceAction.KEEP
    reason: str = ""
    severity: RiskLevel = RiskLevel.MEDIUM
    confidence: float = 0.5
    posterior: PosteriorDistribution = field(
        default_factory=lambda: PosteriorDistribution()
    )
    tsla_action_map: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_action": self.recommended_action.value,
            "reason": self.reason,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "posterior": self.posterior.to_dict(),
            "tsla_action_map": self.tsla_action_map,
        }


# ─── 20-5: Fix Package ────────────────────────────────────────────

@dataclass(frozen=True)
class FixPackage:
    """Generated fix package — saved as file, not applied automatically."""
    package_id: str = field(default_factory=lambda: str(uuid4()))
    package_type: str = ""
    target_failure_types: tuple[FailureCategory, ...] = ()
    risk_decision: RiskDecision | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    applied: bool = False
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "package_type": self.package_type,
            "target_failure_types": [f.value for f in self.target_failure_types],
            "risk_decision": self.risk_decision.to_dict() if self.risk_decision else None,
            "payload": self.payload,
            "generated_at": self.generated_at,
            "applied": self.applied,
            "verified": self.verified,
        }


# ─── 20-6: Replay Result ──────────────────────────────────────────

@dataclass(frozen=True)
class ReplayResult:
    """Offline replay verification result for one test set."""
    test_set_name: str
    total_cases: int = 0
    passed_cases: int = 0
    failure_fix_rate: float = 0.0
    regression_pass_rate: float = 0.0
    tsla_safety_intercept: float = 0.0
    false_kill_rate: float = 0.0
    memory_contamination_count: int = 0
    details: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_set_name": self.test_set_name,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failure_fix_rate": self.failure_fix_rate,
            "regression_pass_rate": self.regression_pass_rate,
            "tsla_safety_intercept": self.tsla_safety_intercept,
            "false_kill_rate": self.false_kill_rate,
            "memory_contamination_count": self.memory_contamination_count,
            "details": list(self.details),
        }


# ─── 20-7: Shadow Result ──────────────────────────────────────────

@dataclass(frozen=True)
class ShadowResult:
    """Shadow running comparison: Stage18_frozen vs Stage20_candidate."""
    total_requests: int = 0
    response_quality_delta: float = 0.0
    retrieval_hit_delta: float = 0.0
    tsla_action_delta: dict[str, int] = field(default_factory=dict)
    latency_delta_pct: float = 0.0
    memory_write_delta: int = 0
    user_visible_risk_delta: float = 0.0
    new_high_risk_failures: int = 0
    memory_write_errors: int = 0
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "response_quality_delta": self.response_quality_delta,
            "retrieval_hit_delta": self.retrieval_hit_delta,
            "tsla_action_delta": self.tsla_action_delta,
            "latency_delta_pct": self.latency_delta_pct,
            "memory_write_delta": self.memory_write_delta,
            "user_visible_risk_delta": self.user_visible_risk_delta,
            "new_high_risk_failures": self.new_high_risk_failures,
            "memory_write_errors": self.memory_write_errors,
            "passed": self.passed,
        }


# ─── 20-8: Rollout Result ─────────────────────────────────────────

@dataclass(frozen=True)
class RolloutResult:
    """Simulated rollout acceptance result for one round."""
    round_name: str
    overall_pass_rate: float = 0.0
    known_failure_reduction_pct: float = 0.0
    new_failure_rate: float = 0.0
    critical_failures: int = 0
    memory_contamination: int = 0
    rollback_test_pass: bool = False
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_name": self.round_name,
            "overall_pass_rate": self.overall_pass_rate,
            "known_failure_reduction_pct": self.known_failure_reduction_pct,
            "new_failure_rate": self.new_failure_rate,
            "critical_failures": self.critical_failures,
            "memory_contamination": self.memory_contamination,
            "rollback_test_pass": self.rollback_test_pass,
            "passed": self.passed,
        }


# ─── 20-9: Version Freeze Point ───────────────────────────────────

@dataclass(frozen=True)
class VersionFreezePoint:
    """Version freeze + rollback snapshot."""
    version: str
    freeze_timestamp: str = ""
    component_versions: dict[str, str] = field(default_factory=dict)
    snapshot_ids: dict[str, str] = field(default_factory=dict)
    rollback_runbook_path: str = ""
    auto_rollback_conditions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "freeze_timestamp": self.freeze_timestamp,
            "component_versions": self.component_versions,
            "snapshot_ids": self.snapshot_ids,
            "rollback_runbook_path": self.rollback_runbook_path,
            "auto_rollback_conditions": self.auto_rollback_conditions,
        }
