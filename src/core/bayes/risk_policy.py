"""Risk Policy Decider — governance action decisions (Stage 20-4).

Maps Bayesian posterior distributions to governance actions,
aligned with the existing TSLAActionType 8-action framework.
"""

from typing import Any

from ..tsla.hard_veto import HardVetoChecker
from .schema import (
    EvidenceVector,
    FailureCategory,
    FailureSample,
    GovernanceAction,
    PosteriorDistribution,
    PosteriorState,
    RiskDecision,
    RiskLevel,
)

# GovernanceAction → TSLAActionType value mapping
GOVERNANCE_TO_TSLA: dict[GovernanceAction, str] = {
    GovernanceAction.KEEP: "keep",
    GovernanceAction.RETRIEVAL_PATCH: "review_backflow",
    GovernanceAction.MEMORY_PATCH: "isolate",
    GovernanceAction.GENERATION_PATCH: "review_backflow",
    GovernanceAction.ROUTE_PATCH: "downgrade",
    GovernanceAction.TSLA_THRESHOLD_TUNE: "keep",
    GovernanceAction.QUARANTINE: "isolate",
    GovernanceAction.ROLLBACK: "review_backflow",
    GovernanceAction.HUMAN_REVIEW: "review_backflow",
}

# Dominant cause → default action
CAUSE_TO_ACTION: dict[PosteriorState, GovernanceAction] = {
    PosteriorState.KNOWLEDGE_GAP: GovernanceAction.KEEP,
    PosteriorState.RETRIEVAL_FAILURE: GovernanceAction.RETRIEVAL_PATCH,
    PosteriorState.GOVERNANCE_FAILURE: GovernanceAction.TSLA_THRESHOLD_TUNE,
    PosteriorState.GENERATION_FAILURE: GovernanceAction.GENERATION_PATCH,
    PosteriorState.MEMORY_FAILURE: GovernanceAction.MEMORY_PATCH,
    PosteriorState.ROUTE_FAILURE: GovernanceAction.ROUTE_PATCH,
}


class RiskPolicyDecider:
    """Decides governance actions from Bayesian posteriors.

    Aligns GovernanceAction values with the existing TSLAActionType
    from action_router.py via the GOVERNANCE_TO_TSLA mapping.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        hard_veto_checker: HardVetoChecker | None = None,
    ):
        self.thresholds = thresholds or {
            "min_confidence_for_action": 0.5,
            "min_confidence_for_auto_action": 0.6,
            "quarantine_risk_threshold": 0.8,
            "rollback_risk_threshold": 0.9,
            "escalation_entropy_threshold": 2.0,
            "contamination_escalation_threshold": 0.7,
        }
        self.hard_veto_checker = hard_veto_checker or HardVetoChecker()

    def decide(
        self,
        posterior: PosteriorDistribution,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
    ) -> RiskDecision:
        """Determine governance action from posterior distribution.

        Decision logic (v0.1):
        1. Check for escalation triggers (safety, contamination, entropy)
        2. If confidence >= min_for_auto_action: auto-route by dominant cause
        3. If confidence >= min_for_action but < auto: route with review flag
        4. If confidence < min_for_action: escalate to HUMAN_REVIEW
        """
        # Step 1: Escalation check
        escalation = self._escalation_check(posterior, evidence)
        if escalation is not None:
            tsla_action = GOVERNANCE_TO_TSLA.get(escalation, "review_backflow")
            return RiskDecision(
                recommended_action=escalation,
                reason=f"Escalation triggered: contamination={evidence.memory_contamination_score:.2f}, "
                       f"entropy={posterior.entropy:.2f}",
                severity=RiskLevel.HIGH,
                confidence=posterior.confidence,
                posterior=posterior,
                tsla_action_map=tsla_action,
            )

        # Step 2: Map dominant cause to action
        base_action = CAUSE_TO_ACTION.get(
            posterior.dominant_cause, GovernanceAction.HUMAN_REVIEW
        )

        # Step 3: Confidence-based decision
        if posterior.confidence >= self.thresholds["min_confidence_for_auto_action"]:
            action = base_action
            reason = (
                f"Auto-decision: dominant cause={posterior.dominant_cause.value} "
                f"with confidence={posterior.confidence:.2f}"
            )
            severity = self._severity_from_evidence(evidence)
        elif posterior.confidence >= self.thresholds["min_confidence_for_action"]:
            action = base_action
            reason = (
                f"Action with review flag: dominant cause={posterior.dominant_cause.value} "
                f"confidence={posterior.confidence:.2f}"
            )
            severity = RiskLevel.MEDIUM
        else:
            action = GovernanceAction.HUMAN_REVIEW
            reason = (
                f"Low confidence ({posterior.confidence:.2f}) — escalating to human review. "
                f"Dominant cause was {posterior.dominant_cause.value}"
            )
            severity = RiskLevel.LOW

        tsla_action = GOVERNANCE_TO_TSLA.get(action, "review_backflow")
        return RiskDecision(
            recommended_action=action,
            reason=reason,
            severity=severity,
            confidence=posterior.confidence,
            posterior=posterior,
            tsla_action_map=tsla_action,
        )

    def decide_batch(
        self,
        posteriors: list[PosteriorDistribution],
        evidence_batch: list[EvidenceVector],
        failure_samples: list[FailureSample],
    ) -> list[RiskDecision]:
        return [
            self.decide(p, e, f)
            for p, e, f in zip(posteriors, evidence_batch, failure_samples)
        ]

    def _escalation_check(
        self,
        posterior: PosteriorDistribution,
        evidence: EvidenceVector,
    ) -> GovernanceAction | None:
        """Check if action needs safety/risk escalation.

        Returns escalated action or None if no escalation needed.
        """
        thresholds = self.thresholds

        # Memory contamination escalates to QUARANTINE
        if (
            evidence.memory_contamination_score
            >= thresholds["contamination_escalation_threshold"]
        ):
            return GovernanceAction.QUARANTINE

        # High entropy (= high uncertainty across causes) → HUMAN_REVIEW
        if posterior.entropy >= thresholds["escalation_entropy_threshold"]:
            return GovernanceAction.HUMAN_REVIEW

        return None

    def _severity_from_evidence(self, evidence: EvidenceVector) -> RiskLevel:
        """Determine risk severity from evidence scores."""
        high_scores = sum(
            1 for s in evidence.to_list() if s >= self.thresholds["quarantine_risk_threshold"]
        )
        if high_scores >= 3:
            return RiskLevel.CRITICAL
        if high_scores >= 2:
            return RiskLevel.HIGH
        if high_scores >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
