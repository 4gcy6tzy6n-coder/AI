"""B-TSLA Orchestrator — internal diagnosis pipeline (Stage 20-2 through 20-5).

Strings together:
    EvidenceCollector → BayesianInferencer → RiskPolicyDecider → CorrectionExecutor

This is the engine called by the Stage 20 pipeline for each failure
sample. It does NOT handle trigger conditions (20-0/20-1), offline
replay (20-6), shadow running (20-7), rollout (20-8), or version
freeze (20-9).
"""

from typing import Any
from uuid import UUID

from ..memory.memory_events import MemoryEventLog
from ..tsla.action_router import ActionDecision
from ..tsla.audit_trace import AuditTrace
from ..tsla.scorer import ScoringResult
from ..unit.models import Unit
from .bayesian_inferencer import BayesianInferencer
from .correction_executor import CorrectionExecutor
from .evidence_collector import EvidenceCollector
from .risk_policy import RiskPolicyDecider
from .schema import FailureSample


class BTSLAOrchestrator:
    """Internal orchestrator for the B-TSLA diagnosis pipeline.

    Coordinates the B-TSLA loop for each failure sample:
        1. Extract evidence (EvidenceCollector)         — 20-2
        2. Compute posteriors (BayesianInferencer)       — 20-3
        3. Make risk decision (RiskPolicyDecider)        — 20-4
        4. Generate fix package (CorrectionExecutor)     — 20-5
        5. Record audit trail (AuditTrace)
    """

    def __init__(
        self,
        evidence_collector: EvidenceCollector | None = None,
        bayesian_inferencer: BayesianInferencer | None = None,
        risk_policy: RiskPolicyDecider | None = None,
        correction_executor: CorrectionExecutor | None = None,
        audit_trace: AuditTrace | None = None,
    ):
        self.evidence_collector = evidence_collector or EvidenceCollector()
        self.bayesian_inferencer = bayesian_inferencer or BayesianInferencer()
        self.risk_policy = risk_policy or RiskPolicyDecider()
        self.correction_executor = correction_executor or CorrectionExecutor()
        self.audit_trace = audit_trace

    def diagnose(
        self,
        failure_sample: FailureSample,
        unit: Unit,
        scoring_result: ScoringResult,
        action_decision: ActionDecision,
        query: str = "",
        response: str = "",
        retrieved_context: list[str] | None = None,
        memory_events: MemoryEventLog | None = None,
        existing_memory: list[Unit] | None = None,
    ) -> dict[str, Any]:
        """Run the full B-TSLA diagnosis for a single failure.

        Returns a dict with evidence, posterior, risk_decision,
        fix_package, and optional audit_record.
        """
        query = query or failure_sample.user_query
        response = response or failure_sample.system_response
        context = retrieved_context or list(failure_sample.retrieved_context)

        # 20-2: Extract 8-dim evidence
        evidence = self.evidence_collector.extract(
            unit=unit,
            scoring_result=scoring_result,
            query=query,
            response=response,
            retrieved_context=context,
            action_decision=action_decision,
            memory_events=memory_events,
        )

        # 20-3: Bayesian posterior inference
        posterior = self.bayesian_inferencer.infer(evidence)

        # 20-4: Risk policy decision
        risk_decision = self.risk_policy.decide(
            posterior=posterior,
            evidence=evidence,
            failure_sample=failure_sample,
        )

        # 20-5: Generate fix package
        fix_package = self.correction_executor.generate(
            decision=risk_decision,
            evidence=evidence,
            failure_sample=failure_sample,
            existing_memory=existing_memory,
        )

        # Audit trail
        audit_record = None
        if self.audit_trace:
            audit_record = self.audit_trace.record(
                unit_id=unit.unit_id,
                scores=scoring_result.current_scores.to_dict(),
                action={
                    "action_type": risk_decision.tsla_action_map,
                    "governance_action": risk_decision.recommended_action.value,
                    "posterior": posterior.to_dict(),
                    "evidence": evidence.to_dict(),
                },
                context={"diagnosis": "B-TSLA v0.1"},
            )

        return {
            "sample_id": failure_sample.sample_id,
            "evidence": evidence,
            "posterior": posterior,
            "risk_decision": risk_decision,
            "fix_package": fix_package,
            "audit_record": audit_record,
        }

    def diagnose_batch(
        self,
        failure_samples: list[FailureSample],
        units: list[Unit],
        scoring_results: list[ScoringResult],
        action_decisions: list[ActionDecision],
        queries: list[str] | None = None,
        responses: list[str] | None = None,
        retrieved_contexts: list[list[str]] | None = None,
        memory_events: MemoryEventLog | None = None,
        existing_memory: list[Unit] | None = None,
    ) -> list[dict[str, Any]]:
        """Run B-TSLA diagnosis for a batch of failure samples."""
        n = len(failure_samples)
        queries = queries or [fs.user_query for fs in failure_samples]
        responses = responses or [fs.system_response for fs in failure_samples]
        contexts = retrieved_contexts or [
            list(fs.retrieved_context) for fs in failure_samples
        ]

        results: list[dict[str, Any]] = []
        for i in range(n):
            result = self.diagnose(
                failure_sample=failure_samples[i],
                unit=units[i],
                scoring_result=scoring_results[i],
                action_decision=action_decisions[i],
                query=queries[i],
                response=responses[i],
                retrieved_context=contexts[i],
                memory_events=memory_events,
                existing_memory=existing_memory,
            )
            results.append(result)

        return results
