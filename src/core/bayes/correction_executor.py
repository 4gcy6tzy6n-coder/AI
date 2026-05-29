"""Correction Executor — fix package generation (Stage 20-5).

Generates 5 types of patch packages as JSON/JSONL files.
Does NOT apply them automatically — that is controlled by
the Orchestrator after offline replay verification.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..unit.models import Unit
from .schema import (
    EvidenceVector,
    FailureCategory,
    FailureSample,
    FixPackage,
    PosteriorDistribution,
    RiskDecision,
)

PACKAGE_TYPES = [
    "knowledge",
    "retrieval",
    "generation",
    "context",
    "tsla_threshold",
]


class CorrectionExecutor:
    """Generates fix packages from risk decisions.

    Writes JSON/JSONL files to output_dir. Packages are idempotent:
    re-running with the same decision will skip existing packages.
    """

    def __init__(
        self,
        output_dir: str = "data/stage20/stage20_patches",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None = None,
    ) -> FixPackage:
        """Route to the correct patch generator based on decision action."""
        action = decision.recommended_action

        generators = {
            "keep": self._generate_keep_package,
            "retrieval_patch": self._generate_retrieval_patch,
            "memory_patch": self._generate_memory_patch,
            "generation_patch": self._generate_generation_patch,
            "route_patch": self._generate_route_patch,
            "tsla_threshold_tune": self._generate_tsla_threshold_patch,
            "quarantine": self._generate_quarantine_package,
            "rollback": self._generate_rollback_package,
            "human_review": self._generate_human_review_package,
        }

        gen = generators.get(action.value, self._generate_keep_package)
        package = gen(decision, evidence, failure_sample, existing_memory)

        self.save_package(package)
        return package

    # ── Patch generators ───────────────────────────────────────────

    def _generate_keep_package(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """No fix needed — record observation for future analysis."""
        return FixPackage(
            package_type="keep",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "keep",
                "note": "No fix generated. Failure recorded for trend analysis.",
                "evidence_summary": evidence.to_dict(),
                "sample_id": failure_sample.sample_id,
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_knowledge_patch(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate knowledge_patch.jsonl entry.

        New knowledge entries default to long_term_review zone,
        never directly to permanent layers.
        """
        entries: list[dict[str, Any]] = []
        if failure_sample.expected_behavior:
            entries.append({
                "entry_id": f"kp_{failure_sample.sample_id}",
                "content": failure_sample.expected_behavior,
                "source_refs": list(failure_sample.memory_used),
                "memory_layer_target": "long_term_review",
                "original_query": failure_sample.user_query,
                "failure_type": failure_sample.failure_type.value,
                "risk_level": failure_sample.risk_level.value,
            })

        return FixPackage(
            package_type="knowledge",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "knowledge_patch",
                "new_entries": entries,
                "total_entries": len(entries),
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_retrieval_patch(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate retrieval_patch.json."""
        query = failure_sample.user_query
        keywords = query.lower().split() if query else []

        return FixPackage(
            package_type="retrieval",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "retrieval_patch",
                "query_rewrite_rules": [
                    {
                        "pattern": query,
                        "rewrite": failure_sample.expected_behavior or query,
                        "priority": 1,
                    }
                ],
                "keyword_alias_map": {kw: [] for kw in keywords[:10]},
                "retrieval_threshold_tuning": {
                    "min_similarity": 0.65,
                    "rerank_top_k": 5,
                    "hybrid_weight_dense": 0.6,
                    "hybrid_weight_sparse": 0.4,
                },
                "sample_id": failure_sample.sample_id,
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_memory_patch(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate memory patch for contaminated or stale memory entries."""
        contaminated_ids: list[str] = []
        if existing_memory:
            for unit in existing_memory:
                if unit.conflict_cleanliness < 0.5:
                    contaminated_ids.append(str(unit.unit_id))

        return FixPackage(
            package_type="memory",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "memory_patch",
                "contaminated_unit_ids": contaminated_ids,
                "recommended_action": "isolate_and_review",
                "target_zone": "long_term_isolation",
                "memory_contamination_score": evidence.memory_contamination_score,
                "sample_id": failure_sample.sample_id,
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_generation_patch(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate generation_patch.json."""
        avoid_phrases: list[str] = []
        if evidence.generation_fluency_score < 0.5:
            # Extract potential problematic phrases from response
            resp = failure_sample.system_response
            if resp:
                avoid_phrases = [resp[:100]]  # Truncated for analysis

        return FixPackage(
            package_type="generation",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "generation_patch",
                "style_constraints": {
                    "tone": "neutral",
                    "max_sentence_length": 50,
                },
                "avoid_phrases": avoid_phrases,
                "clarity_rules": [
                    {
                        "rule_id": 1,
                        "description": "Ensure responses are grounded in retrieved evidence",
                        "check": "response_claims in retrieved_context",
                    }
                ],
                "sample_id": failure_sample.sample_id,
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_route_patch(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate context/route patch."""
        return FixPackage(
            package_type="context",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "route_patch",
                "history_compression_rules": [
                    {
                        "max_turns": 5,
                        "strategy": "summarize",
                        "target_tokens": 200,
                    }
                ],
                "turn_state_tracking_rules": [
                    {
                        "state_key": "user_topic",
                        "persist_across_turns": True,
                    }
                ],
                "strategy_route_correction": {
                    "original_route": failure_sample.strategy_route,
                    "suggested_route": "RETRIEVAL_FIRST",
                },
                "sample_id": failure_sample.sample_id,
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_tsla_threshold_patch(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate tsla_threshold_patch.json.

        Threshold values are v0.1 examples — must be validated
        through prototype experiments, ablation tests, and
        long-term replay before use as final values.
        """
        return FixPackage(
            package_type="tsla_threshold",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "tsla_threshold_tune",
                "H_signal_adjustments": {
                    "safety_threshold": {"from": 20.0, "to": 25.0},
                    "liability_threshold": {"from": 25.0, "to": 30.0},
                    "accountability_threshold": {"from": 30.0, "to": 35.0},
                },
                "Q_threshold_changes": {
                    "min_promote_score": {"from": 65.0, "to": 70.0},
                    "min_keep_score": {"from": 50.0, "to": 55.0},
                },
                "action_priority_changes": [
                    {"action": "isolate", "priority": {"from": 5, "to": 4}},
                ],
                "reason": decision.reason,
                "evidence": evidence.to_dict(),
                "posterior": decision.posterior.to_dict(),
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_quarantine_package(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate quarantine package — isolate affected memory/units."""
        return FixPackage(
            package_type="quarantine",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "quarantine",
                "target_zone": "long_term_isolation",
                "sample_id": failure_sample.sample_id,
                "quarantine_reason": (
                    f"Memory contamination score: "
                    f"{evidence.memory_contamination_score:.2f}"
                ),
                "review_deadline_days": 90,
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_rollback_package(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate rollback package."""
        return FixPackage(
            package_type="rollback",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "rollback",
                "sample_id": failure_sample.sample_id,
                "rollback_scope": "unit",
                "rollback_reason": decision.reason,
                "posterior": decision.posterior.to_dict(),
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _generate_human_review_package(
        self,
        decision: RiskDecision,
        evidence: EvidenceVector,
        failure_sample: FailureSample,
        existing_memory: list[Unit] | None,
    ) -> FixPackage:
        """Generate human review package — flag for manual review."""
        return FixPackage(
            package_type="human_review",
            target_failure_types=(failure_sample.failure_type,),
            risk_decision=decision,
            payload={
                "action": "human_review",
                "sample_id": failure_sample.sample_id,
                "reason": decision.reason,
                "evidence_summary": evidence.to_dict(),
                "posterior": decision.posterior.to_dict(),
            },
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Persistence ─────────────────────────────────────────────────

    def save_package(self, package: FixPackage) -> str:
        """Persist fix package to disk. Returns file path."""
        suffix = "jsonl" if package.package_type == "knowledge" else "json"
        filename = f"{package.package_type}_{package.package_id[:8]}.{suffix}"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(package.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        return str(filepath)

    def load_package(self, package_id: str) -> FixPackage | None:
        """Load a fix package from disk by package_id prefix."""
        for fpath in self.output_dir.iterdir():
            if package_id[:8] in fpath.name:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return FixPackage(
                    package_id=data.get("package_id", ""),
                    package_type=data.get("package_type", ""),
                    payload=data.get("payload", {}),
                    generated_at=data.get("generated_at", ""),
                )
        return None

    def list_packages(self) -> list[str]:
        """List all generated package file paths."""
        return sorted(str(p) for p in self.output_dir.iterdir() if p.suffix in (".json", ".jsonl"))
