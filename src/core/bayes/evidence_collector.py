"""Evidence Collector — 8-dimension evidence vector extraction (Stage 20-2).

Translates TSLA scoring output, action decisions, and memory events
into the 8-dimension EvidenceVector. Does NOT compute TSLA scores
itself — that is the scorer's responsibility.
"""

from typing import Any

from ..memory.memory_events import MemoryEventLog, MemoryEventType
from ..tsla.action_router import ActionDecision, TSLAActionType
from ..tsla.scorer import ScoringResult
from ..unit.models import Unit
from .schema import EvidenceVector


class EvidenceCollector:
    """Extracts 8-dimension evidence vectors from existing TSLA scoring data.

    Input signals:
    - ScoringResult from TSLAScorerWithHistory.score_with_history()
    - ActionDecision from ActionRouter.route()
    - MemoryEventLog for contamination signals
    - Unit for TSLA field values
    - Raw query/response strings for surface analysis
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        # v0.1 example thresholds
        self._low_evidence_threshold: float = cfg.get("low_evidence_threshold", 0.4)
        self._high_evidence_threshold: float = cfg.get("high_evidence_threshold", 0.8)
        self._low_fluency_threshold: float = cfg.get("low_fluency_threshold", 0.3)
        self._tsla_volatility_threshold: float = cfg.get("tsla_volatility_threshold", 15.0)

    def extract(
        self,
        unit: Unit,
        scoring_result: ScoringResult,
        query: str,
        response: str,
        retrieved_context: list[str],
        action_decision: ActionDecision,
        memory_events: MemoryEventLog | None = None,
    ) -> EvidenceVector:
        """Compute the full 8-dimension evidence vector.

        Returns an EvidenceVector with all values clamped to [0.0, 1.0].
        """
        return EvidenceVector(
            retrieval_gap_score=self._extract_retrieval_gap(
                unit, retrieved_context
            ),
            context_conflict_score=self._extract_context_conflict(unit),
            generation_fluency_score=self._extract_generation_fluency(
                response, scoring_result
            ),
            answer_faithfulness_score=self._extract_answer_faithfulness(
                scoring_result
            ),
            memory_contamination_score=self._extract_memory_contamination(
                unit.unit_id, memory_events
            ),
            strategy_mismatch_score=self._extract_strategy_mismatch(
                action_decision, scoring_result
            ),
            tsla_confidence_score=self._extract_tsla_confidence(
                scoring_result
            ),
            user_intent_clarity_score=self._extract_user_intent_clarity(
                query, unit
            ),
        )

    # ── Dimension extractors ──────────────────────────────────────

    def _extract_retrieval_gap(
        self,
        unit: Unit,
        retrieved_context: list[str],
    ) -> float:
        """Evidence dimension 1: how much the retrieval missed needed info.

        0.0 = no gap (retrieval was adequate)
        1.0 = complete retrieval failure

        Uses unit.evidence_score and context emptiness.
        """
        if not retrieved_context:
            return 0.9
        evidence = unit.evidence_score
        gap = 1.0 - evidence
        return self._clamp(gap)

    def _extract_context_conflict(self, unit: Unit) -> float:
        """Evidence dimension 2: how much conflict exists in context.

        0.0 = perfectly clean
        1.0 = severe unresolvable conflict

        Uses unit.conflict_cleanliness (C in TSLA).
        """
        return self._clamp(1.0 - unit.conflict_cleanliness)

    def _extract_generation_fluency(
        self,
        response: str,
        scoring_result: ScoringResult,
    ) -> float:
        """Evidence dimension 3: generation fluency / naturalness.

        0.0 = broken/unreadable
        1.0 = perfectly fluent

        Heuristic: checks for common broken-pattern markers and
        uses the response text length as a coarse signal.
        """
        if not response:
            return 0.0

        broken_markers = ["[", "]", "{", "}", "<UNK>", "���", "�"]
        marker_count = sum(1 for m in broken_markers if m in response)
        tokens = response.split()
        token_count = max(len(tokens), 1)

        # Penalty for broken markers per token
        marker_penalty = min(marker_count / token_count, 1.0)

        # Short responses may be fine; extremely short ones get a slight penalty
        length_penalty = 0.0
        if token_count < 3:
            length_penalty = 0.3
        elif token_count < 5:
            length_penalty = 0.1

        fluency = 1.0 - marker_penalty - length_penalty
        return self._clamp(fluency)

    def _extract_answer_faithfulness(
        self,
        scoring_result: ScoringResult,
    ) -> float:
        """Evidence dimension 4: how faithful the answer is to evidence.

        0.0 = hallucinated
        1.0 = fully grounded in evidence

        Uses the TSLA T (truth) score as primary signal.
        """
        t_score = scoring_result.current_scores.T / 100.0
        return self._clamp(t_score)

    def _extract_memory_contamination(
        self,
        unit_id: Any,
        memory_events: MemoryEventLog | None,
    ) -> float:
        """Evidence dimension 5: how contaminated the memory is.

        0.0 = clean
        1.0 = heavily contaminated

        Uses MemoryEventLog: counts ISOLATION and ERROR_ARCHIVAL
        events relative to total events for this unit.
        """
        if memory_events is None:
            return 0.0

        history = memory_events.get_unit_history(unit_id)
        total = len(history)
        if total == 0:
            return 0.0

        contamination_events = sum(
            1 for e in history
            if e.event_type in (MemoryEventType.ISOLATION, MemoryEventType.ERROR_ARCHIVAL)
        )
        return self._clamp(contamination_events / total)

    def _extract_strategy_mismatch(
        self,
        action_decision: ActionDecision,
        scoring_result: ScoringResult,
    ) -> float:
        """Evidence dimension 6: how mismatched the routing strategy is.

        0.0 = correct route
        1.0 = completely wrong route

        Heuristic: if hard_veto was triggered and the composite score
        was high, that suggests the route was wrong. If composite is low
        and no veto, the route was likely correct.
        """
        composite = scoring_result.current_scores.Q / 100.0

        if action_decision.hard_veto_triggered:
            # Hard veto when composite was high = likely route mismatch
            mismatch = composite * 0.8
        elif action_decision.action in (
            TSLAActionType.ISOLATE,
            TSLAActionType.EXCLUDE,
        ):
            # Negative action suggests some mismatch
            mismatch = self._clamp(1.0 - composite)
        else:
            # KEEP or PROMOTE with low composite = possible false positive
            if composite < self._low_evidence_threshold:
                mismatch = 0.5
            else:
                mismatch = 0.0

        return self._clamp(mismatch)

    def _extract_tsla_confidence(
        self,
        scoring_result: ScoringResult,
    ) -> float:
        """Evidence dimension 7: TSLA scoring confidence/stability.

        0.0 = low confidence (high variance in scores)
        1.0 = high confidence (stable scores)

        Uses the composite Q score and volatility of rolling stats.
        """
        q_score = scoring_result.current_scores.Q / 100.0
        stats = scoring_result.rolling_stats

        # Penalize if Q_std is high (volatile scoring = low confidence)
        volatility_penalty = min(stats.q_std / self._tsla_volatility_threshold, 1.0)
        confidence = q_score * (1.0 - volatility_penalty * 0.5)
        return self._clamp(confidence)

    def _extract_user_intent_clarity(
        self,
        query: str,
        unit: Unit,
    ) -> float:
        """Evidence dimension 8: how clear the user's intent is.

        0.0 = completely ambiguous
        1.0 = crystal clear

        Heuristic: checks query length, ambiguity markers, and
        whether the Unit has a defined core_meaning.
        """
        if not query:
            return 0.0

        tokens = query.split()
        token_count = len(tokens)

        # Very short queries are inherently ambiguous
        if token_count <= 2:
            base_clarity = 0.3
        elif token_count <= 5:
            base_clarity = 0.5
        elif token_count <= 10:
            base_clarity = 0.7
        else:
            base_clarity = 0.85

        # Ambiguity markers reduce clarity
        ambiguity_markers = [
            "?", "maybe", "perhaps", "大概", "可能", "也许",
            "anything", "whatever", "随便", "无所谓",
        ]
        marker_hits = sum(1 for m in ambiguity_markers if m.lower() in query.lower())
        marker_penalty = min(marker_hits * 0.1, 0.5)

        # If Unit has a defined core_meaning, intent was likely clear
        meaning_bonus = 0.15 if unit.has_meaning() else 0.0

        clarity = base_clarity - marker_penalty + meaning_bonus
        return self._clamp(clarity)

    # ── Utility ────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))
