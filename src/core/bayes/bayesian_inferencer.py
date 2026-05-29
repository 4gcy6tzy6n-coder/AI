"""Bayesian Inferencer — 6-posterior root-cause attribution (Stage 20-3).

Computes P(cause | evidence) for 6 failure causes using naive Bayes
with expert-crafted heuristic priors and likelihood tables.

v0.1 approach:
- Hand-crafted priors (not learned from data yet)
- Naive Bayes independence assumption
- Soft-likelihood table mapping evidence dimensions to causes
- Output normalized to sum to 1.0

This is explicitly a bootstrap; future stages can replace with a
learned conditional probability table.
"""

import math
from typing import Any

from .schema import EvidenceVector, PosteriorDistribution, PosteriorState


# ─── v0.1 Likelihood Table ─────────────────────────────────────────
# P(high_dim_score | cause) — how strongly each evidence dimension
# signals each failure cause when scored high (>= 0.5).
# Complement P(low | cause) = 1.0 - P(high | cause).

DEFAULT_LIKELIHOOD_HIGH: dict[str, dict[str, float]] = {
    "retrieval_gap": {
        "knowledge_gap": 0.40,
        "retrieval_failure": 0.80,
        "governance_failure": 0.10,
        "generation_failure": 0.05,
        "memory_failure": 0.15,
        "route_failure": 0.10,
    },
    "context_conflict": {
        "knowledge_gap": 0.15,
        "retrieval_failure": 0.25,
        "governance_failure": 0.70,
        "generation_failure": 0.20,
        "memory_failure": 0.50,
        "route_failure": 0.15,
    },
    "generation_fluency": {
        "knowledge_gap": 0.05,
        "retrieval_failure": 0.05,
        "governance_failure": 0.10,
        "generation_failure": 0.80,
        "memory_failure": 0.05,
        "route_failure": 0.10,
    },
    "answer_faithfulness": {
        "knowledge_gap": 0.40,
        "retrieval_failure": 0.20,
        "governance_failure": 0.25,
        "generation_failure": 0.50,
        "memory_failure": 0.10,
        "route_failure": 0.10,
    },
    "memory_contamination": {
        "knowledge_gap": 0.05,
        "retrieval_failure": 0.10,
        "governance_failure": 0.20,
        "generation_failure": 0.05,
        "memory_failure": 0.90,
        "route_failure": 0.05,
    },
    "strategy_mismatch": {
        "knowledge_gap": 0.10,
        "retrieval_failure": 0.15,
        "governance_failure": 0.30,
        "generation_failure": 0.10,
        "memory_failure": 0.10,
        "route_failure": 0.75,
    },
    "tsla_confidence": {
        "knowledge_gap": 0.10,
        "retrieval_failure": 0.10,
        "governance_failure": 0.40,
        "generation_failure": 0.10,
        "memory_failure": 0.20,
        "route_failure": 0.15,
    },
    "user_intent_clarity": {
        "knowledge_gap": 0.25,
        "retrieval_failure": 0.15,
        "governance_failure": 0.20,
        "generation_failure": 0.30,
        "memory_failure": 0.10,
        "route_failure": 0.25,
    },
}

DEFAULT_PRIORS: dict[str, float] = {
    "knowledge_gap": 0.25,
    "retrieval_failure": 0.15,
    "governance_failure": 0.10,
    "generation_failure": 0.20,
    "memory_failure": 0.15,
    "route_failure": 0.15,
}

# Map cause string → PosteriorState enum
CAUSE_TO_STATE: dict[str, PosteriorState] = {
    "knowledge_gap": PosteriorState.KNOWLEDGE_GAP,
    "retrieval_failure": PosteriorState.RETRIEVAL_FAILURE,
    "governance_failure": PosteriorState.GOVERNANCE_FAILURE,
    "generation_failure": PosteriorState.GENERATION_FAILURE,
    "memory_failure": PosteriorState.MEMORY_FAILURE,
    "route_failure": PosteriorState.ROUTE_FAILURE,
}

# Evidence dimension keys matching EvidenceVector fields
EVIDENCE_DIMS = [
    "retrieval_gap",
    "context_conflict",
    "generation_fluency",
    "answer_faithfulness",
    "memory_contamination",
    "strategy_mismatch",
    "tsla_confidence",
    "user_intent_clarity",
]

CAUSE_KEYS = [
    "knowledge_gap",
    "retrieval_failure",
    "governance_failure",
    "generation_failure",
    "memory_failure",
    "route_failure",
]


class BayesianInferencer:
    """Naive Bayes inference over 6 failure causes given 8-dim evidence.

    Uses the formula:
        P(cause | E) ∝ P(cause) * ∏_i P(E_i | cause)

    Where:
    - P(cause) is a heuristic prior
    - P(E_i | cause) is interpolated between P(high|cause) and P(low|cause)
      based on the observed evidence score E_i.
    """

    def __init__(
        self,
        priors: dict[str, float] | None = None,
        likelihood_high: dict[str, dict[str, float]] | None = None,
        config: dict[str, Any] | None = None,
    ):
        cfg = config or {}
        self.priors = priors or DEFAULT_PRIORS
        self._lh_high = likelihood_high or DEFAULT_LIKELIHOOD_HIGH
        self._min_prob: float = cfg.get("min_prob", 1e-10)

        self._validate_likelihood_map()

    def infer(self, evidence: EvidenceVector) -> PosteriorDistribution:
        """Compute posterior distribution from evidence vector.

        Returns PosteriorDistribution with:
        - 6 posterior probabilities (sum to 1.0)
        - dominant_cause
        - confidence (max posterior value)
        - entropy
        """
        ev_scores = dict(zip(EVIDENCE_DIMS, evidence.to_list()))

        unnormalized: dict[str, float] = {}
        for cause in CAUSE_KEYS:
            prior = self.priors[cause]
            likelihood = self._compute_likelihood(cause, ev_scores)
            unnormalized[cause] = prior * likelihood

        # Normalize
        total = sum(unnormalized.values())
        if total <= self._min_prob:
            # Degenerate case: return uniform
            uniform = 1.0 / len(CAUSE_KEYS)
            return PosteriorDistribution(
                p_knowledge_gap=uniform,
                p_retrieval_failure=uniform,
                p_governance_failure=uniform,
                p_generation_failure=uniform,
                p_memory_failure=uniform,
                p_route_failure=uniform,
                dominant_cause=PosteriorState.KNOWLEDGE_GAP,
                confidence=uniform,
                entropy=math.log(len(CAUSE_KEYS)),
            )

        normalized = {c: v / total for c, v in unnormalized.items()}

        # Dominant cause and confidence
        dominant = max(normalized, key=normalized.get)  # type: ignore[arg-type]
        confidence = normalized[dominant]

        # Entropy: -sum(p * log(p))
        entropy = -sum(p * math.log(max(p, self._min_prob)) for p in normalized.values())

        return PosteriorDistribution(
            p_knowledge_gap=normalized["knowledge_gap"],
            p_retrieval_failure=normalized["retrieval_failure"],
            p_governance_failure=normalized["governance_failure"],
            p_generation_failure=normalized["generation_failure"],
            p_memory_failure=normalized["memory_failure"],
            p_route_failure=normalized["route_failure"],
            dominant_cause=CAUSE_TO_STATE[dominant],
            confidence=confidence,
            entropy=entropy,
        )

    def infer_batch(
        self,
        evidence_batch: list[EvidenceVector],
    ) -> list[PosteriorDistribution]:
        return [self.infer(ev) for ev in evidence_batch]

    def _compute_likelihood(
        self,
        cause: str,
        ev_scores: dict[str, float],
    ) -> float:
        """Compute P(E | cause) = ∏_i P(E_i | cause).

        For each evidence dimension, interpolate between the high and
        low likelihood values based on the observed score.
        - If score >= 0.5: interpolate toward P(high|cause)
        - If score < 0.5: interpolate toward P(low|cause)
        """
        product = 1.0
        for dim in EVIDENCE_DIMS:
            score = ev_scores.get(dim, 0.5)
            p_high = self._lh_high.get(dim, {}).get(cause, 0.1)
            p_low = 1.0 - p_high

            # Interpolate: weight high by score, low by (1-score)
            p_dim = score * p_high + (1.0 - score) * p_low
            product *= max(p_dim, self._min_prob)

        return product

    def _validate_likelihood_map(self) -> None:
        """Ensure every evidence dim covers all 6 causes."""
        for dim in EVIDENCE_DIMS:
            if dim not in self._lh_high:
                raise ValueError(
                    f"Missing evidence dimension '{dim}' in likelihood map"
                )
            for cause in CAUSE_KEYS:
                if cause not in self._lh_high[dim]:
                    raise ValueError(
                        f"Missing cause '{cause}' in likelihood map dim '{dim}'"
                    )
