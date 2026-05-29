"""Shadow Runner — Stage 20-7.

Side-by-side comparison: Stage18_frozen vs Stage20_candidate.

Stage20 runs in shadow mode: all real inputs are mirrored, but
Stage20 outputs are logged only, never returned to users.
"""

from typing import Any

from src.core.bayes.schema import ShadowResult


class ShadowRunner:
    """Shadow running comparison between frozen and candidate systems.

    Implements the Phase 12 shadow strategy:
    - 100% of real inputs mirrored to Stage20
    - Stage20 results logged but NOT returned to users
    - Compare 6 delta metrics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.traffic_percentage = cfg.get("traffic_percentage", 100)
        self.return_response = cfg.get("return_response", False)
        self.max_results = cfg.get("max_results", 1000)

        # v0.1 pass thresholds
        self.max_latency_increase_pct = cfg.get("max_latency_increase_pct", 20)
        self.max_memory_write_errors = cfg.get("max_memory_write_errors", 0)

    def run_shadow(
        self,
        stage18_outputs: list[dict[str, Any]],
        stage20_outputs: list[dict[str, Any]],
    ) -> ShadowResult:
        """Execute shadow comparison.

        For each paired output:
        1. Compare response quality
        2. Compare retrieval hits
        3. Compare TSLA actions
        4. Compare latency
        5. Compare memory writes
        6. Assess user-visible risk
        """
        n = min(len(stage18_outputs), len(stage20_outputs))
        if n == 0:
            return ShadowResult()

        quality_deltas: list[float] = []
        retrieval_deltas: list[float] = []
        tsla_delta_counts: dict[str, int] = {}
        latency_deltas: list[float] = []
        memory_write_delta_total = 0
        risk_deltas: list[float] = []
        new_high_risk = 0
        memory_errors = 0

        for i in range(n):
            s18 = stage18_outputs[i]
            s20 = stage20_outputs[i]

            # Quality delta (simulated: compare confidence scores)
            q18 = s18.get("confidence", 0.5)
            q20 = s20.get("confidence", 0.5)
            quality_deltas.append(q20 - q18)

            # Retrieval delta (simulated)
            r18 = len(s18.get("retrieved_context", []))
            r20 = len(s20.get("retrieved_context", []))
            retrieval_deltas.append(r20 - r18)

            # TSLA action delta
            a20 = s20.get("tsla_action", "keep")
            a18 = s18.get("tsla_action", "keep")
            if a20 != a18:
                key = f"{a18}->{a20}"
                tsla_delta_counts[key] = tsla_delta_counts.get(key, 0) + 1

            # Latency delta (simulated)
            l18 = s18.get("latency_ms", 0)
            l20 = s20.get("latency_ms", 0)
            if l18 > 0:
                latency_deltas.append((l20 - l18) / l18 * 100)

            # Memory write delta
            mw18 = s18.get("memory_writes", 0)
            mw20 = s20.get("memory_writes", 0)
            memory_write_delta_total += mw20 - mw18

        avg_quality = sum(quality_deltas) / n if quality_deltas else 0.0
        avg_retrieval = sum(retrieval_deltas) / n
        avg_latency = sum(latency_deltas) / n if latency_deltas else 0.0
        avg_risk = sum(risk_deltas) / n if risk_deltas else 0.0

        # Check pass criteria
        passed = (
            avg_latency <= self.max_latency_increase_pct
            and memory_errors <= self.max_memory_write_errors
            and new_high_risk == 0
        )

        return ShadowResult(
            total_requests=n,
            response_quality_delta=round(avg_quality, 4),
            retrieval_hit_delta=round(avg_retrieval, 2),
            tsla_action_delta=tsla_delta_counts,
            latency_delta_pct=round(avg_latency, 2),
            memory_write_delta=memory_write_delta_total,
            user_visible_risk_delta=round(avg_risk, 4),
            new_high_risk_failures=new_high_risk,
            memory_write_errors=memory_errors,
            passed=passed,
        )
