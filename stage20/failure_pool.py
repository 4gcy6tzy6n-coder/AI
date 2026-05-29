"""Failure Pool Builder — Stage 20-0 and 20-1.

20-0: Confirm trigger conditions from Stage 19 metrics.
20-1: Build and freeze the stage20 failure pool as JSONL.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.bayes.schema import FailureCategory, FailureSample, RiskLevel


class TriggerCondition:
    """Stage 19 trigger conditions that may fire Stage 20."""

    def __init__(
        self,
        knowledge_miss_count: int = 0,
        unnatural_generation_count: int = 0,
        multiturn_anomaly_count: int = 0,
        critical_safety_events: int = 0,
        thresholds: dict[str, int] | None = None,
    ):
        self.knowledge_miss = knowledge_miss_count
        self.unnatural_generation = unnatural_generation_count
        self.multiturn_anomaly = multiturn_anomaly_count
        self.critical_safety = critical_safety_events

        t = thresholds or {}
        self._threshold_knowledge = t.get("knowledge_miss", 20)
        self._threshold_generation = t.get("unnatural_generation", 10)
        self._threshold_multiturn = t.get("multiturn_anomaly", 10)

    def is_triggered(self) -> tuple[bool, str]:
        """Check if any Stage 19 trigger condition fires.

        Returns (triggered, reason).
        """
        reasons: list[str] = []

        if self.knowledge_miss >= self._threshold_knowledge:
            reasons.append(
                f"knowledge_miss={self.knowledge_miss} >= {self._threshold_knowledge}"
            )
        if self.unnatural_generation >= self._threshold_generation:
            reasons.append(
                f"unnatural_generation={self.unnatural_generation} >= "
                f"{self._threshold_generation}"
            )
        if self.multiturn_anomaly >= self._threshold_multiturn:
            reasons.append(
                f"multiturn_anomaly={self.multiturn_anomaly} >= "
                f"{self._threshold_multiturn}"
            )
        if self.critical_safety > 0:
            reasons.append(
                f"critical_safety_events={self.critical_safety} > 0"
            )

        if reasons:
            return True, "; ".join(reasons)
        return False, "No trigger conditions met"


class FailurePoolBuilder:
    """Builds and manages the stage20 failure pool.

    20-0: Confirm trigger conditions.
    20-1: Build frozen failure pool from collected cases.
    """

    def __init__(
        self,
        output_path: str = "data/stage20/stage20_failure_pool.jsonl",
        existing_failures_path: str | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.existing_failures_path = existing_failures_path
        self._pool: list[FailureSample] = []
        self._frozen: bool = False

    def confirm_trigger(self, condition: TriggerCondition) -> tuple[bool, str]:
        """20-0: Confirm Stage 19 trigger conditions.

        Also allows override via critical safety event regardless of counts.
        """
        triggered, reason = condition.is_triggered()
        return triggered, reason

    def build_pool(
        self,
        existing_failures: list[dict[str, Any]] | None = None,
        target_size: int = 50,
        require_coverage: bool = True,
    ) -> list[FailureSample]:
        """20-1: Build the frozen failure pool.

        1. Load existing failure cases
        2. Classify into K1/G1/M1/R1/T1/T2/S1/W1 taxonomy
        3. Ensure minimum coverage across all 8 types
        4. Freeze — no further modifications
        """
        if self._frozen:
            return self._pool

        samples: list[FailureSample] = []

        # Load existing failures
        failures = existing_failures or self._load_existing_failures()
        for fdata in failures:
            sample = FailureSample.from_dict(fdata)
            samples.append(sample)

        # Synthesize edge cases for coverage if needed
        if require_coverage:
            samples = self._ensure_type_coverage(samples)

        self._pool = samples[:target_size] if len(samples) > target_size else samples
        self._frozen = True
        self._save_pool()
        return self._pool

    def load_pool(self) -> list[FailureSample]:
        """Load pool from JSONL."""
        if not self.output_path.exists():
            return []

        samples: list[FailureSample] = []
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    samples.append(FailureSample.from_dict(data))
        self._pool = samples
        self._frozen = True
        return samples

    def get_pool_stats(self) -> dict[str, Any]:
        """Get summary statistics of the failure pool."""
        if not self._pool:
            return {"total": 0, "by_type": {}}

        by_type: dict[str, int] = {}
        by_risk: dict[str, int] = {}
        for sample in self._pool:
            ft = sample.failure_type.value
            by_type[ft] = by_type.get(ft, 0) + 1
            rl = sample.risk_level.value
            by_risk[rl] = by_risk.get(rl, 0) + 1

        return {
            "total": len(self._pool),
            "by_type": by_type,
            "by_risk": by_risk,
            "frozen": self._frozen,
        }

    def _load_existing_failures(self) -> list[dict[str, Any]]:
        """Load failures from existing collection."""
        if not self.existing_failures_path:
            return []

        path = Path(self.existing_failures_path)
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get("failures", [])
            except json.JSONDecodeError:
                return []

    def _ensure_type_coverage(
        self, samples: list[FailureSample]
    ) -> list[FailureSample]:
        """Ensure at least 1 sample per failure type for v0.1 coverage."""
        covered = {s.failure_type for s in samples}
        missing = set(FailureCategory) - covered

        for cat in missing:
            samples.append(FailureSample(
                sample_id=f"synth_coverage_{cat.value}",
                user_query=f"[Synthesized] Coverage sample for {cat.value}",
                failure_type=cat,
                risk_level=RiskLevel.LOW,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"synthetic": True, "purpose": "coverage"},
            ))

        return samples

    def _save_pool(self) -> str:
        """Save pool as JSONL."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for sample in self._pool:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        return str(self.output_path)
