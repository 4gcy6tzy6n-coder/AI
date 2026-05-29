"""Version Freezer — Stage 20-9.

Version freeze + rollback point establishment.

Produces:
- Version freeze manifest (JSON)
- Rollback runbook (Markdown)
- Auto-rollback condition spec
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.bayes.schema import VersionFreezePoint


class VersionFreezer:
    """Version freeze and rollback point manager.

    Freezes at multiple levels:
    - Component version manifest
    - Auto-rollback condition specification
    - Rollback runbook generation
    """

    def __init__(
        self,
        output_dir: str = "data/stage20/",
        component_versions: dict[str, str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.component_versions = component_versions or {
            "tsla_scorer": "v0.4",
            "hard_veto": "v0.2",
            "action_router": "v0.2",
            "stability_gate": "v0.3",
            "promotion_gate": "v0.3",
            "rollback_gate": "v0.1",
            "permanent_protection_gate": "v0.1",
            "deep_permanent_gate": "v0.1",
            "zone_manager": "v0.2",
            "btsta_orchestrator": "v0.1",
            "bayesian_inferencer": "v0.1",
            "evidence_collector": "v0.1",
            "risk_policy": "v0.1",
            "correction_executor": "v0.1",
        }

    def freeze(self, version: str = "Stage20_Baseline_v1.0") -> VersionFreezePoint:
        """Execute version freeze.

        1. Record component versions
        2. Define auto-rollback conditions
        3. Generate and save rollback runbook
        4. Write freeze manifest
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        auto_rollback = self._define_auto_rollback_conditions()

        runbook_path = self._generate_rollback_runbook(
            version=version,
            timestamp=timestamp,
            components=self.component_versions,
            auto_rollback=auto_rollback,
        )

        freeze_point = VersionFreezePoint(
            version=version,
            freeze_timestamp=timestamp,
            component_versions=dict(self.component_versions),
            snapshot_ids={},
            rollback_runbook_path=runbook_path,
            auto_rollback_conditions=auto_rollback,
        )

        self._save_freeze_manifest(freeze_point)
        return freeze_point

    def _define_auto_rollback_conditions(self) -> dict[str, Any]:
        """Define auto-rollback trigger conditions.

        v0.1 thresholds — must be validated and calibrated.
        """
        return {
            "error_rate_spike": {
                "threshold": 0.05,
                "window_minutes": 5,
                "description": "Error rate exceeds 5% sustained for 5 minutes",
            },
            "latency_p95_spike": {
                "threshold_ms": 1000,
                "window_minutes": 5,
                "description": "P95 latency exceeds 1000ms sustained for 5 minutes",
            },
            "memory_contamination_detected": {
                "threshold": 0,
                "description": "Any memory contamination detected",
            },
            "critical_safety_event": {
                "threshold": 1,
                "description": "Any single critical safety event",
            },
            "regression_pass_rate_drop": {
                "threshold": 0.98,
                "description": "Regression pass rate drops below 98%",
            },
            "tsla_false_pass": {
                "threshold": 1,
                "description": "TSLA lets through a failure it should have caught",
            },
            "multiturn_anomaly_spike": {
                "threshold": 3,
                "window_hours": 1,
                "description": "3+ multi-turn anomalies within 1 hour",
            },
        }

    def _generate_rollback_runbook(
        self,
        version: str,
        timestamp: str,
        components: dict[str, str],
        auto_rollback: dict[str, Any],
    ) -> str:
        """Generate Markdown rollback runbook."""
        lines: list[str] = []
        lines.append(f"# Rollback Runbook — {version}")
        lines.append("")
        lines.append(f"**Frozen at**: {timestamp}")
        lines.append("")
        lines.append("## Component Versions at Freeze Point")
        lines.append("")
        lines.append("| Component | Version |")
        lines.append("|-----------|---------|")
        for comp, ver in sorted(components.items()):
            lines.append(f"| {comp} | {ver} |")
        lines.append("")
        lines.append("## Auto-Rollback Triggers")
        lines.append("")
        lines.append("If any of the following conditions are met, initiate rollback:")
        lines.append("")
        for cond, spec in sorted(auto_rollback.items()):
            desc = spec.get("description", cond)
            lines.append(f"- **{cond}**: {desc}")
        lines.append("")
        lines.append("## Rollback Procedure")
        lines.append("")
        lines.append("1. Halt all incoming traffic to Stage20 candidate")
        lines.append("2. Restore Stage18 frozen baseline snapshot")
        lines.append("3. Verify regression pass rate >= 98% after rollback")
        lines.append("4. Quarantine all memory writes made by Stage20 candidate")
        lines.append("5. Record rollback reason and time in audit log")
        lines.append("6. Notify on-call with rollback summary")
        lines.append("")
        lines.append("## Recovery Checklist")
        lines.append("")
        lines.append("- [ ] Rollback snapshot verified")
        lines.append("- [ ] Regression tests pass")
        lines.append("- [ ] Memory contamination resolved")
        lines.append("- [ ] Root cause analysis started")
        lines.append("- [ ] Fix package re-evaluated before next deployment")

        content = "\n".join(lines)
        path = self.output_dir / "stage20_rollback_runbook.md"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _save_freeze_manifest(self, freeze_point: VersionFreezePoint) -> str:
        """Write freeze manifest as JSON."""
        path = self.output_dir / "stage20_freeze_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            import json
            json.dump(freeze_point.to_dict(), f, ensure_ascii=False, indent=2)
        return str(path)
