from typing import Any

from ..thinking.gap_detector import GapDetectionResult
from ..unit.models import CandidateConclusion, TSLAAction, TSLAResult, Unit


class TSLAReviewer:
    """
    TSLA 审查器

    第一阶段：基于简单规则返回动作建议
    目标：让 TSLA 成为控制器，而不是旁白解释器

    支持 5 个动作：
    - keep: 接受并保留
    - review: 需要人工审查
    - split: 需要拆分
    - isolate: 需要隔离
    - reject: 拒绝
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        max_conflict_threshold: float = 0.3
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.max_conflict_threshold = max_conflict_threshold

    def review(
        self,
        candidate: CandidateConclusion,
        gap_result: GapDetectionResult | None = None,
        units: list[Unit] | None = None
    ) -> TSLAResult:
        """
        审查候选结论并返回动作建议

        输入：
        - 候选结论
        - 涉及的 Unit
        - evidence 数量
        - 是否有冲突
        - 是否存在明显缺口

        输出：TSLA 动作建议
        """
        units = units or []

        # 1. 检查是否有严重缺口
        if gap_result and gap_result.has_gap:
            if gap_result.severity == "high":
                return TSLAResult(
                    action=TSLAAction.REJECT,
                    confidence=0.8,
                    reason=f"存在严重信息缺口: {', '.join(gap_result.reasons)}",
                    details={
                        "gap_types": gap_result.gap_types,
                        "severity": gap_result.severity
                    }
                )
            elif gap_result.severity == "medium":
                return TSLAResult(
                    action=TSLAAction.REVIEW,
                    confidence=0.6,
                    reason=f"存在中等信息缺口: {', '.join(gap_result.reasons)}",
                    details={
                        "gap_types": gap_result.gap_types,
                        "severity": gap_result.severity
                    }
                )

        # 2. 检查是否有冲突
        if candidate.conflict_flags:
            return TSLAResult(
                action=TSLAAction.ISOLATE,
                confidence=0.7,
                reason=f"检测到冲突: {', '.join(candidate.conflict_flags)}",
                details={
                    "conflict_flags": candidate.conflict_flags
                }
            )

        # 3. 检查置信度
        if candidate.confidence_stub < self.min_confidence_threshold:
            return TSLAResult(
                action=TSLAAction.REVIEW,
                confidence=0.5,
                reason=f"置信度过低: {candidate.confidence_stub:.2f} < {self.min_confidence_threshold}",
                details={
                    "confidence": candidate.confidence_stub,
                    "threshold": self.min_confidence_threshold
                }
            )

        # 4. 检查是否需要拆分（多个不同含义的 Unit）
        script_forms = [u.script_form for u in units if u.script_form]
        unique_forms = set(script_forms)
        if len(script_forms) != len(unique_forms):
            return TSLAResult(
                action=TSLAAction.SPLIT,
                confidence=0.6,
                reason="检测到需要拆分的多义对象",
                details={
                    "duplicate_forms": list(set([f for f in script_forms if script_forms.count(f) > 1]))
                }
            )

        # 5. 检查证据充分性
        if not candidate.used_evidence:
            return TSLAResult(
                action=TSLAAction.REVIEW,
                confidence=0.5,
                reason="缺少支撑证据",
                details={
                    "evidence_count": len(candidate.used_evidence)
                }
            )

        # 6. 通过所有检查，可以保留
        return TSLAResult(
            action=TSLAAction.KEEP,
            confidence=candidate.confidence_stub,
            reason="通过所有 TSLA 检查",
            details={
                "confidence": candidate.confidence_stub,
                "evidence_count": len(candidate.used_evidence),
                "unit_count": len(units)
            }
        )

    def review_unit(self, unit: Unit) -> TSLAResult:
        """审查单个 Unit"""
        # 检查结构合法性
        if not unit.is_structurally_valid():
            return TSLAResult(
                action=TSLAAction.REJECT,
                confidence=0.9,
                reason="Unit 结构不合法",
                details={
                    "legality_score": unit.legality_score
                }
            )

        # 检查冲突清洁度
        if unit.conflict_cleanliness < self.max_conflict_threshold:
            return TSLAResult(
                action=TSLAAction.ISOLATE,
                confidence=0.7,
                reason=f"冲突清洁度过低: {unit.conflict_cleanliness}",
                details={
                    "conflict_cleanliness": unit.conflict_cleanliness
                }
            )

        # 检查真实性
        if unit.truth_score < 0.3:
            return TSLAResult(
                action=TSLAAction.REVIEW,
                confidence=0.6,
                reason=f"真实性分数过低: {unit.truth_score}",
                details={
                    "truth_score": unit.truth_score
                }
            )

        return TSLAResult(
            action=TSLAAction.KEEP,
            confidence=unit.governance_score,
            reason="Unit 通过审查",
            details={
                "truth_score": unit.truth_score,
                "legality_score": unit.legality_score
            }
        )

    def get_decision_matrix(self) -> dict[str, Any]:
        """获取决策矩阵（用于调试）"""
        return {
            "high_gap_severity": TSLAAction.REJECT.value,
            "medium_gap_severity": TSLAAction.REVIEW.value,
            "has_conflict": TSLAAction.ISOLATE.value,
            "low_confidence": TSLAAction.REVIEW.value,
            "needs_split": TSLAAction.SPLIT.value,
            "no_evidence": TSLAAction.REVIEW.value,
            "all_good": TSLAAction.KEEP.value
        }
