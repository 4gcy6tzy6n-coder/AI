"""
Action Router - TSLA 动作路由器

第二阶段核心：
8 动作正式框架：
1. keep - 保留
2. promote - 晋升（第二阶段只到长期正常区）
3. isolate - 隔离
4. error_archive - 错误归档
5. downgrade - 降级
6. review_backflow - 回流重审
7. split - 拆分
8. exclude - 排除

触发顺序：先危险动作 → 再治理动作 → 最后正向动作
"""

from enum import Enum
from typing import Any, Optional

from ..memory.memory_events import MemoryZone
from ..unit.models import Unit
from .hard_veto import HardVetoChecker, HardVetoResult


class TSLAActionType(Enum):
    """TSLA 8 动作类型"""
    # 危险动作（优先）
    REVIEW_BACKFLOW = "review_backflow"    # 回流重审
    SPLIT = "split"                        # 拆分
    ERROR_ARCHIVE = "error_archive"        # 错误归档
    EXCLUDE = "exclude"                    # 排除

    # 治理动作
    ISOLATE = "isolate"                    # 隔离
    DOWNGRADE = "downgrade"                # 降级

    # 正向动作
    KEEP = "keep"                          # 保留
    PROMOTE = "promote"                    # 晋升


class ActionDecision:
    """动作决策结果"""

    def __init__(
        self,
        action: TSLAActionType,
        reason: str,
        target_zone: MemoryZone,
        confidence: float = 0.5,
        hard_veto_triggered: bool = False,
        scores: dict[str, float] | None = None
    ):
        self.action = action
        self.reason = reason
        self.target_zone = target_zone
        self.confidence = confidence
        self.hard_veto_triggered = hard_veto_triggered
        self.scores = scores or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "target_zone": self.target_zone.value,
            "confidence": self.confidence,
            "hard_veto_triggered": self.hard_veto_triggered,
            "scores": self.scores
        }


class ActionRouter:
    """
    TSLA 动作路由器

    根据评分和硬否决结果，路由到正确的治理动作
    """

    def __init__(
        self,
        hard_veto_checker: Optional[HardVetoChecker] = None,
        min_promote_score: float = 65.0,
        min_keep_score: float = 50.0
    ):
        self.hard_veto_checker = hard_veto_checker or HardVetoChecker()
        self.min_promote_score = min_promote_score
        self.min_keep_score = min_keep_score

    def route(
        self,
        unit: Unit,
        composite_score: float,
        gap_detected: bool = False,
        gap_severity: str = "low",
        conflict_detected: bool = False,
        current_zone: MemoryZone = MemoryZone.TRANSIENT
    ) -> ActionDecision:
        """
        路由到正确的 TSLA 动作

        触发顺序：
        1. 危险动作（硬否决）
        2. 治理动作
        3. 正向动作
        """
        # 第一步：检查硬否决
        hard_veto = self.hard_veto_checker.check(
            unit, composite_score, gap_detected, gap_severity, conflict_detected
        )

        if hard_veto.triggered:
            return self._handle_hard_veto(hard_veto, unit, composite_score)

        # 第二步：检查治理动作
        governance_action = self._check_governance_actions(
            unit, composite_score, conflict_detected, current_zone
        )
        if governance_action:
            return governance_action

        # 第三步：检查正向动作
        return self._check_positive_actions(
            unit, composite_score, current_zone
        )

    def _handle_hard_veto(
        self,
        hard_veto: HardVetoResult,
        unit: Unit,
        composite_score: float
    ) -> ActionDecision:
        """处理硬否决结果"""
        action_map = {
            "review_backflow": (TSLAActionType.REVIEW_BACKFLOW, MemoryZone.TRANSIENT),
            "split": (TSLAActionType.SPLIT, MemoryZone.TRANSIENT),
            "error_archive": (TSLAActionType.ERROR_ARCHIVE, MemoryZone.LONG_TERM_ERROR),
            "exclude": (TSLAActionType.EXCLUDE, MemoryZone.TRANSIENT)
        }

        action, zone = action_map.get(
            hard_veto.recommended_action,
            (TSLAActionType.EXCLUDE, MemoryZone.TRANSIENT)
        )

        return ActionDecision(
            action=action,
            reason=f"硬否决触发: {hard_veto.reason}",
            target_zone=zone,
            confidence=0.9 if hard_veto.severity == "critical" else 0.7,
            hard_veto_triggered=True,
            scores={"composite": composite_score}
        )

    def _check_governance_actions(
        self,
        unit: Unit,
        composite_score: float,
        conflict_detected: bool,
        current_zone: MemoryZone
    ) -> Optional[ActionDecision]:
        """检查治理动作"""
        # 冲突检测 → 隔离
        if conflict_detected or unit.conflict_cleanliness < 0.50:
            return ActionDecision(
                action=TSLAActionType.ISOLATE,
                reason=f"检测到冲突或清洁度不足({unit.conflict_cleanliness:.2f})",
                target_zone=MemoryZone.LONG_TERM_ISOLATION,
                confidence=0.7,
                scores={
                    "composite": composite_score,
                    "conflict_cleanliness": unit.conflict_cleanliness
                }
            )

        # 降级检查
        if current_zone == MemoryZone.LONG_TERM_NORMAL and composite_score < self.min_keep_score:
            return ActionDecision(
                action=TSLAActionType.DOWNGRADE,
                reason=f"分数下降({composite_score:.1f})，需要降级审查",
                target_zone=MemoryZone.LONG_TERM_REVIEW,
                confidence=0.6,
                scores={"composite": composite_score}
            )

        return None

    def _check_positive_actions(
        self,
        unit: Unit,
        composite_score: float,
        current_zone: MemoryZone
    ) -> ActionDecision:
        """检查正向动作"""
        # 晋升检查（第二阶段只到长期正常区）
        if composite_score >= self.min_promote_score:
            if current_zone == MemoryZone.TRANSIENT:
                return ActionDecision(
                    action=TSLAActionType.PROMOTE,
                    reason=f"综合治理分达标({composite_score:.1f} >= {self.min_promote_score})",
                    target_zone=MemoryZone.LONG_TERM_REVIEW,  # 先到受审区
                    confidence=composite_score / 100,
                    scores={
                        "composite": composite_score,
                        "truth": unit.truth_score,
                        "stability": unit.stability_score
                    }
                )
            elif current_zone == MemoryZone.LONG_TERM_REVIEW:
                return ActionDecision(
                    action=TSLAActionType.PROMOTE,
                    reason="通过受审区审查，晋升到正常区",
                    target_zone=MemoryZone.LONG_TERM_NORMAL,
                    confidence=composite_score / 100,
                    scores={"composite": composite_score}
                )

        # 保留
        return ActionDecision(
            action=TSLAActionType.KEEP,
            reason="保持当前状态",
            target_zone=current_zone,
            confidence=composite_score / 100,
            scores={"composite": composite_score}
        )

    def get_action_priority(self) -> list[str]:
        """获取动作优先级顺序"""
        return [
            # 危险动作（优先）
            "exclude",
            "review_backflow",
            "split",
            "error_archive",
            # 治理动作
            "isolate",
            "downgrade",
            # 正向动作
            "promote",
            "keep"
        ]
