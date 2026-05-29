"""
Write Gate - 写入门

第二阶段核心：控制从瞬时层到长期受审区的迁移

规则：
1. 所有输入必须先进入瞬时层
2. 只有部分对象可通过写入门进入长期受审区
3. 写入门基于 TSLA 评分和硬否决规则
"""

from typing import Any, Optional
from uuid import UUID

from ..memory.memory_events import MemoryEvent, MemoryEventLog, MemoryEventType, MemoryZone
from ..unit.models import Unit


class WriteGateDecision:
    """写入门决策结果"""

    def __init__(
        self,
        allowed: bool,
        reason: str,
        target_zone: MemoryZone = MemoryZone.TRANSIENT,
        scores: dict[str, float] | None = None,
        hard_veto_triggered: bool = False
    ):
        self.allowed = allowed
        self.reason = reason
        self.target_zone = target_zone
        self.scores = scores or {}
        self.hard_veto_triggered = hard_veto_triggered

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "target_zone": self.target_zone.value,
            "scores": self.scores,
            "hard_veto_triggered": self.hard_veto_triggered
        }


class WriteGate:
    """
    写入门

    判断 Unit 是否允许从瞬时层进入长期层
    """

    def __init__(
        self,
        min_composite_score: float = 65.0,  # 0-100 范围
        min_truth_score: float = 0.50,      # 0-1 范围
        min_legality_score: float = 0.70,   # 0-1 范围
        event_log: Optional[MemoryEventLog] = None
    ):
        # 阈值配置
        # 注意：composite_score 是 0-100 范围
        # unit.truth_score 和 unit.legality_score 是 0-1 范围
        self.min_composite_score = min_composite_score
        self.min_truth_score = min_truth_score
        self.min_legality_score = min_legality_score

        self._event_log = event_log

    def evaluate(
        self,
        unit: Unit,
        composite_score: float,
        session_id: Optional[str] = None
    ) -> WriteGateDecision:
        """
        评估 Unit 是否允许进入长期层

        返回决策结果
        """
        # 1. 硬否决检查
        hard_veto = self._check_hard_veto(unit, composite_score)
        if hard_veto:
            decision = WriteGateDecision(
                allowed=False,
                reason=f"硬否决触发: {hard_veto}",
                target_zone=MemoryZone.TRANSIENT,
                scores={"composite": composite_score},
                hard_veto_triggered=True
            )
            self._log_decision(unit, decision, session_id)
            return decision

        # 2. 结构合法性检查
        if unit.legality_score < self.min_legality_score:
            decision = WriteGateDecision(
                allowed=False,
                reason=f"结构合法性不足: {unit.legality_score:.1f} < {self.min_legality_score}",
                target_zone=MemoryZone.TRANSIENT,
                scores={
                    "legality": unit.legality_score,
                    "composite": composite_score
                }
            )
            self._log_decision(unit, decision, session_id)
            return decision

        # 3. 真实性检查
        if unit.truth_score < self.min_truth_score:
            decision = WriteGateDecision(
                allowed=False,
                reason=f"真实性不足: {unit.truth_score:.1f} < {self.min_truth_score}",
                target_zone=MemoryZone.TRANSIENT,
                scores={
                    "truth": unit.truth_score,
                    "composite": composite_score
                }
            )
            self._log_decision(unit, decision, session_id)
            return decision

        # 4. 综合治理分检查
        if composite_score < self.min_composite_score:
            decision = WriteGateDecision(
                allowed=False,
                reason=f"综合治理分不足: {composite_score:.1f} < {self.min_composite_score}",
                target_zone=MemoryZone.TRANSIENT,
                scores={"composite": composite_score}
            )
            self._log_decision(unit, decision, session_id)
            return decision

        # 5. 通过所有检查，允许进入长期受审区
        decision = WriteGateDecision(
            allowed=True,
            reason="通过写入门所有检查",
            target_zone=MemoryZone.LONG_TERM_REVIEW,
            scores={
                "composite": composite_score,
                "truth": unit.truth_score,
                "legality": unit.legality_score
            }
        )
        self._log_decision(unit, decision, session_id)
        return decision

    def _check_hard_veto(
        self,
        unit: Unit,
        composite_score: float
    ) -> Optional[str]:
        """
        检查硬否决条件

        返回否决原因，如果没有触发则返回 None
        """
        # 硬否决：严重安全/责任问题
        # 注意：composite_score 是 0-100 范围，unit.truth_score 是 0-1 范围
        if composite_score < 20:
            return "综合治理分低于安全阈值(20)"

        if unit.truth_score < 0.15:  # 0-1 范围
            return "真实性严重存疑"

        if unit.conflict_cleanliness < 0.10:  # 0-1 范围
            return "冲突清洁度严重不合格"

        # 检查元数据中的硬否决标记
        if unit.metadata.get("hard_veto_safety"):
            return "安全硬否决标记"

        if unit.metadata.get("hard_veto_liability"):
            return "责任硬否决标记"

        return None

    def _log_decision(
        self,
        unit: Unit,
        decision: WriteGateDecision,
        session_id: Optional[str]
    ) -> None:
        """记录决策事件"""
        if not self._event_log:
            return

        event_type = (
            MemoryEventType.PROMOTION_SUCCESS
            if decision.allowed
            else MemoryEventType.PROMOTION_REJECTED
        )

        event = MemoryEvent(
            event_type=event_type,
            unit_id=unit.unit_id,
            session_id=session_id,
            source_zone=MemoryZone.TRANSIENT,
            target_zone=decision.target_zone if decision.allowed else None,
            details={
                "reason": decision.reason,
                "hard_veto": decision.hard_veto_triggered
            },
            tsla_scores=decision.scores
        )
        self._event_log.log(event)

    def get_thresholds(self) -> dict[str, float]:
        """获取当前阈值"""
        return {
            "min_composite_score": self.min_composite_score,
            "min_truth_score": self.min_truth_score,
            "min_legality_score": self.min_legality_score
        }
