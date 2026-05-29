"""
Hard Veto - 硬否决系统

第二阶段核心：
第一层硬触发规则：
- 幻觉/明显冲突 → review_backflow
- 多义混装/结构非法 → split
- 明确错误且有保留价值 → error_archive
- 明确无价值且不成立 → exclude
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..unit.models import Unit


class HardVetoType(Enum):
    """硬否决类型"""
    SAFETY = "safety"          # 安全硬否决
    LIABILITY = "liability"    # 责任硬否决
    ACCOUNTABILITY = "accountability"  # 问责硬否决


class HardVetoResult:
    """硬否决结果"""

    def __init__(
        self,
        triggered: bool,
        veto_type: Optional[HardVetoType] = None,
        reason: str = "",
        recommended_action: str = "",
        severity: str = "medium"
    ):
        self.triggered = triggered
        self.veto_type = veto_type
        self.reason = reason
        self.recommended_action = recommended_action
        self.severity = severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "veto_type": self.veto_type.value if self.veto_type else None,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "severity": self.severity
        }


class HardVetoChecker:
    """
    硬否决检查器

    触发顺序：先危险动作，再治理动作，最后正向动作
    """

    def __init__(
        self,
        safety_threshold: float = 20.0,      # composite_score 阈值 (0-100)
        liability_threshold: float = 25.0,   # composite_score 阈值 (0-100)
        accountability_threshold: float = 30.0  # composite_score 阈值 (0-100)
    ):
        # 注意：这些阈值用于检查 composite_score (0-100 范围)
        # unit.truth_score 等是 0-1 范围，在各自检查中处理
        self.safety_threshold = safety_threshold
        self.liability_threshold = liability_threshold
        self.accountability_threshold = accountability_threshold

    def check(
        self,
        unit: Unit,
        composite_score: float,
        gap_detected: bool = False,
        gap_severity: str = "low",
        conflict_detected: bool = False
    ) -> HardVetoResult:
        """
        执行硬否决检查

        返回硬否决结果
        """
        # 1. 安全硬否决（最高优先级）
        safety_check = self._check_safety_veto(
            unit, composite_score, gap_severity
        )
        if safety_check:
            return safety_check

        # 2. 责任硬否决
        liability_check = self._check_liability_veto(
            unit, composite_score, conflict_detected
        )
        if liability_check:
            return liability_check

        # 3. 问责硬否决
        accountability_check = self._check_accountability_veto(
            unit, composite_score, gap_detected
        )
        if accountability_check:
            return accountability_check

        # 无硬否决触发
        return HardVetoResult(triggered=False)

    def _check_safety_veto(
        self,
        unit: Unit,
        composite_score: float,
        gap_severity: str
    ) -> Optional[HardVetoResult]:
        """检查安全硬否决"""
        # 综合治理分极低
        if composite_score < self.safety_threshold:
            # 检查是否有学习价值，如果有则错误归档而不是排除
            if unit.metadata.get("confirmed_error") and unit.metadata.get("learning_value"):
                return HardVetoResult(
                    triggered=True,
                    veto_type=HardVetoType.SAFETY,
                    reason=f"综合治理分({composite_score:.1f})低于安全阈值，但有学习价值",
                    recommended_action="error_archive",
                    severity="high"
                )
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.SAFETY,
                reason=f"综合治理分({composite_score:.1f})低于安全阈值({self.safety_threshold})",
                recommended_action="exclude",
                severity="critical"
            )

        # 真实性严重存疑 (unit.truth_score 是 0-1 范围)
        if unit.truth_score < 0.15:
            # 检查是否有学习价值
            if unit.metadata.get("confirmed_error") and unit.metadata.get("learning_value"):
                return HardVetoResult(
                    triggered=True,
                    veto_type=HardVetoType.SAFETY,
                    reason=f"真实性分数({unit.truth_score:.2f})严重不合格，但有学习价值",
                    recommended_action="error_archive",
                    severity="high"
                )
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.SAFETY,
                reason=f"真实性分数({unit.truth_score:.2f})严重不合格",
                recommended_action="exclude",
                severity="critical"
            )

        # 严重幻觉（高缺口 + 低置信）
        if gap_severity == "high" and composite_score < 40:
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.SAFETY,
                reason="严重信息缺口且置信度不足，疑似幻觉",
                recommended_action="review_backflow",
                severity="high"
            )

        return None

    def _check_liability_veto(
        self,
        unit: Unit,
        composite_score: float,
        conflict_detected: bool
    ) -> Optional[HardVetoResult]:
        """检查责任硬否决"""
        # 冲突清洁度严重不合格 (0-1 范围)
        if unit.conflict_cleanliness < 0.10:
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.LIABILITY,
                reason=f"冲突清洁度({unit.conflict_cleanliness:.2f})严重不合格",
                recommended_action="isolate",
                severity="high"
            )

        # 明显冲突且无法解决
        if conflict_detected and composite_score < self.liability_threshold:
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.LIABILITY,
                reason=f"检测到冲突且综合治理分({composite_score:.1f})低于责任阈值",
                recommended_action="isolate",
                severity="high"
            )

        # 结构非法 (0-1 范围)
        if unit.legality_score < 0.20:
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.LIABILITY,
                reason=f"结构合法性({unit.legality_score:.2f})严重不合格",
                recommended_action="split",
                severity="high"
            )

        return None

    def _check_accountability_veto(
        self,
        unit: Unit,
        composite_score: float,
        gap_detected: bool
    ) -> Optional[HardVetoResult]:
        """检查问责硬否决"""
        # 多义混装
        if unit.metadata.get("ambiguous_structure"):
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.ACCOUNTABILITY,
                reason="检测到多义混装结构",
                recommended_action="split",
                severity="medium"
            )

        # 明确错误但有学习价值
        if unit.metadata.get("confirmed_error") and unit.metadata.get("learning_value"):
            return HardVetoResult(
                triggered=True,
                veto_type=HardVetoType.ACCOUNTABILITY,
                reason="已确认错误但具有学习价值",
                recommended_action="error_archive",
                severity="medium"
            )

        # 无价值且不成立 (unit.truth_score 和 unit.evidence_score 是 0-1 范围)
        if composite_score < self.accountability_threshold and not gap_detected:
            if unit.truth_score < 0.30 and unit.evidence_score < 0.30:
                return HardVetoResult(
                    triggered=True,
                    veto_type=HardVetoType.ACCOUNTABILITY,
                    reason="无价值且不成立",
                    recommended_action="exclude",
                    severity="medium"
                )

        return None

    def get_thresholds(self) -> dict[str, float]:
        """获取当前阈值"""
        return {
            "safety_threshold": self.safety_threshold,
            "liability_threshold": self.liability_threshold,
            "accountability_threshold": self.accountability_threshold
        }
