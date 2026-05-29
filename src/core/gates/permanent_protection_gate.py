"""
Permanent Protection Gate - 永久层保护门

第四阶段核心组件：
- 控制长期正常区 → 浅层永久的晋升
- 禁止跨层直写
- 多周期稳定性验证
- 来源策略检查

保护原则：
- 数量极少
- 质量极高
- 修改极慢
- 回退极严
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from datetime import datetime


class PermanentZone(Enum):
    """永久层区域"""
    SHALLOW_PERMANENT = "shallow_permanent"  # 浅层永久
    DEEP_PERMANENT = "deep_permanent"        # 深层永久（第四阶段暂不开放）


class PermanentDecision(Enum):
    """永久层决策"""
    PROMOTE_TO_SHALLOW = "promote_to_shallow_permanent"
    BLOCKED_BY_STABILITY = "blocked_by_stability"
    BLOCKED_BY_SOURCE_POLICY = "blocked_by_source_policy"
    BLOCKED_BY_CONFLICT = "blocked_by_conflict"
    BLOCKED_BY_INSUFFICIENT_CYCLES = "blocked_by_insufficient_cycles"
    BLOCKED_BY_RECENT_PROMOTION = "blocked_by_recent_promotion"


@dataclass
class PermanentProtectionResult:
    """永久层保护结果"""
    unit_id: str
    current_zone: str
    target_zone: str
    decision: PermanentDecision
    reason: str
    confidence: float
    stability_cycles: int
    source_type: str
    conflict_free_rounds: int
    blockers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class PermanentProtectionGate:
    """
    永久层保护门

    功能：
    1. 检查对象是否来自长期正常区
    2. 检查是否满足永久候选阈值
    3. 检查是否近期无重大冲突
    4. 检查是否已通过多周期稳定性验证
    5. 检查是否不是"刚刚晋升"的新对象
    6. 检查来源类型是否允许进入浅层永久
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._init_thresholds()

    def _init_thresholds(self):
        """初始化阈值"""
        # 浅层永久候选阈值（高于正常区阈值）
        self.shallow_min_q = self.config.get("shallow_min_q", 78)
        self.shallow_min_t = self.config.get("shallow_min_t", 80)
        self.shallow_min_s = self.config.get("shallow_min_s", 75)
        self.shallow_min_c = self.config.get("shallow_min_c", 85)
        self.shallow_min_l = self.config.get("shallow_min_l", 85)

        # 周期要求
        self.min_stability_cycles = self.config.get("min_stability_cycles", 5)
        self.min_conflict_free_rounds = self.config.get("min_conflict_free_rounds", 3)
        self.min_days_since_promotion = self.config.get("min_days_since_promotion", 7)

        # 允许的来源类型
        self.allowed_source_types = self.config.get(
            "allowed_source_types",
            ["retrieved", "verified_inference"]
        )

        # 禁止的来源类型
        self.blocked_source_types = self.config.get(
            "blocked_source_types",
            ["user_input", "unverified_inference", "ambiguous"]
        )

    def evaluate(
        self,
        unit_id: str,
        current_zone: str,
        current_scores: dict[str, float],
        source_type: str,
        stability_cycles: int,
        conflict_free_rounds: int,
        days_since_promotion: int,
        recent_conflict: bool = False,
        recent_backflow: bool = False,
        hard_veto_history: list = None,
        config: Optional[dict] = None
    ) -> PermanentProtectionResult:
        """
        评估永久层晋升申请

        Args:
            unit_id: 对象ID
            current_zone: 当前区域（必须是normal）
            current_scores: 当前8维分数
            source_type: 来源类型
            stability_cycles: 稳定性验证周期数
            conflict_free_rounds: 无冲突轮数
            days_since_promotion: 晋升到normal区的天数
            recent_conflict: 近期是否有冲突
            recent_backflow: 近期是否有回流
            hard_veto_history: 硬否决历史

        Returns:
            PermanentProtectionResult: 保护门评估结果
        """
        cfg = config or self.config
        blockers = []
        recommendations = []

        # Step 1: 检查区位前提
        zone_ok, zone_blocker = self._check_zone_prerequisite(current_zone)
        if not zone_ok:
            blockers.append(zone_blocker)

        # Step 2: 检查来源策略
        source_ok, source_blocker = self._check_source_policy(source_type)
        if not source_ok:
            blockers.append(source_blocker)

        # Step 3: 检查周期要求
        cycles_ok, cycles_blocker = self._check_cycle_requirements(
            stability_cycles, conflict_free_rounds, days_since_promotion
        )
        if not cycles_ok:
            blockers.append(cycles_blocker)

        # Step 4: 检查阈值要求
        thresholds_ok, threshold_blockers = self._check_shallow_thresholds(current_scores)
        if not thresholds_ok:
            blockers.extend(threshold_blockers)

        # Step 5: 检查冲突历史
        conflict_ok, conflict_blocker = self._check_conflict_status(
            recent_conflict, recent_backflow, hard_veto_history
        )
        if not conflict_ok:
            blockers.append(conflict_blocker)

        # 生成建议
        if blockers:
            recommendations = self._generate_recommendations(blockers, current_scores)
            decision = self._determine_block_reason(blockers)
            confidence = 0.7
            reason = f"永久层晋升被阻止: {'; '.join(blockers)}"
        else:
            decision = PermanentDecision.PROMOTE_TO_SHALLOW
            confidence = 0.85
            reason = "通过所有永久层保护检查，允许进入浅层永久"
            recommendations = [
                "建议持续监控稳定性",
                "建议记录支持证据摘要",
                "建议设置定期复查机制"
            ]

        return PermanentProtectionResult(
            unit_id=unit_id,
            current_zone=current_zone,
            target_zone=PermanentZone.SHALLOW_PERMANENT.value,
            decision=decision,
            reason=reason,
            confidence=confidence,
            stability_cycles=stability_cycles,
            source_type=source_type,
            conflict_free_rounds=conflict_free_rounds,
            blockers=blockers,
            recommendations=recommendations
        )

    def _check_zone_prerequisite(self, current_zone: str) -> tuple[bool, str]:
        """检查区位前提 - 必须来自长期正常区"""
        if current_zone != "normal":
            return False, f"当前区域为{current_zone}，永久层只能由长期正常区晋升"
        return True, ""

    def _check_source_policy(self, source_type: str) -> tuple[bool, str]:
        """检查来源策略"""
        if source_type in self.blocked_source_types:
            return False, f"来源类型'{source_type}'不允许进入永久层"

        if source_type not in self.allowed_source_types:
            return False, f"来源类型'{source_type}'未在允许列表中"

        return True, ""

    def _check_cycle_requirements(
        self,
        stability_cycles: int,
        conflict_free_rounds: int,
        days_since_promotion: int
    ) -> tuple[bool, str]:
        """检查周期要求"""
        if stability_cycles < self.min_stability_cycles:
            return False, f"稳定性周期不足({stability_cycles}/{self.min_stability_cycles})"

        if conflict_free_rounds < self.min_conflict_free_rounds:
            return False, f"无冲突轮数不足({conflict_free_rounds}/{self.min_conflict_free_rounds})"

        if days_since_promotion < self.min_days_since_promotion:
            return False, f"晋升时间太短({days_since_promotion}/{self.min_days_since_promotion}天)"

        return True, ""

    def _check_shallow_thresholds(
        self,
        current_scores: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """检查浅层永久阈值"""
        blockers = []

        checks = [
            ("Q", current_scores.get("Q", 0), self.shallow_min_q),
            ("T", current_scores.get("T", 0), self.shallow_min_t),
            ("S", current_scores.get("S", 0), self.shallow_min_s),
            ("C", current_scores.get("C", 0), self.shallow_min_c),
            ("L", current_scores.get("L", 0), self.shallow_min_l),
        ]

        for dim, actual, threshold in checks:
            if actual < threshold:
                blockers.append(f"{dim}分数不足({actual:.1f}/{threshold})")

        return len(blockers) == 0, blockers

    def _check_conflict_status(
        self,
        recent_conflict: bool,
        recent_backflow: bool,
        hard_veto_history: list
    ) -> tuple[bool, str]:
        """检查冲突状态"""
        if recent_conflict:
            return False, "近期存在冲突"

        if recent_backflow:
            return False, "近期发生过回流"

        if hard_veto_history and len(hard_veto_history) > 0:
            return False, f"存在硬否决历史({len(hard_veto_history)}次)"

        return True, ""

    def _determine_block_reason(self, blockers: list[str]) -> PermanentDecision:
        """确定阻止原因"""
        blocker_text = "; ".join(blockers).lower()

        if "周期" in blocker_text or "天数" in blocker_text:
            return PermanentDecision.BLOCKED_BY_INSUFFICIENT_CYCLES
        elif "来源" in blocker_text:
            return PermanentDecision.BLOCKED_BY_SOURCE_POLICY
        elif "冲突" in blocker_text or "回流" in blocker_text or "硬否决" in blocker_text:
            return PermanentDecision.BLOCKED_BY_CONFLICT
        else:
            return PermanentDecision.BLOCKED_BY_STABILITY

    def _generate_recommendations(
        self,
        blockers: list[str],
        current_scores: dict[str, float]
    ) -> list[str]:
        """生成改进建议"""
        recommendations = []

        for blocker in blockers:
            if "周期" in blocker:
                recommendations.append(f"建议继续积累稳定性周期")
            elif "来源" in blocker:
                recommendations.append(f"建议寻找更高质量的来源支持")
            elif "冲突" in blocker:
                recommendations.append(f"建议解决冲突后再申请")
            elif "分数" in blocker:
                # 提取维度
                for dim in ["Q", "T", "S", "C", "L"]:
                    if dim in blocker:
                        recommendations.append(f"建议提升{dim}维度质量")

        return recommendations

    def get_shallow_thresholds(self) -> dict[str, float]:
        """获取浅层永久阈值"""
        return {
            "Q": self.shallow_min_q,
            "T": self.shallow_min_t,
            "S": self.shallow_min_s,
            "C": self.shallow_min_c,
            "L": self.shallow_min_l
        }

    def get_gap_analysis(
        self,
        current_scores: dict[str, float],
        stability_cycles: int,
        conflict_free_rounds: int
    ) -> dict:
        """获取与阈值的差距分析"""
        thresholds = self.get_shallow_thresholds()

        gaps = {}
        for dim, threshold in thresholds.items():
            actual = current_scores.get(dim, 0)
            gaps[dim] = {
                "actual": actual,
                "threshold": threshold,
                "gap": threshold - actual,
                "passed": actual >= threshold
            }

        return {
            "score_gaps": gaps,
            "cycle_gap": {
                "actual": stability_cycles,
                "required": self.min_stability_cycles,
                "gap": self.min_stability_cycles - stability_cycles
            },
            "conflict_free_gap": {
                "actual": conflict_free_rounds,
                "required": self.min_conflict_free_rounds,
                "gap": self.min_conflict_free_rounds - conflict_free_rounds
            }
        }
