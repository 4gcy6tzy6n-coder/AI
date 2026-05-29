"""
Parameter Promotion Gate - 参数晋升门控

第六阶段核心组件：
控制从 deep_permanent 到 self_learned_param_store 的晋升

检查项：
- 是否来自 deep_permanent
- 是否满足参数晋升阈值（比深层永久更高）
- 是否通过跨任务验证
- 是否满足长期稳定性要求
- 是否通过跨版本验证
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime


class ParamPromotionDecision(Enum):
    """参数晋升决策"""
    APPROVED = "approved"                                  # 批准晋升
    BLOCKED_BY_SOURCE = "blocked_by_source"               # 被来源阻止
    BLOCKED_BY_THRESHOLD = "blocked_by_threshold"         # 被阈值阻止
    BLOCKED_BY_CROSS_TASK = "blocked_by_cross_task"       # 被跨任务验证阻止
    BLOCKED_BY_STABILITY = "blocked_by_stability"         # 被稳定性阻止
    BLOCKED_BY_VERSION = "blocked_by_version"             # 被版本验证阻止
    NEEDS_MORE_VERIFICATION = "needs_more_verification"   # 需要更多验证


@dataclass
class ParamPromotionResult:
    """参数晋升评估结果"""
    candidate_id: str
    decision: ParamPromotionDecision
    confidence: float
    reason: str
    recommendations: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ParameterPromotionGate:
    """
    参数晋升门控

    功能：
    1. 来源检查 - 只能来自 deep_permanent
    2. 阈值检查 - 更高的质量门槛
    3. 跨任务验证 - 检查在不同任务类型下的表现
    4. 长期稳定性 - 多周期稳定性验证
    5. 跨版本验证 - 确保不破坏旧能力

    与深层永久门控的区别：
    - 来源限制更严格（仅限 deep_permanent）
    - 阈值更高（Q≥90, T≥92, S≥90, C≥95, L≥93）
    - 需要跨任务验证
    - 需要跨版本验证
    - 需要更长验证周期（8轮 vs 5轮）
    """

    def __init__(self):
        # 参数晋升阈值（更高）
        self.param_thresholds = {
            "Q": 90,  # 质量
            "T": 92,  # 透明度
            "S": 90,  # 稳定性
            "C": 95,  # 一致性
            "L": 93   # 合法性
        }

        # 额外验证要求
        self.verification_requirements = {
            "min_verification_rounds": 8,
            "min_cross_task_score": 0.90,
            "min_long_term_stability": 0.95,
            "min_cross_version_score": 0.85
        }

        # 来源限制
        self.allowed_sources = ["deep_permanent"]
        self.blocked_sources = [
            "shallow_permanent",
            "user_path",
            "model_self_constructed_direct",
            "training_normal"
        ]

    def evaluate(
        self,
        candidate_id: str,
        source_zone: str,
        quality_scores: Dict[str, float],
        verification_rounds: int,
        cross_task_score: float,
        long_term_stability: float,
        cross_version_score: float = 0.0
    ) -> ParamPromotionResult:
        """
        评估是否允许参数晋升

        Args:
            candidate_id: 候选ID
            source_zone: 来源区域
            quality_scores: 质量分数
            verification_rounds: 验证轮数
            cross_task_score: 跨任务分数
            long_term_stability: 长期稳定性
            cross_version_score: 跨版本分数
        """
        blockers = []

        # 1. 来源检查 - 只能来自 deep_permanent
        source_ok, source_blocker = self._check_source(source_zone)
        if not source_ok:
            blockers.append(source_blocker)

        # 2. 阈值检查
        thresholds_ok, threshold_blockers = self._check_thresholds(quality_scores)
        if not thresholds_ok:
            blockers.extend(threshold_blockers)

        # 3. 验证轮数检查
        rounds_ok, rounds_blocker = self._check_verification_rounds(verification_rounds)
        if not rounds_ok:
            blockers.append(rounds_blocker)

        # 4. 跨任务验证
        cross_task_ok, cross_task_blocker = self._check_cross_task(cross_task_score)
        if not cross_task_ok:
            blockers.append(cross_task_blocker)

        # 5. 长期稳定性
        stability_ok, stability_blocker = self._check_stability(long_term_stability)
        if not stability_ok:
            blockers.append(stability_blocker)

        # 6. 跨版本验证
        version_ok, version_blocker = self._check_cross_version(cross_version_score)
        if not version_ok:
            blockers.append(version_blocker)

        # 生成决策
        if blockers:
            decision = self._determine_block_reason(blockers)
            confidence = 0.7
            reason = f"参数晋升被阻止: {'; '.join(blockers)}"
            recommendations = self._generate_recommendations(blockers)
        else:
            decision = ParamPromotionDecision.APPROVED
            confidence = 0.95
            reason = "通过所有参数晋升保护检查，允许进入自构建学习参数区"
            recommendations = [
                "建议持续监控跨任务表现",
                "建议记录完整的验证证明",
                "建议设置定期复查机制",
                "建议准备回滚方案"
            ]

        return ParamPromotionResult(
            candidate_id=candidate_id,
            decision=decision,
            confidence=confidence,
            reason=reason,
            recommendations=recommendations,
            blockers=blockers
        )

    def _check_source(self, source_zone: str) -> tuple[bool, str]:
        """检查来源"""
        if source_zone in self.allowed_sources:
            return True, f"来源 {source_zone} 允许参数晋升"
        elif source_zone in self.blocked_sources:
            return False, f"来源 {source_zone} 被阻止进入参数层"
        else:
            return False, f"来源 {source_zone} 未在允许列表中"

    def _check_thresholds(self, scores: Dict[str, float]) -> tuple[bool, List[str]]:
        """检查阈值"""
        blockers = []

        for metric, threshold in self.param_thresholds.items():
            score = scores.get(metric, 0)
            if score < threshold:
                blockers.append(f"{metric}分数 {score} < 阈值 {threshold}")

        if blockers:
            return False, blockers
        return True, []

    def _check_verification_rounds(self, rounds: int) -> tuple[bool, str]:
        """检查验证轮数"""
        min_rounds = self.verification_requirements["min_verification_rounds"]
        if rounds >= min_rounds:
            return True, f"验证轮数 {rounds} >= 阈值 {min_rounds}"
        else:
            return False, f"验证轮数 {rounds} < 阈值 {min_rounds}"

    def _check_cross_task(self, score: float) -> tuple[bool, str]:
        """检查跨任务验证"""
        min_score = self.verification_requirements["min_cross_task_score"]
        if score >= min_score:
            return True, f"跨任务分数 {score} >= 阈值 {min_score}"
        else:
            return False, f"跨任务分数 {score} < 阈值 {min_score}"

    def _check_stability(self, stability: float) -> tuple[bool, str]:
        """检查长期稳定性"""
        min_stability = self.verification_requirements["min_long_term_stability"]
        if stability >= min_stability:
            return True, f"长期稳定性 {stability} >= 阈值 {min_stability}"
        else:
            return False, f"长期稳定性 {stability} < 阈值 {min_stability}"

    def _check_cross_version(self, score: float) -> tuple[bool, str]:
        """检查跨版本验证"""
        min_score = self.verification_requirements["min_cross_version_score"]
        if score >= min_score:
            return True, f"跨版本分数 {score} >= 阈值 {min_score}"
        else:
            return False, f"跨版本分数 {score} < 阈值 {min_score}"

    def _determine_block_reason(self, blockers: List[str]) -> ParamPromotionDecision:
        """确定阻止原因"""
        blocker_str = " ".join(blockers).lower()

        if "来源" in blocker_str or "source" in blocker_str:
            return ParamPromotionDecision.BLOCKED_BY_SOURCE
        elif "跨任务" in blocker_str or "cross_task" in blocker_str:
            return ParamPromotionDecision.BLOCKED_BY_CROSS_TASK
        elif "稳定性" in blocker_str or "stability" in blocker_str:
            return ParamPromotionDecision.BLOCKED_BY_STABILITY
        elif "版本" in blocker_str or "version" in blocker_str:
            return ParamPromotionDecision.BLOCKED_BY_VERSION
        elif "轮数" in blocker_str or "verification" in blocker_str:
            return ParamPromotionDecision.NEEDS_MORE_VERIFICATION
        else:
            return ParamPromotionDecision.BLOCKED_BY_THRESHOLD

    def _generate_recommendations(self, blockers: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []

        for blocker in blockers:
            if "来源" in blocker:
                recommendations.append("建议先进入深层永久层")
            elif "跨任务" in blocker:
                recommendations.append("建议在更多任务类型上验证")
            elif "稳定性" in blocker:
                recommendations.append("建议继续观察长期稳定性")
            elif "版本" in blocker:
                recommendations.append("建议验证对旧能力的影响")
            elif "轮数" in blocker:
                recommendations.append("建议完成更多验证轮次")
            elif "分数" in blocker:
                recommendations.append("建议提升内容质量")

        return recommendations


def demo_param_gate():
    """参数晋升门控演示"""
    print("\n" + "=" * 70)
    print("Parameter Promotion Gate Demo - 参数晋升门控演示")
    print("=" * 70)

    gate = ParameterPromotionGate()

    # 案例1: 完美候选
    print("\n案例 1: 完美候选")
    result1 = gate.evaluate(
        candidate_id="param_candidate_001",
        source_zone="deep_permanent",
        quality_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
        verification_rounds=10,
        cross_task_score=0.92,
        long_term_stability=0.96,
        cross_version_score=0.88
    )

    print(f"  决策: {result1.decision.value}")
    print(f"  置信度: {result1.confidence}")
    print(f"  阻止项: {result1.blockers}")

    # 案例2: 跨任务验证不足
    print("\n案例 2: 跨任务验证不足")
    result2 = gate.evaluate(
        candidate_id="param_candidate_002",
        source_zone="deep_permanent",
        quality_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
        verification_rounds=10,
        cross_task_score=0.80,  # 不足
        long_term_stability=0.96,
        cross_version_score=0.88
    )

    print(f"  决策: {result2.decision.value}")
    print(f"  阻止项: {result2.blockers}")
    print(f"  建议: {result2.recommendations}")

    # 案例3: 来源不符（尝试从浅层永久直接晋升）
    print("\n案例 3: 来源不符（浅层永久尝试）")
    result3 = gate.evaluate(
        candidate_id="param_candidate_003",
        source_zone="shallow_permanent",  # 不允许
        quality_scores={"Q": 95, "T": 95, "S": 95, "C": 95, "L": 95},
        verification_rounds=10,
        cross_task_score=0.95,
        long_term_stability=0.98,
        cross_version_score=0.90
    )

    print(f"  决策: {result3.decision.value}")
    print(f"  阻止项: {result3.blockers}")


if __name__ == "__main__":
    demo_param_gate()
