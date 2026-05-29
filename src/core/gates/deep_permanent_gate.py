"""
Deep Permanent Gate - 深层永久门控

第五阶段核心组件：
控制训练路径长期正常区 → 深层永久的晋升

检查项：
- 是否来自训练路径正常区
- 是否满足深层永久阈值
- 是否通过强审查
- 是否完成多轮验证
- 是否满足最小训练周期
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime


class DeepPermanentDecision(Enum):
    """深层永久决策"""
    PROMOTE_TO_DEEP = "promote_to_deep"                    # 晋升到深层永久
    BLOCKED_BY_ZONE = "blocked_by_zone"                    # 被区域阻止
    BLOCKED_BY_THRESHOLD = "blocked_by_threshold"          # 被阈值阻止
    BLOCKED_BY_REVIEW = "blocked_by_review"                # 被审查阻止
    BLOCKED_BY_VERIFICATION = "blocked_by_verification"    # 被验证阻止
    BLOCKED_BY_CYCLES = "blocked_by_cycles"                # 被周期阻止
    NEEDS_MORE_VERIFICATION = "needs_more_verification"    # 需要更多验证


@dataclass
class DeepPermanentResult:
    """深层永久评估结果"""
    unit_id: str
    decision: DeepPermanentDecision
    confidence: float
    reason: str
    recommendations: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DeepPermanentGate:
    """
    深层永久门控
    
    功能：
    1. 区域检查 - 只能来自训练路径正常区
    2. 阈值检查 - 更高的质量门槛
    3. 审查检查 - 必须通过强审查
    4. 验证检查 - 必须完成多轮验证
    5. 周期检查 - 必须满足最小训练周期
    
    与浅层永久门控的区别：
    - 来源限制更严格（仅限训练路径）
    - 阈值更高（Q≥85, T≥88, S≥85, C≥92, L≥90）
    - 需要强审查证明
    - 需要验证证明
    - 需要更长周期（10周期 vs 5周期）
    """
    
    def __init__(self):
        # 深层永久阈值（更高）
        self.deep_thresholds = {
            "Q": 85,  # 质量
            "T": 88,  # 透明度
            "S": 85,  # 稳定性
            "C": 92,  # 一致性
            "L": 90   # 合法性
        }
        
        # 周期要求
        self.cycle_requirements = {
            "min_stability_cycles": 10,
            "min_conflict_free_rounds": 8,
            "min_verification_rounds": 5,
            "min_training_iterations": 100
        }
        
        # 来源限制
        self.allowed_sources = ["training_path", "multi_round_verified"]
        self.blocked_sources = ["user_path", "single_round", "model_self_constructed"]
    
    def evaluate(
        self,
        unit_id: str,
        current_zone: str,
        current_scores: Dict[str, float],
        source_type: str,
        review_passed: bool,
        review_confidence: float,
        verification_passed: bool,
        verification_rounds: int,
        stability_cycles: int,
        conflict_free_rounds: int,
        training_iterations: int,
        recent_conflict: bool = False
    ) -> DeepPermanentResult:
        """
        评估是否允许进入深层永久
        
        Args:
            unit_id: 单元ID
            current_zone: 当前区域
            current_scores: 当前分数
            source_type: 来源类型
            review_passed: 是否通过强审查
            review_confidence: 审查置信度
            verification_passed: 是否通过验证
            verification_rounds: 验证轮数
            stability_cycles: 稳定周期数
            conflict_free_rounds: 无冲突轮数
            training_iterations: 训练迭代数
            recent_conflict: 近期是否有冲突
        """
        blockers = []
        
        # 1. 区域检查 - 只能来自训练路径正常区
        zone_ok, zone_blocker = self._check_zone(current_zone)
        if not zone_ok:
            blockers.append(zone_blocker)
        
        # 2. 来源检查
        source_ok, source_blocker = self._check_source(source_type)
        if not source_ok:
            blockers.append(source_blocker)
        
        # 3. 审查检查
        review_ok, review_blocker = self._check_review(
            review_passed, review_confidence
        )
        if not review_ok:
            blockers.append(review_blocker)
        
        # 4. 验证检查
        verify_ok, verify_blocker = self._check_verification(
            verification_passed, verification_rounds
        )
        if not verify_ok:
            blockers.append(verify_blocker)
        
        # 5. 周期检查
        cycles_ok, cycles_blocker = self._check_cycles(
            stability_cycles, conflict_free_rounds, training_iterations
        )
        if not cycles_ok:
            blockers.append(cycles_blocker)
        
        # 6. 阈值检查
        thresholds_ok, threshold_blockers = self._check_thresholds(current_scores)
        if not thresholds_ok:
            blockers.extend(threshold_blockers)
        
        # 7. 冲突检查
        conflict_ok, conflict_blocker = self._check_conflict(recent_conflict)
        if not conflict_ok:
            blockers.append(conflict_blocker)
        
        # 生成决策
        if blockers:
            decision = self._determine_block_reason(blockers)
            confidence = 0.7
            reason = f"深层永久晋升被阻止: {'; '.join(blockers)}"
            recommendations = self._generate_recommendations(blockers)
        else:
            decision = DeepPermanentDecision.PROMOTE_TO_DEEP
            confidence = 0.95
            reason = "通过所有深层永久保护检查，允许进入深层永久"
            recommendations = [
                "建议持续监控长期稳定性",
                "建议记录完整的验证证明",
                "建议设置定期复查机制",
                "建议跟踪训练迭代进度"
            ]
        
        return DeepPermanentResult(
            unit_id=unit_id,
            decision=decision,
            confidence=confidence,
            reason=reason,
            recommendations=recommendations,
            blockers=blockers
        )
    
    def _check_zone(self, current_zone: str) -> tuple[bool, str]:
        """检查区域"""
        if current_zone == "training_normal":
            return True, "当前区域为训练路径正常区"
        else:
            return False, f"当前区域为{current_zone}，深层永久只能由训练路径正常区晋升"
    
    def _check_source(self, source_type: str) -> tuple[bool, str]:
        """检查来源"""
        if source_type in self.allowed_sources:
            return True, f"来源 {source_type} 允许进入深层永久"
        elif source_type in self.blocked_sources:
            return False, f"来源 {source_type} 被阻止进入深层永久"
        else:
            return False, f"来源 {source_type} 未在允许列表中"
    
    def _check_review(self, passed: bool, confidence: float) -> tuple[bool, str]:
        """检查审查"""
        if not passed:
            return False, "未通过强审查"
        if confidence < 0.90:
            return False, f"审查置信度 {confidence:.2f} < 0.90"
        return True, f"强审查通过，置信度 {confidence:.2f}"
    
    def _check_verification(
        self, passed: bool, rounds: int
    ) -> tuple[bool, str]:
        """检查验证"""
        if not passed:
            return False, "未通过验证"
        if rounds < self.cycle_requirements["min_verification_rounds"]:
            return False, f"验证轮数 {rounds} < {self.cycle_requirements['min_verification_rounds']}"
        return True, f"验证通过，完成 {rounds} 轮验证"
    
    def _check_cycles(
        self,
        stability_cycles: int,
        conflict_free_rounds: int,
        training_iterations: int
    ) -> tuple[bool, str]:
        """检查周期"""
        blockers = []
        
        if stability_cycles < self.cycle_requirements["min_stability_cycles"]:
            blockers.append(
                f"稳定周期 {stability_cycles} < {self.cycle_requirements['min_stability_cycles']}"
            )
        
        if conflict_free_rounds < self.cycle_requirements["min_conflict_free_rounds"]:
            blockers.append(
                f"无冲突轮数 {conflict_free_rounds} < {self.cycle_requirements['min_conflict_free_rounds']}"
            )
        
        if training_iterations < self.cycle_requirements["min_training_iterations"]:
            blockers.append(
                f"训练迭代 {training_iterations} < {self.cycle_requirements['min_training_iterations']}"
            )
        
        if blockers:
            return False, "; ".join(blockers)
        return True, "周期要求满足"
    
    def _check_thresholds(self, scores: Dict[str, float]) -> tuple[bool, List[str]]:
        """检查阈值"""
        blockers = []
        
        for metric, threshold in self.deep_thresholds.items():
            score = scores.get(metric, 0)
            if score < threshold:
                blockers.append(f"{metric}分数 {score} < 阈值 {threshold}")
        
        if blockers:
            return False, blockers
        return True, []
    
    def _check_conflict(self, recent_conflict: bool) -> tuple[bool, str]:
        """检查冲突"""
        if recent_conflict:
            return False, "近期存在冲突，不适合进入深层永久"
        return True, "近期无冲突"
    
    def _determine_block_reason(self, blockers: List[str]) -> DeepPermanentDecision:
        """确定阻止原因"""
        blocker_str = " ".join(blockers).lower()
        
        if "区域" in blocker_str or "zone" in blocker_str:
            return DeepPermanentDecision.BLOCKED_BY_ZONE
        elif "审查" in blocker_str or "review" in blocker_str:
            return DeepPermanentDecision.BLOCKED_BY_REVIEW
        elif "验证" in blocker_str or "verification" in blocker_str:
            return DeepPermanentDecision.BLOCKED_BY_VERIFICATION
        elif "周期" in blocker_str or "轮数" in blocker_str or "iteration" in blocker_str:
            return DeepPermanentDecision.BLOCKED_BY_CYCLES
        else:
            return DeepPermanentDecision.BLOCKED_BY_THRESHOLD
    
    def _generate_recommendations(self, blockers: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for blocker in blockers:
            if "区域" in blocker:
                recommendations.append("建议先进入训练路径正常区")
            elif "审查" in blocker:
                recommendations.append("建议补充证据并重新申请强审查")
            elif "验证" in blocker:
                recommendations.append("建议完成更多验证轮次")
            elif "周期" in blocker or "轮数" in blocker:
                recommendations.append("建议继续观察稳定性")
            elif "分数" in blocker:
                recommendations.append("建议提升内容质量")
        
        return recommendations


def demo_deep_gate():
    """深层永久门控演示"""
    print("\n" + "=" * 70)
    print("Deep Permanent Gate Demo - 深层永久门控演示")
    print("=" * 70)
    
    gate = DeepPermanentGate()
    
    # 案例1: 完美候选
    print("\n案例 1: 完美候选")
    result1 = gate.evaluate(
        unit_id="deep_candidate_001",
        current_zone="training_normal",
        current_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
        source_type="training_path",
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_rounds=5,
        stability_cycles=12,
        conflict_free_rounds=10,
        training_iterations=150,
        recent_conflict=False
    )
    
    print(f"  决策: {result1.decision.value}")
    print(f"  置信度: {result1.confidence}")
    print(f"  阻止项: {result1.blockers}")
    
    # 案例2: 周期不足
    print("\n案例 2: 周期不足")
    result2 = gate.evaluate(
        unit_id="deep_candidate_002",
        current_zone="training_normal",
        current_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
        source_type="training_path",
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_rounds=5,
        stability_cycles=6,  # 不足
        conflict_free_rounds=5,
        training_iterations=80,  # 不足
        recent_conflict=False
    )
    
    print(f"  决策: {result2.decision.value}")
    print(f"  阻止项: {result2.blockers}")
    print(f"  建议: {result2.recommendations}")
    
    # 案例3: 来源不符
    print("\n案例 3: 来源不符（用户路径尝试）")
    result3 = gate.evaluate(
        unit_id="deep_candidate_003",
        current_zone="normal",  # 不是 training_normal
        current_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
        source_type="user_path",
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_rounds=5,
        stability_cycles=12,
        conflict_free_rounds=10,
        training_iterations=150,
        recent_conflict=False
    )
    
    print(f"  决策: {result3.decision.value}")
    print(f"  阻止项: {result3.blockers}")


if __name__ == "__main__":
    demo_deep_gate()
