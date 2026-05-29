"""
Verifier - 验证节点

第五阶段核心组件：
负责训练路径的多轮验证：
- 多轮一致性验证
- 跨任务调用验证
- 长期稳定性验证
- 是否满足深层永久最小周期
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class VerificationType(Enum):
    """验证类型"""
    CONSISTENCY = "consistency"           # 一致性验证
    CROSS_TASK = "cross_task"            # 跨任务验证
    LONG_TERM_STABILITY = "long_term_stability"  # 长期稳定性验证
    BOUNDARY_STRESS = "boundary_stress"  # 边界压力测试
    ADVERSARIAL = "adversarial"          # 对抗性验证


class VerificationStatus(Enum):
    """验证状态"""
    PASSED = "passed"                    # 通过
    FAILED = "failed"                    # 失败
    PENDING = "pending"                  # 待验证
    INCONCLUSIVE = "inconclusive"        # 无结论


@dataclass
class VerificationRound:
    """验证轮次"""
    round_number: int
    verification_type: VerificationType
    status: VerificationStatus
    score: float
    details: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class VerificationResult:
    """验证结果"""
    case_id: str
    total_rounds: int
    passed_rounds: int
    failed_rounds: int
    overall_status: VerificationStatus
    rounds: List[VerificationRound] = field(default_factory=list)
    meets_deep_permanent_requirements: bool = False
    verified_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class StabilityRecord:
    """稳定性记录"""
    cycle: int
    score: float
    variance: float
    timestamp: str


class Verifier:
    """
    验证器
    
    功能：
    1. 多轮一致性验证 - 检查同一内容多次调用的结果一致性
    2. 跨任务验证 - 检查在不同任务场景下的表现
    3. 长期稳定性验证 - 跟踪多周期的稳定性
    4. 边界压力测试 - 测试边界条件的处理能力
    5. 对抗性验证 - 测试对噪声和攻击的鲁棒性
    
    原则：
    - 深层永久需要至少5轮验证
    - 每轮验证必须达到一定分数
    - 长期稳定性需要跟踪多周期
    """
    
    def __init__(self):
        # 验证阈值
        self.thresholds = {
            "min_rounds": 3,
            "deep_permanent_min_rounds": 5,
            "min_score_per_round": 0.75,
            "min_overall_score": 0.80,
            "deep_permanent_min_score": 0.90,
            "max_variance": 0.15,
            "min_stability_cycles": 10
        }
        
        # 稳定性历史
        self.stability_history: Dict[str, List[StabilityRecord]] = {}
    
    def verify(
        self,
        case_id: str,
        content: str,
        verification_types: List[VerificationType] = None,
        min_rounds: int = None,
        target_deep_permanent: bool = False
    ) -> VerificationResult:
        """
        执行完整验证
        
        Args:
            case_id: 案例ID
            content: 内容
            verification_types: 验证类型列表
            min_rounds: 最少验证轮数
            target_deep_permanent: 是否目标深层永久
        """
        if verification_types is None:
            verification_types = [
                VerificationType.CONSISTENCY,
                VerificationType.CROSS_TASK,
                VerificationType.LONG_TERM_STABILITY
            ]
        
        # 确定最少轮数
        if min_rounds is None:
            min_rounds = (
                self.thresholds["deep_permanent_min_rounds"]
                if target_deep_permanent
                else self.thresholds["min_rounds"]
            )
        
        rounds = []
        passed = 0
        failed = 0
        
        # 执行多轮验证
        for i in range(min_rounds):
            for vtype in verification_types:
                round_result = self._execute_verification_round(
                    case_id, content, i + 1, vtype
                )
                rounds.append(round_result)
                
                if round_result.status == VerificationStatus.PASSED:
                    passed += 1
                elif round_result.status == VerificationStatus.FAILED:
                    failed += 1
        
        # 计算总体状态
        total = len(rounds)
        pass_rate = passed / total if total > 0 else 0
        
        if pass_rate >= 0.95:
            overall_status = VerificationStatus.PASSED
        elif pass_rate >= 0.75:
            overall_status = VerificationStatus.PASSED
        elif pass_rate >= 0.50:
            overall_status = VerificationStatus.INCONCLUSIVE
        else:
            overall_status = VerificationStatus.FAILED
        
        # 检查是否满足深层永久要求
        meets_deep = self._check_deep_permanent_requirements(
            rounds, target_deep_permanent
        )
        
        return VerificationResult(
            case_id=case_id,
            total_rounds=total,
            passed_rounds=passed,
            failed_rounds=failed,
            overall_status=overall_status,
            rounds=rounds,
            meets_deep_permanent_requirements=meets_deep
        )
    
    def _execute_verification_round(
        self,
        case_id: str,
        content: str,
        round_number: int,
        vtype: VerificationType
    ) -> VerificationRound:
        """执行单轮验证"""
        
        if vtype == VerificationType.CONSISTENCY:
            return self._verify_consistency(case_id, content, round_number)
        elif vtype == VerificationType.CROSS_TASK:
            return self._verify_cross_task(case_id, content, round_number)
        elif vtype == VerificationType.LONG_TERM_STABILITY:
            return self._verify_long_term_stability(case_id, content, round_number)
        elif vtype == VerificationType.BOUNDARY_STRESS:
            return self._verify_boundary_stress(case_id, content, round_number)
        elif vtype == VerificationType.ADVERSARIAL:
            return self._verify_adversarial(case_id, content, round_number)
        else:
            return VerificationRound(
                round_number=round_number,
                verification_type=vtype,
                status=VerificationStatus.INCONCLUSIVE,
                score=0.0,
                details="未知验证类型"
            )
    
    def _verify_consistency(
        self, case_id: str, content: str, round_number: int
    ) -> VerificationRound:
        """一致性验证"""
        # 模拟：检查多次调用结果是否一致
        # 实际实现应该调用模型多次并比较结果
        
        # 简化：基于内容长度计算一致性分数
        consistency_score = min(1.0, len(content) / 100)
        
        if consistency_score >= self.thresholds["min_score_per_round"]:
            status = VerificationStatus.PASSED
            details = f"一致性验证通过，分数 {consistency_score:.2f}"
        else:
            status = VerificationStatus.FAILED
            details = f"一致性验证失败，分数 {consistency_score:.2f}"
        
        return VerificationRound(
            round_number=round_number,
            verification_type=VerificationType.CONSISTENCY,
            status=status,
            score=consistency_score,
            details=details
        )
    
    def _verify_cross_task(
        self, case_id: str, content: str, round_number: int
    ) -> VerificationRound:
        """跨任务验证"""
        # 模拟：检查在不同任务场景下的表现
        
        # 简化：假设跨任务表现良好
        cross_task_score = 0.85
        
        if cross_task_score >= self.thresholds["min_score_per_round"]:
            status = VerificationStatus.PASSED
            details = f"跨任务验证通过，分数 {cross_task_score:.2f}"
        else:
            status = VerificationStatus.FAILED
            details = f"跨任务验证失败，分数 {cross_task_score:.2f}"
        
        return VerificationRound(
            round_number=round_number,
            verification_type=VerificationType.CROSS_TASK,
            status=status,
            score=cross_task_score,
            details=details
        )
    
    def _verify_long_term_stability(
        self, case_id: str, content: str, round_number: int
    ) -> VerificationRound:
        """长期稳定性验证"""
        # 获取历史稳定性记录
        history = self.stability_history.get(case_id, [])
        
        if len(history) >= self.thresholds["min_stability_cycles"]:
            # 计算方差
            scores = [r.score for r in history[-10:]]
            variance = max(scores) - min(scores) if scores else 1.0
            
            if variance <= self.thresholds["max_variance"]:
                status = VerificationStatus.PASSED
                details = f"长期稳定性验证通过，方差 {variance:.2f}"
                score = 0.90
            else:
                status = VerificationStatus.FAILED
                details = f"长期稳定性验证失败，方差 {variance:.2f}"
                score = 0.60
        else:
            # 记录不足，视为待定
            status = VerificationStatus.PENDING
            details = f"稳定性记录不足 ({len(history)}/{self.thresholds['min_stability_cycles']})"
            score = 0.50
        
        return VerificationRound(
            round_number=round_number,
            verification_type=VerificationType.LONG_TERM_STABILITY,
            status=status,
            score=score,
            details=details
        )
    
    def _verify_boundary_stress(
        self, case_id: str, content: str, round_number: int
    ) -> VerificationRound:
        """边界压力测试"""
        # 模拟边界条件测试
        
        boundary_score = 0.80
        
        if boundary_score >= self.thresholds["min_score_per_round"]:
            status = VerificationStatus.PASSED
            details = f"边界压力测试通过，分数 {boundary_score:.2f}"
        else:
            status = VerificationStatus.FAILED
            details = f"边界压力测试失败，分数 {boundary_score:.2f}"
        
        return VerificationRound(
            round_number=round_number,
            verification_type=VerificationType.BOUNDARY_STRESS,
            status=status,
            score=boundary_score,
            details=details
        )
    
    def _verify_adversarial(
        self, case_id: str, content: str, round_number: int
    ) -> VerificationRound:
        """对抗性验证"""
        # 模拟对抗性测试
        
        adversarial_score = 0.75
        
        if adversarial_score >= self.thresholds["min_score_per_round"]:
            status = VerificationStatus.PASSED
            details = f"对抗性验证通过，分数 {adversarial_score:.2f}"
        else:
            status = VerificationStatus.FAILED
            details = f"对抗性验证失败，分数 {adversarial_score:.2f}"
        
        return VerificationRound(
            round_number=round_number,
            verification_type=VerificationType.ADVERSARIAL,
            status=status,
            score=adversarial_score,
            details=details
        )
    
    def _check_deep_permanent_requirements(
        self, rounds: List[VerificationRound], target_deep: bool
    ) -> bool:
        """检查是否满足深层永久要求"""
        if not target_deep:
            return False
        
        # 检查轮数
        if len(rounds) < self.thresholds["deep_permanent_min_rounds"]:
            return False
        
        # 检查分数
        avg_score = sum(r.score for r in rounds) / len(rounds)
        if avg_score < self.thresholds["deep_permanent_min_score"]:
            return False
        
        # 检查失败率
        failed_count = sum(1 for r in rounds if r.status == VerificationStatus.FAILED)
        if failed_count > len(rounds) * 0.1:  # 允许10%失败
            return False
        
        return True
    
    def record_stability(
        self, case_id: str, cycle: int, score: float, variance: float
    ):
        """记录稳定性数据"""
        if case_id not in self.stability_history:
            self.stability_history[case_id] = []
        
        self.stability_history[case_id].append(StabilityRecord(
            cycle=cycle,
            score=score,
            variance=variance,
            timestamp=datetime.utcnow().isoformat()
        ))
    
    def get_stability_history(self, case_id: str) -> List[StabilityRecord]:
        """获取稳定性历史"""
        return self.stability_history.get(case_id, [])


def demo_verification():
    """验证演示"""
    print("\n" + "=" * 70)
    print("Verifier Demo - 验证节点演示")
    print("=" * 70)
    
    verifier = Verifier()
    
    # 案例1: 普通验证
    print("\n案例 1: 普通验证 (3轮)")
    result1 = verifier.verify(
        case_id="verify_001",
        content="人工智能是指由人制造出来的系统所表现出来的智能。",
        min_rounds=3,
        target_deep_permanent=False
    )
    
    print(f"  总轮数: {result1.total_rounds}")
    print(f"  通过: {result1.passed_rounds}")
    print(f"  失败: {result1.failed_rounds}")
    print(f"  总体状态: {result1.overall_status.value}")
    print(f"  满足深层永久: {result1.meets_deep_permanent_requirements}")
    
    # 案例2: 深层永久验证
    print("\n案例 2: 深层永久验证 (5轮)")
    
    # 先记录一些稳定性历史
    for i in range(12):
        verifier.record_stability("verify_002", i, 0.88 + (i % 3) * 0.02, 0.05)
    
    result2 = verifier.verify(
        case_id="verify_002",
        content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。",
        min_rounds=5,
        target_deep_permanent=True
    )
    
    print(f"  总轮数: {result2.total_rounds}")
    print(f"  通过: {result2.passed_rounds}")
    print(f"  失败: {result2.failed_rounds}")
    print(f"  总体状态: {result2.overall_status.value}")
    print(f"  满足深层永久: {result2.meets_deep_permanent_requirements}")


if __name__ == "__main__":
    demo_verification()
