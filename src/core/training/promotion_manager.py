"""
Promotion Manager - 参数晋升管理器

第六阶段核心组件：
管理从知识库到参数的晋升流程：
- 评估 deep_permanent 对象是否适合参数晋升
- 管理参数晋升候选队列
- 协调参数写回流程
- 维护回滚/冻结机制

原则：
- 只有 deep_permanent 中经过长期稳定验证的对象可晋升
- 默认只允许写入 self_learned_param_store
- 禁止直接覆盖 manual_param_store
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class PromotionStatus(Enum):
    """晋升状态"""
    PENDING = "pending"                    # 待评估
    EVALUATING = "evaluating"              # 评估中
    APPROVED = "approved"                  # 已批准
    REJECTED = "rejected"                  # 已拒绝
    WRITTEN = "written"                    # 已写回
    ROLLED_BACK = "rolled_back"            # 已回滚
    FROZEN = "frozen"                      # 已冻结


class ParamWriteStatus(Enum):
    """参数写回状态"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class PromotionCandidate:
    """晋升候选"""
    candidate_id: str
    source_unit_id: str
    content: str
    source_zone: str  # 只能是 "deep_permanent"
    
    # 质量指标
    quality_scores: Dict[str, float]
    verification_rounds: int
    cross_task_score: float
    long_term_stability: float
    
    # 状态
    status: PromotionStatus = PromotionStatus.PENDING
    write_status: ParamWriteStatus = ParamWriteStatus.PENDING
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    evaluated_at: Optional[str] = None
    written_at: Optional[str] = None
    
    # 评估结果
    evaluation_notes: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


@dataclass
class RollbackRecord:
    """回滚记录"""
    timestamp: str
    candidate_id: str
    reason: str
    previous_state: str
    rollback_type: str  # "full" 或 "partial"


class PromotionManager:
    """
    参数晋升管理器
    
    功能：
    1. 接收 deep_permanent 对象作为晋升候选
    2. 评估是否满足参数晋升门槛
    3. 管理候选队列
    4. 协调参数写回
    5. 维护回滚/冻结机制
    
    晋升门槛（比 deep_permanent 更高）：
    - Q ≥ 90, T ≥ 92, S ≥ 90, C ≥ 95, L ≥ 93
    - 跨任务分数 ≥ 0.90
    - 长期稳定性 ≥ 0.95
    - 验证轮数 ≥ 8
    
    保护原则：
    - 只能写入 self_learned_param_store
    - 禁止直接覆盖 manual_param_store
    - 写回后必须有验证期
    """
    
    def __init__(self):
        # 晋升门槛
        self.param_thresholds = {
            "Q": 90,
            "T": 92,
            "S": 90,
            "C": 95,
            "L": 93,
            "cross_task": 0.90,
            "long_term_stability": 0.95,
            "min_verification_rounds": 8
        }
        
        # 候选队列
        self._candidates: Dict[str, PromotionCandidate] = {}
        
        # 回滚历史
        self._rollback_history: List[RollbackRecord] = []
        
        # 统计
        self._stats = {
            "total_evaluated": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "total_written": 0,
            "total_rolled_back": 0
        }
    
    def submit_candidate(
        self,
        unit_id: str,
        content: str,
        source_zone: str,
        quality_scores: Dict[str, float],
        verification_rounds: int,
        cross_task_score: float,
        long_term_stability: float
    ) -> PromotionCandidate:
        """
        提交晋升候选
        
        注意：只能来自 deep_permanent
        """
        if source_zone != "deep_permanent":
            raise ValueError(f"参数晋升只能来自 deep_permanent，当前来源: {source_zone}")
        
        candidate_id = f"param_candidate_{unit_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        candidate = PromotionCandidate(
            candidate_id=candidate_id,
            source_unit_id=unit_id,
            content=content,
            source_zone=source_zone,
            quality_scores=quality_scores,
            verification_rounds=verification_rounds,
            cross_task_score=cross_task_score,
            long_term_stability=long_term_stability
        )
        
        self._candidates[candidate_id] = candidate
        
        return candidate
    
    def evaluate_candidate(self, candidate_id: str) -> PromotionCandidate:
        """评估候选"""
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise ValueError(f"候选 {candidate_id} 不存在")
        
        candidate.status = PromotionStatus.EVALUATING
        self._stats["total_evaluated"] += 1
        
        notes = []
        passed = True
        
        # 1. 检查质量分数
        for metric, threshold in self.param_thresholds.items():
            if metric in ["cross_task", "long_term_stability", "min_verification_rounds"]:
                continue
            
            score = candidate.quality_scores.get(metric, 0)
            if score < threshold:
                notes.append(f"{metric}分数 {score} < 阈值 {threshold}")
                passed = False
        
        # 2. 检查跨任务分数
        if candidate.cross_task_score < self.param_thresholds["cross_task"]:
            notes.append(f"跨任务分数 {candidate.cross_task_score} < 阈值 {self.param_thresholds['cross_task']}")
            passed = False
        
        # 3. 检查长期稳定性
        if candidate.long_term_stability < self.param_thresholds["long_term_stability"]:
            notes.append(f"长期稳定性 {candidate.long_term_stability} < 阈值 {self.param_thresholds['long_term_stability']}")
            passed = False
        
        # 4. 检查验证轮数
        if candidate.verification_rounds < self.param_thresholds["min_verification_rounds"]:
            notes.append(f"验证轮数 {candidate.verification_rounds} < 阈值 {self.param_thresholds['min_verification_rounds']}")
            passed = False
        
        # 更新状态
        candidate.evaluation_notes = notes
        candidate.evaluated_at = datetime.utcnow().isoformat()
        
        if passed:
            candidate.status = PromotionStatus.APPROVED
            self._stats["total_approved"] += 1
        else:
            candidate.status = PromotionStatus.REJECTED
            candidate.rejection_reason = "; ".join(notes)
            self._stats["total_rejected"] += 1
        
        return candidate
    
    def approve_for_write(self, candidate_id: str) -> bool:
        """批准参数写回"""
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return False
        
        if candidate.status != PromotionStatus.APPROVED:
            return False
        
        # 模拟参数写回
        candidate.write_status = ParamWriteStatus.SUCCESS
        candidate.written_at = datetime.utcnow().isoformat()
        candidate.status = PromotionStatus.WRITTEN
        
        self._stats["total_written"] += 1
        
        return True
    
    def rollback_candidate(
        self,
        candidate_id: str,
        reason: str,
        rollback_type: str = "full"
    ) -> bool:
        """
        回滚候选
        
        当写回后发现问题时触发
        """
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return False
        
        # 创建回滚记录
        record = RollbackRecord(
            timestamp=datetime.utcnow().isoformat(),
            candidate_id=candidate_id,
            reason=reason,
            previous_state=candidate.status.value,
            rollback_type=rollback_type
        )
        
        self._rollback_history.append(record)
        
        # 更新状态
        candidate.status = PromotionStatus.ROLLED_BACK
        candidate.write_status = ParamWriteStatus.FAILED
        candidate.evaluation_notes.append(f"回滚: {reason}")
        
        self._stats["total_rolled_back"] += 1
        
        return True
    
    def freeze_candidate(self, candidate_id: str, reason: str) -> bool:
        """冻结候选"""
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            return False
        
        candidate.status = PromotionStatus.FROZEN
        candidate.evaluation_notes.append(f"冻结: {reason}")
        
        return True
    
    def get_candidate(self, candidate_id: str) -> Optional[PromotionCandidate]:
        """获取候选"""
        return self._candidates.get(candidate_id)
    
    def get_approved_candidates(self) -> List[PromotionCandidate]:
        """获取已批准的候选列表"""
        return [
            c for c in self._candidates.values()
            if c.status == PromotionStatus.APPROVED
        ]
    
    def get_written_candidates(self) -> List[PromotionCandidate]:
        """获取已写回的候选列表"""
        return [
            c for c in self._candidates.values()
            if c.status == PromotionStatus.WRITTEN
        ]
    
    def get_rollback_history(self) -> List[RollbackRecord]:
        """获取回滚历史"""
        return self._rollback_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "pending_count": sum(1 for c in self._candidates.values() if c.status == PromotionStatus.PENDING),
            "approved_count": sum(1 for c in self._candidates.values() if c.status == PromotionStatus.APPROVED),
            "rejected_count": sum(1 for c in self._candidates.values() if c.status == PromotionStatus.REJECTED),
            "written_count": sum(1 for c in self._candidates.values() if c.status == PromotionStatus.WRITTEN),
            "rolled_back_count": sum(1 for c in self._candidates.values() if c.status == PromotionStatus.ROLLED_BACK),
            "total_candidates": len(self._candidates)
        }


def demo_promotion_manager():
    """演示参数晋升管理"""
    print("\n" + "=" * 70)
    print("Promotion Manager Demo - 参数晋升管理演示")
    print("=" * 70)
    
    manager = PromotionManager()
    
    # 案例1: 高质量候选
    print("\n案例 1: 高质量候选")
    candidate1 = manager.submit_candidate(
        unit_id="deep_001",
        content="高质量知识",
        source_zone="deep_permanent",
        quality_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
        verification_rounds=10,
        cross_task_score=0.92,
        long_term_stability=0.96
    )
    
    print(f"  提交候选: {candidate1.candidate_id}")
    
    result1 = manager.evaluate_candidate(candidate1.candidate_id)
    print(f"  评估结果: {result1.status.value}")
    
    if result1.status.value == "approved":
        success = manager.approve_for_write(result1.candidate_id)
        print(f"  参数写回: {'成功' if success else '失败'}")
    
    # 案例2: 低质量候选
    print("\n案例 2: 低质量候选")
    candidate2 = manager.submit_candidate(
        unit_id="deep_002",
        content="低质量知识",
        source_zone="deep_permanent",
        quality_scores={"Q": 85, "T": 88, "S": 85, "C": 90, "L": 89},
        verification_rounds=5,
        cross_task_score=0.80,
        long_term_stability=0.85
    )
    
    print(f"  提交候选: {candidate2.candidate_id}")
    
    result2 = manager.evaluate_candidate(candidate2.candidate_id)
    print(f"  评估结果: {result2.status.value}")
    print(f"  拒绝原因: {result2.rejection_reason}")
    
    # 案例3: 回滚演示
    print("\n案例 3: 回滚演示")
    rollback_success = manager.rollback_candidate(
        candidate_id=candidate1.candidate_id,
        reason="写回后发现能力漂移",
        rollback_type="full"
    )
    print(f"  回滚结果: {'成功' if rollback_success else '失败'}")
    
    # 统计
    print("\n统计信息:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_promotion_manager()
