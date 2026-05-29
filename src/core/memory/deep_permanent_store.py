"""
Deep Permanent Store - 深层永久存储

第五阶段核心组件：
- 仅接收训练路径长期验证后的高质量知识
- 比浅层永久更高的门槛
- 更强的保护机制
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class DeepPermanentStatus(Enum):
    """深层永久状态"""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    DOWNGRADED = "downgraded"
    ARCHIVED = "archived"


@dataclass
class VerificationProof:
    """验证证明"""
    verification_rounds: int
    avg_score: float
    consistency_score: float
    cross_task_score: float
    long_term_stability: float
    verified_at: str
    verifier_id: str


@dataclass
class TrainingOrigin:
    """训练来源"""
    training_iteration: int
    source_dataset: str
    review_passed: bool
    review_confidence: float
    verification_passed: bool
    verification_proof: VerificationProof


@dataclass
class DeepPermanentEntry:
    """深层永久条目"""
    unit_id: str
    content: str
    core_meaning: str
    
    # 晋升信息
    promoted_at: str
    promoted_from: str  # 只能是 "training_normal"
    
    # 训练来源
    training_origin: TrainingOrigin
    
    # 质量分数
    promotion_scores: Dict[str, float]
    
    # 状态
    status: str = DeepPermanentStatus.ACTIVE.value
    
    # 保护历史
    conflict_history: List[Dict] = field(default_factory=list)
    downgrade_history: List[Dict] = field(default_factory=list)
    review_history: List[Dict] = field(default_factory=list)
    
    # 元数据
    access_count: int = 0
    last_accessed: Optional[str] = None


class DeepPermanentStore:
    """
    深层永久存储
    
    特性：
    1. 仅允许训练路径晋升
    2. 需要强审查和验证证明
    3. 更高的稳定性要求
    4. 更强的保护（禁止直接删除）
    5. 降级只能回训练路径，不能跨层
    
    与浅层永久的区别：
    - 来源：仅限训练路径（vs 用户路径/检索）
    - 门槛：更高（10周期 vs 5周期）
    - 审查：强审查 + 多轮验证
    - 保护：更严格
    """
    
    def __init__(self):
        self._storage: Dict[str, DeepPermanentEntry] = {}
        self._access_log: List[Dict] = []
    
    def promote_to_deep_permanent(
        self,
        unit_id: str,
        content: str,
        core_meaning: str,
        promotion_scores: Dict[str, float],
        training_origin: TrainingOrigin,
        review_confidence: float
    ) -> DeepPermanentEntry:
        """
        晋升到深层永久
        
        注意：
        - 只能由训练路径长期正常区晋升
        - 必须有验证证明
        - 必须通过强审查
        """
        if unit_id in self._storage:
            raise ValueError(f"对象 {unit_id} 已存在于深层永久层")
        
        # 验证来源
        if not training_origin.verification_passed:
            raise ValueError(f"对象 {unit_id} 未通过验证")
        
        if not training_origin.review_passed:
            raise ValueError(f"对象 {unit_id} 未通过强审查")
        
        # 验证分数门槛
        min_score = min(promotion_scores.values())
        if min_score < 0.85:
            raise ValueError(f"对象 {unit_id} 最低分 {min_score:.2f} < 0.85")
        
        entry = DeepPermanentEntry(
            unit_id=unit_id,
            content=content,
            core_meaning=core_meaning,
            promoted_at=datetime.utcnow().isoformat(),
            promoted_from="training_normal",
            training_origin=training_origin,
            promotion_scores=promotion_scores,
            status=DeepPermanentStatus.ACTIVE.value
        )
        
        self._storage[unit_id] = entry
        self._log_access(unit_id, "promote", "对象晋升到深层永久")
        
        return entry
    
    def get(self, unit_id: str) -> Optional[DeepPermanentEntry]:
        """获取条目"""
        entry = self._storage.get(unit_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow().isoformat()
            self._log_access(unit_id, "access", "访问深层永久对象")
        return entry
    
    def is_in_deep_permanent(self, unit_id: str) -> bool:
        """检查是否在深层永久"""
        return unit_id in self._storage
    
    def mark_for_review(self, unit_id: str, reason: str) -> bool:
        """标记为待审查"""
        entry = self._storage.get(unit_id)
        if not entry:
            return False
        
        entry.status = DeepPermanentStatus.UNDER_REVIEW.value
        entry.review_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "action": "marked_for_review"
        })
        
        self._log_access(unit_id, "review", f"标记待审查: {reason}")
        return True
    
    def downgrade_to_training(
        self,
        unit_id: str,
        target_zone: str,
        reason: str
    ) -> Dict:
        """
        降级回训练路径
        
        注意：
        - 只能降级回训练路径
        - 不能跨层降级到用户路径
        """
        entry = self._storage.get(unit_id)
        if not entry:
            return {"success": False, "error": "对象不存在"}
        
        # 创建降级记录
        downgrade_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_status": entry.status,
            "to_target": target_zone,
            "reason": reason,
            "source_restriction": "training_path_only"
        }
        
        entry.downgrade_history.append(downgrade_record)
        entry.status = DeepPermanentStatus.DOWNGRADED.value
        
        self._log_access(unit_id, "downgrade", f"降级到训练路径 {target_zone}: {reason}")
        
        return {
            "success": True,
            "downgrade_record": downgrade_record,
            "total_downgrades": len(entry.downgrade_history)
        }
    
    def get_verification_proof(self, unit_id: str) -> Optional[VerificationProof]:
        """获取验证证明"""
        entry = self._storage.get(unit_id)
        if entry:
            return entry.training_origin.verification_proof
        return None
    
    def get_training_origin(self, unit_id: str) -> Optional[TrainingOrigin]:
        """获取训练来源"""
        entry = self._storage.get(unit_id)
        if entry:
            return entry.training_origin
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._storage)
        active = sum(1 for e in self._storage.values() if e.status == DeepPermanentStatus.ACTIVE.value)
        under_review = sum(1 for e in self._storage.values() if e.status == DeepPermanentStatus.UNDER_REVIEW.value)
        downgraded = sum(1 for e in self._storage.values() if e.status == DeepPermanentStatus.DOWNGRADED.value)
        
        return {
            "total_entries": total,
            "active": active,
            "under_review": under_review,
            "downgraded": downgraded,
            "total_accesses": len(self._access_log)
        }
    
    def _log_access(self, unit_id: str, action: str, details: str):
        """记录访问日志"""
        self._access_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "unit_id": unit_id,
            "action": action,
            "details": details
        })


def demo_deep_permanent():
    """深层永久演示"""
    print("\n" + "=" * 70)
    print("Deep Permanent Store Demo - 深层永久存储演示")
    print("=" * 70)
    
    store = DeepPermanentStore()
    
    # 创建验证证明
    proof = VerificationProof(
        verification_rounds=5,
        avg_score=0.92,
        consistency_score=0.95,
        cross_task_score=0.90,
        long_term_stability=0.93,
        verified_at=datetime.utcnow().isoformat(),
        verifier_id="verifier_v1"
    )
    
    # 创建训练来源
    origin = TrainingOrigin(
        training_iteration=150,
        source_dataset="curated_v2",
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_proof=proof
    )
    
    # 晋升到深层永久
    print("\n晋升到深层永久:")
    entry = store.promote_to_deep_permanent(
        unit_id="deep_001",
        content="深度学习是机器学习的一个分支，使用多层神经网络。",
        core_meaning="深度学习是使用多层神经网络的机器学习方法",
        promotion_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
        training_origin=origin,
        review_confidence=0.95
    )
    
    print(f"  对象ID: {entry.unit_id}")
    print(f"  状态: {entry.status}")
    print(f"  验证轮数: {entry.training_origin.verification_proof.verification_rounds}")
    print(f"  平均分数: {entry.training_origin.verification_proof.avg_score:.2f}")
    
    # 获取统计
    print("\n统计信息:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 标记审查
    print("\n标记待审查:")
    store.mark_for_review("deep_001", "新证据需要复核")
    entry = store.get("deep_001")
    print(f"  新状态: {entry.status}")
    
    # 降级
    print("\n降级回训练路径:")
    result = store.downgrade_to_training(
        unit_id="deep_001",
        target_zone="training_review",
        reason="需要重新验证"
    )
    print(f"  降级成功: {result['success']}")
    print(f"  降级历史数: {result['total_downgrades']}")
    
    entry = store.get("deep_001")
    print(f"  最终状态: {entry.status}")


if __name__ == "__main__":
    demo_deep_permanent()
