"""
Self-Learned Parameter Store - 自构建参数存储

第六阶段核心组件：
存储模型自构建学习参数

核心原则：
- 允许受控写回
- 只能接收通过 parameter_promotion_gate 的内容
- 必须来自 deep_permanent
- 有完整的追溯和回滚机制
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime


class SelfLearnedParamStatus(Enum):
    """自构建参数状态"""
    PENDING = "pending"                    # 待写入
    ACTIVE = "active"                      # 已激活
    VALIDATING = "validating"              # 验证中
    FROZEN = "frozen"                      # 已冻结
    ROLLED_BACK = "rolled_back"            # 已回滚
    DEPRECATED = "deprecated"              # 已弃用


@dataclass
class SelfLearnedParam:
    """自构建参数"""
    param_id: str
    source_unit_id: str                    # 来源知识单元
    source_zone: str                       # 来源区域（必须是 deep_permanent）
    
    # 参数值（简化表示）
    value_shape: tuple
    value_hash: str
    
    # 晋升证明
    promotion_scores: Dict[str, float]
    verification_rounds: int
    cross_task_score: float
    long_term_stability: float
    
    # 状态
    status: SelfLearnedParamStatus = SelfLearnedParamStatus.PENDING
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    activated_at: Optional[str] = None
    
    # 追溯信息
    gate_decision: str = ""
    gate_confidence: float = 0.0
    
    # 验证期
    validation_period_days: int = 7
    validation_end_at: Optional[str] = None
    
    # 回滚信息
    rollback_reason: Optional[str] = None
    rollback_at: Optional[str] = None
    
    # 历史
    access_history: List[Dict] = field(default_factory=list)


class SelfLearnedParamStore:
    """
    自构建参数存储
    
    功能：
    1. 存储模型自构建学习参数
    2. 提供受控写入接口
    3. 维护参数追溯信息
    4. 支持回滚机制
    
    写入流程：
    1. 接收来自 promotion_manager 的候选
    2. 验证 gate 决策
    3. 写入参数
    4. 进入验证期
    5. 验证通过后激活
    
    回滚机制：
    - 发现问题时可以回滚
    - 记录回滚原因
    - 保留历史版本
    """
    
    def __init__(self):
        self._storage: Dict[str, SelfLearnedParam] = {}
        self._access_log: List[Dict] = []
        
        # 统计
        self._stats = {
            "total_written": 0,
            "total_activated": 0,
            "total_rolled_back": 0,
            "total_frozen": 0
        }
    
    def write_param(
        self,
        param_id: str,
        source_unit_id: str,
        source_zone: str,
        value_shape: tuple,
        value_hash: str,
        promotion_scores: Dict[str, float],
        verification_rounds: int,
        cross_task_score: float,
        long_term_stability: float,
        gate_decision: str,
        gate_confidence: float
    ) -> SelfLearnedParam:
        """
        写入自构建参数
        
        注意：
        - 只能来自 deep_permanent
        - 必须通过 parameter_promotion_gate
        - 需要完整的晋升证明
        """
        # 验证来源
        if source_zone != "deep_permanent":
            raise ValueError(f"自构建参数只能来自 deep_permanent，当前: {source_zone}")
        
        # 验证 gate 决策
        if gate_decision != "approved":
            raise ValueError(f"必须通过 parameter_promotion_gate，当前决策: {gate_decision}")
        
        # 创建参数
        param = SelfLearnedParam(
            param_id=param_id,
            source_unit_id=source_unit_id,
            source_zone=source_zone,
            value_shape=value_shape,
            value_hash=value_hash,
            promotion_scores=promotion_scores,
            verification_rounds=verification_rounds,
            cross_task_score=cross_task_score,
            long_term_stability=long_term_stability,
            gate_decision=gate_decision,
            gate_confidence=gate_confidence
        )
        
        self._storage[param_id] = param
        self._stats["total_written"] += 1
        
        self._log_access(param_id, "write", "success", source_unit_id)
        
        return param
    
    def activate_param(self, param_id: str) -> bool:
        """
        激活参数
        
        验证期通过后激活
        """
        param = self._storage.get(param_id)
        if not param:
            return False
        
        if param.status != SelfLearnedParamStatus.PENDING:
            return False
        
        param.status = SelfLearnedParamStatus.ACTIVE
        param.activated_at = datetime.utcnow().isoformat()
        
        self._stats["total_activated"] += 1
        self._log_access(param_id, "activate", "success")
        
        return True
    
    def rollback_param(
        self,
        param_id: str,
        reason: str,
        preserve_history: bool = True
    ) -> bool:
        """
        回滚参数
        
        当发现问题时回滚到之前状态
        """
        param = self._storage.get(param_id)
        if not param:
            return False
        
        # 记录回滚信息
        param.status = SelfLearnedParamStatus.ROLLED_BACK
        param.rollback_reason = reason
        param.rollback_at = datetime.utcnow().isoformat()
        
        self._stats["total_rolled_back"] += 1
        self._log_access(param_id, "rollback", "success", reason)
        
        return True
    
    def freeze_param(self, param_id: str, reason: str) -> bool:
        """冻结参数"""
        param = self._storage.get(param_id)
        if not param:
            return False
        
        param.status = SelfLearnedParamStatus.FROZEN
        self._stats["total_frozen"] += 1
        self._log_access(param_id, "freeze", "success", reason)
        
        return True
    
    def get_param(self, param_id: str) -> Optional[SelfLearnedParam]:
        """获取参数"""
        param = self._storage.get(param_id)
        if param:
            self._log_access(param_id, "read", "success")
        return param
    
    def read_param_value(self, param_id: str) -> Optional[Dict]:
        """读取参数值"""
        param = self._storage.get(param_id)
        if not param:
            return None
        
        self._log_access(param_id, "read_value", "success")
        
        return {
            "param_id": param_id,
            "shape": param.value_shape,
            "hash": param.value_hash,
            "status": param.status.value,
            "source": param.source_unit_id
        }
    
    def list_params(
        self,
        status: Optional[SelfLearnedParamStatus] = None
    ) -> List[SelfLearnedParam]:
        """列出参数"""
        params = list(self._storage.values())
        
        if status:
            params = [p for p in params if p.status == status]
        
        return params
    
    def get_active_params(self) -> List[SelfLearnedParam]:
        """获取已激活的参数"""
        return self.list_params(SelfLearnedParamStatus.ACTIVE)
    
    def get_rollback_history(self) -> List[Dict]:
        """获取回滚历史"""
        history = []
        for param in self._storage.values():
            if param.status == SelfLearnedParamStatus.ROLLED_BACK:
                history.append({
                    "param_id": param.param_id,
                    "rollback_at": param.rollback_at,
                    "reason": param.rollback_reason,
                    "source_unit_id": param.source_unit_id
                })
        return history
    
    def _log_access(self, param_id: str, action: str, result: str, source: str = ""):
        """记录访问日志"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "param_id": param_id,
            "action": action,
            "result": result,
            "source": source
        }
        
        self._access_log.append(log_entry)
        
        # 更新参数历史
        param = self._storage.get(param_id)
        if param:
            param.access_history.append(log_entry)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "total_params": len(self._storage),
            "active_count": len(self.get_active_params()),
            "pending_count": len(self.list_params(SelfLearnedParamStatus.PENDING)),
            "frozen_count": len(self.list_params(SelfLearnedParamStatus.FROZEN)),
            "rolled_back_count": len(self.list_params(SelfLearnedParamStatus.ROLLED_BACK))
        }


def demo_self_learned_store():
    """演示自构建参数存储"""
    print("\n" + "=" * 70)
    print("Self-Learned Parameter Store Demo - 自构建参数存储演示")
    print("=" * 70)
    
    store = SelfLearnedParamStore()
    
    # 1. 写入参数（成功）
    print("\n1. 写入参数（通过 gate）:")
    try:
        param = store.write_param(
            param_id="self_learned_001",
            source_unit_id="deep_unit_001",
            source_zone="deep_permanent",
            value_shape=(768, 768),
            value_hash="hash_sl_001",
            promotion_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
            verification_rounds=10,
            cross_task_score=0.92,
            long_term_stability=0.96,
            gate_decision="approved",
            gate_confidence=0.95
        )
        print(f"  写入成功: {param.param_id}")
        print(f"  状态: {param.status.value}")
    except ValueError as e:
        print(f"  写入失败: {e}")
    
    # 2. 尝试写入（来源不符）
    print("\n2. 尝试写入（来源不符）:")
    try:
        store.write_param(
            param_id="self_learned_002",
            source_unit_id="shallow_unit_001",
            source_zone="shallow_permanent",  # 错误来源
            value_shape=(768, 768),
            value_hash="hash_sl_002",
            promotion_scores={"Q": 95, "T": 95, "S": 95, "C": 95, "L": 95},
            verification_rounds=10,
            cross_task_score=0.95,
            long_term_stability=0.98,
            gate_decision="approved",
            gate_confidence=0.95
        )
    except ValueError as e:
        print(f"  预期错误: {e}")
    
    # 3. 激活参数
    print("\n3. 激活参数:")
    success = store.activate_param("self_learned_001")
    print(f"  激活结果: {'成功' if success else '失败'}")
    
    param = store.get_param("self_learned_001")
    if param:
        print(f"  新状态: {param.status.value}")
        print(f"  激活时间: {param.activated_at}")
    
    # 4. 回滚参数
    print("\n4. 回滚参数:")
    success = store.rollback_param(
        param_id="self_learned_001",
        reason="发现能力漂移"
    )
    print(f"  回滚结果: {'成功' if success else '失败'}")
    
    param = store.get_param("self_learned_001")
    if param:
        print(f"  新状态: {param.status.value}")
        print(f"  回滚原因: {param.rollback_reason}")
    
    # 5. 统计信息
    print("\n5. 统计信息:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_self_learned_store()
