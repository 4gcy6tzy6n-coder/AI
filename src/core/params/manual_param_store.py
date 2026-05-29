"""
Manual Parameter Store - 手工参数存储

第六阶段核心组件：
存储基础手工训练参数

核心原则：
- 受保护存储，禁止直接覆盖
- 只允许读取
- 模型自构建内容不能直接写入
- 只能通过受控流程更新
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime


class ParamType(Enum):
    """参数类型"""
    ATTENTION = "attention"
    FEEDFORWARD = "feedforward"
    EMBEDDING = "embedding"
    LAYER_NORM = "layer_norm"
    BIAS = "bias"


class ParamProtectionLevel(Enum):
    """参数保护级别"""
    CRITICAL = "critical"      # 关键参数，完全禁止修改
    STANDARD = "standard"      # 标准参数，需要特殊流程修改
    ADJUSTABLE = "adjustable"  # 可调参数，允许微调


@dataclass
class ProtectedParam:
    """受保护参数"""
    param_id: str
    param_type: ParamType
    protection_level: ParamProtectionLevel
    
    # 参数值（简化表示）
    value_shape: tuple
    value_hash: str
    
    # 元数据
    created_at: str
    created_by: str
    description: str
    
    # 访问控制
    allow_read: bool = True
    allow_write: bool = False
    
    # 历史
    access_history: List[Dict] = field(default_factory=list)


class ManualParamStore:
    """
    手工参数存储
    
    功能：
    1. 存储基础手工训练参数
    2. 提供受保护的读取接口
    3. 禁止直接写入
    4. 记录所有访问历史
    
    保护原则：
    - 基础手工训练参数独立存放
    - 模型自构建学习参数独立存放
    - 禁止直接覆盖 manual_param_store
    - 所有写操作必须通过受控流程
    
    更新流程（如果需要）：
    1. 人工审核
    2. 版本控制
    3. 备份旧参数
    4. 原子性更新
    5. 验证测试
    """
    
    def __init__(self):
        self._storage: Dict[str, ProtectedParam] = {}
        self._access_log: List[Dict] = []
        
        # 初始化一些示例参数
        self._initialize_default_params()
    
    def _initialize_default_params(self):
        """初始化默认参数"""
        default_params = [
            ProtectedParam(
                param_id="attention_weights_layer_0",
                param_type=ParamType.ATTENTION,
                protection_level=ParamProtectionLevel.CRITICAL,
                value_shape=(768, 768),
                value_hash="hash_attention_0",
                created_at="2026-01-01T00:00:00",
                created_by="manual_training",
                description="注意力层0权重 - 关键参数"
            ),
            ProtectedParam(
                param_id="feedforward_weights_layer_0",
                param_type=ParamType.FEEDFORWARD,
                protection_level=ParamProtectionLevel.STANDARD,
                value_shape=(768, 3072),
                value_hash="hash_ff_0",
                created_at="2026-01-01T00:00:00",
                created_by="manual_training",
                description="前馈层0权重 - 标准参数"
            ),
            ProtectedParam(
                param_id="embedding_matrix",
                param_type=ParamType.EMBEDDING,
                protection_level=ParamProtectionLevel.CRITICAL,
                value_shape=(50000, 768),
                value_hash="hash_emb",
                created_at="2026-01-01T00:00:00",
                created_by="manual_training",
                description="词嵌入矩阵 - 关键参数"
            ),
        ]
        
        for param in default_params:
            self._storage[param.param_id] = param
    
    def get_param(self, param_id: str) -> Optional[ProtectedParam]:
        """
        获取参数（只读）
        
        所有读取操作都会被记录
        """
        param = self._storage.get(param_id)
        if param:
            self._log_access(param_id, "read", "success")
        else:
            self._log_access(param_id, "read", "not_found")
        
        return param
    
    def read_param_value(self, param_id: str) -> Optional[Any]:
        """
        读取参数值
        
        返回参数值的引用（只读）
        """
        param = self._storage.get(param_id)
        if not param:
            return None
        
        if not param.allow_read:
            self._log_access(param_id, "read_value", "access_denied")
            return None
        
        self._log_access(param_id, "read_value", "success")
        
        # 返回模拟的参数值
        return {
            "param_id": param_id,
            "shape": param.value_shape,
            "hash": param.value_hash,
            "read_only": True
        }
    
    def write_param(self, param_id: str, value: Any, source: str) -> bool:
        """
        写入参数
        
        默认禁止写入！
        只有通过受控流程才能更新
        """
        self._log_access(param_id, "write", "blocked", source)
        
        # 检查参数是否存在
        param = self._storage.get(param_id)
        if not param:
            return False
        
        # 检查是否允许写入
        if not param.allow_write:
            print(f"  [BLOCKED] 禁止直接写入受保护参数: {param_id}")
            print(f"  来源: {source}")
            print(f"  保护级别: {param.protection_level.value}")
            return False
        
        # 即使允许写入，也需要特殊流程
        if param.protection_level == ParamProtectionLevel.CRITICAL:
            print(f"  [BLOCKED] 关键参数禁止修改: {param_id}")
            return False
        
        return False  # 默认仍然拒绝
    
    def attempt_write(
        self,
        param_id: str,
        value: Any,
        source: str,
        approval_token: Optional[str] = None
    ) -> bool:
        """
        尝试写入（需要审批令牌）
        
        这是受控写入的唯一入口
        """
        self._log_access(param_id, "attempt_write", "checking", source)
        
        # 检查审批令牌
        if not approval_token:
            print(f"  [BLOCKED] 缺少审批令牌: {param_id}")
            return False
        
        if not self._validate_approval_token(approval_token):
            print(f"  [BLOCKED] 无效的审批令牌: {param_id}")
            return False
        
        # 检查参数
        param = self._storage.get(param_id)
        if not param:
            return False
        
        if param.protection_level == ParamProtectionLevel.CRITICAL:
            print(f"  [BLOCKED] 关键参数即使有审批也不能修改: {param_id}")
            return False
        
        # 执行写入（简化实现）
        print(f"  [ALLOWED] 受控写入: {param_id}")
        self._log_access(param_id, "controlled_write", "success", source)
        
        return True
    
    def _validate_approval_token(self, token: str) -> bool:
        """验证审批令牌"""
        # 简化实现，实际应该有复杂的验证逻辑
        return token.startswith("APPROVED_")
    
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
        
        # 同时更新参数的历史
        param = self._storage.get(param_id)
        if param:
            param.access_history.append(log_entry)
    
    def list_params(self) -> List[ProtectedParam]:
        """列出所有参数"""
        return list(self._storage.values())
    
    def get_access_log(self) -> List[Dict]:
        """获取访问日志"""
        return self._access_log.copy()
    
    def get_protection_summary(self) -> Dict[str, Any]:
        """获取保护摘要"""
        total = len(self._storage)
        critical = sum(1 for p in self._storage.values() 
                      if p.protection_level == ParamProtectionLevel.CRITICAL)
        standard = sum(1 for p in self._storage.values() 
                      if p.protection_level == ParamProtectionLevel.STANDARD)
        adjustable = sum(1 for p in self._storage.values() 
                        if p.protection_level == ParamProtectionLevel.ADJUSTABLE)
        
        blocked_writes = sum(1 for log in self._access_log 
                           if log["action"] == "write" and log["result"] == "blocked")
        
        return {
            "total_params": total,
            "critical_params": critical,
            "standard_params": standard,
            "adjustable_params": adjustable,
            "blocked_write_attempts": blocked_writes,
            "protection_active": True
        }


def demo_manual_store():
    """演示手工参数存储"""
    print("\n" + "=" * 70)
    print("Manual Parameter Store Demo - 手工参数存储演示")
    print("=" * 70)
    
    store = ManualParamStore()
    
    # 1. 列出所有参数
    print("\n1. 参数列表:")
    for param in store.list_params():
        print(f"  {param.param_id}: {param.protection_level.value}")
    
    # 2. 读取参数
    print("\n2. 读取参数:")
    value = store.read_param_value("attention_weights_layer_0")
    print(f"  读取结果: {value}")
    
    # 3. 尝试直接写入（应该被阻止）
    print("\n3. 尝试直接写入（应该被阻止）:")
    success = store.write_param(
        param_id="attention_weights_layer_0",
        value={"new": "value"},
        source="model_self_constructed"
    )
    print(f"  写入结果: {'成功' if success else '被阻止'}")
    
    # 4. 尝试无令牌写入（应该被阻止）
    print("\n4. 尝试无令牌写入（应该被阻止）:")
    success = store.attempt_write(
        param_id="feedforward_weights_layer_0",
        value={"new": "value"},
        source="training_loop"
    )
    print(f"  写入结果: {'成功' if success else '被阻止'}")
    
    # 5. 尝试有关键参数的写入（应该被阻止）
    print("\n5. 尝试有关键参数的写入（应该被阻止）:")
    success = store.attempt_write(
        param_id="attention_weights_layer_0",
        value={"new": "value"},
        source="training_loop",
        approval_token="APPROVED_12345"
    )
    print(f"  写入结果: {'成功' if success else '被阻止'}")
    
    # 6. 保护摘要
    print("\n6. 保护摘要:")
    summary = store.get_protection_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_manual_store()
