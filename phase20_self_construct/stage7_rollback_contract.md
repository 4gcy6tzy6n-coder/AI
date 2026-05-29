# Stage 7 Rollback 完整快照契约

**日期**: 2026-04-19  
**阶段**: Stage 7 - Task 2  
**目标**: 将 rollback 恢复率从 32.6% 提升到 >90%  
**前提**: 不破坏 Stage 6 官方基线 V1.0 的四个必须项

---

## 一、问题定义

### 当前状态

- **rollback 恢复率**: 32.6% (不达标)
- **目标**: > 90%
- **差距**: 57.4个百分点

### 核心问题

Stage 6 的单存 `model.state_dict()` 远远不够。回滚后系统状态要真实恢复。

---

## 二、完整快照范围

### 2.1 必须包含的状态

| 状态类型 | 当前覆盖 | 必须覆盖 | 优先级 |
|----------|----------|----------|--------|
| Model Parameters | ✅ | ✅ | P0 |
| Optimizer State | ❌ | ✅ | P0 |
| Scheduler State | ❌ | ✅ | P0 |
| Random State | ❌ | ✅ | P0 |
| Writeback Runtime | ❌ | ✅ | P1 |
| Governance Runtime | ❌ | ✅ | P1 |
| Evaluation Cache | ❌ | ✅ | P1 |
| Memory Runtime | ❌ | ✅ | P2 |

### 2.2 状态依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                     System State                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Model     │  │  Optimizer   │  │  Scheduler   │      │
│  │  Parameters  │  │    State     │  │    State     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                  ┌─────────────────┐                        │
│                  │   Random State  │                        │
│                  │  (torch, numpy, │                        │
│                  │   python random)│                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐               │
│         ▼                 ▼                 ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Writeback  │  │  Governance  │  │   Memory     │      │
│  │   Runtime    │  │   Runtime    │  │   Runtime    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Evaluation Cache State                  │   │
│  │  (temporary results, cached embeddings, etc.)       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、契约定义

### 3.1 快照契约 (Snapshot Contract)

```python
@dataclass
class RollbackSnapshotV2:
    """
    Stage 7 完整回滚快照
    
    契约保证: 从此快照恢复后，系统状态与创建快照时完全一致
    """
    
    # 1. 模型参数 (P0)
    model_state: Dict[str, torch.Tensor]
    
    # 2. 优化器状态 (P0)
    optimizer_state: Dict[str, Any]
    
    # 3. 学习率调度器状态 (P0)
    scheduler_state: Dict[str, Any]
    
    # 4. 随机数状态 (P0)
    random_states: Dict[str, Any]  # torch, numpy, random
    
    # 5. Writeback 运行时状态 (P1)
    writeback_runtime: Dict[str, Any]
    
    # 6. Governance 运行时状态 (P1)
    governance_runtime: Dict[str, Any]
    
    # 7. 评估缓存状态 (P1)
    evaluation_cache: Dict[str, Any]
    
    # 8. 记忆运行时状态 (P2)
    memory_runtime: Dict[str, Any]
    
    # 元数据
    metadata: Dict[str, Any]  # 时间戳、版本、校验和等
```

### 3.2 恢复契约 (Recovery Contract)

```python
class RollbackRecoveryContract:
    """
    回滚恢复契约
    
    恢复后必须满足:
    1. 模型参数完全一致
    2. 优化器状态完全一致
    3. 随机数序列可复现
    4. 评估结果与快照时一致
    """
    
    @staticmethod
    def verify_recovery(original_snapshot: RollbackSnapshotV2, 
                       recovered_system) -> RecoveryVerificationResult:
        """
        验证恢复是否成功
        
        Returns:
            RecoveryVerificationResult with:
            - model_match: bool
            - optimizer_match: bool
            - random_reproducible: bool
            - evaluation_consistent: bool
            - overall_recovery_rate: float
        """
        pass
```

### 3.3 验证指标

| 验证项 | 通过标准 | 当前 | 目标 |
|--------|----------|------|------|
| 模型参数匹配 | 100% | ? | 100% |
| 优化器状态匹配 | 100% | ? | 100% |
| 随机数可复现 | 是 | ? | 是 |
| 评估结果一致 | 误差 < 1% | ? | < 1% |
| **总体恢复率** | **> 90%** | **32.6%** | **> 90%** |

---

## 四、实现要点

### 4.1 随机数状态捕获

```python
def capture_random_states() -> Dict[str, Any]:
    """捕获所有随机数生成器状态"""
    return {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    }

def restore_random_states(states: Dict[str, Any]):
    """恢复所有随机数生成器状态"""
    torch.set_rng_state(states['torch'])
    if states['torch_cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])
    np.random.set_state(states['numpy'])
    random.setstate(states['random'])
```

### 4.2 运行时状态捕获

```python
def capture_runtime_states(orchestrator) -> Dict[str, Dict[str, Any]]:
    """捕获所有运行时状态"""
    return {
        'writeback': {
            'current_writeback': orchestrator.writeback_head.state_dict(),
            'writeback_cache': getattr(orchestrator, 'writeback_cache', {}),
        },
        'governance': {
            'governance_state': orchestrator.governance_head.state_dict(),
            'policy_cache': getattr(orchestrator, 'policy_cache', {}),
        },
        'memory': {
            'memory_state': orchestrator.memory.state_dict() if hasattr(orchestrator, 'memory') else {},
            'retrieval_cache': getattr(orchestrator, 'retrieval_cache', {}),
        },
    }
```

### 4.3 评估缓存清除

```python
def clear_evaluation_cache():
    """清除所有评估相关缓存"""
    # 清除 torch 缓存
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 清除自定义缓存
    # (具体实现取决于评估器的实现)
```

---

## 五、验证策略

### 5.1 恢复验证流程

```
1. 创建基线快照
   ↓
2. 运行 N 步训练
   ↓
3. 记录训练后状态
   ↓
4. 执行回滚
   ↓
5. 验证恢复状态
   ├─ 模型参数对比
   ├─ 优化器状态对比
   ├─ 随机数复现测试
   └─ 评估结果对比
   ↓
6. 计算恢复率
```

### 5.2 测试用例

| 测试用例 | 描述 | 通过标准 |
|----------|------|----------|
| TC-1 | 单步训练后回滚 | 100% 恢复 |
| TC-2 | 10步训练后回滚 | > 90% 恢复 |
| TC-3 | 回滚后重复评估 | 结果一致 |
| TC-4 | 多次回滚 | 每次恢复率 > 90% |
| TC-5 | 回滚后 old ability | 恢复到基线水平 |

---

## 六、交付物

### 6.1 契约文档

- ✅ `stage7_rollback_contract.md` (本文档)

### 6.2 实现代码

- ⏳ `stage7_rollback_snapshot_v2.py` - 完整快照实现
- ⏳ `stage7_rollback_recovery_test.py` - 恢复测试框架

### 6.3 测试报告

- ⏳ `stage7_rollback_validation_report.md` - 验证报告

---

## 七、下一步行动

1. **实现完整快照捕获** (所有 P0/P1 状态)
2. **实现完整快照恢复**
3. **运行恢复验证测试**
4. **验证不破坏 Stage 6 必须项**

---

**契约定义完成时间**: 2026-04-19  
**状态**: Task 2 完成，准备进入实现阶段
