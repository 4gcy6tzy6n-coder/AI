# Stage 6 Phase 3 Execution Plan

**阶段**: Stage 6.3 - 系统级评估  
**状态**: Task 1 完成，准备 Task 2/3  
**日期**: 2026-04-19

---

## Task 1 完成情况

### 已交付: Phase 3 统一官方系统壳

**文件**: [stage6_system_orchestrator.py](stage6_system_orchestrator.py)

#### 1. 系统总入口 ✓

```python
class Stage6SystemOrchestrator:
    def __init__(self, experiment_id, custom_config):
        # 配置加载 (官方基线 V1.0)
        # 模型加载
        # 记忆/治理模块加载
        # 评估器接口挂载
        # 日志与实验编号
```

#### 2. 系统级运行拓扑 ✓

```python
def run_system_loop(self, queries, max_steps):
    # 输入 → 检索/上下文 → 推理 → writeback → rollback → 评估 → 落盘
    for step_num in range(max_steps):
        trace = self._execute_step(step_num, query)
```

#### 3. 环境一致性规范 ✓

```python
class EnvironmentSpec:
    BASELINE_VERSION = '1.0'
    CHECKPOINT_FORMAT = '{exp_id}_{timestamp}_{step}'
    DATA_SPLIT_SEED = 42
    TORCH_SEED = 42
    NUMPY_SEED = 42
    RANDOM_SEED = 42
```

#### 4. 监控与追踪 ✓

```python
@dataclass
class StepTrace:
    step_number, timestamp, state
    input_query, gap_detected, policy_selected
    target_improvement, old_ability_drop, writeback_change
    rollback_triggered, rollback_success
    error_message
```

#### 5. 预留接口 ✓

- `run_official_phase3_test()` - 官方测试入口
- `test_rollback_recovery()` - rollback 测试入口
- `export_report()` - 报告生成接口
- 回调注册: `register_step_callback()`, `register_error_callback()`

---

## Task 2: 端到端行为测试 (待执行)

### 目标

在系统级环境中验证基线配置的端到端行为。

### 测试场景

#### 2.1 单轮查询测试

```python
# 测试文件: stage6_end_to_end_test.py

class SingleTurnTest:
    """单轮查询测试"""
    
    def test_basic_query(self):
        """基础查询"""
        pass
    
    def test_gap_detection(self):
        """GAP 检测"""
        pass
    
    def test_policy_selection(self):
        """策略选择"""
        pass
    
    def test_writeback(self):
        """Writeback 执行"""
        pass
```

#### 2.2 多轮对话测试

```python
class MultiTurnTest:
    """多轮对话测试"""
    
    def test_conversation_flow(self):
        """对话流程"""
        pass
    
    def test_context_retention(self):
        """上下文保持"""
        pass
    
    def test_capability_accumulation(self):
        """能力累积"""
        pass
```

#### 2.3 复杂场景测试

```python
class ComplexScenarioTest:
    """复杂场景测试"""
    
    def test_multi_gap_scenario(self):
        """多 GAP 场景"""
        pass
    
    def test_rollback_scenario(self):
        """回滚场景"""
        pass
    
    def test_stress_scenario(self):
        """压力场景"""
        pass
```

### 验收标准

- [ ] 单轮查询成功率 > 95%
- [ ] 多轮对话连贯性 > 90%
- [ ] 复杂场景处理率 > 80%

---

## Task 3: 长期稳定性测试 (待执行)

### 目标

验证系统在长时间运行下的稳定性。

### 测试内容

#### 3.1 100+ 步连续晋升测试

```python
# 测试文件: stage6_stability_test.py

class LongRunningTest:
    """长期运行测试"""
    
    def test_100_steps(self):
        """100 步连续运行"""
        system = Stage6SystemOrchestrator()
        queries = generate_100_queries()
        metrics = system.run_system_loop(queries, max_steps=100)
        
        # 验证:
        # - 无崩溃
        # - 目标能力持续提升
        # - 旧能力保持在阈值内
```

#### 3.2 内存泄漏检查

```python
def test_memory_leak(self):
    """内存泄漏测试"""
    # 运行 100 步
    # 监控内存使用
    # 验证无持续增长
```

#### 3.3 性能衰减检查

```python
def test_performance_decay(self):
    """性能衰减测试"""
    # 记录每步执行时间
    # 验证无显著衰减
```

### 验收标准

- [ ] 100 步无崩溃
- [ ] 内存增长 < 10%
- [ ] 执行时间衰减 < 20%

---

## Task 4: 已知限制优化 (可选)

### 4.1 Writeback 保护优化

**当前**: 5.6-12% 变化  
**目标**: < 5%

**方案**:
1. 调整 KL 权重
2. 优化 replay 比例
3. 重写 writeback isolation (完全重写，不基于失败分支)

### 4.2 Rollback 恢复优化

**当前**: 32.6% 恢复率  
**目标**: > 90%

**方案**:
1. 改进快照机制
2. 优化恢复策略
3. 增加恢复验证

---

## Task 5: 最终验收 (待执行)

### 验收流程

```python
def run_final_acceptance():
    """最终验收"""
    
    # 1. 系统级评估
    system = Stage6SystemOrchestrator(experiment_id='final_acceptance')
    
    # 2. 运行完整测试套件
    run_single_turn_tests()
    run_multi_turn_tests()
    run_complex_scenario_tests()
    run_stability_tests()
    
    # 3. 生成验收报告
    report = generate_acceptance_report()
    
    # 4. 判断验收结果
    if report['all_pass']:
        print("Stage 6 验收通过！")
    else:
        print("需要进一步修复")
```

### 验收标准

| 指标 | 目标 | 权重 |
|------|------|------|
| 目标提升 | > 10% | 必须 |
| 旧能力掉落 | < 15% | 必须 |
| 端到端成功率 | > 90% | 必须 |
| 100 步稳定性 | 通过 | 必须 |
| writeback 变化 | < 5% | 可选 |
| rollback 恢复 | > 90% | 可选 |

---

## 执行顺序

```
Task 1 (完成)
    ↓
Task 2: 端到端行为测试
    ↓
Task 3: 长期稳定性测试
    ↓
Task 4: 已知限制优化 (可选，可并行)
    ↓
Task 5: 最终验收
```

---

## 下一步行动

**立即开始 Task 2: 端到端行为测试**

建议产出:
1. `stage6_end_to_end_test.py` - 端到端测试套件
2. `stage6_test_data.py` - 测试数据生成
3. `stage6_phase3_task2_report.md` - Task 2 进展报告

是否开始 Task 2?
