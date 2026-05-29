# Stage 7 深度链路排查报告

**日期**: 2026-04-19  
**问题**: 所有消融测试策略结果完全一致  
**根因确认**: writeback guard 未实际集成到训练流程

---

## 问题现象

| 策略 | target_gain | old_ability_drop | writeback_change |
|------|-------------|------------------|------------------|
| baseline | +17.49% | 2.26% | +13.60% |
| A (特征隔离) | +17.49% | 2.26% | +13.60% |
| B (梯度屏蔽) | +17.49% | 2.26% | +13.60% |
| C (独立优化器) | +17.49% | 2.26% | +13.60% |
| D variants | +17.49% | 2.26% | +13.60% |

**异常信号**: 所有策略的核心数值逐项完全一致

---

## 根因分析

### 第一层检查: guard 是否真的被调用

**检查结果**: ❌ **未调用**

在 `stage7_ablation_test.py` 中:
- ✅ 导入了 `build_feature_isolation_only` 等函数 (line 19-22)
- ❌ **但从未在测试中实际调用**

```python
# _test_strategy_A 函数 (line 147-163)
def _test_strategy_A(self, num_steps: int) -> AblationTestResult:
    test_id = f"ablation_A_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建系统
    system = Stage6SystemOrchestrator(...)
    
    # 应用特征隔离 (简化版: 在训练时应用)
    # 实际实现需要修改 orchestrator  ← 注释说明未实现
    
    # 运行测试 - 直接运行回归检查，没有应用任何 guard
    gate = Stage7RegressionGate(self.config_name)
    result = gate.run_full_regression_check(num_steps)  # ← 和 baseline 完全一样
```

**结论**: 所有策略测试都运行了完全相同的代码路径

### 第二层检查: guard 是否 attach 到训练流程

**检查结果**: ❌ **未 attach**

`stage7_writeback_guard.py` 实现了:
- `FeatureIsolator` - 特征隔离模块
- `GradientShield` - 梯度屏蔽模块  
- `IndependentOptimizerManager` - 独立优化器模块
- `WritebackGuardV2` - 综合保护框架

但 `stage6_orchestrator.py` 中:
- ❌ 没有导入任何 writeback guard 模块
- ❌ 没有调用 `build_writeback_guard()`
- ❌ 没有修改 forward/backward/optimizer.step()

**结论**: guard 对象被创建后没有被挂到 orchestrator 真正使用的位置

### 第三层检查: 评估是否读取正确对象

**检查结果**: ⚠️ **可能有问题**

所有策略使用相同的:
- 实验 ID 生成方式
- checkpoint 保存路径
- 评估协议

但没有证据表明评估读取了错误对象，因为**训练本身就没有差异**。

---

## 问题定位

### 核心问题

**writeback guard 根本没有真正进入训练路径**

虽然实现了:
```python
build_feature_isolation_only()
build_gradient_mask_only()
build_independent_optimizer_only()
build_hybrid_guard()
```

但它们只是"被创建了"，没有真正影响:
- forward
- backward
- optimizer.step()
- writeback update path

### 为什么所有结果一致

1. **不同策略名在跑** - 是的，测试函数被调用了
2. **同一条 baseline 训练链实际上在执行** - **是的，这就是问题**

---

## 修复方案

### 必须完成: 将 writeback guard 集成到 orchestrator

修改 `stage6_orchestrator.py`:

1. **导入 guard**
```python
from stage7_writeback_guard import build_writeback_guard, WritebackGuardMode
```

2. **初始化时创建 guard**
```python
def __init__(self, config):
    # ... 现有代码 ...
    self.writeback_guard = None
    if config.get('writeback_protection_mode'):
        self.writeback_guard = build_writeback_guard(
            self.model,
            self.writeback_head,
            mode=config['writeback_protection_mode'],
        )
```

3. **训练时应用 guard**
```python
def _train_step(self, batch):
    # forward
    features = self.backbone(batch)
    
    # 应用特征隔离
    if self.writeback_guard and self.writeback_guard.feature_isolator:
        features = self.writeback_guard.apply_feature_isolation(features)
    
    # writeback 计算
    writeback_output = self.writeback_head(features)
    
    # backward
    loss = self.compute_loss(writeback_output)
    
    # 应用梯度屏蔽
    if self.writeback_guard and self.writeback_guard.gradient_shield:
        self.writeback_guard.enable_gradient_shield()
    
    loss.backward()
    
    # 应用独立优化器
    if self.writeback_guard and self.writeback_guard.independent_optimizer:
        self.writeback_guard.independent_optimizer.step_writeback(loss)
    else:
        self.optimizer.step()
```

### 必须完成: 修复消融测试

修改 `stage7_ablation_test.py`:

```python
def _test_strategy_A(self, num_steps: int) -> AblationTestResult:
    # 创建带特征隔离配置的系统
    config = self.config.copy()
    config['writeback_protection_mode'] = 'feature_isolation'
    
    system = Stage6SystemOrchestrator(
        experiment_id=test_id,
        custom_config=config,
    )
    
    # 确认 guard 已创建
    assert system.orchestrator.writeback_guard is not None
    assert system.orchestrator.writeback_guard.feature_isolator is not None
    
    # 运行测试
    # ...
```

---

## 验证计划

### 实验 1: 单步差异实验

同一 batch，同一 seed，分别跑 baseline / A / B / C，只做 1 step，打印:
- 梯度 norm
- 参数 delta norm
- writeback head 的 delta

**预期**: 如果 guard 生效，四组应该有明显差异

### 实验 2: 参数哈希实验

训练完每个策略后，输出:
- checkpoint 参数 hash
- writeback head 参数 hash
- optimizer state hash

**预期**: 如果 guard 生效，hash 应该不同

### 实验 3: 强制破坏实验

在 B 策略里把 writeback 相关梯度全清零，看结果是否变化。

**预期**: 如果 guard 生效，结果应该明显不同

---

## 当前结论

**是的，当前"所有数据一致"几乎可以判定为实验链路存在错误，而不是策略真实等价。**

**最可能的根因**: writeback guard 未实际接入训练流程

**优先级**:
1. 🔴 **最高**: 将 writeback guard 集成到 `stage6_orchestrator.py`
2. 🟡 **中**: 修复消融测试，确保策略真正被应用
3. 🟢 **低**: 重新运行消融测试验证

---

**排查完成时间**: 2026-04-19  
**状态**: 根因已确认，等待修复
