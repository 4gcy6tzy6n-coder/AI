# EXP-6B-001: Writeback Isolation 修复实验

**状态**: ❌ **失败实验 - 已归档**  
**日期**: 2026-04-19  
**实验目的**: 通过架构隔离解决 writeback 保护问题

---

## 实验背景

Stage 6 第二阶段发现 writeback 保护不足 (5.6-12% 变化，目标 <5%)。

假设: writeback 与其他 head 共享表示，导致 KL 约束无法完全隔离。

方案: 实现独立的 writeback head，冻结其参数防止被影响。

---

## 实验实现

**实现文件**: [stage6_writeback_isolation_fix.py](../stage6_writeback_isolation_fix.py)

**核心设计**:
```python
class IsolatedWritebackHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_writeback_types):
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_writeback_types)
```

**关键机制**:
- 独立的 feature extractor
- freeze/unfreeze 控制
- 兼容原始接口

---

## 实验结果

### 消融测试 (初步)

| 配置 | Target Gain | 初步结论 |
|------|-------------|----------|
| baseline | +14.03% | 基线 |
| writeback_only | +17.66% | 提升 3.63% |

**初步判断**: 架构修复能提升 target gain

### 最终验收 (使用统一协议)

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| 目标提升 | **-3.81%** | >10% | ❌ 严重不达标 |
| 旧能力掉落 | **50.00%** | <15% | ❌ 计算异常 |
| writeback 变化 | **36.77%** | <5% | ❌ 保护失效 |
| rollback 恢复 | 0.0% | >90% | ❌ 恢复失败 |

---

## 失败原因分析

### 1. 权重复制失败

```python
# 尝试复制原始权重到新结构
try:
    original_state = self.original_writeback_head.state_dict()
    self.isolated_writeback_head.load_state_dict(original_state, strict=False)
except:
    print("Warning: Could not copy weights to isolated writeback head")
```

**结果**: 新 head 使用随机初始化，行为与原始完全不同。

### 2. 冻结机制副作用

冻结 writeback head 影响了整体梯度流，导致：
- 目标能力反而下降 (-3.81%)
- 模型行为异常

### 3. 接口兼容性问题

虽然添加了 gap_probs 参数，但处理方式不正确：
```python
def forward(self, x, gap_probs=None):
    # 暂时忽略 gap_probs - 这可能导致行为不一致
    logits = self.classifier(features)
```

---

## 关键教训

1. **架构修改需要更严格的验证** - 不能只看消融测试，需要完整验收
2. **权重复制是关键** - 新结构必须能正确继承原始行为
3. **冻结机制有风险** - 需要验证对整体训练的影响

---

## 后续建议

**如需继续研究 writeback 隔离**:

1. **完全重写实现**，不基于当前失败分支
2. **确保权重复制成功**
3. **在隔离环境中充分测试**后再接入主线

**当前建议**:
- 使用基线配置进入第三阶段
- writeback 优化作为并行次线推进
- 不在主线阻塞第三阶段

---

## 归档信息

**归档日期**: 2026-04-19  
**归档原因**: 实现有严重缺陷，导致系统行为全面恶化  
**后续行动**: 如需继续研究，应完全重写实现

---

## 相关文件

- [stage6_writeback_isolation_fix.py](../../stage6_writeback_isolation_fix.py) - 实现代码
- [stage6_phase2_conclusion.md](../../stage6_phase2_conclusion.md) - 第二阶段结论
- [stage6_official_baseline_v1.md](../../stage6_official_baseline_v1.md) - 官方基线
