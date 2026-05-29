# Stage 6 评估指标契约

**版本**: v1.0  
**日期**: 2026-04-18  
**用途**: 定义 Stage 6 真实能力评估的统一标准

---

## 1. 评估体系概述

### 1.1 三条评估轴

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 6 能力评估体系                      │
├─────────────────────────────────────────────────────────────┤
│  目标能力 (Target)    │  旧能力保持 (Old)    │  Writeback   │
│  ─────────────────    │  ────────────────    │  ──────────  │
│  • 目标领域准确率     │  • 检索能力          │  • 写回决策  │
│  • 新知识整合         │  • 策略选择          │  • 关键场景  │
│  • 任务完成率         │  • 治理动作          │  • 边界处理  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 评估原则

1. **标准化**: 使用统一测试集，确保可重复
2. **量化**: 所有指标必须可量化、可比较
3. **可追溯**: 每次评估记录完整上下文
4. **可对比**: 支持与基线对比

---

## 2. 评估指标定义

### 2.1 目标能力 (Target Capability)

**定义**: 模型在目标领域/任务上的表现

**测试集**: `StandardTestSets.get_target_capability_tests()`

**指标计算**:
```python
target_score = (correct_gap_predictions + correct_policy_predictions) / total_predictions
```

**评分标准**:
| 等级 | 分数范围 | 说明 |
|------|----------|------|
| 优秀 | ≥ 0.80 | 目标能力显著增强 |
| 良好 | 0.60 - 0.80 | 目标能力有提升 |
| 及格 | 0.40 - 0.60 | 目标能力轻微提升 |
| 不足 | < 0.40 | 目标能力无明显变化 |

**验收阈值**:
- 单步提升: > 0% (至少不下降)
- 5步累积提升: > +10%
- 10步累积提升: > +15%

### 2.2 旧能力保持 (Old Capability Preservation)

**定义**: 模型在已有能力上的稳定性

**子维度**:
1. **检索能力 (Retrieval)**: 正确触发检索的能力
2. **策略能力 (Policy)**: 正确选择策略的能力
3. **治理能力 (Governance)**: 正确执行治理动作的能力

**测试集**: `StandardTestSets.get_old_capability_tests()`

**指标计算**:
```python
retrieval_score = correct_retrieval / total_retrieval_tests
policy_score = correct_policy / total_policy_tests
governance_score = correct_governance / total_governance_tests

old_ability_drop = max(
    abs(retrieval_current - retrieval_baseline),
    abs(policy_current - policy_baseline),
    abs(governance_current - governance_baseline)
)
```

**验收阈值**:
| 步骤 | 最大允许掉落 |
|------|-------------|
| 单步 | < 10% |
| 5步后 | < 15% |
| 10步后 | < 20% |

### 2.3 Writeback 稳定性 (Writeback Stability)

**定义**: 写回能力的稳定性

**测试集**: `StandardTestSets.get_writeback_tests()`

**指标计算**:
```python
base_score = correct_predictions / total_tests
critical_score = correct_critical / total_critical_tests

writeback_score = 0.7 * base_score + 0.3 * critical_score
writeback_change = abs(writeback_current - writeback_baseline)
```

**验收阈值**:
- 始终: writeback_change < 5%
- 关键场景: 准确率 > 95%

---

## 3. 综合评估指标

### 3.1 核心指标

| 指标名 | 符号 | 计算方式 | 目标值 |
|--------|------|----------|--------|
| 目标提升 | ΔT | target_current - target_baseline | > +10% (5步) |
| 旧能力掉落 | ΔO | max(old_ability_drops) | < 15% (5步) |
| Writeback变化 | ΔW | \|writeback_current - writeback_baseline\| | < 5% |
| 单步最大掉落 | ΔS | max(single_step_drops) | < 10% |
| 回滚恢复率 | RR | recovered_abilities / lost_abilities | > 90% |

### 3.2 综合评分

```python
overall_score = (
    0.35 * normalize(ΔT) +      # 目标提升权重 35%
    0.35 * (1 - normalize(ΔO)) + # 旧能力保持权重 35%
    0.20 * (1 - normalize(ΔW)) + # Writeback权重 20%
    0.10 * RR                    # 回滚恢复权重 10%
)
```

**综合评级**:
| 评级 | 分数范围 | 说明 |
|------|----------|------|
| A (优秀) | ≥ 0.85 | 所有指标达标，系统稳定 |
| B (良好) | 0.70 - 0.85 | 主要指标达标， minor issues |
| C (及格) | 0.55 - 0.70 | 部分指标达标，需优化 |
| D (不足) | < 0.55 | 多项指标不达标，需重构 |

---

## 4. 评估流程

### 4.1 基线评估

```
1. 加载初始模型
2. 执行完整评估 (target + old + writeback)
3. 记录基线分数
4. 保存基线结果
```

### 4.2 步骤评估

```
1. 执行晋升步骤
2. 评估当前能力
3. 计算与基线差异
4. 判断是否触发回滚
5. 记录评估结果
```

### 4.3 累积评估

```
1. 汇总所有步骤评估结果
2. 计算累积指标
3. 分析趋势
4. 生成报告
```

---

## 5. 验收标准

### 5.1 Stage 6 第二阶段验收

**必须通过项**:
- [ ] 5步后目标提升 > +10%
- [ ] 5步后旧能力掉落 < 15%
- [ ] 10步后旧能力掉落 < 20%
- [ ] writeback 变化始终 < 5%
- [ ] 单步最大掉落 < 10%
- [ ] 回滚恢复率 > 90%

**综合评级**: 必须达到 B 级或以上

### 5.2 6B 配置专项验证

使用 6B 配置 (base_kl=0.30/0.38, replay=0.35) 时:

| 指标 | 6B 目标 |
|------|---------|
| 5步目标提升 | > +12% |
| 5步旧能力掉落 | < 12% |
| writeback 掉落 | < 3% |
| 回滚触发率 | < 20% |

---

## 6. 接口规范

### 6.1 评估器接口

```python
class Stage6RealEvaluator:
    """统一评估接口"""
    
    def evaluate_target_capability(self) -> float:
        """评估目标能力"""
        pass
    
    def evaluate_old_capability(self) -> Dict[str, float]:
        """评估旧能力保持"""
        pass
    
    def evaluate_writeback_stability(self) -> float:
        """评估 writeback 稳定性"""
        pass
    
    def evaluate_all(self) -> EvaluationResult:
        """执行完整评估"""
        pass
```

### 6.2 结果格式

```json
{
  "timestamp": "2026-04-18T10:30:00",
  "scores": {
    "target_score": 0.75,
    "retrieval_score": 0.88,
    "policy_score": 0.82,
    "governance_score": 0.85,
    "writeback_score": 0.96
  },
  "target_improvement": 0.15,
  "old_ability_drop": 0.08,
  "writeback_change": 0.02,
  "details": {
    "num_samples": 100,
    "old_capability_breakdown": {...}
  }
}
```

---

## 7. 报告模板

### 7.1 单步评估报告

```markdown
## 步骤 X 评估报告

**时间**: 2026-04-18 10:30:00

### 能力评分
- 目标能力: 75% (+15%)
- 检索能力: 88% (-2%)
- 策略能力: 82% (-3%)
- 治理能力: 85% (-5%)
- writeback: 96% (-2%)

### 关键指标
- 目标提升: +15%
- 旧能力掉落: 5%
- writeback 变化: 2%

### 结论
✓ 通过 / ✗ 失败
```

### 7.2 累积评估报告

```markdown
## N 步累积评估报告

**总步骤**: 10
**通过步骤**: 8
**回滚次数**: 2

### 累积指标
- 总目标提升: +45%
- 最大旧能力掉落: 18%
- 平均单步掉落: 6%
- writeback 最大变化: 4%

### 趋势分析
- 目标能力: ↗ 持续上升
- 旧能力: → 基本稳定
- writeback: → 轻微波动

### 综合评级: B (良好)
```

---

## 8. 附录

### 8.1 测试集详情

**目标能力测试集**:
- 数量: 5个核心场景 + 95个随机样本
- 覆盖: gap判断、policy选择
- 关键场景: 新知识需求、策略切换

**旧能力测试集**:
- 数量: 6个核心场景 + 94个随机样本
- 覆盖: 检索、策略、治理
- 关键能力: 全部标记为 critical

**Writeback 测试集**:
- 数量: 5个核心场景 + 95个随机样本
- 覆盖: 写回决策、边界场景
- 关键场景: 2个零容忍场景

### 8.2 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-18 | 初始版本，基于 6B 配置 |

---

## 参考文档

- [stage6_real_evaluator.py](stage6_real_evaluator.py) - 评估器实现
- [stage6_long_term_validation.py](stage6_long_term_validation.py) - 长期验证脚本
- [STAGE6_ENTRY.md](STAGE6_ENTRY.md) - Stage 6 入口文档
