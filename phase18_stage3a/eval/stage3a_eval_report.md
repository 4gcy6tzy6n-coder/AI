# Stage 3A 评估报告

## 概述

**阶段**: Stage 3A - 真实数据基础行为训练  
**日期**: 2026-04-18  
**目标**: 在真实数据上训练 Policy + Gap + Governance

---

## 训练配置

| 配置项 | 值 |
|--------|-----|
| 总样本数 | 500 |
| D1 (高质量骨架) | 60% (300) |
| D2 (策略治理) | 30% (150) |
| D3 (对抗样本) | 10% (50) |
| Epochs | 10 |
| Batch Size | 16 |
| Learning Rate | 5e-4 |
| 训练目标 | Policy + Gap + Governance |
| 冻结模块 | WritebackHead |

---

## 最终指标

### 准确率

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| Gap 准确率 | ≥ 70% | TBD | ⏳ |
| 策略准确率 | ≥ 50% | TBD | ⏳ |
| 治理准确率 | - | TBD | ⏳ |

### 策略分布

| 策略 | 目标分布 | 实际分布 | 状态 |
|------|----------|----------|------|
| DIRECT | 20-40% | TBD | ⏳ |
| RETRIEVAL_FIRST | 20-40% | TBD | ⏳ |
| CONSERVATIVE | 15-30% | TBD | ⏳ |
| DECLINE | 5-15% | TBD | ⏳ |
| REVIEW | 5-15% | TBD | ⏳ |

---

## 训练历史

### Loss 曲线

```
Epoch | Train Loss | Val Loss
------|------------|----------
  1   |    TBD     |   TBD
  2   |    TBD     |   TBD
  ... |    ...     |   ...
```

### 准确率曲线

```
Epoch | Gap Acc | Policy Acc | Gov Acc
------|---------|------------|--------
  1   |   TBD   |    TBD     |  TBD
  2   |   TBD   |    TBD     |  TBD
  ... |   ...   |    ...     |  ...
```

---

## 行为分析

### 高风险误答率

- 目标: < 10%
- 实际: TBD

### 检索触发合理性

- RETRIEVAL_FIRST 触发时机是否合理
- 误触发率: TBD

### 治理动作分布

| 动作 | 触发次数 | 占比 |
|------|----------|------|
| TSLA | TBD | TBD |
| STRONG_REVIEW | TBD | TBD |
| ISOLATE | TBD | TBD |
| ARCHIVE | TBD | TBD |
| ... | ... | ... |

---

## 对比基线

### 与 Stage 2.3b 对比

| 指标 | Stage 2.3b | Stage 3A | 变化 |
|------|------------|----------|------|
| Gap 准确率 | 80.0% | TBD | TBD |
| 策略准确率 | 81.7% | TBD | TBD |

---

## 问题与发现

### 发现的问题

1. TBD
2. TBD
3. TBD

### 改进建议

1. TBD
2. TBD
3. TBD

---

## 进入 Stage 3B 建议

### 验收标准检查

| 标准 | 要求 | 实际 | 通过 |
|------|------|------|------|
| Gap 准确率 | ≥ 70% | TBD | ⏳ |
| 策略准确率 | ≥ 50% | TBD | ⏳ |
| 高风险误答率 | 下降 | TBD | ⏳ |
| 策略分布合理 | 是 | TBD | ⏳ |

### 建议

- [ ] 继续优化当前模型
- [ ] 进入 Stage 3B (D4 + D5 + Writeback)
- [ ] 增加训练数据
- [ ] 调整超参数

---

## 附录

### 检查点文件

- 最佳模型: `phase18_stage3a/checkpoints/best_model.pt`
- 训练报告: `phase18_stage3a/eval/stage3a_report.json`

### 相关文件

- 训练脚本: `phase18_stage3a/stage3a_training_runner.py`
- 数据加载器: `phase18_stage3a/stage3a_dataloader.py`
