# Phase 14 Training Report - 训练报告

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 14 - Controlled Training, Behavior Internalization & Governance-Aligned Learning

---

## 1. 阶段概述

### 1.1 阶段目标

Phase 14 的核心目标：**把平台能力变成模型习惯**

不是传统意义上的"把模型喂更多数据"，而是：
- 把治理逻辑内化成默认反应
- 把检索习惯训成本能
- 把回答协议变成稳定行为

### 1.2 核心交付

| 组件 | 文件 | 状态 |
|------|------|------|
| 训练数据 Schema | `training_data_schema_v1.md` | ✅ 完成 |
| 训练指标契约 | `training_metric_contract_v1.md` | ✅ 完成 |
| D1 骨架数据 | `datasets/d1_seed_quality.jsonl` | ✅ 完成 |
| D2 策略数据 | `datasets/d2_policy_governance.jsonl` | ✅ 完成 |
| D3 冲突数据 | `datasets/d3_conflict_noise.jsonl` | ✅ 完成 |
| D4 记忆数据 | `datasets/d4_multiturn_memory.jsonl` | ✅ 完成 |
| D5 真实数据 | `datasets/d5_real_case_replay.jsonl` | ✅ 完成 |
| 策略头训练 | `models/train_policy_head.py` | ✅ 完成 |
| 检索治理训练 | `models/train_retrieval_governance.py` | ✅ 完成 |
| 记忆写回训练 | `models/train_memory_writeback.py` | ✅ 完成 |
| 产品行为评估 | `eval/eval_product_behavior.py` | ✅ 完成 |

---

## 2. 训练体系架构

### 2.1 数据分层

```
训练数据体系 (100%)
├── D1: 高质量骨架数据 (30%)
│   └── 基础理解、表达、任务遵循
├── D2: 检索与治理决策数据 (25%)
│   └── 五类响应策略、检索触发
├── D3: 冲突/噪声/对抗数据 (20%)
│   └── 回流、隔离、拒绝装懂
├── D4: 多轮对话与记忆数据 (15%)
│   └── 长期行为、记忆判断
└── D5: 真实样本回放数据 (10%)
    └── 后期对齐、不用于早期训练
```

### 2.2 多目标损失函数

```
L_total = 
    0.20 * L_resp      (回答质量)
  + 0.25 * L_policy    (策略分类)
  + 0.20 * L_retrieval (检索触发)
  + 0.10 * L_confidence (置信度校准)
  + 0.15 * L_governance (治理动作)
  + 0.05 * L_memory    (写回正确性)
  + 0.05 * L_consistency (多轮一致性)
```

### 2.3 训练阶段

| Stage | 时间 | 目标 | 关键输出 |
|-------|------|------|----------|
| 14.0 | 3-5天 | 基线建立 | 训练前指标 |
| 14.1 | 1-2周 | 基础骨架 | 策略头模型 |
| 14.2 | 1-2周 | 检索习惯 | 检索治理模型 |
| 14.3 | 1周 | 记忆治理 | 记忆写回模型 |
| 14.4 | 1-2周 | 自构建 | 自构建模型 |
| 14.5 | 并行 | 专项精修 | 英文关系、双语一致性 |
| 14.6 | 3-5天 | 闭环评估 | 产品行为报告 |

---

## 3. 核心训练组件

### 3.1 策略头 (Policy Head)

**职责**: 五类响应策略选择

**输入特征**:
- 查询长度
- 检索关键词
- 疑问词
- 历史长度
- 语言特征
- 风险关键词

**输出**:
- DIRECT (直接回答)
- RETRIEVAL_FIRST (检索优先)
- CONSERVATIVE (保守回答)
- DECLINE (拒绝回答)
- REVIEW (需要审查)

**验收标准**: 策略准确率 > 85%

### 3.2 检索治理头 (Retrieval & Governance Head)

**职责**: 检索触发决策 + 治理动作选择

**输入特征**:
- 证据状态
- 风险指标
- 查询类型
- 历史长度

**输出**:
- 检索概率
- 必要性评分
- 治理动作 (TSLA/STRONG_REVIEW/VALIDATION/ISOLATE/ARCHIVE/PROMOTE)

**验收标准**: 检索触发 F1 > 80%

### 3.3 记忆写回头 (Memory Writeback Head)

**职责**: 记忆写回决策 + 层级选择

**输入特征**:
- 内容价值
- 信息密度
- 事实性
- 持久性指标

**输出**:
- 写回概率
- 目标层级 (EPHEMERAL/LONG_TERM/DEEP_PERMANENT)

**验收标准**: 写回准确率 > 85%，误写回率 < 5%

---

## 4. 评估体系

### 4.1 训练期指标

| 指标 | 权重 | 目标 | 验证方式 |
|------|------|------|----------|
| 策略准确率 | 25% | > 85% | 验证集 |
| 检索触发 F1 | 20% | > 80% | 验证集 |
| 置信度 ECE | 15% | < 0.05 | 校准曲线 |
| 治理匹配率 | 15% | > 85% | 验证集 |
| 记忆写回准确率 | 10% | > 85% | 验证集 |
| 多轮一致性 | 10% | > 90% | 长对话测试 |
| 回答质量 | 5% | > 4.0/5 | 人工评估 |

### 4.2 产品期指标 (六大维度)

| 维度 | 目标 | 验证方式 |
|------|------|----------|
| 系统特色感知 | > 70% | 用户访谈 |
| AI 人格一致性 | > 90% | 盲测评估 |
| 治理可见性 | > 50% | 场景测试 |
| 检索价值感知 | > 4.2/5 | A/B 测试 |
| 记忆效果 | > 4.0/5 | 长对话测试 |
| 差异化 | > 4.0/5 | 盲测对比 |

---

## 5. 硬性禁止项

训练中**绝对禁止**：

1. ❌ 直接拿真实用户对话大规模训练主模型
2. ❌ 让用户数据直接进长期正常区或永久层
3. ❌ 把"回答更像普通 LLM"当成训练成功
4. ❌ 为提高答题率弱化 CONSERVATIVE/DECLINE/REVIEW
5. ❌ 让自构建内容跳过 TSLA 和强审查直接晋升

---

## 6. 使用指南

### 6.1 快速开始

```bash
# 1. 训练策略头
cd phase14/models
python train_policy_head.py

# 2. 训练检索治理
python train_retrieval_governance.py

# 3. 训练记忆写回
python train_memory_writeback.py

# 4. 评估产品行为
python eval/eval_product_behavior.py
```

### 6.2 训练流程

```python
# 完整训练流程示例
from train_policy_head import PolicyHead, PolicyTrainer
from train_retrieval_governance import RetrievalGovernanceHead, RetrievalGovernanceTrainer
from train_memory_writeback import MemoryWritebackHead, MemoryWritebackTrainer

# Stage 14.1: 训练策略头
policy_model = PolicyHead(input_dim=7, hidden_dim=64, num_strategies=5)
policy_trainer = PolicyTrainer(policy_model, train_loader, val_loader)
policy_history = policy_trainer.train(epochs=20)

# Stage 14.2: 训练检索治理
retrieval_model = RetrievalGovernanceHead(input_dim=10, hidden_dim=64)
retrieval_trainer = RetrievalGovernanceTrainer(retrieval_model, train_loader, val_loader)
retrieval_history = retrieval_trainer.train(epochs=15)

# Stage 14.3: 训练记忆写回
memory_model = MemoryWritebackHead(input_dim=8, hidden_dim=64)
memory_trainer = MemoryWritebackTrainer(memory_model, train_loader, val_loader)
memory_history = memory_trainer.train(epochs=15)

# Stage 14.6: 产品行为评估
evaluator = ProductBehaviorEvaluator()
report = evaluator.generate_report()
```

---

## 7. 下一步行动

### 7.1 立即执行

1. **运行训练脚本** - 验证训练流程
2. **准备真实数据** - 替换合成数据
3. **建立评估流水线** - 自动化评估

### 7.2 近期优化

1. **超参数调优** - 学习率、批量大小
2. **模型架构优化** - 尝试 Transformer-based
3. **数据增强** - 扩充训练样本

### 7.3 长期规划

1. **持续学习** - 在线学习机制
2. **A/B 测试** - 对比实验
3. **用户反馈闭环** - 收集真实反馈

---

## 8. 附录

### 8.1 文件清单

```
phase14/
├── phase14_training_report.md          # 本报告
├── training_data_schema_v1.md          # 数据模式
├── training_metric_contract_v1.md      # 指标契约
├── datasets/
│   ├── d1_seed_quality.jsonl          # D1数据
│   ├── d2_policy_governance.jsonl     # D2数据
│   ├── d3_conflict_noise.jsonl        # D3数据
│   ├── d4_multiturn_memory.jsonl      # D4数据
│   └── d5_real_case_replay.jsonl      # D5数据
├── models/
│   ├── train_policy_head.py           # 策略头训练
│   ├── train_retrieval_governance.py  # 检索治理训练
│   └── train_memory_writeback.py      # 记忆写回训练
├── eval/
│   └── eval_product_behavior.py       # 产品行为评估
├── checkpoints/                        # 模型检查点
└── logs/                              # 训练日志
```

### 8.2 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本 |

---

**文档状态**: 生效中  
**负责人**: Phase 14 训练负责人

---

## 核心结论

**Phase 14 不是传统预训练，而是治理优先的受控训练。**

目标是把你们已经做好的：
- ✅ 平台规则
- ✅ 治理逻辑
- ✅ 对话协议
- ✅ 产品行为

**系统性地内化成模型的默认行为。**

这才是你们项目下一步真正该做的训练。
