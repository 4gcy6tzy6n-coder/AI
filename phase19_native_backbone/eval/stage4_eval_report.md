# Stage 4 评估报告：原生主干替代实验

## 概述

**阶段**: Stage 4 - 原生主干替代实验  
**日期**: 2026-04-18  
**目标**: 验证 tiny 版原生主干（GRU-based）能否替代 Transformer 主干

---

## 实验设计

### 对比模型

| 模型 | 架构 | 参数量 |
|------|------|--------|
| **Native Backbone Tiny V1** | 双向 GRU (2层) | 3,150,115 |
| **Transformer Baseline Tiny** | Transformer (2层, 4头) | 2,887,587 |

### 数据配置

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| 训练集 | 80 (80%) | 训练 |
| 验证集 | 20 (20%) | 早停验证 |
| 留出集 (test_holdout) | 9 | 私有知识/高风险/中英混合测试 |
| 同义改写 (test_paraphrase) | 9 | 同义表达鲁棒性测试 |
| 长对话 (test_longdialog) | 4 | 多轮对话稳定性测试 |

### 训练配置

- **Epochs**: 15 (早停 patience=5)
- **Batch Size**: 16
- **Learning Rate**: 1e-3 (Cosine Annealing)
- **Weight Decay**: 1e-4
- **损失函数**: Gap + Policy + Writeback (等权重)

---

## 训练结果

### 验证集表现

| 模型 | 最佳验证准确率 | 早停 Epoch |
|------|----------------|------------|
| Native (GRU) | **100.0%** | 13 |
| Transformer | **100.0%** | 10 |

**结论**: 两个模型在验证集上都达到了 100% 准确率，说明都能拟合训练数据。

### 训练收敛速度

| 模型 | Gap 达到 100% | Policy 达到 100% | Writeback 达到 100% |
|------|---------------|------------------|---------------------|
| Native (GRU) | Epoch 7 | Epoch 7 | Epoch 8 |
| Transformer | Epoch 3 | Epoch 4 | Epoch 5 |

**结论**: Transformer 收敛更快，但两者最终都达到相同水平。

---

## 测试集评估

### 1. 留出集 (test_holdout)

测试场景：私有知识、高风险问题、中英混合

| 指标 | Native (GRU) | Transformer | 优势方 |
|------|--------------|-------------|--------|
| Gap Acc | 22.2% | 22.2% | 平手 |
| Policy Acc | **33.3%** | 22.2% | Native |
| Writeback Acc | **66.7%** | 33.3% | Native |

### 2. 同义改写 (test_paraphrase)

测试场景：同一意图的不同表达方式

| 指标 | Native (GRU) | Transformer | 优势方 |
|------|--------------|-------------|--------|
| Gap Acc | 33.3% | **55.6%** | Transformer |
| Policy Acc | 33.3% | **44.4%** | Transformer |
| Writeback Acc | **66.7%** | 44.4% | Native |

### 3. 长对话 (test_longdialog)

测试场景：10轮/20轮对话、冲突信息、历史回顾

| 指标 | Native (GRU) | Transformer | 优势方 |
|------|--------------|-------------|--------|
| Gap Acc | 25.0% | **50.0%** | Transformer |
| Policy Acc | 25.0% | 25.0% | 平手 |
| Writeback Acc | 0.0% | **25.0%** | Transformer |

---

## 关键发现

### 1. 可训练性 ✅

- **Native (GRU) 能够稳定训练**
- Loss 能正常下降
- 不炸梯度
- Checkpoint 正常保存/加载

### 2. 核心行为保持 ⚠️

| 行为 | Native 表现 | 评估 |
|------|-------------|------|
| Gap Detection | 验证集 100%，测试集 20-33% | 过拟合严重 |
| Policy | 验证集 100%，测试集 25-33% | 过拟合严重 |
| Writeback | 验证集 100%，测试集 0-66% | 相对较好 |

**问题**: 数据量太小（100条），导致严重过拟合。

### 3. 与 Transformer 对比

| 维度 | Native (GRU) | Transformer | 结论 |
|------|--------------|-------------|------|
| 参数量 | 3.15M | 2.89M | 相当 |
| 训练速度 | 较慢 | 较快 | Transformer 更快 |
| 泛化能力 | 较弱 | 较弱 | 两者都过拟合 |
| 长对话 | 25% | 25-50% | Transformer 略好 |

### 4. 优势场景

在当前小数据设置下，**未发现 Native 明显优势场景**。

---

## 问题分析

### 过拟合原因

1. **数据量不足**: 训练集仅 80 条样本
2. **模型容量过大**: 3M 参数对 80 条样本来说过大
3. **缺乏正则化**: 只有基本的 Dropout

### 改进建议

1. **增加数据量**
   - 使用完整的 Stage 3B 数据（300条）
   - 或生成更多合成数据

2. **减小模型容量**
   - hidden_dim: 128 → 64
   - num_layers: 2 → 1

3. **增强正则化**
   - 增加 Dropout 率
   - 添加 L2 正则
   - 使用数据增强

---

## Stage 4 验收标准检查

| 标准 | 要求 | 实际 | 通过 |
|------|------|------|------|
| 可训练性 | loss 能下降，不炸梯度 | ✅ 满足 | ✅ |
| 核心行为保持 | Gap/Policy/Governance 不明显塌陷 | ⚠️ 测试集塌陷 | ❌ |
| 优势场景 | 至少 2 个主打场景体现优势 | ❌ 未发现 | ❌ |
| 检索/治理/记忆闭环 | 继续成立 | ⚠️ 部分成立 | ⚠️ |
| 长对话稳定性 | 不明显崩 | ⚠️ 25% 准确率 | ❌ |

### 结论

**Stage 4 当前实验未通过验收**。

原因：数据量过小导致过拟合，无法公平评估 Native 主干的真实能力。

---

## 下一步建议

### 选项 1：增加数据量（推荐）

- 使用完整的 Stage 3B 数据（300条）
- 重新训练 Native 和 Transformer
- 预期：过拟合减轻，能更好评估泛化能力

### 选项 2：减小模型容量

- Native hidden_dim: 128 → 64
- Transformer 保持相同比例缩小
- 预期：减少过拟合，但可能欠拟合

### 选项 3：数据增强

- 对训练数据进行同义改写
- 增加噪声样本
- 预期：提升泛化能力

---

## 阶段 4 文件清单

```
phase19_native_backbone/
├── native_backbone_tiny_v1.py      # Native 主干 (GRU-based)
├── transformer_baseline_tiny.py    # Transformer 基线
├── native_vs_transformer_comparison.py  # 对比实验
├── stage4_training_with_real_data.py    # 真实数据训练
├── datasets/
│   ├── prepare_test_datasets.py    # 数据集生成脚本
│   ├── train.jsonl                 # 训练集
│   ├── val.jsonl                   # 验证集
│   ├── test_holdout.jsonl          # 留出集
│   ├── test_paraphrase.jsonl       # 同义改写
│   └── test_longdialog.jsonl       # 长对话
├── checkpoints/
│   ├── native_stage4.pt            # Native 模型权重
│   └── transformer_stage4.pt       # Transformer 模型权重
└── eval/
    ├── comparison_results.json     # 对比结果
    ├── stage4_results.json         # Stage 4 结果
    └── stage4_eval_report.md       # 本报告
```

---

## 总结

### 已完成

1. ✅ 创建 tiny 版 Native 主干（GRU-based）
2. ✅ 创建同参数量 Transformer 基线
3. ✅ 准备严格测试数据集
4. ✅ 运行对比实验

### 发现的问题

- 数据量过小（100条）导致严重过拟合
- 无法公平评估 Native 主干的泛化能力
- 需要增加数据量或减小模型容量

### 关键结论

**Native 主干可训练，但当前实验设置无法证明其替代 Transformer 的可行性。**

需要更多数据和更严格的实验设计才能得出可靠结论。
