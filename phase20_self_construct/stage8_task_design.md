# Stage 8 真实目标任务设计方案

## 文档信息

- **阶段**: Stage 8
- **版本**: 1.0
- **创建日期**: 2026-04-19
- **状态**: 设计阶段

---

## 1. 任务选择决策

### 推荐任务: 知识问答 + 推理能力综合任务

**选择理由**:
1. ✅ 能够体现 Target Improvement（知识获取 + 推理提升）
2. ✅ 包含多轮决策（gap 判断 → 检索/推理 → policy 选择）
3. ✅ 可量化评估（准确率、完整度、writeback 变化）
4. ✅ 风险可控（标准 QA 数据集，噪声可预期）
5. ✅ TSLA 审计可观测（每个环节都有明确触发点）

---

## 2. 任务定义

### 任务名称: 结构化知识问答 (Structured Knowledge QA)

### 任务描述
模型需要回答关于特定领域知识的问题，问题分为三个难度层级：

| 层级 | 描述 | 示例 |
|------|------|------|
| **L1 - 直接检索** | 知识在记忆库中，直接检索回答 | "什么是光合作用？" |
| **L2 - 简单推理** | 需要结合多个知识点推理 | "如果植物没有阳光，光合作用会怎样？" |
| **L3 - 复杂推理** | 需要多步推理 + 外部知识整合 | "比较光合作用和呼吸作用的能量转换效率" |

### 输入/输出格式

```python
# 输入
{
    "question": "问题文本",
    "context": "可选上下文",
    "difficulty": "L1/L2/L3",
    "domain": "生物学/物理学/化学/..."
}

# 输出
{
    "gap_analysis": "是否需要新知识 (0/1)",
    "retrieval_triggered": "是否触发检索 (0/1)",
    "reasoning_steps": ["推理步骤1", "推理步骤2", ...],
    "answer": "最终答案",
    "confidence": "置信度 (0-1)"
}
```

---

## 3. 数据集设计

### 数据集结构

```
stage8_dataset/
├── train/
│   ├── l1_direct_retrieval.jsonl    # 300 条
│   ├── l2_simple_reasoning.jsonl    # 200 条
│   └── l3_complex_reasoning.jsonl   # 100 条
├── val/
│   └── validation_set.jsonl         # 100 条 (混合)
└── test/
    └── test_set.jsonl               # 100 条 (混合，留作最终评估)
```

### 样本格式

```json
{
    "id": "qa_001",
    "question": "光合作用的主要产物是什么？",
    "difficulty": "L1",
    "domain": "生物学",
    "expected_gap": 0,
    "expected_retrieval": 1,
    "expected_policy": 0,
    "answer": "葡萄糖和氧气",
    "explanation": "光合作用将二氧化碳和水转化为葡萄糖和氧气",
    "knowledge_units": ["光合作用定义", "光合作用产物"],
    "reasoning_chain": []
}
```

### 数据来源建议

1. **基础科学知识**: 从公开教育资源整理（如 Khan Academy、Wikipedia 基础条目）
2. **结构化 QA**: 参考 Natural Questions、SQuAD 等数据集的简化版本
3. **自构建**: 基于系统已有的 Unit 结构生成问题

---

## 4. 评估指标

### 主要指标

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **Answer Accuracy** | 正确答案数 / 总题数 | > 70% | 核心指标 |
| **Gap Detection Accuracy** | gap 判断正确率 | > 80% | 信息缺口识别 |
| **Retrieval Precision** | 正确触发检索 / 总触发 | > 75% | 检索决策质量 |
| **Policy Selection Accuracy** | policy 选择正确率 | > 80% | 策略选择能力 |

### Stage 7 保持指标

| 指标 | 目标 | 监控方式 |
|------|------|----------|
| **Writeback Change** | < 5% | 每 20 步检查 |
| **Guard Grad Ratio** | >= 0.5% | 实时监控 |
| **Old Ability Drop** | < 15% | 每 50 步检查 |

### TSLA 行为指标

| 指标 | 说明 |
|------|------|
| **Promotion Events** | 晋升事件次数和分布 |
| **Writeback Events** | 写回事件次数和成功率 |
| **Isolation Events** | 隔离事件次数和原因 |
| **Rollback Events** | 回滚事件次数和触发条件 |

---

## 5. 训练流程设计

### 训练步骤

```python
# Stage 8 训练流程
def stage8_training():
    # 1. 加载数据集
    dataset = load_stage8_dataset()
    
    # 2. 建立基线（使用 Stage 7 配置）
    baseline = establish_baseline_with_guard()
    
    # 3. 分阶段训练
    for epoch in range(num_epochs):
        for batch in dataset:
            # 3.1 前向传播
            outputs = model(batch['input'])
            
            # 3.2 计算任务损失
            task_loss = compute_qa_loss(outputs, batch['answer'])
            
            # 3.3 计算 Output KL Guard 损失
            guard_loss = output_kl_guard.compute_kl_guard_loss()
            
            # 3.4 总损失
            total_loss = task_loss + guard_loss
            
            # 3.5 反向传播
            total_loss.backward()
            
            # 3.6 TSLA 门控检查
            if tsla_gate.should_writeback(outputs):
                tsla_gate.execute_writeback()
            
            # 3.7 优化器步骤
            optimizer.step()
        
        # 4. 阶段性评估
        if epoch % eval_interval == 0:
            evaluate_and_log()
    
    # 5. 最终评估
    final_evaluation()
```

### 检查点设置

| 检查点 | 评估内容 |
|--------|----------|
| Step 20 | 初步收敛检查、writeback 稳定性 |
| Step 50 | 中期评估、指标趋势分析 |
| Step 100 | 最终验收、完整报告生成 |

---

## 6. 风险管控

### 潜在风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数据质量问题 | 训练不稳定 | 人工审核 + 自动过滤 |
| 任务过难 | 无法收敛 | 从 L1 开始，逐步增加难度 |
| Guard 失效 | writeback 漂移 | 实时监控 + 自动报警 |
| TSLA 误判 | 错误晋升/回滚 | 保守阈值 + 人工复核 |

### 应急预案

1. **Writeback 漂移 > 5%**: 立即暂停，检查 Guard 配置
2. **Answer Accuracy < 50%**: 降低任务难度，检查数据质量
3. **训练不稳定**: 降低学习率，增加 batch size

---

## 7. 实施计划

### Phase 1: 数据准备 (1-2 天)

- [ ] 收集/生成 600 条训练样本
- [ ] 标注验证集 100 条
- [ ] 数据质量审核
- [ ] 格式标准化

### Phase 2: 基线建立 (0.5 天)

- [ ] 使用 Stage 7 配置建立基线
- [ ] 验证基线性能
- [ ] 记录初始指标

### Phase 3: 训练实施 (3-5 天)

- [ ] 启动训练
- [ ] 监控关键指标
- [ ] 记录中间检查点
- [ ] 处理异常情况

### Phase 4: 评估分析 (1-2 天)

- [ ] 最终评估
- [ ] 数据分析
- [ ] 报告生成

---

## 8. 快速启动

### 立即可以开始的工作

1. **数据收集**: 从公开资源整理 100 条 L1 级别样本作为试点
2. **基线验证**: 确认 Stage 7 配置可以正常加载
3. **监控搭建**: 准备实时监控 Dashboard

### 最小可行实验 (MVP)

**目标**: 用 100 条 L1 样本验证完整流程

**时间**: 1-2 天

**成功标准**:
- Answer Accuracy > 60%
- Writeback Change < 5%
- 训练过程无异常

---

## 9. 下一步行动

请选择以下选项之一：

**A. 立即开始数据收集**
- 我将帮您从公开资源整理初始数据集

**B. 先进行 MVP 实验**
- 使用模拟数据验证完整流程

**C. 细化某个具体环节**
- 深入设计数据格式、评估指标或训练流程

**D. 其他**
- 请描述您的具体需求

---

**准备就绪，等待您的选择！**
