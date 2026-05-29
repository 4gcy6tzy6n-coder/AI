# Prototype Evaluation Plan - 原型评估计划

**版本**: v0.4  
**阶段**: Phase 4 - Permanent Memory & Minimal Experiment Route  
**创建日期**: 2026-04-17

---

## 1. 实验目标

将"理论正确"转化为"可验证、可回退、可迭代"的实验闭环。

### 1.1 核心目标

1. **验证瞬时层拦截能力** - 噪声是否主要停留在瞬时层
2. **验证长期层治理能力** - 各区域分流、回流、降级是否有效
3. **验证浅层永久晋升能力** - 哪些对象能进入浅层永久
4. **验证永久层回退保护能力** - 暴露问题时能否优先修正/拆分/降级

### 1.2 实验原则

- **可验证**: 每个假设都有明确的通过/失败标准
- **可回退**: 实验失败时能回到上一稳定版本
- **可迭代**: 实验结果能指导下一次改进

---

## 2. 实验包设计

### 实验 A: 瞬时层拦截能力 (Experiment A)

#### A.1 实验目的
验证噪声和用户错误是否主要停留在瞬时层，不会直接进入长期正常区。

#### A.2 测试场景

| 场景ID | 场景描述 | 输入特征 | 预期结果 |
|--------|----------|----------|----------|
| A-01 | 用户输入噪声 | 模糊、不完整、无来源 | 停留在transient，不进入长期层 |
| A-02 | 检索结果低质量 | 来源不可靠、证据弱 | 进入review或isolation，不进入normal |
| A-03 | 推理结果高冲突 | 与现有知识冲突 | 进入isolation，标记冲突 |
| A-04 | 硬否决触发 | 真实性极低或非法 | 直接exclude或error_archive |

#### A.3 通过标准

- 90%以上的噪声输入停留在瞬时层或隔离区
- 无重大噪声直接进入normal区
- 硬否决触发率符合预期

#### A.4 评估指标

```python
{
    "noise_interception_rate": 0.90,  # 噪声拦截率
    "false_positive_to_normal": 0.05,  # 误进入normal的比例
    "hard_veto_trigger_rate": 0.10     # 硬否决触发率
}
```

---

### 实验 B: 长期层治理能力 (Experiment B)

#### B.1 实验目的
验证受审区、隔离区、错误区、正常区的分流是否合理，回流、降级、错误归档是否有效。

#### B.2 测试场景

| 场景ID | 场景描述 | 初始区域 | 治理动作 | 预期结果 |
|--------|----------|----------|----------|----------|
| B-01 | review区稳定晋升 | review | keep多轮 | 晋升到normal |
| B-02 | review区不稳定隔离 | review | 波动大 | 降级到isolation |
| B-03 | isolation区改善 | isolation | 冲突解决 | 回流到review |
| B-04 | normal区退化 | normal | 质量下降 | 降级到review |
| B-05 | 多次硬否决 | review | 反复失败 | error_archive |

#### B.3 通过标准

- 各区域分流符合设计意图
- 回流机制有效（isolation → review）
- 降级机制有效（normal → review/isolation）
- 错误归档机制有效

#### B.4 评估指标

```python
{
    "review_to_normal_promotion_rate": 0.30,  # review晋升率
    "isolation_reflow_rate": 0.40,            # isolation回流率
    "normal_downgrade_rate": 0.10,            # normal降级率
    "error_archive_accuracy": 0.95            # 错误归档准确率
}
```

---

### 实验 C: 浅层永久晋升能力 (Experiment C)

#### C.1 实验目的
验证哪些对象能进入浅层永久，误晋升率是多少，永久候选失败的主要原因是什么。

#### C.2 测试场景

| 场景ID | 场景描述 | 当前区域 | 条件 | 预期结果 |
|--------|----------|----------|------|----------|
| C-01 | 完美候选 | normal | 高分+长周期+无冲突 | promote_to_shallow |
| C-02 | 周期不足 | normal | 高分+短周期 | blocked_by_cycles |
| C-03 | 来源不符 | normal | user_input来源 | blocked_by_source |
| C-04 | 近期冲突 | normal | 有近期冲突 | blocked_by_conflict |
| C-05 | 分数不足 | normal | 低于阈值 | blocked_by_stability |

#### C.3 通过标准

- 只有长期正常区对象能申请永久层
- 禁止跨层直写（review/isolation不能直接到permanent）
- 误晋升率低于5%

#### C.4 评估指标

```python
{
    "permanent_promotion_rate": 0.10,      # normal区晋升永久比例
    "blocked_by_cycles_rate": 0.30,        # 因周期不足被阻止比例
    "blocked_by_source_rate": 0.20,        # 因来源被阻止比例
    "false_promotion_rate": 0.05           # 误晋升率
}
```

---

### 实验 D: 永久层回退保护能力 (Experiment D)

#### D.1 实验目的
验证浅层永久对象在后续暴露问题时，是否能优先修正/拆分/降级，而不是直接消失。

#### D.2 测试场景

| 场景ID | 场景描述 | 初始状态 | 问题类型 | 处理策略 | 预期结果 |
|--------|----------|----------|----------|----------|----------|
| D-01 | 局部可修正 | active | 小错误 | local_fix | 修正后保持active |
| D-02 | 多义需拆分 | active | 多义混装 | split | 原对象split_pending，子对象重新治理 |
| D-03 | 重大冲突 | active | 重大冲突 | downgrade | 降级到review/isolation |
| D-04 | 证据失效 | active | 支持证据被推翻 | downgrade | 降级并记录原因 |
| D-05 | 尝试删除 | any | 任何情况 | delete | 拒绝删除，强制降级 |

#### D.3 通过标准

- 永久层对象禁止直接删除
- 优先尝试局部修正
- 多义对象能拆分
- 无法修正时降级到长期层

#### D.4 评估指标

```python
{
    "local_fix_success_rate": 0.60,        # 局部修正成功率
    "split_initiation_rate": 0.20,         # 拆分发起率
    "downgrade_rate": 0.15,                # 降级率
    "delete_attempt_blocked": 1.00         # 删除尝试被阻止率
}
```

---

## 3. 实验执行计划

### 3.1 实验顺序

```
Week 1: 实验 A + B
  - 瞬时层拦截能力验证
  - 长期层治理能力验证

Week 2: 实验 C
  - 浅层永久晋升能力验证

Week 3: 实验 D
  - 永久层回退保护能力验证

Week 4: 综合分析与迭代
  - 结果汇总
  - 问题诊断
  - 参数调优
```

### 3.2 样本需求

| 实验包 | 样本数量 | 样本来源 |
|--------|----------|----------|
| 实验 A | 50 | 合成噪声 + 真实用户输入 |
| 实验 B | 30 | phase3_cases.json + 扩展 |
| 实验 C | 20 | phase4_cases.json |
| 实验 D | 15 | 永久层模拟场景 |

### 3.3 通过阈值

| 实验包 | 最低通过率 | 目标通过率 |
|--------|------------|------------|
| 实验 A | 85% | 90% |
| 实验 B | 80% | 85% |
| 实验 C | 90% | 95% |
| 实验 D | 95% | 100% |

---

## 4. 实验工具

### 4.1 已有工具

- `test_phase3_calibration.py` - 第三阶段定标测试
- `test_phase4_permanent_layer.py` - 第四阶段永久层测试
- `calibration_pipeline.py` - 阈值扫描定标

### 4.2 需开发工具

- `evaluation_pipeline.py` - 统一实验执行管道
- `experiment_report.py` - 实验报告生成
- `regression_test.py` - 回归测试套件

---

## 5. 风险与应对

### 5.1 潜在风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 实验A失败率高 | 瞬时层拦截不足 | 调整transient_store阈值 |
| 实验C误晋升率高 | 永久层质量下降 | 提高permanent_gate阈值 |
| 实验D删除发生 | 违反保护原则 | 强制禁止delete操作 |

### 5.2 回退策略

- **v0.3_calibrated** - 第三阶段稳定版本
- **v0.4_permanent** - 第四阶段开发版本
- 实验失败时回退到v0.3，分析问题后重新迭代

---

## 6. 成功标准

第四阶段完成的5条标志：

1. ✅ 长期正常区对象可以被判定为浅层永久候选
2. ✅ 永久层禁止跨层直写
3. ✅ 永久对象暴露问题时不会直接删除
4. ⏳ 最小实验路线图正式形成（本文档）
5. ⏳ 至少跑通一组"浅层永久晋升 + 回退保护"实验

---

## 7. 后续阶段预览

### Phase 5: Training Path Integration
- 训练路径接入
- 深层永久层
- 模型自构建进入知识库/参数双链

### Phase 6: Full Training Loop
- 完整训练闭环
- 知识库与参数协同更新
- 长期稳定性验证

---

**文档版本**: v0.4-draft  
**最后更新**: 2026-04-17  
**状态**: 开发中
