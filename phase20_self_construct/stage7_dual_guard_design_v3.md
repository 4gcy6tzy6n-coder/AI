# Stage 7 Dual-Level Guard Design V3

## 核心结论 (基于正确来源拆解实验)

### 实验结果 (50-step)

| 实验 | Freeze BB | Freeze Head | Score Δ | 说明 |
|------|-----------|-------------|---------|------|
| A | False | True | **-0.0635** | Backbone训练损害writeback |
| B | True | False | **+0.0291** | Head训练改善writeback |
| C | True | True | +0.0000 | 控制组验证通过 |
| D | False | False | **+0.0432** | 协同训练效果最佳 |

### 关键洞察

1. **Backbone训练是writeback漂移的主要来源** (影响更大，负向)
2. **Head训练是有益补偿** (影响较小，正向)
3. **协同训练(D)优于单独训练(B)** - 说明backbone学习带来主任务收益，head学习提供补偿

### 旧假设 vs 新结论

**旧假设 (已推翻)**:
- 保护writeback = 尽量不让writeback head变化
- 冻结head或限制其更新是主要保护手段

**新结论**:
- head的变化是有益补偿
- 真正需要约束的是backbone对writeback相关表征的破坏
- 约束必须是选择性的，不能把协同学习一起冻死

---

## 新设计方向: Dual-Level Coordinated Guard

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Dual-Level Guard                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: Backbone Feature Guard (主保护)                   │
│  ├── 识别writeback-sensitive特征子空间                      │
│  ├── 约束这些特征在训练中的漂移                             │
│  └── 允许backbone其他部分自由学习                           │
│                                                             │
│  Level 2: Head Compensation (保留学习)                      │
│  ├── 继续训练writeback head                                 │
│  ├── 补偿backbone分布变化                                   │
│  └── 温和约束(小lr, 梯度裁剪)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 总损失函数

```python
L_total = L_main + λ_wb * L_wb + β * L_feat_guard

其中:
- L_main: 主任务损失 (gap + policy)
- L_wb: writeback监督损失
- L_feat_guard: backbone特征稳定约束
- λ_wb: writeback损失权重 (建议0.1)
- β: 特征保护权重 (建议0.01-0.1)
```

---

## Level 1: Backbone Feature Guard

### 方案A: 特征保持损失 (Feature Preservation Loss)

**原理**: 对固定的anchor样本集，约束训练前后backbone在writeback入口处的特征不要漂移太多。

```python
# 训练前保存参考特征
with torch.no_grad():
    ref_features = backbone(anchor_samples)

# 训练中计算特征保持损失
L_feat_preservation = ||F_wb(x) - F_wb_ref(x)||^2
# 或使用cosine similarity
L_feat_preservation = 1 - cosine_similarity(F_wb(x), F_wb_ref(x))
```

**优点**:
- 简单直接
- 约束明确

**缺点**:
- 全局约束可能过于严格
- 可能限制backbone学习主任务所需的新特征

### 方案B: 子空间投影保护 (Subspace Projection Guard) ⭐推荐

**原理**: 先识别writeback高敏感维度，只对这些维度加约束。

```python
# 1. 识别writeback敏感维度 (离线分析)
sensitive_dims = identify_writeback_sensitive_dimensions(backbone, writeback_head)

# 2. 只约束这些维度
L_feat_guard = ||F_wb(x)[sensitive_dims] - F_wb_ref(x)[sensitive_dims]||^2
```

**优点**:
- 选择性约束，不伤害主任务
- 更精准地保护writeback相关表征

**实现步骤**:
1. 使用sensitivity analysis识别关键维度
2. 创建投影mask
3. 只约束mask内的特征

---

## Level 2: Head Compensation

### 保留策略

**不再冻结head**，而是:

```python
# 1. 保留writeback损失
L_wb = binary_cross_entropy(writeback_pred, writeback_target)

# 2. 温和约束
- 较小学习率 (如主任务的0.1-0.5倍)
- 梯度裁剪 (max_grad_norm=1.0)
- 可选: weight decay (1e-4)
```

### 作用

- 补偿backbone分布变化
- 维持writeback性能
- 与backbone协同适配

---

## 实验矩阵

### 4组50-step对照实验

| 实验 | Backbone Guard | Head Training | 预期结果 |
|------|----------------|---------------|----------|
| baseline | None | Normal | writeback下降 (参考A组) |
| feature_guard_only | ✓ | Normal | 减少负漂移 |
| head_comp_only | None | ✓ (保留) | 正向补偿 (参考B组) |
| dual_guard | ✓ | ✓ (保留) | 最佳效果 (参考D组) |

### 评估指标

1. **writeback score change** - 主要指标
2. **output KL** - 输出稳定性
3. **target gain** - 主任务收益
4. **old ability drop** - 旧能力保持

### 成功标准

- dual_guard的writeback变化优于baseline
- dual_guard同时保住target gain和old ability
- feature_guard_only能减少负漂移，验证主因

---

## 实现计划

### 文件1: stage7_backbone_feature_guard.py

实现Backbone-level Feature Guard:
- FeaturePreservationGuard (方案A)
- SubspaceProjectionGuard (方案B)
- 特征敏感度分析工具

### 文件2: stage7_dual_guard_orchestrator.py

集成Dual-Level Guard到训练流程:
- 修改训练损失计算
- 集成feature guard
- 保留head训练

### 文件3: stage7_dual_guard_validation.py

验证实验:
- 4组50-step对照
- 指标收集
- 结果分析

---

## 预期判断逻辑

```
if feature_guard_only 明显减少负漂移:
    → 主因确实在backbone特征
    → Backbone Feature Guard有效

if dual_guard 优于 head_comp_only:
    → 双层机制成立
    → 协同保护优于单一保护

if dual_guard 同时保住 target / old ability:
    → 这就是Stage 7新主候选
    → 可以进入系统集成阶段
```

---

## 总结

Stage 7的正确问题已从:
> "如何保护writeback head不被破坏？"

转变为:
> "如何在允许head继续学习补偿的前提下，限制backbone对writeback关键表征的破坏性漂移？"

新设计明确转向:
**"选择性backbone特征保护 + 保留head补偿学习"的双层协同机制**。
