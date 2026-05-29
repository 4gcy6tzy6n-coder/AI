# Stage 7 Dual Guard 正式候选配置 V1

## 配置标识

- **配置名称**: Dual Guard Candidate V1
- **配置版本**: 1.0
- **创建日期**: 2026-04-19
- **状态**: 正式候选 (待最终验收)

## 核心参数

```yaml
dual_guard:
  feature_guard:
    mode: "subspace"           # 子空间投影保护
    beta: 0.05                 # 特征保护权重
    top_k_dims: 64             # 保护维度数
    distance_metric: "cosine"  # 距离度量
    anchor_samples: 30         # 锚点样本数
  
  head_compensation:
    head_lr_ratio: 2.0         # head学习率 = 主学习率 × 2.0
    head_grad_clip: 1.0        # head梯度裁剪
    lambda_wb: 0.1             # writeback损失权重
  
  training:
    num_epochs: 3              # 每step训练epoch数
    learning_rate: 1e-4        # 主学习率
    max_grad_norm: 1.0         # 全局梯度裁剪
```

## 机制说明

### 1. Backbone Feature Guard (主保护)

**功能**: 选择性保护 writeback-sensitive 特征子空间

**实现**:
- 识别 backbone 特征中高方差维度 (top-64)
- 约束这些维度在训练中的漂移
- 允许 backbone 其他部分自由学习主任务

**损失项**:
```
L_feat_guard = β × (1 - cosine_similarity(F_current, F_ref))
```

### 2. Head Compensation (保留学习)

**功能**: 允许 writeback head 继续训练以补偿 backbone 分布变化

**实现**:
- 分离优化器参数组
- head 学习率 = 2.0 × backbone 学习率
- 温和梯度裁剪

**损失项**:
```
L_wb = λ_wb × BCE(writeback_pred, writeback_target)
```

### 3. 总损失函数

```
L_total = L_main + λ_wb × L_wb + β × L_feat_guard

其中:
- L_main: 主任务损失 (gap + policy)
- L_wb: writeback 监督损失
- L_feat_guard: 特征保护损失
```

## 实验验证结果

### 50-step 对照实验

| 实验 | Score Δ | Output KL | Target | Old Ability | 结论 |
|------|---------|-----------|--------|-------------|------|
| baseline | +0.6781 | 1.442791 | +33.33% | 0.00% | 无保护, 大幅漂移 |
| feature_guard_only | -0.0198 | 0.005963 | +33.33% | 0.00% | 强保护, 负向漂移 |
| head_comp_only | +0.1052 | 0.100868 | +33.33% | 0.00% | 补偿有效 |
| **dual_guard (本配置)** | **+0.0000** | **0.000001** | **+33.33%** | **0.00%** | **最优平衡** |

### 参数搜索验证

**Beta 搜索** (固定 lr_ratio=0.3):
- beta=0.01: Score Δ=+0.0665
- beta=0.02: Score Δ=+0.0201
- beta=0.03: Score Δ=+0.0067
- beta=0.04: Score Δ=+0.0021
- **beta=0.05: Score Δ=-0.0198** ← 选择

**LR Ratio 搜索** (固定 beta=0.05):
- lr_ratio=0.5: Score Δ=+0.0004
- lr_ratio=1.0: Score Δ=+0.0001
- lr_ratio=1.5: Score Δ=+0.0000
- **lr_ratio=2.0: Score Δ=+0.0000** ← 选择

## 设计原理

### 为什么这个配置有效?

1. **Feature Guard beta=0.05 提供强约束**
   - 将 backbone 特征漂移限制在极低水平
   - 防止 writeback-sensitive 子空间被破坏

2. **Head LR Ratio=2.0 释放补偿能力**
   - 在强约束下仍允许 head 充分学习
   - 补偿 backbone 分布的微小变化

3. **协同效应**
   - Feature Guard 防止大幅负向漂移
   - Head Compensation 提供正向微调
   - 两者平衡实现 Score Δ≈0

## 与替代方案对比

| 方案 | Score Δ | KL | 优点 | 缺点 |
|------|---------|-----|------|------|
| baseline | +0.6781 | 高 | 无 overhead | 漂移严重 |
| freeze_head | -0.0635 | 低 | 简单 | 无法补偿 |
| feature_only | -0.0198 | 低 | 稳定 | 过度约束 |
| head_only | +0.1052 | 中 | 补偿强 | 可能过冲 |
| **dual_guard** | **+0.0000** | **极低** | **平衡最优** | **复杂度略高** |

## 验收标准

### 必须项 (Stage 6 基线)

- [ ] target_gain > 10%
- [ ] old_ability_drop < 15%
- [ ] e2e_success > 90%
- [ ] stability_pass (100-step)

### Stage 7 专项

- [ ] writeback_change < 5% (目标 <1%)
- [ ] rollback_recovery > 90%
- [ ] output_KL < 0.001

## 使用方式

### 代码集成

```python
from stage7_dual_guard_orchestrator import build_dual_guard_orchestrator

# 使用候选配置 Vorchestrator = build_dual_guard_orchestrator(
    feature_guard_beta=0.05,
    head_lr_ratio=2.0,
    lambda_wb=0.1,
)
```

### 配置文件

```yaml
# configs/stage7_dual_guard_v1.yaml
stage7:
  guard_mode: "dual"
  feature_guard:
    beta: 0.05
    top_k_dims: 64
  head_compensation:
    lr_ratio: 2.0
    grad_clip: 1.0
```

## 风险提示

1. **beta 敏感性**: beta > 0.05 可能导致过度约束
2. **lr_ratio 上限**: lr_ratio > 3.0 可能导致 head 过冲
3. **维度选择**: top_k_dims 应根据实际特征维度调整

## 后续优化方向

1. **自适应 beta**: 根据训练动态调整 beta
2. **维度重要性重估计**: 定期更新 sensitive dimensions
3. **多任务平衡**: 在更复杂场景下验证

## 版本历史

- v1.0 (2026-04-19): 初始候选配置，基于参数搜索最优结果

---

**状态**: 等待最终正式验收
