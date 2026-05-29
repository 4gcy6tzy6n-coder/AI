# Stage 9 入口文档：多层任务平衡与优化

**状态**: 🚀 准备启动  
**前置阶段**: Stage 8 (已完成)  
**日期**: 2026-04-20

---

## 1. 阶段目标

Stage 9 的核心目标是解决 Stage 8 发现的多层任务跷跷板效应，实现 L1/L2/L3 任务的平衡训练。

### 1.1 主要目标
- 解决多层任务训练的跷跷板效应
- 实现 L1/L2/L3 同时保持合理准确率
- 优化课程学习和采样策略
- 保持 Guard 和 TSLA 配置冻结

### 1.2 成功标准
| 指标 | 目标 | 说明 |
|------|------|------|
| L1 Accuracy | > 60% | 基础层保持 |
| L2 Accuracy | > 50% | 进阶层达标 |
| L3 Accuracy | > 30% | 复杂层可接受 |
| Overall | > 55% | 综合表现 |
| Writeback Δ | < 5% | Guard 保护有效 |
| 跷跷板检查 | 通过 | L1>50% & L2>40% |

---

## 2. 前置条件 (Stage 8 成果)

### 2.1 已验证稳定的配置
```python
# 冻结配置 - Stage 9 保持不变
Output KL Guard:
  - beta: 0.2
  - use_probs: True
  - num_samples: 30

TSLA 门控:
  - promotion_threshold: 0.8
  - isolation_threshold: 0.3

基础训练:
  - learning_rate: 1e-4 (tiny model)
  - learning_rate: 5e-5 (large model)
  - grad_clip: 1.0
```

### 2.2 可用数据集
- `train.jsonl` - L1 数据 (70条)
- `synthetic_l2.jsonl` - L2 数据 (500条)
- `synthetic_l3.jsonl` - L3 数据 (300条)

### 2.3 关键认知
1. **跷跷板效应**是任务/训练策略问题，不是系统故障
2. **Guard 和 TSLA**在单层任务下完全稳定
3. **需要专门优化**多层任务的采样和课程策略

---

## 3. Stage 9 实验策略

### 3.1 实验方向

#### 方向 A: 渐进式混合训练
- 从 L1 开始，逐步增加 L2/L3 比例
- 使用动态权重调整
- 保持 L1 replay 机制

#### 方向 B: 任务隔离训练
- 分别训练 L1/L2/L3 专家
- 使用门控机制组合
- 减少任务间干扰

#### 方向 C: 元学习策略
- 学习如何平衡多任务
- 动态调整采样比例
- 基于验证反馈优化

### 3.2 推荐策略

**主推方向 A: 渐进式混合训练**

理由：
- 与 Stage 8-R2 类似但优化参数
- 可复用现有代码基础
- 渐进式更容易控制和调试

---

## 4. 实验设计

### 4.1 实验轮次规划

| 轮次 | 名称 | 策略 | 预期结果 |
|------|------|------|----------|
| R1 | 动态权重 | 根据表现动态调整 L1/L2/L3 权重 | 找到平衡点 |
| R2 | 交替冻结 | 交替冻结某些层参数 | 减少干扰 |
| R3 | 专家混合 | 训练子专家 + 门控 | 任务隔离 |
| R4 | 元学习 | 学习最优采样策略 | 自适应平衡 |

### 4.2 关键参数探索

```python
# 需要探索的参数空间
l1_weights = [1.0, 1.2, 1.5]  # L1 保护权重
sampling_ratios = [
    (2, 2, 1),  # L1:L2:L3
    (1, 1, 1),
    (3, 2, 1),
]
replay_intervals = [20, 25, 30]  # L1 replay 频率
phase_ratios = [
    (0.3, 0.3, 0.4),  # 三阶段比例
    (0.2, 0.3, 0.5),
]
```

---

## 5. 技术方案

### 5.1 动态权重调整

```python
class DynamicWeightScheduler:
    """动态权重调度器"""
    
    def __init__(self):
        self.weights = {'L1': 1.2, 'L2': 1.0, 'L3': 1.0}
        self.performance_history = []
    
    def update_weights(self, metrics: Dict):
        """根据表现更新权重"""
        # 如果 L1 下降，增加 L1 权重
        # 如果 L2/L3 过低，暂时降低 L1 权重
        pass
```

### 5.2 验证驱动的早停

```python
# 每 N 步验证，保存最佳 checkpoint
# 选择标准：composite_score = 0.35*L1 + 0.35*L2 + 0.20*L3 + 0.10*WB
```

### 5.3 梯度冲突检测

```python
# 检测 L1/L2/L3 任务的梯度方向
# 如果冲突严重，使用梯度投影或加权平均
```

---

## 6. 验收标准

### 6.1 必达指标
- [ ] L1 Accuracy > 60%
- [ ] L2 Accuracy > 50%
- [ ] L3 Accuracy > 30%
- [ ] Overall > 55%
- [ ] Writeback Δ < 5%
- [ ] 跷跷板检查通过

### 6.2 观察指标
- [ ] 最佳 checkpoint 出现位置
- [ ] 训练稳定性 (loss 曲线)
- [ ] TSLA 事件分布
- [ ] 梯度比例变化

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| 跷跷板无法消除 | 中 | 高 | 考虑任务隔离或专家混合 |
| 训练不稳定 | 低 | 中 | 降低学习率，增加 warmup |
| 过拟合 L1 | 中 | 中 | 增加 L2/L3 采样比例 |
| Guard 失效 | 低 | 高 | 检查配置，保持冻结 |

---

## 8. 里程碑

### Milestone 1: 策略验证 (R1)
- 完成动态权重实验
- 找到初步平衡点
- 时间预估: 1-2 天

### Milestone 2: 优化迭代 (R2-R3)
- 尝试交替冻结和专家混合
- 对比不同策略效果
- 时间预估: 2-3 天

### Milestone 3: 最终验证 (R4)
- 确定最优策略
- 完整验收测试
- 生成最终报告
- 时间预估: 1-2 天

---

## 9. 文件规划

```
stage9_dynamic_weight.py      # R1 动态权重实验
stage9_alternate_freeze.py    # R2 交替冻结实验
stage9_mixture_of_experts.py  # R3 专家混合实验
stage9_meta_learning.py       # R4 元学习实验

STAGE9_FINAL_REPORT.md        # 最终报告
```

---

## 10. 启动检查清单

- [x] Stage 8 已完成并验收
- [x] Guard 和 TSLA 配置已冻结
- [x] 数据集已准备就绪
- [x] 实验策略已确定
- [x] 验收标准已明确

**准备就绪，可以启动 Stage 9！**

---

## 附录: Stage 8 关键数据参考

### Stage 8-R4 最佳表现
```
Accuracy: 100% (单层 L2)
Writeback Δ: +0.0007
TSLA Events: 110
Loss: 0.08
```

### Stage 8-R2 跷跷板现象
```
Step 126: L1=100%, L2=0%, L3=0%
Step 151: L1=0%, L2=100%, L3=100%
```

### 基线模型配置
```python
hidden_size: 128
num_layers: 4
num_heads: 4
params: 3.1M
```

---

**文档版本**: 1.0  
**创建日期**: 2026-04-20  
**最后更新**: 2026-04-20
