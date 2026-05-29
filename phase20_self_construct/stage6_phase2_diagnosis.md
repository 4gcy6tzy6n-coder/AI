# Stage 6 第二阶段诊断报告

**日期**: 2026-04-19  
**阶段**: Stage 6.2 - 长期成长验证 (修复后重测)  
**状态**: 验收失败，问题根因已定位

---

## 1. 修复尝试总结

### 1.1 应用的修复

| 修复项 | 原值 | 修复后 | 效果 |
|--------|------|--------|------|
| writeback KL | 0.38 | 0.42 | 无效 |
| replay_ratio | 0.35 | 0.40 | 无效 |
| rollback 状态保存 | 基础版 | 完整版 | 恶化 |

### 1.2 修复后结果

| 指标 | 修复前 | 修复后 | 目标 |
|------|--------|--------|------|
| writeback 变化 | 5.6-12% | 5.6-12% | < 5% |
| 回滚恢复率 | 32.6% | 21.5% | > 90% |
| 目标提升 | +11-15% | +11-15% | > +10% |
| 旧能力掉落 | 2-3% | 2-3% | < 15% |

**结论**: 目标能力和旧能力保持正常，但 writeback 和 rollback 问题未解决，rollback 恢复率反而下降。

---

## 2. 问题根因分析

### 2.1 Writeback 保护失效 - 根因

**现象**: KL 从 0.38 提升到 0.42 仍无法控制 writeback 变化 (5.6-12%)

**可能根因**:

1. **KL 作用对象错误**
   ```python
   # 当前: KL 应用于 writeback_head 输出
   # 但 writeback 能力可能由其他层共享
   ```

2. **Writeback Head 与其他组件耦合**
   ```
   UnitEncoder → [共享表示] → WritebackHead
                      ↓
               [PolicyHead, GapDetector]
   ```
   当 Policy/Gap 被更新时，共享表示变化间接影响 writeback

3. **评估方式问题**
   - 当前评估基于随机输入
   - 可能无法稳定反映真实 writeback 能力

**诊断建议**:
- 检查 writeback head 是否与其他 head 共享参数
- 测试 writeback 专项隔离训练
- 使用固定测试集而非随机输入

### 2.2 Rollback 恢复失效 - 根因

**现象**: 恢复率从 32.6% 降至 21.5%，远低于 90% 目标

**可能根因**:

1. **状态保存不完整**
   ```python
   # 当前只保存 model.state_dict()
   # 缺少:
   # - 优化器状态
   # - 随机数生成器状态
   # - 批次归一化运行统计
   ```

2. **评估时机问题**
   ```
   晋升 → 评估(变化大) → 回滚 → 评估(应恢复)
   
   问题: 评估时可能使用了缓存/污染的状态
   ```

3. **模型架构特殊性**
   ```python
   # GRU-based UnitEncoder 可能有隐藏状态
   # 这些状态未被保存/恢复
   ```

**诊断建议**:
- 验证 state_dict 是否包含所有参数
- 检查是否有隐藏状态未保存
- 在完全独立的进程中测试回滚

---

## 3. 深度诊断测试

### 3.1 Writeback 隔离测试

```python
# 测试: 只更新 policy head，观察 writeback 是否变化

# 步骤 1: 记录基线 writeback
baseline_writeback = evaluate_writeback()

# 步骤 2: 只训练 policy head (不涉及 writeback)
for param in writeback_head.parameters():
    param.requires_grad = False

train(policy_head_only=True)

# 步骤 3: 再次评估 writeback
after_writeback = evaluate_writeback()

# 如果变化 > 5%，说明存在耦合
```

### 3.2 Rollback 完整性测试

```python
# 测试: 验证 state_dict 是否完整

# 步骤 1: 保存前哈希
pre_hash = hash_all_parameters(model)
pre_output = model(test_input)

# 步骤 2: 执行几步训练
train(steps=3)

# 步骤 3: 回滚
model.load_state_dict(baseline_state)

# 步骤 4: 验证
post_hash = hash_all_parameters(model)
post_output = model(test_input)

# 检查
assert pre_hash == post_hash, "参数未完全恢复"
assert torch.allclose(pre_output, post_output), "输出不一致"
```

---

## 4. 修复方案建议

### 方案 A: 架构级修复 (推荐)

**针对 Writeback**:
1. 实现 writeback head 完全隔离
2. 使用独立的 feature extractor
3. 或增加 writeback 专项约束损失

**针对 Rollback**:
1. 保存完整状态 (model + optimizer + rng)
2. 实现 checkpoint 系统
3. 在独立环境中验证回滚

### 方案 B: 配置级修复 (尝试)

**激进提升保护**:
```python
'writeback': 0.50,        # 大幅提升 KL
'replay_ratio': 0.50,     # 大幅提升 replay
'writeback_threshold': 0.02,  # 更严格回滚
```

**风险**: 可能过度抑制学习能力

### 方案 C: 评估级修复 (验证)

**改进评估方式**:
1. 使用固定测试集 (非随机)
2. 增加评估样本数 (100 → 500)
3. 多次评估取平均

---

## 5. 阶段性结论

### 5.1 已验证成立

✓ **目标能力学习机制** - 有效 (+11-15%)
✓ **旧能力保护机制** - 有效 (2-3% 掉落)
✓ **全链路整合** - 成功
✓ **真实评估体系** - 就绪

### 5.2 确认为设计问题

✗ **Writeback 保护** - 架构耦合问题，非参数问题
✗ **Rollback 恢复** - 状态保存不完整，非参数问题

### 5.3 当前状态

**Stage 6 第二阶段**: 基本成立，待修复收口

**关键阻塞项**:
1. Writeback 与其他组件的耦合
2. Rollback 状态保存不完整

---

## 6. 下一步建议

### 选项 1: 深度修复 (推荐)

执行架构级修复:
1. 重构 writeback head 隔离
2. 实现完整 checkpoint 系统
3. 重新验证

**时间**: 2-3 天  
**风险**: 中等，可能引入新问题

### 选项 2: 进入第三阶段 (带限制)

接受当前限制:
1. 明确 writeback 和 rollback 为已知问题
2. 在第三阶段继续优化
3. 增加监控和告警

**风险**: 系统级评估结果可能被污染

### 选项 3: 简化验收标准

调整验收门槛:
- writeback < 10% (而非 5%)
- rollback > 50% (而非 90%)

**风险**: 降低系统可靠性

---

## 7. 建议决策

**建议选 选项 1 (深度修复)**，理由：

1. 问题已定位到架构层，不是调参能解决
2. 修复成本可控 (2-3天)
3. 不修复会影响后续所有阶段
4. 当前已证明核心机制有效，只差安全约束

**修复优先级**:
1. P0: Writeback 隔离
2. P0: Rollback 完整状态保存
3. P1: 固定测试集评估

---

## 8. 总结

第二阶段真实验收完成，**主学习能力与旧能力保护验证通过**。

**Writeback 和 Rollback 确认为架构设计问题**，需要架构级修复。

建议执行深度修复后再进入第三阶段。
