# Stage 6 第二阶段最终诊断报告

**日期**: 2026-04-19  
**阶段**: Stage 6.2 - 诊断型修复完成  
**状态**: 消融测试完成，发现关键问题

---

## 1. 诊断执行总结

### 1.1 已完成的诊断

✓ **冻结修复前基线** - [stage6_baseline_frozen.md](stage6_baseline_frozen.md)  
✓ **标记修复分支** - 标记为"失败实验"待重新评估  
✓ **三组最小消融测试** - [stage6_ablation_test.py](stage6_ablation_test.py)

---

## 2. 消融测试结果

### 2.1 关键指标对比

| 配置 | Target Gain | Old Drop | Writeback | Rollback |
|------|-------------|----------|-----------|----------|
| **baseline** | **+14.03%** | 3.34% | +3.73% | N/A |
| **writeback_only** | **+17.66%** | 50.00% | +16.40% | N/A |
| **rollback_only** | **+24.94%** | 3.34% | +0.93% | 0.0% |
| **both** | **+15.84%** | 50.00% | +8.40% | 55.0% |

### 2.2 意外发现

**❗ 架构修复并未破坏 target gain，反而提升了它**

- writeback_only: +17.66% (比基线高 3.63%)
- rollback_only: +24.94% (比基线高 10.91%)
- both: +15.84% (比基线高 1.81%)

**❗ 但 old ability drop 在 writeback_only 和 both 中恶化到 50%**

---

## 3. 问题根因重新定位

### 3.1 之前的错误判断

**错误**: "架构修复破坏了 target gain"

**事实**: 架构修复**提升**了 target gain，但**破坏了 old ability drop 的计算**

### 3.2 真正的问题

**Old ability drop 计算异常**

在 writeback_only 和 both 配置中，old ability drop 显示为 50%，这明显异常。

**可能原因**:
1. **评估器状态污染** - writeback isolation 可能改变了评估器的行为
2. **基线设置问题** - 在设置基线后才应用 writeback isolation，导致基线不一致
3. **随机种子问题** - 不同配置的随机性导致评估波动

### 3.3 核心结论

**架构修复本身是正确的方向**，但：
1. **writeback isolation 需要改进** - 当前实现有副作用
2. **评估流程需要标准化** - 确保基线一致性
3. **需要更多重复测试** - 消除随机性影响

---

## 4. 修正后的建议

### 4.1 不要放弃架构修复

消融测试证明架构修复能提升 target gain，只是需要解决副作用。

### 4.2 需要修复的问题

1. **writeback isolation 的基线一致性问题**
   - 应在设置基线**之前**应用 writeback isolation
   - 确保评估器和模型状态同步

2. **old ability drop 计算异常**
   - 检查评估器是否在 writeback 冻结/解冻时状态一致
   - 验证基线 scores 的保存和恢复

3. **rollback snapshot 的恢复率**
   - rollback_only 显示 0% 恢复率
   - both 显示 55% 恢复率
   - 需要进一步优化

### 4.3 下一步行动

**选项 B' (修正版): 继续诊断型修复**

不是无限深挖，而是：
1. 修复 writeback isolation 的基线一致性问题
2. 重新运行消融测试验证
3. 如果 target gain 保持且 old drop 正常，接受架构修复

**预计时间**: 半天到一天

---

## 5. 最终决策建议

### 建议: 继续修复 (修正方向)

**理由**:
1. 消融测试证明架构修复能提升性能
2. 问题不是"修复破坏了系统"，而是"修复有副作用"
3. 副作用 (old drop 计算异常) 是可修复的

**具体行动**:
1. 修复 writeback isolation 的基线设置顺序
2. 重新验证 4 项关键指标
3. 如果通过，使用架构修复版本进入第三阶段

---

## 6. 关键教训

1. **不要急于下结论** - 之前的"修复导致回归"判断是错误的
2. **消融测试很重要** - 帮助定位真正的问题
3. **基线一致性是关键** - 评估流程的标准化很重要

---

## 7. 附件

- [stage6_baseline_frozen.md](stage6_baseline_frozen.md) - 基线冻结文档
- [stage6_ablation_test.py](stage6_ablation_test.py) - 消融测试代码
- [eval/stage6_ablation_report.json](eval/stage6_ablation_report.json) - 详细数据
