# Stage 6 Phase 3 Entry

**阶段**: Stage 6.3 - 系统级评估  
**日期**: 2026-04-19  
**状态**: 第三阶段入口，使用官方基线 V1.0

---

## 第二阶段正式结论

**Stage 6 第二阶段已完成真实验收，以下机制正式验证通过：**

✓ **目标能力学习机制** - 稳定提升 +11-14%  
✓ **旧能力保护机制** - 掉落控制在 2-3%  
✓ **全链路整合** - 成功运行  
✓ **真实评估体系** - 建立并标准化  
✓ **评估协议** - 已修复并统一

**当前 writeback isolation 修复分支判定为失败实现，已归档退出主线。**

---

## 第三阶段官方配置

**唯一官方配置**: [stage6_official_baseline_v1.md](stage6_official_baseline_v1.md)

```python
OFFICIAL_BASELINE_CONFIG = {
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}
```

---

## 第三阶段目标

### 核心目标

1. **在系统级环境中验证基线配置**
   - 端到端行为测试
   - 多轮对话测试
   - 长时间运行稳定性

2. **继续优化已知限制**
   - writeback 变化: 5.6-12% → < 5%
   - rollback 恢复率: 32.6% → > 90%

3. **完成 Stage 6 最终验收**
   - 所有指标达标
   - 系统级验证通过

---

## 第三阶段任务清单

### Task 1: 系统级环境搭建

- [ ] 搭建端到端测试环境
- [ ] 集成所有模块到统一系统
- [ ] 建立系统级监控和日志

### Task 2: 端到端行为测试

- [ ] 单轮查询测试
- [ ] 多轮对话测试
- [ ] 复杂场景测试

### Task 3: 长期稳定性测试

- [ ] 100+ 步连续晋升测试
- [ ] 内存泄漏检查
- [ ] 性能衰减检查

### Task 4: 已知限制优化

- [ ] writeback 保护优化 (可选，可并行)
- [ ] rollback 恢复优化 (可选，可并行)

### Task 5: 最终验收

- [ ] 完整系统级评估
- [ ] 生成最终验收报告
- [ ] Stage 6 正式完成

---

## 已知限制 (第三阶段优化目标)

| 指标 | 当前值 | 目标值 | 优先级 |
|------|--------|--------|--------|
| writeback 变化 | 5.6% ~ 12% | < 5% | P1 |
| rollback 恢复率 | 32.6% | > 90% | P1 |

**说明**: 这些限制不影响核心功能，但需要在第三阶段优化。

---

## 入口检查清单

进入第三阶段前，确认以下事项：

- [x] 第二阶段正式结论已生成
- [x] 官方基线 V1.0 已冻结
- [x] 失败实验已归档
- [x] 统一评估协议已建立
- [x] 第三阶段任务清单已定义

**所有检查项通过，可以进入第三阶段。**

---

## 关键文件

- [stage6_official_baseline_v1.md](stage6_official_baseline_v1.md) - 官方基线配置
- [stage6_phase2_conclusion.md](stage6_phase2_conclusion.md) - 第二阶段完整结论
- [stage6_evaluation_protocol.py](stage6_evaluation_protocol.py) - 统一评估协议
- [experiments/failed/EXP-6B-001-writeback-isolation.md](experiments/failed/EXP-6B-001-writeback-isolation.md) - 失败实验归档

---

## 下一步行动

**立即开始 Task 1: 系统级环境搭建**

建议产出：
1. `stage6_system_integration.py` - 系统集成框架
2. `stage6_end_to_end_test.py` - 端到端测试
3. `stage6_phase3_report.md` - 第三阶段进展报告

---

**Stage 6 第三阶段正式开始。**
