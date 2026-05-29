# Stage 6 官方基线 V1.0

**版本**: V1.0  
**日期**: 2026-04-19  
**状态**: Stage 6 第二阶段正式通过版本，第三阶段唯一官方配置  
**来源**: Stage 6.2 真实验收 (修复前基线)

---

## 官方配置

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

## 验证通过的指标

| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| **目标能力** | **+11% ~ +14%** | > +10% | ✓ **正式通过** |
| **旧能力掉落** | **2% ~ 3%** | < 15% | ✓ **正式通过** |
| 全链路整合 | 成功 | - | ✓ 正式通过 |
| 真实评估体系 | 就绪 | - | ✓ 正式通过 |

---

## 已知限制 (第三阶段优化目标)

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| **writeback 变化** | **5.6% ~ 12%** | < 5% | ⚠️ 已知限制 |
| **rollback 恢复率** | **32.6%** | > 90% | ⚠️ 已知限制 |

---

## 第二阶段正式结论

**Stage 6 第二阶段已完成真实验收，以下机制正式验证通过：**

✓ **目标能力学习机制** - 稳定提升 +11-14%  
✓ **旧能力保护机制** - 掉落控制在 2-3%  
✓ **全链路整合** - 成功运行  
✓ **真实评估体系** - 建立并标准化  

**当前 writeback isolation 修复分支判定为失败实现，退出主线。**

后续如需继续研究 writeback 隔离，应以"重写实现"方式单独推进，不再基于当前失败分支。

---

## 第三阶段入口

此基线配置作为 Stage 6 第三阶段**唯一官方配置**。

**第三阶段任务**:
1. 在系统级环境中验证基线配置
2. 继续优化 writeback 和 rollback 指标
3. 完成 Stage 6 最终验收

---

## 附件

- [stage6_phase2_conclusion.md](stage6_phase2_conclusion.md) - 第二阶段完整结论
- [stage6_evaluation_protocol.py](stage6_evaluation_protocol.py) - 统一评估协议
