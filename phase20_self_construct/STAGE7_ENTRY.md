# Stage 7 Entry

**日期**: 2026-04-19  
**上一阶段**: Stage 6 (✅ 已完成)  
**当前阶段**: Stage 7 - 系统优化与扩展

---

## Stage 6 完成确认

### 核心结论

✅ **Stage 6 已正式完成，必须项 4/4 通过。**

### 关键成果

- 目标提升: +15.67% (> 10% 目标)
- 旧能力保护: 2.26% (< 15% 目标)
- 官方基线 V1.0: 已冻结
- 系统级评估框架: 已建立

### 已知限制 (Stage 7 优化任务)

| 限制项 | 当前值 | 目标 | 优先级 |
|--------|--------|------|--------|
| writeback 变化 | 13.60% | < 5% | P1 |
| rollback 恢复 | 32.6% | > 90% | P1 |

---

## Stage 7 总目标

**在不破坏 Stage 6 官方基线 V1.0 的前提下，专项优化两个遗留问题。**

### 核心原则

1. **保持 Stage 6 主链**: 四个必须项不能退化
2. **专项优化**: writeback 和 rollback 作为独立专题
3. **带护栏优化**: 每次改动自动检查回归

---

## Stage 7 四块内容

### 1. Writeback 专项优化 (第一主线)

**目标**: 13.60% → < 5%

**设计文档**: [stage7_writeback_design_v2.md](stage7_writeback_design_v2.md)

**核心方案**:
- 方案 A: 特征隔离 (Feature Isolation)
- 方案 B: 梯度屏蔽 (Gradient Masking)
- 方案 C: 独立优化器 (Independent Optimizer)
- 方案 D: 综合保护 (Combined Protection)

**交付物**:
- ✅ `stage7_writeback_design_v2.md` (设计文档)
- ✅ `stage7_writeback_guard.py` (可插拔框架)
  - `build_writeback_guard(mode, config)`
  - `apply_feature_isolation()`
  - `apply_gradient_mask()`
  - `build_independent_optimizer()`
  - `run_writeback_protected_step()`
- ⏳ `stage7_writeback_validation.py` (验证)

### 2. Rollback 专项优化 (第二主线)

**目标**: 32.6% → > 90%

**契约文档**: [stage7_rollback_contract.md](stage7_rollback_contract.md)

**完整快照范围**:
- Model Parameters (P0)
- Optimizer State (P0)
- Scheduler State (P0)
- Random State (P0)
- Writeback/Governance Runtime (P1)
- Evaluation Cache (P1)
- Memory Runtime (P2)

**交付物**:
- ✅ `stage7_rollback_contract.md` (契约定义)
- ✅ `stage7_rollback_snapshot_v2.py` (完整实现)
  - `save_full_snapshot()`
  - `load_full_snapshot()`
  - `verify_snapshot_integrity()`
  - `run_recovery_check()`
- ⏳ `stage7_rollback_recovery_test.py` (测试)

### 3. 回归保护与统一验收 (第三块)

**目标**: 建立带护栏优化机制

**实现**: [stage7_regression_gate.py](stage7_regression_gate.py)

**检查项**:
- Stage 6 必须项 (不可突破):
  - target gain > 10%
  - old ability drop < 15%
  - E2E success > 90%
  - 100-step stability pass
- Stage 7 优化目标:
  - writeback change < 5%
  - rollback recovery > 90%

**交付物**:
- ✅ `stage7_regression_gate.py` (回归护栏)
- ✅ `stage7_ablation_test.py` (消融测试框架)
- ⏳ `stage7_acceptance_suite.py` (验收套件)
- ⏳ `stage7_metric_dashboard.md` (指标看板)

### 4. 最终集成与 Stage 7 验收 (第四块)

**目标**: 产出 Stage 7 优化版官方候选基线

**交付物**:
- ⏳ `stage7_final_validation.py`
- ⏳ `STAGE7_FINAL_REPORT.md`
- ⏳ `stage7_candidate_baseline_v1.md`

---

## 执行顺序

### 已完成 ✅

1. ✅ **Task 1**: writeback 根因重建 (设计完成)
2. ✅ **Task 2**: rollback 完整快照重建 (契约完成)
3. ✅ **Task 3**: 回归护栏搭建 (实现完成)
4. ✅ **Task 4.1**: 实现 `stage7_rollback_snapshot_v2.py`
5. ✅ **Task 4.2**: 实现 `stage7_writeback_guard.py` (可插拔框架)
6. ✅ **Task 4.3**: 单策略实验框架 (`stage7_ablation_test.py`)

### 进行中 ⏳

7. ⏳ **Task 4.4**: 运行消融测试，通过 regression gate 筛选候选
8. ⏳ **Task 5**: 系统级集成验收

---

## Stage 7 验收标准

### 必须项

| 指标 | 当前 | 目标 | 状态 |
|------|------|------|------|
| writeback change | 13.60% | < 5% | ⏳ |
| rollback recovery | 32.6% | > 90% | ⏳ |
| target gain | 15.67% | > 10% | ✅ |
| old ability drop | 2.26% | < 15% | ✅ |
| E2E success | 95% | > 90% | ✅ |
| 100-step stability | 通过 | 通过 | ✅ |

### 结果要求

**只有当新指标达标且旧指标不退化，Stage 7 才算完成。**

---

## 最重要的边界

### Stage 6 官方基线 V1.0 不能动

Stage 7 的所有实现都应该：
- 在新分支上做
- 跟基线做对照
- 通过后再升级为新候选基线

**不能再回头修改 Stage 6 主结论。**

---

## 一句话总结

> Stage 7 的任务，就是在不破坏 Stage 6 已验证主链的前提下，完成 writeback 与 rollback 两条安全链的正式工程化修复，并产出新的候选官方基线。

---

## 关键文件

### 已完成

- [stage7_writeback_design_v2.md](stage7_writeback_design_v2.md) - Writeback V2 设计
- [stage7_rollback_contract.md](stage7_rollback_contract.md) - Rollback 契约
- [stage7_regression_gate.py](stage7_regression_gate.py) - 回归护栏
- [stage7_rollback_snapshot_v2.py](stage7_rollback_snapshot_v2.py) - Rollback V2 实现
- [stage7_writeback_guard.py](stage7_writeback_guard.py) - Writeback 保护框架
- [stage7_ablation_test.py](stage7_ablation_test.py) - 消融测试框架

### 待实现

- `stage7_final_validation.py` - 最终验证
- `STAGE7_FINAL_REPORT.md` - 最终报告
- `stage7_candidate_baseline_v1.md` - 候选基线

---

## 下一步行动

1. **运行消融测试** (`stage7_ablation_test.py`)
2. **通过 regression gate 筛选最优策略**
3. **进入系统级集成验收**

---

**Stage 7 进行中。**

**入口时间**: 2026-04-19  
**上一阶段**: Stage 6 (✅ 完成)  
**当前状态**: 核心实现完成，准备运行消融测试
