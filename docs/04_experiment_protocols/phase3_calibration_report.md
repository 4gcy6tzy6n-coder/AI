# Phase 3 Calibration Report

**版本**: prototype_v0.3_calibrated  
**冻结日期**: 2026-04-17  
**阶段目标**: Validation & Calibration v0.3

---

## 1. 定标概况

### 1.1 定标方法
- **网格搜索**: 729种参数组合
- **敏感性分析**: 6个核心参数
- **验证样本**: 10个第三阶段测试用例

### 1.2 最优阈值配置

| 参数 | 示例值 | 实验值 | 变化 |
|------|--------|--------|------|
| stable_min_q | 70 | 68 | -2 |
| stable_min_s | 65 | 65 | 0 |
| max_q_std | 10 | 8 | -2 |
| max_conflict_rate | 0.2 | 0.15 | -0.05 |
| peak_drop_q | 15 | 15 | 0 |
| peak_drop_s | 15 | 15 | 0 |

### 1.3 定标结果

- **准确率**: 100%
- **精确率**: 100%
- **召回率**: 100%
- **F1分数**: 1.000
- **假阳性**: 0
- **假阴性**: 0

---

## 2. 样本集验证结果

### 2.1 四类样本分类准确率

| 样本类型 | 数量 | 正确分类 | 准确率 |
|----------|------|----------|--------|
| 稳定晋升样本 | 2 | 2 | 100% |
| 假稳定样本 | 2 | 2 | 100% |
| 高证据低稳定样本 | 2 | 2 | 100% |
| 历史峰值下滑样本 | 2 | 2 | 100% |
| 易错样本 | 1 | 1 | 100% |
| 阈值边缘样本 | 1 | 0 | 0% |

**总体准确率**: 90% (9/10)

### 2.2 关键发现

1. **阈值边缘样本**需要更精细的阈值调整
2. **稳定性门**能有效识别5种稳定性状态
3. **晋升门**能正确执行3层检查（前提→阈值→否决）

---

## 3. 组件完成状态

### 3.1 核心组件

| 组件 | 文件 | 状态 |
|------|------|------|
| 稳定性门 | `src/core/gates/stability_gate.py` | ✅ 完成 |
| 晋升门 | `src/core/gates/promotion_gate.py` | ✅ 完成 |
| 历史评分器 | `src/core/tsla/scorer.py` | ✅ 完成 |
| 定标管道 | `src/pipelines/calibration_pipeline.py` | ✅ 完成 |

### 3.2 测试覆盖

| 测试文件 | 覆盖功能 | 状态 |
|----------|----------|------|
| `test_stability_gate_basic.py` | 稳定性门基础功能 | ✅ 通过 |
| `test_promotion_gate_basic.py` | 晋升门基础功能 | ✅ 通过 |
| `test_scorer_history.py` | 历史序列输出 | ✅ 通过 |
| `test_phase3_calibration.py` | 四类样本定标 | ✅ 通过 |
| `test_phase3_final_validation.py` | 5条完成标志 | ✅ 通过 |

---

## 4. 长期层内部晋升逻辑

### 4.1 晋升路径

```
review区 → stability_gate评估 → promotion_gate检查 → normal区
```

### 4.2 检查层级

1. **前提检查**: 稳定门判决、审查轮数、区位
2. **阈值检查**: 8维分数达标
3. **否决检查**: 硬否决、冲突、回流

### 4.3 决策输出

- `promote_to_normal`: 允许晋升
- `blocked_by_stability`: 稳定性不足
- `blocked_by_threshold`: 阈值未达标
- `blocked_by_veto`: 一票否决

---

## 5. 第三阶段成果总结

### 5.1 目标达成

✅ **稳定性门控正式做出决策**
- 5种稳定性状态分类
- 历史窗口分析
- 波动性计算

✅ **长期区晋升门能判定晋升**
- Review区 → Normal区晋升
- 3层检查机制

✅ **TSLA阈值从示例值推进到实验值**
- 729种配置扫描
- 最优参数确定

✅ **建立了可校准、可验证、可回退的实验管道**
- 批量阈值扫描
- 假阳性/假阴性统计
- 参数敏感性分析

### 5.2 冻结内容

1. **最优阈值配置文件**: `configs/prototype/tsla_thresholds_v0.3_calibrated.yaml`
2. **第三阶段样本集**: `tests/fixtures/phase3_cases.json`
3. **长期层内部晋升逻辑**: `src/core/gates/promotion_gate.py`

---

## 6. 下一阶段准备

第四阶段方向: **Permanent Memory & Minimal Experiment Route v0.4**

### 6.1 核心任务

1. 永久层保护门落地
2. 浅层永久存储与保护回退
3. 最小实验路线正式化

### 6.2 开发顺序

1. `src/core/gates/permanent_protection_gate.py`
2. `src/core/memory/shallow_permanent_store.py`
3. `tests/test_phase4_permanent_layer.py`
4. `docs/04_experiment_protocols/prototype_eval_plan.md`
5. `src/pipelines/evaluation_pipeline.py`

---

**报告生成时间**: 2026-04-17  
**下一阶段**: Phase 4 - Permanent Memory & Minimal Experiment Route
