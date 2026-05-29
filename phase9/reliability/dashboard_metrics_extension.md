# Dashboard Metrics Extension - 仪表板指标扩展

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 9 - WP4: 工程可靠性与可观测性增强

---

## 1. 概述

本文档定义 Phase 9 新增的仪表板指标，用于增强系统的可观测性。

---

## 2. 延迟指标

### 2.1 检索延迟

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `retrieval_latency_p50` | Gauge | 检索 P50 延迟 | ms |
| `retrieval_latency_p95` | Gauge | 检索 P95 延迟 | ms |
| `retrieval_latency_p99` | Gauge | 检索 P99 延迟 | ms |
| `retrieval_latency_avg` | Gauge | 检索平均延迟 | ms |

### 2.2 治理延迟

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `governance_latency_p50` | Gauge | 治理 P50 延迟 | ms |
| `governance_latency_p95` | Gauge | 治理 P95 延迟 | ms |
| `governance_latency_p99` | Gauge | 治理 P99 延迟 | ms |

### 2.3 写回延迟

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `writeback_latency_p50` | Gauge | 写回 P50 延迟 | ms |
| `writeback_latency_p95` | Gauge | 写回 P95 延迟 | ms |
| `writeback_latency_p99` | Gauge | 写回 P99 延迟 | ms |

---

## 3. 各层对象分布

### 3.1 Unit 分布

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `units_short_term` | Gauge | 短期层 Units 数量 | count |
| `units_long_term` | Gauge | 长期层 Units 数量 | count |
| `units_permanent` | Gauge | 永久层 Units 数量 | count |
| `units_isolated` | Gauge | 隔离区 Units 数量 | count |
| `units_error` | Gauge | 错误区 Units 数量 | count |

### 3.2 Unit 类型分布

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `units_by_type{unit_type="concept"}` | Gauge | 概念型 Units | count |
| `units_by_type{unit_type="relation"}` | Gauge | 关系型 Units | count |
| `units_by_type{unit_type="rule"}` | Gauge | 规则型 Units | count |
| `units_by_type{unit_type="task_pattern"}` | Gauge | 任务模式型 Units | count |

---

## 4. TSLA 动作分布

### 4.1 动作计数

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `tsla_actions_total{action="promote"}` | Counter | 晋升动作总数 | count |
| `tsla_actions_total{action="isolate"}` | Counter | 隔离动作总数 | count |
| `tsla_actions_total{action="retry"}` | Counter | 重试动作总数 | count |
| `tsla_actions_total{action="discard"}` | Counter | 丢弃动作总数 | count |

### 4.2 动作速率

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `tsla_action_rate{action="promote"}` | Gauge | 晋升速率 | count/s |
| `tsla_action_rate{action="isolate"}` | Gauge | 隔离速率 | count/s |

---

## 5. 隔离区/错误区负载趋势

### 5.1 隔离区指标

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `isolation_zone_size` | Gauge | 隔离区大小 | count |
| `isolation_zone_growth_rate` | Gauge | 隔离区增长率 | count/h |
| `isolation_zone_contamination_rate` | Gauge | 隔离区污染率 | % |
| `isolation_by_reason{reason="quality_threshold"}` | Counter | 质量阈值隔离 | count |
| `isolation_by_reason{reason="conflict_detected"}` | Counter | 冲突检测隔离 | count |
| `isolation_by_reason{reason="manual_review"}` | Counter | 人工审查隔离 | count |

### 5.2 错误区指标

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `error_zone_size` | Gauge | 错误区大小 | count |
| `error_zone_growth_rate` | Gauge | 错误区增长率 | count/h |
| `error_by_type{type="syntax"}` | Counter | 语法错误 | count |
| `error_by_type{type="semantic"}` | Counter | 语义错误 | count |
| `error_by_type{type="conflict"}` | Counter | 冲突错误 | count |

---

## 6. 英文 Relation Error 分布

### 6.1 错误类型分布

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `english_relation_errors_total{type="FN-1"}` | Counter | 介词关系漏检 | count |
| `english_relation_errors_total{type="FN-2"}` | Counter | 从句关系漏检 | count |
| `english_relation_errors_total{type="FP-1"}` | Counter | 虚词误检 | count |
| `english_relation_errors_total{type="FP-2"}` | Counter | 修饰语误检 | count |
| `english_relation_errors_total{type="MC-1"}` | Counter | 关系类型错分 | count |
| `english_relation_errors_total{type="MC-2"}` | Counter | 方向性错误 | count |

### 6.2 检测质量

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `english_relation_precision` | Gauge | 精确率 | % |
| `english_relation_recall` | Gauge | 召回率 | % |
| `english_relation_f1` | Gauge | F1 分数 | % |

---

## 7. 双语一致性指标

### 7.1 对齐分数

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `bilingual_semantic_alignment` | Gauge | 语义对齐分数 | 0-1 |
| `bilingual_governance_alignment` | Gauge | 治理动作对齐分数 | 0-1 |
| `bilingual_quality_alignment` | Gauge | 质量分数对齐 | 0-1 |
| `bilingual_unit_type_consistency` | Gauge | Unit 类型一致性 | 0-1 |
| `bilingual_overall_score` | Gauge | 综合对齐分数 | 0-1 |

---

## 8. 缓存指标

### 8.1 缓存性能

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `cache_hit_rate` | Gauge | 缓存命中率 | % |
| `cache_l1_hits` | Counter | L1 缓存命中 | count |
| `cache_l2_hits` | Counter | L2 缓存命中 | count |
| `cache_misses` | Counter | 缓存未命中 | count |
| `cache_size` | Gauge | 缓存大小 | count |

---

## 9. 并发指标

### 9.1 并发控制

| 指标名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| `concurrent_reads` | Gauge | 当前并发读数 | count |
| `concurrent_writes_kb` | Gauge | 当前并发知识库写 | count |
| `concurrent_writes_param` | Gauge | 当前并发参数写 | count |
| `queue_size` | Gauge | 任务队列大小 | count |
| `rate_limit_rejections` | Counter | 限流拒绝数 | count |

---

## 10. 仪表板布局建议

### 10.1 概览面板

```
┌─────────────────────────────────────────────────────────────┐
│  系统概览                                                   │
├─────────────────────────────────────────────────────────────┤
│  [延迟趋势图]  [Unit 分布饼图]  [TSLA 动作柱状图]           │
├─────────────────────────────────────────────────────────────┤
│  [隔离区/错误区趋势]  [缓存命中率]  [并发数]                │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 性能面板

```
┌─────────────────────────────────────────────────────────────┐
│  性能指标                                                   │
├─────────────────────────────────────────────────────────────┤
│  [P50/P95/P99 延迟]  [检索延迟分布]  [写回延迟分布]         │
├─────────────────────────────────────────────────────────────┤
│  [缓存性能]  [并发控制状态]  [队列深度]                     │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 质量面板

```
┌─────────────────────────────────────────────────────────────┐
│  质量指标                                                   │
├─────────────────────────────────────────────────────────────┤
│  [英文 Relation F1]  [双语对齐分数]  [治理通过率]           │
├─────────────────────────────────────────────────────────────┤
│  [英文错误类型分布]  [隔离原因分布]  [错误类型分布]         │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义 Phase 9 新增指标 |

---

**文档状态**: 生效中  
**仪表板负责人**: Phase 9 WP4 负责人
