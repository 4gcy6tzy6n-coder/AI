# Stage 6 指标契约文档

**版本**: v1.0  
**日期**: 2026-04-18  
**用途**: 定义 Stage 6 全链路整合中的统一指标收集规范

---

## 1. 指标设计原则

### 1.1 核心目标

Stage 6 的指标设计服务于一个核心问题：

> **"已经成立的受控成长机制，在完整系统中能否长期稳定工作？"**

### 1.2 指标层级

```
Level 1: 单步指标 (Per-Step Metrics)
    ↓ 聚合
Level 2: 累积指标 (Cumulative Metrics)
    ↓ 分析
Level 3: 系统指标 (System Metrics)
```

### 1.3 数据完整性要求

- **必须记录**: 所有单步指标不可缺失
- **时间戳**: 每个指标必须带 ISO 8601 时间戳
- **可追溯**: 每个指标能追溯到具体候选和晋升步骤
- **可对比**: 支持基线对比和跨实验对比

---

## 2. 单步指标 (StepMetrics)

### 2.1 必录指标

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `step_number` | int | 步骤序号 | 1, 2, 3... |
| `timestamp` | str | ISO 8601 时间戳 | "2026-04-18T10:30:00" |
| `candidate_id` | str | 候选 ID | "cand_20260418_103000_1" |
| `candidate_type` | str | 候选类型 | "RELATION" |
| `promotion_type` | str | 晋升类型 | "PARAM" / "KB" |

### 2.2 能力提升指标

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| `target_improvement` | float | 目标能力提升 | current_target - baseline_target |
| `target_absolute` | float | 目标能力绝对值 | 0.0 - 1.0 |
| `learning_efficiency` | float | 学习效率 | target_improvement / step_cost |

### 2.3 旧能力保护指标

| 字段名 | 类型 | 说明 | 阈值参考 |
|--------|------|------|----------|
| `old_ability_drop` | float | 最大旧能力掉落 | < 0.10 (10%) |
| `retrieval_change` | float | retrieval 变化 | 监控项 |
| `policy_change` | float | policy 变化 | 监控项 |
| `governance_change` | float | governance 变化 | 监控项 |
| `writeback_change` | float | writeback 变化 | < 0.05 (5%) |

### 2.4 保护机制指标

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `kl_weights` | dict | 各头 KL 权重 |
| `kl_weights.gap` | float | gap KL 权重 |
| `kl_weights.policy` | float | policy KL 权重 |
| `kl_weights.governance` | float | governance KL 权重 |
| `kl_weights.writeback` | float | writeback KL 权重 |
| `replay_ratio` | float | 回放比例 |
| `update_cap` | float | 更新幅度上限 |

### 2.5 回滚指标

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `rollback_triggered` | bool | 是否触发回滚 |
| `rollback_reason` | str | 回滚原因 |
| `rollback_success` | bool | 回滚是否成功 |
| `pre_rollback_metrics` | dict | 回滚前指标快照 |

### 2.6 资源消耗指标

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `training_time_ms` | int | 训练耗时 (毫秒) |
| `memory_usage_mb` | float | 内存使用 (MB) |
| `num_epochs` | int | 训练轮数 |
| `num_samples` | int | 样本数 |

### 2.7 JSON Schema

```json
{
  "StepMetrics": {
    "type": "object",
    "required": [
      "step_number",
      "timestamp",
      "candidate_id",
      "target_improvement",
      "old_ability_drop",
      "writeback_change"
    ],
    "properties": {
      "step_number": {"type": "integer", "minimum": 1},
      "timestamp": {"type": "string", "format": "date-time"},
      "candidate_id": {"type": "string"},
      "candidate_type": {"type": "string", "enum": ["RELATION", "EXPLANATION", "RULE", "PATTERN"]},
      "promotion_type": {"type": "string", "enum": ["PARAM", "KB"]},
      "target_improvement": {"type": "number"},
      "target_absolute": {"type": "number", "minimum": 0, "maximum": 1},
      "old_ability_drop": {"type": "number", "minimum": 0},
      "retrieval_change": {"type": "number"},
      "policy_change": {"type": "number"},
      "governance_change": {"type": "number"},
      "writeback_change": {"type": "number"},
      "kl_weights": {
        "type": "object",
        "properties": {
          "gap": {"type": "number"},
          "policy": {"type": "number"},
          "governance": {"type": "number"},
          "writeback": {"type": "number"}
        }
      },
      "rollback_triggered": {"type": "boolean"},
      "rollback_reason": {"type": "string"},
      "rollback_success": {"type": "boolean"}
    }
  }
}
```

---

## 3. 累积指标 (CumulativeMetrics)

### 3.1 聚合指标

| 字段名 | 类型 | 说明 | 计算方式 |
|--------|------|------|----------|
| `total_steps` | int | 总步骤数 | count |
| `total_candidates` | int | 总候选数 | count |
| `param_promotions` | int | 参数晋升次数 | count |
| `kb_promotions` | int | 知识库晋升次数 | count |
| `discards` | int | 丢弃数 | count |

### 3.2 统计指标

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `avg_target_improvement` | float | 平均目标提升 |
| `max_target_improvement` | float | 最大目标提升 |
| `min_target_improvement` | float | 最小目标提升 |
| `std_target_improvement` | float | 目标提升标准差 |
| `max_old_ability_drop` | float | 最大旧能力掉落 |
| `avg_old_ability_drop` | float | 平均旧能力掉落 |
| `cumulative_old_drop` | float | 累积旧能力变化 |

### 3.3 稳定性指标

| 字段名 | 类型 | 说明 | 理想值 |
|--------|------|------|--------|
| `rollback_rate` | float | 回滚率 | < 0.20 |
| `success_rate` | float | 成功率 | > 0.80 |
| `stability_score` | float | 稳定性评分 | > 0.90 |
| `trend_consistency` | float | 趋势一致性 | > 0.85 |

### 3.4 效率指标

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `total_training_time_ms` | int | 总训练时间 |
| `avg_time_per_step_ms` | float | 平均每步时间 |
| `throughput_candidates_per_hour` | float | 每小时处理候选数 |

---

## 4. 系统指标 (SystemMetrics)

### 4.1 端到端行为指标

| 字段名 | 类型 | 说明 | 评估方式 |
|--------|------|------|----------|
| `end_to_end_accuracy` | float | 端到端准确率 | 人工评估 |
| `response_quality_score` | float | 回复质量分 | 自动+人工 |
| `knowledge_retrieval_rate` | float | 知识检索率 | 日志统计 |
| `user_satisfaction` | float | 用户满意度 | 反馈收集 |

### 4.2 长期稳定性指标

| 字段名 | 类型 | 说明 | 监控周期 |
|--------|------|------|----------|
| `drift_score` | float | 漂移分数 | 每日 |
| `consistency_score` | float | 一致性分数 | 每周 |
| `degradation_rate` | float | 退化率 | 每月 |
| `recovery_time_ms` | int | 恢复时间 | 每次回滚 |

### 4.3 资源健康指标

| 字段名 | 类型 | 说明 | 告警阈值 |
|--------|------|------|----------|
| `memory_usage_percent` | float | 内存使用率 | > 80% |
| `cpu_usage_percent` | float | CPU 使用率 | > 90% |
| `disk_usage_percent` | float | 磁盘使用率 | > 85% |
| `gpu_utilization_percent` | float | GPU 利用率 | - |

---

## 5. 指标收集接口

### 5.1 Python API

```python
class MetricsCollector:
    """指标收集器"""
    
    def record_step(self, metrics: StepMetrics) -> None:
        """记录单步指标"""
        pass
    
    def get_step(self, step_number: int) -> Optional[StepMetrics]:
        """获取指定步骤指标"""
        pass
    
    def get_cumulative(self) -> CumulativeMetrics:
        """获取累积指标"""
        pass
    
    def export_json(self, filepath: str) -> None:
        """导出为 JSON"""
        pass
    
    def generate_report(self) -> str:
        """生成文本报告"""
        pass
    
    def check_thresholds(self) -> List[ThresholdViolation]:
        """检查阈值违规"""
        pass
```

### 5.2 存储格式

```python
# 文件命名规范
{experiment_id}_{timestamp}_metrics.json

# 示例
stage6_test1_20260418_103000_metrics.json
```

### 5.3 数据样例

```json
{
  "metadata": {
    "experiment_id": "stage6_smoke_test_001",
    "config_version": "6B",
    "start_time": "2026-04-18T10:30:00",
    "end_time": "2026-04-18T10:35:00"
  },
  "steps": [
    {
      "step_number": 1,
      "timestamp": "2026-04-18T10:30:15",
      "candidate_id": "cand_001",
      "candidate_type": "RELATION",
      "promotion_type": "PARAM",
      "target_improvement": 0.14,
      "target_absolute": 0.64,
      "old_ability_drop": 0.10,
      "retrieval_change": 0.02,
      "policy_change": 0.03,
      "governance_change": 0.00,
      "writeback_change": 0.00,
      "kl_weights": {
        "gap": 0.30,
        "policy": 0.30,
        "governance": 0.30,
        "writeback": 0.38
      },
      "rollback_triggered": false,
      "training_time_ms": 1250
    }
  ],
  "cumulative": {
    "total_steps": 1,
    "avg_target_improvement": 0.14,
    "max_old_ability_drop": 0.10,
    "rollback_rate": 0.0
  }
}
```

---

## 6. 阈值与告警

### 6.1 硬阈值 (Hard Thresholds)

| 指标 | 阈值 | 超限动作 |
|------|------|----------|
| `old_ability_drop` | > 0.10 | 触发回滚 |
| `writeback_change` | > 0.05 | 触发回滚 |
| `rollback_rate` | > 0.30 | 停止实验 |

### 6.2 软阈值 (Soft Thresholds)

| 指标 | 黄色告警 | 红色告警 |
|------|----------|----------|
| `target_improvement` | < 0.05 | < 0.00 |
| `old_ability_drop` | > 0.08 | > 0.10 |
| `stability_score` | < 0.85 | < 0.70 |

### 6.3 告警响应

```python
ALERT_RESPONSES = {
    'old_ability_drop_hard': {
        'action': 'ROLLBACK',
        'notify': True,
        'log_level': 'ERROR',
    },
    'target_improvement_soft': {
        'action': 'LOG_WARNING',
        'notify': False,
        'log_level': 'WARNING',
    },
}
```

---

## 7. 报告模板

### 7.1 单步报告

```
Step {step_number} Report
========================
Time: {timestamp}
Candidate: {candidate_id} ({candidate_type})

Learning:
  Target Improvement: {target_improvement:+.2%}
  Target Absolute: {target_absolute:.2%}

Protection:
  Old Ability Drop: {old_ability_drop:.2%}
  Writeback Change: {writeback_change:+.2%}
  KL Weights: gap={kl_weights.gap}, writeback={kl_weights.writeback}

Status: {status}
```

### 7.2 累积报告

```
Cumulative Report
=================
Total Steps: {total_steps}
Duration: {duration}

Learning Summary:
  Avg Target Improvement: {avg_target_improvement:+.2%}
  Max Target Improvement: {max_target_improvement:+.2%}

Protection Summary:
  Max Old Ability Drop: {max_old_ability_drop:.2%}
  Avg Old Ability Drop: {avg_old_ability_drop:.2%}
  Rollback Rate: {rollback_rate:.1%}

Stability:
  Success Rate: {success_rate:.1%}
  Stability Score: {stability_score:.2f}

Overall Status: {overall_status}
```

---

## 8. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-18 | 初始版本，基于 6B 配置 |

---

## 9. 参考文档

- [stage6_integration_spec_v1.md](stage6_integration_spec_v1.md) - 整合规范
- [STAGE6_ENTRY.md](STAGE6_ENTRY.md) - Stage 6 入口文档
- [eval/stage5g_final_report.md](eval/stage5g_final_report.md) - Stage 5G 报告
