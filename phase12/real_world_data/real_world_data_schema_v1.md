# Real-World Data Schema v1 - 真实数据模式 v1

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 12 - WP2: 真实场景数据收集与分析

---

## 1. 概述

本文档定义 Pilot 试运行期间需要采集的真实数据模式，建立真实使用数据闭环。

### 1.1 目标

建立真实使用数据闭环，把实验室指标转化为真实环境下的有效结论。

### 1.2 数据原则

- 真实数据优先于主观判断
- 全链路可追溯
- 样本可沉淀复用
- 隐私合规

---

## 2. 数据采集体系

### 2.1 请求数据 (Request Data)

```json
{
  "request_id": "req_abc123",
  "timestamp": "2026-04-18T10:30:00Z",
  "user_id": "user_xyz789",
  "session_id": "sess_def456",
  
  "request_content": {
    "query": "用户输入的查询",
    "language": "zh|en",
    "task_type": "simple|complex|multi_step",
    "context": {}
  },
  
  "metadata": {
    "source_ip": "xxx.xxx.xxx.xxx",
    "user_agent": "...",
    "api_version": "v1.2"
  }
}
```

### 2.2 响应数据 (Response Data)

```json
{
  "response_id": "resp_abc123",
  "request_id": "req_abc123",
  "timestamp": "2026-04-18T10:30:01Z",
  
  "response_content": {
    "answer": "系统返回的答案",
    "confidence": 0.85,
    "sources": [],
    "reasoning": "推理过程"
  },
  
  "performance": {
    "total_latency_ms": 450,
    "retrieval_latency_ms": 120,
    "governance_latency_ms": 80,
    "memory_latency_ms": 50
  },
  
  "status": "success|error|timeout|fallback"
}
```

### 2.3 检索数据 (Retrieval Data)

```json
{
  "retrieval_id": "ret_abc123",
  "request_id": "req_abc123",
  "timestamp": "2026-04-18T10:30:00Z",
  
  "retrieval_params": {
    "query": "检索查询",
    "retrieval_type": "memory|external|hybrid",
    "max_results": 10,
    "filters": {}
  },
  
  "retrieval_results": {
    "total_found": 5,
    "results": [],
    "cache_hit": true|false,
    "from_layer": "long_term|deep_permanent|..."
  },
  
  "performance": {
    "latency_ms": 120,
    "index_lookup_ms": 50,
    "result_ranking_ms": 30
  }
}
```

### 2.4 治理数据 (Governance Data)

```json
{
  "governance_id": "gov_abc123",
  "request_id": "req_abc123",
  "timestamp": "2026-04-18T10:30:00Z",
  
  "decisions": [
    {
      "unit_id": "unit_001",
      "tsla_score": 0.75,
      "action": "keep|promote|demote|quarantine|...",
      "target_layer": "long_term|shallow_permanent|...",
      "confidence": 0.85,
      "reasoning": "决策理由",
      "gate_decisions": {
        "promotion_gate": true|false,
        "demotion_gate": true|false,
        "quarantine_gate": true|false
      }
    }
  ],
  
  "statistics": {
    "total_units": 10,
    "action_distribution": {
      "keep": 5,
      "promote": 2,
      "demote": 1,
      "quarantine": 1
    }
  }
}
```

### 2.5 记忆写回数据 (Memory Writeback Data)

```json
{
  "writeback_id": "wb_abc123",
  "request_id": "req_abc123",
  "timestamp": "2026-04-18T10:30:01Z",
  
  "transaction": {
    "transaction_id": "txn_abc123",
    "status": "committed|rolled_back",
    "atomic": true,
    "operations": [
      {
        "operation_type": "create|update|delete|promote|demote",
        "object_id": "obj_001",
        "target_layer": "long_term",
        "status": "success|failed"
      }
    ]
  },
  
  "performance": {
    "total_duration_ms": 50,
    "lock_wait_ms": 10
  }
}
```

### 2.6 错误数据 (Error Data)

```json
{
  "error_id": "err_abc123",
  "request_id": "req_abc123",
  "timestamp": "2026-04-18T10:30:01Z",
  
  "error_info": {
    "type": "retrieval|governance|memory|system",
    "subtype": "timeout|not_found|permission|...",
    "severity": "low|medium|high|critical",
    "message": "错误信息",
    "stack_trace": "..."
  },
  
  "context": {
    "service": "retrieval-service",
    "endpoint": "/api/retrieve",
    "input_params": {}
  },
  
  "recovery": {
    "fallback_triggered": true|false,
    "fallback_result": "..."
  }
}
```

### 2.7 用户反馈数据 (User Feedback Data)

```json
{
  "feedback_id": "fb_abc123",
  "request_id": "req_abc123",
  "response_id": "resp_abc123",
  "timestamp": "2026-04-18T10:35:00Z",
  "user_id": "user_xyz789",
  
  "feedback": {
    "type": "rating|comment|correction|report",
    "rating": 4,
    "comment": "用户评论",
    "correction": "正确内容",
    "issue_type": "accuracy|relevance|clarity|..."
  },
  
  "metadata": {
    "context": "使用场景",
    "expected": "用户期望结果"
  }
}
```

---

## 3. 数据分类体系

### 3.1 请求类型分布

| 类型 | 定义 | 采集字段 |
|------|------|----------|
| 简单查询 | 单步、明确意图 | task_type=simple |
| 复杂查询 | 多条件、需推理 | task_type=complex |
| 多步任务 | 多轮交互 | task_type=multi_step |

### 3.2 语言分布

| 语言 | 检测方式 | 采集字段 |
|------|----------|----------|
| 中文 | 字符集检测 | language=zh |
| 英文 | 字符集检测 | language=en |
| 混合 | 比例判断 | language=mixed |

### 3.3 任务复杂度分布

| 级别 | 定义 | 指标 |
|------|------|------|
| L1 | 直接回答 | 无检索或单次检索 |
| L2 | 简单推理 | 1-2 次检索 |
| L3 | 复杂推理 | 3+ 次检索或治理 |
| L4 | 多步任务 | 多轮交互 |

---

## 4. 真实问题分类

### 4.1 问题分类体系

```
问题
├── 检索问题
│   ├── 检索失败
│   ├── 结果不相关
│   ├── 召回不足
│   └── 排序错误
├── 治理问题
│   ├── TSLA 计算偏差
│   ├── 动作决策错误
│   └── 门限判断失误
├── 英文检测问题
│   ├── 关系漏检
│   ├── 误召回
│   └── 边界错误
├── Memory 问题
│   ├── 写回失败
│   ├── 事务回滚
│   └── 数据不一致
├── 服务稳定性
│   ├── 超时
│   ├── 错误
│   └── 降级触发
├── 产品接口
│   ├── 响应格式
│   ├── 错误提示
│   └── 交互体验
└── 用户理解
    ├── 期望偏差
    ├── 使用困惑
    └── 功能误解
```

### 4.2 问题严重度

| 级别 | 定义 | 响应时间 |
|------|------|----------|
| P0 | 系统不可用 | 立即 |
| P1 | 核心功能受损 | 2 小时 |
| P2 | 功能降级 | 24 小时 |
| P3 | 体验问题 | 72 小时 |

---

## 5. 样本池建设

### 5.1 样本类型

| 类型 | 定义 | 采集策略 |
|------|------|----------|
| 高价值样本 | 正确且高质量 | 高置信度 + 正面反馈 |
| 失败样本 | 明显错误 | 错误日志 + 负面反馈 |
| 难例样本 | 边界情况 | 低置信度 + 人工标注 |
| 高频错误 | 重复出现的问题 | 错误聚类 |
| 中英混合 | 混合语言输入 | language=mixed |
| 长链路问题 | 多环节异常 | 全链路追踪 |

### 5.2 样本存储

```
real_case_pool_v1/
├── high_value/
│   ├── 2026-04/
│   └── metadata.json
├── failures/
│   ├── 2026-04/
│   └── metadata.json
├── edge_cases/
│   ├── 2026-04/
│   └── metadata.json
├── frequent_errors/
│   ├── 2026-04/
│   └── metadata.json
├── mixed_language/
│   ├── 2026-04/
│   └── metadata.json
└── long_chain/
    ├── 2026-04/
    └── metadata.json
```

---

## 6. 指标校准

### 6.1 需要校准的阈值

| 阈值 | 当前值 | 校准依据 | 目标 |
|------|--------|----------|------|
| 告警阈值 | - | 真实错误分布 | 减少误报 |
| 限流阈值 | 100 QPS | 真实负载 | 优化资源 |
| 检索超时 | 1000ms | 真实延迟分布 | 平衡体验 |
| 回流阈值 | - | 真实回流率 | 控制质量 |
| 人工复核触发 | - | 错误率分布 | 覆盖风险 |

### 6.2 校准流程

```
1. 收集真实数据 (1 周)
2. 分析分布特征
3. 识别异常点
4. 调整阈值
5. 验证效果 (3 天)
6. 固化配置
```

---

## 7. 隐私与合规

### 7.1 数据脱敏

| 字段 | 处理方式 |
|------|----------|
| user_id | 哈希化 |
| source_ip | 匿名化 |
| query | 敏感词过滤 |
| response | 敏感信息脱敏 |

### 7.2 保留策略

| 数据类型 | 保留期 | 说明 |
|----------|--------|------|
| 原始日志 | 30 天 | 用于排查 |
| 聚合指标 | 1 年 | 长期分析 |
| 样本池 | 永久 | 训练优化 |
| 错误详情 | 90 天 | 问题追踪 |

---

## 8. 验收标准

| 标准 | 说明 | 验收方式 |
|------|------|----------|
| 数据完整 | 全链路数据可采集 | 数据验证 |
| 问题分类 | 问题类型清楚 | 分类准确率 |
| 样本资产 | 形成样本池 | 样本数量/质量 |
| 阈值校准 | 关键阈值校准 | 对比实验 |

---

## 9. 附录

### 9.1 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本 |

### 9.2 相关文档

- pilot_execution_plan_v1.md
- pilot_issue_taxonomy_v1.md

---

**文档状态**: 生效中  
**负责人**: Phase 12 WP2 负责人
