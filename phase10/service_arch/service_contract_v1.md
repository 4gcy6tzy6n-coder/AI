# Service Contract V1 - 服务契约 V1

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 10 - WP1: 服务化架构落地

---

## 1. 概述

本文档定义 Phase 10 服务化架构的服务契约，包括服务边界、通信协议、状态一致性规则和 API 规范。

---

## 2. 服务角色定义

### 2.1 五类核心服务

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway / Query Orchestrator         │
│                    (请求入口、路由、鉴权、编排)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Retrieval   │ │  Governance  │ │    Memory    │
│   Service    │ │   Service    │ │   Service    │
│              │ │              │ │              │
│ 记忆检索     │ │ TSLA 决策    │ │ 长期层管理   │
│ 知识检索     │ │ 门控决策     │ │ 永久层管理   │
│ 缓存管理     │ │ 分流动作     │ │ 写回队列     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Observability &       │
              │ Regression Service    │
              │                       │
              │ trace、指标、告警     │
              │ 回归任务触发          │
              │ 结果汇总              │
              └───────────────────────┘
```

### 2.2 服务职责详细定义

#### 2.2.1 API Gateway / Query Orchestrator

**职责**:
- 接收外部请求
- 请求鉴权与限流
- 请求路由与编排
- 响应聚合

**核心接口**:
```yaml
endpoints:
  - path: /api/v1/query
    method: POST
    description: 主查询接口
    request:
      content_type: application/json
      schema: QueryRequest
    response:
      content_type: application/json
      schema: QueryResponse
      status_codes: [200, 400, 429, 500]
  
  - path: /api/v1/health
    method: GET
    description: 健康检查
    response:
      status_codes: [200]
```

#### 2.2.2 Retrieval Service

**职责**:
- 内部记忆向量检索
- 外部知识检索
- 缓存管理
- 索引维护

**核心接口**:
```yaml
endpoints:
  - path: /retrieval/v1/memory
    method: POST
    description: 记忆检索
    request:
      schema: MemoryRetrievalRequest
    response:
      schema: MemoryRetrievalResponse
  
  - path: /retrieval/v1/knowledge
    method: POST
    description: 知识检索
    request:
      schema: KnowledgeRetrievalRequest
    response:
      schema: KnowledgeRetrievalResponse
  
  - path: /retrieval/v1/index/update
    method: POST
    description: 索引更新
    request:
      schema: IndexUpdateRequest
    response:
      schema: IndexUpdateResponse
```

#### 2.2.3 Governance Service

**职责**:
- TSLA (Trigger-Select-Loop-Action) 决策
- 五门治理 (QT/SL/T/C/L)
- 分流动作执行
- 回流重审

**核心接口**:
```yaml
endpoints:
  - path: /governance/v1/tsla
    method: POST
    description: TSLA 决策
    request:
      schema: TSLARequest
    response:
      schema: TSLAResponse
  
  - path: /governance/v1/evaluate
    method: POST
    description: 质量评估
    request:
      schema: EvaluationRequest
    response:
      schema: EvaluationResponse
  
  - path: /governance/v1/promote
    method: POST
    description: 晋升决策
    request:
      schema: PromotionRequest
    response:
      schema: PromotionResponse
```

#### 2.2.4 Memory Service

**职责**:
- 长期层管理
- 永久层管理
- 隔离区管理
- 错误区管理
- 写回队列管理

**核心接口**:
```yaml
endpoints:
  - path: /memory/v1/unit/{unit_id}
    method: GET
    description: 获取 Unit
    response:
      schema: UnitResponse
  
  - path: /memory/v1/unit/{unit_id}/promote
    method: POST
    description: 晋升 Unit
    request:
      schema: UnitPromotionRequest
    response:
      schema: UnitPromotionResponse
  
  - path: /memory/v1/writeback/queue
    method: POST
    description: 提交写回任务
    request:
      schema: WritebackRequest
    response:
      schema: WritebackResponse
  
  - path: /memory/v1/zones/status
    method: GET
    description: 获取各层状态
    response:
      schema: ZonesStatusResponse
```

#### 2.2.5 Observability & Regression Service

**职责**:
- Trace 收集与存储
- 指标收集与聚合
- 告警触发
- 回归任务触发
- 结果汇总

**核心接口**:
```yaml
endpoints:
  - path: /obs/v1/trace
    method: POST
    description: 提交 trace
    request:
      schema: TraceSubmission
  
  - path: /obs/v1/metrics
    method: POST
    description: 提交指标
    request:
      schema: MetricsSubmission
  
  - path: /obs/v1/regression/trigger
    method: POST
    description: 触发回归测试
    request:
      schema: RegressionTriggerRequest
    response:
      schema: RegressionTriggerResponse
  
  -path: /obs/v1/dashboard
    method: GET
    description: 获取仪表板数据
    response:
      schema: DashboardDataResponse
```

---

## 3. 通信协议设计

### 3.1 同步请求链路

```
Client -> API Gateway -> [Retrieval | Governance | Memory] -> Response
```

**协议**: HTTP/1.1 或 HTTP/2  
**序列化**: JSON  
**超时**: 见下表  
**重试**: 见下表

| 链路 | 超时 | 重试 | 降级策略 |
|------|------|------|----------|
| Gateway -> Retrieval | 500ms | 2 | 返回空结果 |
| Gateway -> Governance | 200ms | 1 | 直通模式 |
| Gateway -> Memory | 100ms | 1 | 缓存优先 |

### 3.2 异步事件链路

```
Service A -> Event Bus -> Service B (异步消费)
```

**协议**: 内部消息队列  
**保证**: At-least-once  
**顺序**: 同分区有序

| 事件类型 | 生产者 | 消费者 | 优先级 |
|----------|--------|--------|--------|
| UnitPromoted | Governance | Memory | High |
| UnitIsolated | Governance | Memory, Obs | High |
| IndexUpdated | Memory | Retrieval | Normal |
| CacheInvalidated | Retrieval | Retrieval | Normal |
| MetricsReported | All | Obs | Low |

### 3.3 写回队列

```
Governance -> Writeback Queue -> Memory Service (异步处理)
```

**队列类型**: 优先级队列  
**持久化**: 是  
**重试**: 3 次  
**死信队列**: 有

### 3.4 统一 Schema

所有服务间通信使用统一 Schema 版本:

```yaml
schema_version: "1.0"
compatible_versions: ["1.0"]
```

---

## 4. 状态一致性设计

### 4.1 一致性分类

| 数据类型 | 一致性要求 | 实现方式 |
|----------|-----------|----------|
| Unit 元数据 | 强一致 | 同步写入 + 确认 |
| 质量分数 | 强一致 | 同步写入 + 确认 |
| 写回队列 | 强一致 | 持久化队列 |
| 向量索引 | 最终一致 | 异步更新 |
| 缓存数据 | 可不一致 | TTL + 失效 |
| 检索日志 | 最终一致 | 异步批量写入 |
| 治理历史 | 最终一致 | 异步写入 |

### 4.2 强一致路径

```
Governance Service --(同步)--> Memory Service
  └--> 确认写入后才返回
```

适用场景:
- Unit 晋升到长期层
- Unit 晋升到永久层
- 隔离区操作
- 质量分数更新

### 4.3 最终一致路径

```
Governance Service --(异步)--> Retrieval Service
  └--> 索引更新可延迟
```

适用场景:
- 向量索引更新
- 缓存预热
- 统计指标更新
- 日志写入

### 4.4 串行 vs 并行

| 操作 | 方式 | 原因 |
|------|------|------|
| 永久层写入 | 串行 | 防止污染 |
| 长期层写入 | 并行 | 提升吞吐 |
| 索引更新 | 并行 | 无强依赖 |
| 缓存失效 | 并行 | 幂等操作 |

---

## 5. API 详细规范

### 5.1 通用规范

**请求头**:
```http
Content-Type: application/json
X-Request-ID: {uuid}
X-Service-Version: 1.0
```

**响应头**:
```http
Content-Type: application/json
X-Request-ID: {uuid}
X-Response-Time: {ms}
```

**错误格式**:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": {}
  }
}
```

### 5.2 主要 API 详细定义

#### Query API

**Request**:
```json
{
  "query_id": "req_001",
  "text": "用户输入文本",
  "context": {
    "session_id": "sess_001",
    "user_id": "user_001"
  },
  "options": {
    "enable_retrieval": true,
    "enable_governance": true,
    "max_results": 5
  }
}
```

**Response**:
```json
{
  "query_id": "req_001",
  "status": "success",
  "results": {
    "retrieval": [...],
    "governance": {
      "action": "promote",
      "scores": {...}
    }
  },
  "trace_id": "trace_001",
  "latency_ms": 150
}
```

#### TSLA API

**Request**:
```json
{
  "unit_id": "unit_001",
  "unit_type": "concept",
  "content": {...},
  "context": {
    "retrieval_results": [...]
  }
}
```

**Response**:
```json
{
  "unit_id": "unit_001",
  "decision": {
    "trigger": true,
    "selected_action": "promote_to_long_term",
    "loop_check": "passed",
    "action_params": {...}
  },
  "scores": {
    "QT": 0.92,
    "SL": 0.88,
    "T": 0.85,
    "C": 0.90,
    "L": 0.87
  }
}
```

---

## 6. 版本兼容性

### 6.1 版本策略

- **Major**: 不兼容变更
- **Minor**: 向后兼容的功能添加
- **Patch**: Bug 修复

### 6.2 兼容性规则

| 变更类型 | 兼容性 | 示例 |
|----------|--------|------|
| 添加可选字段 | 向后兼容 | 新增 options |
| 添加新端点 | 向后兼容 | 新增 /v2/... |
| 删除字段 | 不兼容 | 删除 required |
| 修改字段类型 | 不兼容 | string -> int |
| 修改端点路径 | 不兼容 | /v1/ -> /v2/ |

---

## 7. 部署拓扑

### 7.1 单机多服务部署

```
┌─────────────────────────────────────┐
│           Host Machine              │
│  ┌─────────┐ ┌─────────┐           │
│  │ Gateway │ │Retrieval│           │
│  │  :8080  │ │  :8081  │           │
│  └────┬────┘ └────┬────┘           │
│       │           │                 │
│  ┌────┴────┐ ┌────┴────┐           │
│  │Governance│ │ Memory │           │
│  │  :8082  │ │  :8083  │           │
│  └─────────┘ └─────────┘           │
│                                     │
│  ┌─────────┐ ┌─────────┐           │
│  │   Obs   │ │  Queue  │           │
│  │  :8084  │ │ (内部)  │           │
│  └─────────┘ └─────────┘           │
└─────────────────────────────────────┘
```

### 7.2 多 Worker 部署

```
┌─────────────────────────────────────┐
│           Host Machine              │
│  ┌─────────┐ ┌─────────┐           │
│  │ Gateway │ │ Worker  │           │
│  │         │ │  Pool   │           │
│  │         │ │ ┌─┬─┬─┐ │           │
│  │         │ │ │W│W│W│ │           │
│  └────┬────┘ │ └─┴─┴─┘ │           │
│       │      └─────────┘           │
│  ┌────┴──────────────────┐         │
│  │    Shared Services    │         │
│  │  (Retrieval/Memory)   │         │
│  └───────────────────────┘         │
└─────────────────────────────────────┘
```

---

## 8. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义 5 类服务契约 |

---

**文档状态**: 冻结  
**服务边界已确定，进入实现阶段**  
**负责人**: Phase 10 WP1 负责人
