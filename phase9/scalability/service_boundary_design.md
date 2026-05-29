# Service Boundary Design - 服务边界设计

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 9 - WP3: 可扩展架构准备

---

## 1. 设计目标

本文档定义系统各服务的边界，为 Phase 9 的可扩展架构准备和未来的分布式演进提供基础。

### 1.1 核心原则

1. **高内聚低耦合**: 每个服务有明确的单一职责
2. **数据边界清晰**: 明确每个服务管理的数据范围
3. **接口稳定**: 服务间通过稳定的接口通信
4. **渐进式拆分**: 支持从单体到分布式的渐进演进

---

## 2. 服务划分

### 2.1 服务总览

```
┌─────────────────────────────────────────────────────────────┐
│                        API Gateway                          │
│                   (路由、认证、限流)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Query      │ │  Retrieval   │ │ Governance   │
│   Service    │ │   Service    │ │   Service    │
│              │ │              │ │              │
│ 请求解析     │ │ 记忆检索     │ │ TSLA 决策    │
│ 缺口识别     │ │ 知识检索     │ │ 五门治理     │
│ 路由分发     │ │ 缓存管理     │ │ 晋升决策     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Memory Management   │
              │       Service         │
              │                       │
              │  长期层管理           │
              │  永久层管理           │
              │  写回队列             │
              └───────────────────────┘
```

### 2.2 服务详细定义

#### 2.2.1 Query Service (查询服务)

**职责**:
- 接收外部请求
- 请求解析和验证
- 缺口识别 (Gap Identification)
- 路由分发到下游服务

**输入**:
- 用户查询文本
- 上下文信息
- 配置参数

**输出**:
- 解析后的查询对象
- 缺口识别结果
- 路由决策

**数据**:
- 不持久化数据
- 使用内存缓存热点查询

**接口**:
```python
class QueryServiceInterface:
    async def parse_request(self, raw_request: Dict) -> ParsedQuery
    async def identify_gap(self, query: ParsedQuery) -> GapAnalysis
    async def route_request(self, query: ParsedQuery, gap: GapAnalysis) -> RouteDecision
```

#### 2.2.2 Retrieval Service (检索服务)

**职责**:
- 内部记忆检索
- 外部知识检索
- 缓存管理
- 索引维护

**输入**:
- 查询向量/文本
- 检索参数 (top_k, filters)
- 缓存策略

**输出**:
- 检索结果列表
- 相关性分数
- 缓存命中信息

**数据**:
- 向量索引 (HNSW)
- 缓存数据 (L1/L2)
- 检索日志

**状态一致性**: 最终一致

**接口**:
```python
class RetrievalServiceInterface:
    async def query_memory(self, vector: List[float], top_k: int) -> List[RetrievalResult]
    async def query_knowledge(self, query: str, domain: Optional[str]) -> List[KnowledgeResult]
    async def update_index(self, units: List[Unit], operation: str) -> bool
    async def warmup_cache(self, hot_queries: List[str]) -> int
```

#### 2.2.3 Governance Service (治理服务)

**职责**:
- TSLA (Trigger-Select-Loop-Action) 决策
- 五门治理 (QT/SL/T/C/L)
- 晋升决策
- 隔离区管理

**输入**:
- Unit 数据
- 检索结果
- 治理上下文

**输出**:
- 治理决策
- 质量分数
- 晋升建议

**数据**:
- Unit 元数据 (强一致)
- 质量分数 (强一致)
- 治理历史

**状态一致性**: 强一致

**接口**:
```python
class GovernanceServiceInterface:
    async def tsla_decision(self, context: GovernanceContext) -> TSLADecision
    async def evaluate_quality(self, unit: Unit) -> QualityScores
    async def make_promotion_decision(self, unit: Unit, scores: QualityScores) -> PromotionDecision
    async def isolate_unit(self, unit_id: str, reason: str) -> bool
```

#### 2.2.4 Memory Management Service (内存管理服务)

**职责**:
- 长期层 (Long-term Layer) 管理
- 永久层 (Permanent Layer) 管理
- 写回队列管理
- 晋升执行

**输入**:
- 晋升请求
- 写回数据
- 优先级

**输出**:
- 写回确认
- 队列状态
- 执行结果

**数据**:
- 长期层 Units
- 永久层 Units
- 写回队列
- 晋升历史

**状态一致性**: 最终一致 (写回可异步)

**接口**:
```python
class MemoryManagementInterface:
    async def queue_promotion(self, unit_id: str, target_layer: str, priority: int) -> str
    async def execute_writeback(self, queue_id: str) -> WritebackResult
    async def get_queue_status(self) -> QueueStatus
    async def replay_failed_writebacks(self) -> int
```

---

## 3. 数据边界

### 3.1 数据所有权

| 数据类型 | 所属服务 | 一致性要求 |
|----------|----------|-----------|
| Unit 元数据 | Governance | 强一致 |
| 质量分数 | Governance | 强一致 |
| 向量索引 | Retrieval | 最终一致 |
| 缓存数据 | Retrieval | 可接受不一致 |
| 长期层 Units | Memory Mgmt | 最终一致 |
| 永久层 Units | Memory Mgmt | 最终一致 |
| 写回队列 | Memory Mgmt | 强一致 |
| 请求日志 | Query | 可接受不一致 |
| 治理历史 | Governance | 最终一致 |

### 3.2 数据流

```
数据流 1: 查询处理
Query Service → Retrieval Service → Governance Service → Response

数据流 2: 知识晋升
Governance Service → Memory Management Service (队列)
Memory Management Service → Long-term/Permanent Layer

数据流 3: 索引更新
Governance Service → Retrieval Service (异步)
Retrieval Service → Index Update

数据流 4: 缓存预热
Query Service → Retrieval Service (热点查询)
Retrieval Service → Cache Warmup
```

---

## 4. 接口契约

### 4.1 同步接口

| 接口 | 调用方 | 提供方 | 超时 | 重试 |
|------|--------|--------|------|------|
| parse_request | Gateway | Query | 1s | 1 |
| identify_gap | Query | Query | 100ms | 0 |
| query_memory | Query/Governance | Retrieval | 500ms | 2 |
| tsla_decision | Query | Governance | 200ms | 1 |
| evaluate_quality | Governance | Governance | 100ms | 0 |

### 4.2 异步接口

| 接口 | 调用方 | 提供方 | 队列 | 延迟 |
|------|--------|--------|------|------|
| queue_promotion | Governance | Memory Mgmt | promotion_queue | < 100ms |
| update_index | Governance | Retrieval | index_update_queue | < 5s |
| execute_writeback | Memory Mgmt | Memory Mgmt | writeback_queue | < 1s |

---

## 5. 服务间通信

### 5.1 通信方式

| 场景 | 通信方式 | 原因 |
|------|----------|------|
| 同步查询 | HTTP/gRPC | 低延迟，需要即时响应 |
| 异步写回 | Message Queue | 解耦，可重试 |
| 事件通知 | Pub/Sub | 广播，解耦 |
| 配置同步 | Config Service | 集中管理 |

### 5.2 消息流

```
事件流:
Governance Service --(UnitPromotedEvent)--> Memory Mgmt Service
Governance Service --(UnitIsolatedEvent)--> Query Service
Retrieval Service --(CacheInvalidatedEvent)--> Query Service
Memory Mgmt Service --(WritebackCompletedEvent)--> Governance Service
```

---

## 6. 故障隔离

### 6.1 熔断策略

| 服务 | 熔断条件 | 降级策略 |
|------|----------|----------|
| Retrieval | 错误率 > 20% | 返回空结果 |
| Governance | 错误率 > 10% | 直通模式 (不治理) |
| Memory Mgmt | 队列积压 > 1000 | 暂停新晋升 |

### 6.2 降级策略

```
正常模式:
Query → Retrieval → Governance → Memory Mgmt

降级模式 1 (Retrieval 故障):
Query → Governance (跳过检索)

降级模式 2 (Governance 故障):
Query → Retrieval → Response (直通)

降级模式 3 (Memory Mgmt 故障):
Query → Retrieval → Governance (暂停晋升)
```

---

## 7. 部署模式

### 7.1 单机模式 (Phase 9)

```
所有服务在同一进程内
通信: 内存调用
部署: 单实例
```

### 7.2 多 Worker 模式 (Phase 9 原型)

```
同机多进程
通信: 本地队列/共享内存
部署: 单实例，多 Worker
```

### 7.3 分布式模式 (Phase 10+)

```
多机部署
通信: RPC/Message Queue
部署: 多实例，负载均衡
```

---

## 8. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义 4 个核心服务边界 |

---

**文档状态**: 生效中  
**架构负责人**: Phase 9 WP3 负责人  
**下次评审**: 原型验证完成后
