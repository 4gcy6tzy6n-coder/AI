# Core Services Contract V1 - 核心服务契约 V1

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 11 - WP1: 核心服务完整化

---

## 1. 概述

本文档定义 Phase 11 核心服务（Retrieval / Governance / Memory）的完整契约，包括服务边界、API 规范、通信协议和联调要求。

### 1.1 服务关系图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   API Gateway   │────▶│ Retrieval Svcs  │────▶│ Governance Svcs │
│   (已完成)      │     │   (本阶段实现)   │     │   (本阶段实现)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │                                               ▼
         │                                      ┌─────────────────┐
         │                                      │   Memory Svcs   │
         │                                      │   (本阶段实现)   │
         │                                      └─────────────────┘
         │                                               │
         └───────────────────────────────────────────────┘
                            (写回流)
```

### 1.2 核心服务职责

| 服务 | 职责 | 状态 |
|------|------|------|
| Retrieval Service | 记忆检索、知识检索、缓存管理 | 本阶段实现 |
| Governance Service | TSLA、八动作、五门治理、回流重审 | 本阶段实现 |
| Memory Service | 长期层、隔离区、写回事务 | 本阶段实现 |

---

## 2. Retrieval Service 契约

### 2.1 服务职责

1. **内部记忆检索**: 从长期层、永久层检索相关记忆
2. **外部知识检索**: 从外部知识源检索信息
3. **分层索引**: 支持多级索引策略
4. **多级缓存**: 热点数据缓存
5. **超时控制**: 检索超时保护
6. **结果排序**: 相关性排序与过滤

### 2.2 API 规范

#### 2.2.1 检索请求

```python
@dataclass
class RetrievalRequest:
    query: str                    # 查询文本
    query_id: str                 # 查询 ID
    context: Dict[str, Any]       # 上下文
    retrieval_type: str           # "memory" | "knowledge" | "hybrid"
    max_results: int = 10         # 最大结果数
    timeout_ms: int = 1000        # 超时时间
    filters: Dict[str, Any] = None  # 过滤条件
```

#### 2.2.2 检索响应

```python
@dataclass
class RetrievalResponse:
    query_id: str
    results: List[RetrievalResult]
    total_found: int
    latency_ms: float
    cache_hit: bool
    from_layer: str               # "short_term" | "long_term" | "permanent" | "external"
    status: str                   # "success" | "timeout" | "error"
    error: Optional[str] = None

@dataclass
class RetrievalResult:
    id: str
    content: str
    score: float                  # 相关性分数
    source: str                   # 来源
    timestamp: datetime
    metadata: Dict[str, Any]
```

#### 2.2.3 关键接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/retrieve` | POST | 执行检索 |
| `/retrieve/memory` | POST | 仅检索内部记忆 |
| `/retrieve/knowledge` | POST | 仅检索外部知识 |
| `/cache/invalidate` | POST | 缓存失效 |
| `/health` | GET | 健康检查 |

### 2.3 内部架构

```
┌─────────────────────────────────────────────────────────┐
│                    Retrieval Service                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Query Parser │─▶│ Index Router │─▶│ Cache Layer  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                   │                │         │
│         ▼                   ▼                ▼         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Short-term   │  │ Long-term    │  │ Permanent    │  │
│  │ Index        │  │ Index        │  │ Index        │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                   │                │         │
│         └───────────────────┴────────────────┘         │
│                             │                          │
│                             ▼                          │
│                    ┌──────────────┐                    │
│                    │ Result Merger│                    │
│                    │ & Ranker     │                    │
│                    └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### 2.4 关键配置

```yaml
retrieval_service:
  # 超时配置
  default_timeout_ms: 1000
  max_timeout_ms: 5000
  
  # 缓存配置
  cache:
    enabled: true
    ttl_seconds: 300
    max_size: 10000
  
  # 索引配置
  index:
    short_term:
      max_entries: 1000
      ttl_hours: 24
    long_term:
      max_entries: 100000
      ttl_days: 90
    permanent:
      max_entries: 1000000
  
  # 结果配置
  results:
    default_max: 10
    max_max: 100
    min_score_threshold: 0.5
```

---

## 3. Governance Service 契约

### 3.1 服务职责

1. **TSLA 初判**: 计算 TSLA 分数
2. **八动作分流**: 执行八种治理动作
3. **五门迁移决策**: 决定对象迁移路径
4. **回流重审**: 处理回流对象
5. **隔离/降级/归档**: 异常处理联动
6. **治理事件日志**: 输出治理事件

### 3.2 API 规范

#### 3.2.1 治理请求

```python
@dataclass
class GovernanceRequest:
    query_id: str
    query: str
    context: Dict[str, Any]
    retrieved_memories: List[RetrievalResult]
    unit_info: Dict[str, Any]       # Unit 信息
    trace_id: str

@dataclass
class GovernanceDecision:
    query_id: str
    trace_id: str
    tsla_score: float               # TSLA 分数
    action: str                     # 八动作之一
    target_layer: str               # 目标层
    confidence: float
    reasoning: str                  # 决策理由
    gate_decisions: Dict[str, Any]  # 五门决策详情
    needs_review: bool              # 是否需要回流重审
```

#### 3.2.2 八动作定义

| 动作 | 代码 | 描述 |
|------|------|------|
| 保留 | KEEP | 保留在当前层 |
| 晋升 | PROMOTE | 晋升到更高层 |
| 降级 | DEMOTE | 降级到更低层 |
| 隔离 | QUARANTINE | 移至隔离区 |
| 归档 | ARCHIVE | 移至错误区归档 |
| 修复 | REPAIR | 触发修复流程 |
| 回流 | RECYCLE | 进入回流重审 |
| 删除 | DELETE | 标记删除 |

#### 3.2.3 五门决策

```python
@dataclass
class GateDecisions:
    gate_1_promotion: bool          # 门 1: 晋升
    gate_2_demotion: bool           # 门 2: 降级
    gate_3_repair: bool             # 门 3: 修复
    gate_4_quarantine: bool         # 门 4: 隔离
    gate_5_archive: bool            # 门 5: 归档
```

#### 3.2.4 关键接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/govern` | POST | 执行治理决策 |
| `/govern/batch` | POST | 批量治理 |
| `/tsla/calculate` | POST | 计算 TSLA |
| `/recycle/review` | POST | 回流重审 |
| `/events` | GET | 获取治理事件 |

### 3.3 内部架构

```
┌─────────────────────────────────────────────────────────┐
│                   Governance Service                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ TSLA Scorer  │  │ 8-Action     │  │ 5-Gate       │  │
│  │              │  │ Router       │  │ Controller   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                   │                │         │
│         ▼                   ▼                ▼         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Confidence   │  │ Layer        │  │ Recycle      │  │
│  │ Evaluator    │  │ Migration    │  │ Reviewer     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                             │                          │
│                             ▼                          │
│                    ┌──────────────┐                    │
│                    │ Event Logger │                    │
│                    └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### 3.4 关键配置

```yaml
governance_service:
  # TSLA 配置
  tsla:
    threshold_high: 0.8
    threshold_medium: 0.5
    threshold_low: 0.3
  
  # 五门配置
  gates:
    gate_1:
      min_confidence: 0.7
      min_stability: 0.6
    gate_2:
      max_contamination: 0.1
    gate_3:
      repairable_threshold: 0.4
    gate_4:
      quarantine_threshold: 0.2
    gate_5:
      archive_threshold: 0.1
  
  # 回流配置
  recycle:
    enabled: true
    review_interval_hours: 24
    max_recycle_count: 3
```

---

## 4. Memory Service 契约

### 4.1 服务职责

1. **长期层管理**: 长期层对象 CRUD
2. **隔离区管理**: 隔离区对象管理
3. **错误区管理**: 错误区归档
4. **晋升控制**: 浅层永久/深层永久晋升
5. **写回事务**: 原子性写回
6. **状态回放**: 历史追踪
7. **生命周期**: 对象生命周期管理

### 4.2 API 规范

#### 4.2.1 写回请求

```python
@dataclass
class WritebackRequest:
    query_id: str
    trace_id: str
    operations: List[MemoryOperation]
    transaction_id: str
    atomic: bool = True             # 是否原子性

@dataclass
class MemoryOperation:
    operation_type: str             # "create" | "update" | "delete" | "promote" | "demote"
    object_id: str
    target_layer: str               # "long_term" | "shallow_permanent" | "deep_permanent" | "quarantine" | "error_zone"
    content: Optional[Dict] = None
    metadata: Optional[Dict] = None
    reason: str = ""

@dataclass
class WritebackResponse:
    transaction_id: str
    status: str                     # "committed" | "rolled_back" | "partial"
    completed_operations: int
    failed_operations: int
    errors: List[str]
    timestamp: datetime
```

#### 4.2.2 查询请求

```python
@dataclass
class MemoryQueryRequest:
    object_id: Optional[str] = None
    layer: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    limit: int = 100

@dataclass
class MemoryQueryResponse:
    objects: List[MemoryObject]
    total_count: int
    layer_breakdown: Dict[str, int]

@dataclass
class MemoryObject:
    id: str
    layer: str
    content: Dict
    metadata: Dict
    created_at: datetime
    updated_at: datetime
    version: int
    lifecycle_state: str            # "active" | "quarantined" | "archived" | "deleted"
```

#### 4.2.3 关键接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/writeback` | POST | 执行写回操作 |
| `/query` | POST | 查询记忆对象 |
| `/history` | GET | 获取对象历史 |
| `/lifecycle` | POST | 管理生命周期 |
| `/snapshot` | POST | 创建快照 |

### 4.3 内部架构

```
┌─────────────────────────────────────────────────────────┐
│                     Memory Service                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Transaction  │  │ Layer        │  │ Lifecycle    │  │
│  │ Manager      │  │ Router       │  │ Manager      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                   │                │         │
│         ▼                   ▼                ▼         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Long-term    │  │ Shallow      │  │ Deep         │  │
│  │ Store        │  │ Permanent    │  │ Permanent    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                   │                │         │
│         ▼                   ▼                ▼         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Quarantine   │  │ Error Zone   │  │ History      │  │
│  │ Store        │  │ Archive      │  │ Log          │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 4.4 关键配置

```yaml
memory_service:
  # 事务配置
  transaction:
    timeout_seconds: 30
    max_retries: 3
    atomic_default: true
  
  # 存储配置
  storage:
    long_term:
      max_size_gb: 10
      retention_days: 90
    shallow_permanent:
      max_size_gb: 50
      retention_days: 365
    deep_permanent:
      max_size_gb: 100
      retention_days: -1  # 永久
    quarantine:
      max_size_gb: 5
      retention_days: 30
    error_zone:
      max_size_gb: 20
      retention_days: 180
  
  # 晋升配置
  promotion:
    shallow_threshold:
      confidence: 0.8
      stability: 0.7
      min_age_days: 7
    deep_threshold:
      confidence: 0.95
      stability: 0.9
      min_age_days: 30
```

---

## 5. 服务联调规范

### 5.1 核心链路

```
Query → Retrieval → Governance → Memory (Writeback)
```

### 5.2 联调检查点

| 检查点 | 验证内容 | 通过标准 |
|--------|----------|----------|
| CP1 | Query → Retrieval | 检索成功，返回结果 |
| CP2 | Retrieval → Governance | TSLA 计算正确 |
| CP3 | Governance → Memory | 写回成功 |
| CP4 | 失败回退 | 单点故障可回退 |
| CP5 | 回流重放 | 回流路径可重放 |
| CP6 | 写回观测 | 写回操作可观测 |

### 5.3 错误处理

| 错误类型 | 处理方式 | 降级策略 |
|----------|----------|----------|
| Retrieval 超时 | 返回空结果 | 使用缓存 |
| Governance 失败 | 默认 KEEP | 记录日志 |
| Memory 写回失败 | 重试 3 次 | 进入队列 |
| 全链路失败 | 返回错误 | 人工介入 |

---

## 6. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义核心服务契约 |

---

**文档状态**: 冻结  
**核心服务边界已确定，进入实现阶段**  
**负责人**: Phase 11 WP1 负责人
