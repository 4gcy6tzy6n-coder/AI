# Scalable Architecture Specification v1 - 可扩展架构规格

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 9 - WP3: 可扩展架构准备

---

## 1. 架构目标

### 1.1 设计原则

1. **分布式就绪，非分布式强制**: Phase 9 只做可迁移设计，不做重型分布式重构
2. **核心治理层不变**: TSLA、五门治理、训练闭环保持稳定
3. **服务边界清晰**: 明确定义各服务职责和数据边界
4. **渐进式扩展**: 支持从单机到多机逐步演进

### 1.2 扩展路径

```
Phase 9 (当前)
    │
    ├── 单机优化版 (已具备)
    │
    ├── 多 Worker 原型 (Phase 9 目标)
    │   ├── 任务队列模式
    │   └── 检索服务独立
    │
    └── 分布式就绪设计 (Phase 9 目标)
        ├── 服务边界定义
        ├── 状态一致性设计
        └── 远程接口预留

Phase 10+ (未来)
    │
    ├── 微服务架构
    │   ├── Query Service 集群
    │   ├── Retrieval Service 集群
    │   ├── Governance Service 集群
    │   └── Memory Management Service 集群
    │
    └── 云原生部署
        ├── 容器化
        ├── 自动扩缩容
        └── 多区域部署
```

---

## 2. 服务拆分设计

### 2.1 服务总览

| 服务 | 职责 | 当前状态 | Phase 9 目标 |
|------|------|----------|-------------|
| Query Service | 请求接入、预处理、路由 | 单体内部 | 可独立运行 |
| Retrieval Service | 记忆检索、知识检索 | 单体内部 | 可独立运行 |
| Governance Service | TSLA、五门治理 | 单体内部 | 保持核心 |
| Memory Management | 长期层/永久层管理 | 单体内部 | 队列化写回 |
| Metrics Service | 指标收集、Trace | 单体内部 | 可独立运行 |

### 2.2 服务边界

```
┌─────────────────────────────────────────────────────────────┐
│                         API Gateway                          │
│                    (Load Balancer)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Query        │ │ Retrieval    │ │ Governance   │
│ Service      │ │ Service      │ │ Service      │
│              │ │              │ │              │
│ - 请求解析   │ │ - 记忆检索   │ │ - TSLA       │
│ - 缺口识别   │ │ - 知识检索   │ │ - 五门治理   │
│ - 路由分发   │ │ - 缓存管理   │ │ - 晋升决策   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Memory Management    │
              │  Service              │
              │                       │
              │ - 长期层管理          │
              │ - 永久层管理          │
              │ - 写回队列            │
              └───────────────────────┘
```

### 2.3 数据流设计

```
Request Flow:
Client → API Gateway → Query Service → [缺口识别]
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            [无需检索]              [需要检索]              [高风险]
                    │                     │                     │
                    ▼                     ▼                     ▼
           Governance Service    Retrieval Service    Retrieval Service
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          │
                                          ▼
                              Governance Service
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
              [直接响应]           [知识库晋升]           [参数晋升]
                    │                     │                     │
                    ▼                     ▼                     ▼
                 Client         Memory Management      Memory Management
                                    Service                Service
```

---

## 3. 状态一致性设计

### 3.1 状态分类

| 状态类型 | 一致性要求 | 同步策略 | 示例 |
|----------|-----------|----------|------|
| 强一致 | 必须实时一致 | 同步写入 | Unit 质量分数 |
| 最终一致 | 可延迟同步 | 异步队列 | 知识库晋升 |
| 会话一致 | 会话内一致 | 会话缓存 | 用户上下文 |
| 可接受不一致 | 可容忍延迟 | 定时同步 | 监控指标 |

### 3.2 状态分布

```
强一致状态 (Strong Consistency):
├── Unit 元数据
├── 质量分数
├── 稳定性标记
└── 隔离区状态

最终一致状态 (Eventual Consistency):
├── 知识库内容
├── 长期层更新
├── 永久层晋升
└── 索引更新

会话状态 (Session Consistency):
├── 对话历史
├── 用户偏好
└── 临时上下文

可接受不一致 (Acceptable Inconsistency):
├── 监控指标
├── 统计信息
└── 日志数据
```

### 3.3 同步机制

```python
# 强一致 - 同步写入
sync_write(unit_metadata)

# 最终一致 - 异步队列
async_queue.put(kb_update_event)

# 会话状态 - 本地缓存
session_cache.set(user_context)

# 可接受不一致 - 批量同步
batch_sync(metrics_data)
```

---

## 4. 分布式就绪接口

### 4.1 检索服务接口

```python
# retrieval_service_interface.py

class RetrievalServiceInterface:
    """检索服务远程接口"""
    
    async def query_memory(
        self, 
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """远程记忆检索"""
        pass
    
    async def query_knowledge(
        self,
        query: str,
        domain: Optional[str] = None
    ) -> List[KnowledgeResult]:
        """远程知识检索"""
        pass
    
    async def update_index(
        self,
        units: List[Unit],
        operation: str = "add"
    ) -> bool:
        """远程索引更新"""
        pass
```

### 4.2 治理事件接口

```python
# governance_event_interface.py

class GovernanceEventInterface:
    """治理事件消息流接口"""
    
    async def publish_governance_event(
        self,
        event_type: str,
        event_data: Dict
    ) -> bool:
        """发布治理事件"""
        pass
    
    async def subscribe_governance_events(
        self,
        event_types: List[str],
        handler: Callable
    ) -> str:
        """订阅治理事件"""
        pass

# 事件类型定义
class GovernanceEventType:
    UNIT_PROMOTED = "unit_promoted"
    UNIT_ISOLATED = "unit_isolated"
    KB_UPDATED = "kb_updated"
    GOVERNANCE_DECISION = "governance_decision"
    ERROR_DETECTED = "error_detected"
```

### 4.3 内存管理队列接口

```python
# memory_management_interface.py

class MemoryManagementInterface:
    """内存管理远程接口"""
    
    async def queue_kb_promotion(
        self,
        unit_id: str,
        target_layer: str,
        priority: int = 0
    ) -> str:
        """队列化知识库晋升"""
        pass
    
    async def queue_param_promotion(
        self,
        unit_id: str,
        validation_result: Dict,
        priority: int = 0
    ) -> str:
        """队列化参数晋升"""
        pass
    
    async def get_queue_status(
        self,
        queue_type: str
    ) -> QueueStatus:
        """获取队列状态"""
        pass
```

---

## 5. 多 Worker 原型设计

### 5.1 Worker 类型

```
Worker Types:
├── QueryWorker
│   ├── 职责: 请求解析、缺口识别
│   ├── 并发: 高 (IO密集)
│   └── 数量: 可扩展
│
├── RetrievalWorker
│   ├── 职责: 记忆检索、知识检索
│   ├── 并发: 中 (CPU+IO)
│   └── 数量: 受限于索引分片
│
├── GovernanceWorker
│   ├── 职责: TSLA、五门治理
│   ├── 并发: 低 (需保证顺序)
│   └── 数量: 固定或少量
│
└── MemoryWorker
    ├── 职责: 长期层/永久层写回
    ├── 并发: 低 (写密集)
    └── 数量: 固定 (避免写冲突)
```

### 5.2 任务队列设计

```python
# task_queue_design.py

class TaskQueue:
    """任务队列管理"""
    
    QUEUES = {
        "query": {
            "priority_levels": 3,
            "max_workers": 10,
            "timeout_seconds": 30
        },
        "retrieval": {
            "priority_levels": 2,
            "max_workers": 5,
            "timeout_seconds": 60
        },
        "governance": {
            "priority_levels": 2,
            "max_workers": 3,
            "timeout_seconds": 120
        },
        "memory_write": {
            "priority_levels": 1,
            "max_workers": 2,
            "timeout_seconds": 300
        }
    }
```

### 5.3 任务分发策略

```python
# task_dispatch_strategy.py

def dispatch_task(task: Task) -> Worker:
    """任务分发策略"""
    
    if task.type == "query":
        # 轮询分发
        return round_robin_select(query_workers)
    
    elif task.type == "retrieval":
        # 基于索引分片哈希
        shard_id = hash(task.query) % num_shards
        return shard_workers[shard_id]
    
    elif task.type == "governance":
        # 基于 Unit ID 一致性哈希
        worker_id = consistent_hash(task.unit_id, governance_workers)
        return governance_workers[worker_id]
    
    elif task.type == "memory_write":
        # 顺序队列，避免写冲突
        return memory_workers[0]
```

---

## 6. 服务独立运行模式

### 6.1 检索服务独立模式

```python
# retrieval_service_standalone.py

class RetrievalServiceStandalone:
    """检索服务独立运行模式"""
    
    def __init__(self, config: Dict):
        self.index_manager = IndexManager(config["index_path"])
        self.cache_manager = CacheManager(config["cache_config"])
        self.rpc_server = RPCServer(config["port"])
    
    async def start(self):
        """启动服务"""
        await self.index_manager.load()
        await self.cache_manager.warmup()
        await self.rpc_server.start()
    
    async def handle_query(self, request: QueryRequest) -> QueryResponse:
        """处理查询"""
        # 先查缓存
        cached = await self.cache_manager.get(request.cache_key)
        if cached:
            return cached
        
        # 再查索引
        results = await self.index_manager.search(
            request.vector,
            request.top_k
        )
        
        # 更新缓存
        await self.cache_manager.set(
            request.cache_key,
            results,
            ttl=300
        )
        
        return results
```

### 6.2 单机 vs 多 Worker 配置

```yaml
# service_modes.yaml

modes:
  standalone:
    description: "单机模式 - 所有服务在同一进程"
    services:
      - query
      - retrieval
      - governance
      - memory
    communication: "in_memory"
    
  multi_worker:
    description: "多 Worker 模式 - 同机多进程"
    services:
      query:
        workers: 5
        communication: "queue"
      retrieval:
        workers: 3
        communication: "queue"
      governance:
        workers: 2
        communication: "queue"
      memory:
        workers: 1
        communication: "queue"
    
  distributed_ready:
    description: "分布式就绪 - 服务可独立部署"
    services:
      query_service:
        host: "query.service"
        port: 8001
      retrieval_service:
        host: "retrieval.service"
        port: 8002
      governance_service:
        host: "governance.service"
        port: 8003
      memory_service:
        host: "memory.service"
        port: 8004
    communication: "grpc"
```

---

## 7. 扩展演进路径

### 7.1 Phase 9 目标 (当前)

- ✅ 单机优化版运行
- 🎯 多 Worker 原型验证
- 🎯 分布式就绪接口预留
- 🎯 服务边界清晰定义

### 7.2 Phase 10 目标 (未来)

- 微服务拆分
- 容器化部署
- 服务发现和负载均衡
- 配置中心

### 7.3 Phase 11+ 目标 (远景)

- 自动扩缩容
- 多区域部署
- 异地容灾
- 云原生架构

---

## 8. 风险评估与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 过早分布式化 | 复杂度激增 | Phase 9 只做就绪设计 |
| 状态一致性难保证 | 数据不一致 | 明确分类，分层处理 |
| 网络延迟 | 性能下降 | 本地缓存 + 异步化 |
| 服务间耦合 | 难以独立演进 | 清晰边界，接口隔离 |
| 故障传播 | 级联故障 | 熔断、降级、隔离 |

---

## 9. 验收标准

### 9.1 Phase 9 完成标准

| 验收项 | 标准 | 验证方法 |
|--------|------|----------|
| 服务边界 | 5+ 服务边界清晰定义 | 架构评审 |
| 状态设计 | 4 类一致性策略定义 | 设计文档 |
| 接口预留 | 3+ 远程接口定义 | 代码审查 |
| 多 Worker | 原型可运行 | 功能测试 |
| 服务独立 | 检索服务可独立启动 | 集成测试 |

### 9.2 可扩展性指标

| 指标 | 当前 | Phase 9 目标 | Phase 10 目标 |
|------|------|-------------|--------------|
| 服务数量 | 1 (单体) | 1 (多 Worker) | 4+ (微服务) |
| 扩展方式 | 垂直扩展 | 水平扩展(同机) | 水平扩展(多机) |
| 部署单元 | 单进程 | 多进程 | 容器 |
| 通信方式 | 内存 | 队列 | RPC |

---

## 10. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义服务拆分和扩展路径 |

---

**文档状态**: 生效中  
**架构负责人**: Phase 9 WP3 负责人  
**下次评审**: Milestone 9.4 完成时
