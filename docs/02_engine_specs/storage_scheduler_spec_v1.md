# 存储调度器规范 v1

## 概述

存储调度器管理数据在不同存储层级间的流动，优化访问速度和存储成本。

## 职责

1. 存储层级管理
2. 数据迁移调度
3. 缓存策略执行
4. 资源监控
5. 性能优化

## 架构

```
┌─────────────────────────────────────┐
│       Storage Scheduler             │
├─────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │GPU Cache│ │RAM Cache│ │SSD Idx ││
│  └────┬────┘ └────┬────┘ └───┬────┘│
│       │           │          │      │
│       └───────────┼──────────┘      │
│                   │                 │
│                   ▼                 │
│  ┌─────────────────────────────┐    │
│  │   Residency Scheduler       │    │
│  └─────────────────────────────┘    │
│                   │                 │
│                   ▼                 │
│  ┌─────────────────────────────┐    │
│  │   Migration Controller      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## 存储层级

### 1. GPU Cache (GPU 缓存)

**特点**:
- 最高访问速度
- 容量最小
- 成本最高
- 用于活跃模型参数

**配置**:
```yaml
gpu_cache:
  max_size_gb: 16
  eviction_policy: "lru"
  prefetch_enabled: true
  dtype: "fp16"
```

### 2. RAM Cache (内存缓存)

**特点**:
- 高访问速度
- 中等容量
- 用于频繁访问数据
- 支持索引

**配置**:
```yaml
ram_cache:
  max_size_gb: 64
  eviction_policy: "lfu"
  compression: "none"
  index_type: "hash"
```

### 3. SSD Index (SSD 索引)

**特点**:
- 中等访问速度
- 大容量
- 持久化存储
- 支持复杂查询

**配置**:
```yaml
ssd_index:
  path: "/data/ssd_index"
  max_size_gb: 500
  index_type: "vector"
  compression: "lz4"
```

## 组件规范

### 1. GPU Cache Manager (GPU 缓存管理器)

**职责**:
- 管理 GPU 内存分配
- 模型参数加载/卸载
- 批处理优化
- 内存碎片整理

**接口**:
```python
class GPUCache:
    def allocate(self, size_bytes: int) -> GPUPointer:
        """分配 GPU 内存"""
        pass
    
    def load_tensor(self, tensor_id: str) -> Tensor:
        """加载张量到 GPU"""
        pass
    
    def evict(self, strategy: EvictStrategy = "lru"):
        """驱逐数据"""
        pass
    
    def get_stats(self) -> GPUStats:
        """获取 GPU 统计"""
        pass
```

### 2. RAM Cache Manager (内存缓存管理器)

**职责**:
- 管理内存缓存
- 数据压缩/解压
- 索引维护
- 命中率优化

**接口**:
```python
class RAMCache:
    def get(self, key: str) -> Optional[bytes]:
        """获取数据"""
        pass
    
    def put(self, key: str, value: bytes, ttl: int = 0):
        """存储数据"""
        pass
    
    def invalidate(self, key: str):
        """使缓存失效"""
        pass
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        pass
```

### 3. SSD Index Manager (SSD 索引管理器)

**职责**:
- 管理 SSD 存储
- 索引构建和维护
- 查询优化
- 数据压缩

**接口**:
```python
class SSDIndex:
    def index(self, data: Data, metadata: dict) -> IndexEntry:
        """索引数据"""
        pass
    
    def search(self, query: Query, top_k: int = 10) -> list[SearchResult]:
        """搜索数据"""
        pass
    
    def compact(self):
        """压缩存储"""
        pass
    
    def backup(self, destination: str):
        """备份索引"""
        pass
```

### 4. Residency Scheduler (驻留调度器)

**职责**:
- 决定数据驻留层级
- 预测访问模式
- 预取决策
- 负载均衡

**接口**:
```python
class ResidencyScheduler:
    def schedule(
        self,
        data: Data,
        access_pattern: AccessPattern
    ) -> ResidencyDecision:
        """调度数据驻留"""
        pass
    
    def predict_access(
        self,
        data_id: str,
        horizon: int = 10
    ) -> AccessPrediction:
        """预测访问"""
        pass
    
    def prefetch(self, data_ids: list[str]):
        """预取数据"""
        pass
```

### 5. Migration Controller (迁移控制器)

**职责**:
- 执行数据迁移
- 管理迁移队列
- 优化迁移路径
- 处理迁移失败

**接口**:
```python
class MigrationController:
    def migrate(
        self,
        data: Data,
        from_tier: StorageTier,
        to_tier: StorageTier
    ) -> MigrationResult:
        """迁移数据"""
        pass
    
    def batch_migrate(
        self,
        migrations: list[MigrationTask]
    ) -> list[MigrationResult]:
        """批量迁移"""
        pass
    
    def get_queue_status(self) -> QueueStatus:
        """获取队列状态"""
        pass
```

## 数据流动

```
访问请求
    │
    ▼
┌─────────────┐ 命中 ──► 返回数据
│  GPU Cache  │
└──────┬──────┘ 未命中
       │
       ▼
┌─────────────┐ 命中 ──► 加载到 GPU ──► 返回
│  RAM Cache  │
└──────┬──────┘ 未命中
       │
       ▼
┌─────────────┐ 命中 ──► 加载到 RAM ──► 返回
│  SSD Index  │
└──────┬──────┘ 未命中
       │
       ▼
   外部存储
```

## 配置参数

```yaml
storage_scheduler:
  tiers:
    gpu:
      enabled: true
      priority: 1
      max_size_gb: 16
    ram:
      enabled: true
      priority: 2
      max_size_gb: 64
    ssd:
      enabled: true
      priority: 3
      max_size_gb: 500
  
  residency:
    prediction_enabled: true
    prefetch_enabled: true
    prefetch_lookahead: 5
  
  migration:
    batch_size: 100
    max_concurrent: 5
    throttle_ms: 10
  
  monitoring:
    enabled: true
    metrics_interval: 30
```

## 性能要求

- GPU 访问延迟: < 1ms
- RAM 访问延迟: < 10ms
- SSD 访问延迟: < 100ms
- 迁移吞吐量: > 500 MB/s
