# 记忆调度器规范 v1

## 概述

记忆调度器管理信息在不同记忆区域间的流动，优化存储效率和检索性能。

## 职责

1. 记忆区域管理
2. 数据流动调度
3. 门控协调
4. 存储优化
5. 过期清理

## 架构

```
┌─────────────────────────────────────┐
│      Memory Scheduler               │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │      Zone Manager           │   │
│  │  ┌─────┬─────┬─────┬─────┐  │   │
│  │  │Trans│ Long│Shall│Deep │  │   │
│  │  └─────┴─────┴─────┴─────┘  │   │
│  └──────────────┬──────────────┘   │
│                 │                  │
│  ┌──────────────┴──────────────┐   │
│  │      Gate Coordinator       │   │
│  │  ┌─────┬─────┬─────┬─────┐  │   │
│  │  │Write│Stab │Prom │Perm │  │   │
│  │  └─────┴─────┴─────┴─────┘  │   │
│  └─────────────────────────────┘   │
│                 │                  │
│                 ▼                  │
│  ┌─────────────────────────────┐   │
│  │    Memory Event Handler     │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## 记忆区域

### 1. Transient (瞬态记忆)

**特点**:
- 生命周期短 (秒级)
- 容量有限
- 访问速度快
- 不持久化

**用途**:
- 当前会话上下文
- 临时计算结果
- 中间推理状态

**配置**:
```yaml
transient:
  max_size: 1000  # units
  ttl_seconds: 300
  eviction_policy: "lru"
```

### 2. Long-term (长期记忆)

**特点**:
- 生命周期中等 (小时-天)
- 容量较大
- 定期持久化
- 支持索引

**用途**:
- 会话历史
- 频繁访问信息
- 待验证知识

**配置**:
```yaml
long_term:
  max_size: 10000  # units
  ttl_hours: 168  # 7 days
  persistence: "async"
  index_type: "vector"
```

### 3. Shallow Permanent (浅层永久记忆)

**特点**:
- 永久存储
- 压缩存储
- 快速检索
- 定期归档

**用途**:
- 验证后的知识
- 用户偏好
- 常用事实

**配置**:
```yaml
shallow_permanent:
  storage: "ssd"
  compression: "lz4"
  index_update_interval: 3600
```

### 4. Deep Permanent (深层永久记忆)

**特点**:
- 永久存储
- 多重备份
- 严格保护
- 版本控制

**用途**:
- 核心知识
- 系统配置
- 审计日志

**配置**:
```yaml
deep_permanent:
  storage: "distributed"
  replication_factor: 3
  backup_schedule: "daily"
  access_control: "strict"
```

## 组件规范

### 1. Zone Manager (区域管理器)

**职责**:
- 管理各记忆区域
- 监控区域状态
- 协调区域间数据流动
- 处理区域满的情况

**接口**:
```python
class ZoneManager:
    def get_zone(self, zone_type: MemoryZone) -> Zone:
        """获取记忆区域"""
        pass
    
    def move_data(
        self,
        data: Memory,
        from_zone: MemoryZone,
        to_zone: MemoryZone
    ) -> bool:
        """移动数据"""
        pass
    
    def get_zone_stats(self, zone_type: MemoryZone) -> ZoneStats:
        """获取区域统计"""
        pass
```

### 2. Gate Coordinator (门控协调器)

**职责**:
- 协调各门控检查
- 管理门控顺序
- 处理门控决策
- 记录门控日志

**接口**:
```python
class GateCoordinator:
    def evaluate(
        self,
        memory: Memory,
        operation: Operation,
        target_zone: MemoryZone
    ) -> GateDecision:
        """评估门控"""
        pass
    
    def register_gate(self, gate: Gate, priority: int):
        """注册门控"""
        pass
```

### 3. Memory Event Handler (记忆事件处理器)

**职责**:
- 处理记忆事件
- 触发调度操作
- 生成事件日志
- 通知订阅者

**接口**:
```python
class MemoryEventHandler:
    def handle_event(self, event: MemoryEvent):
        """处理事件"""
        pass
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """订阅事件"""
        pass
```

## 数据流动

```
新记忆
  │
  ▼
┌─────────────┐
│  Transient  │ ◄── 写入门
└──────┬──────┘
       │ TTL 过期 / 稳定性检查
       ▼
┌─────────────┐
│  Long-term  │ ◄── 稳定性门
└──────┬──────┘
       │ 访问频率 / 重要性
       ▼
┌─────────────┐
│   Shallow   │ ◄── 提升门
│   Permanent │
└──────┬──────┘
       │ 核心知识 / 验证
       ▼
┌─────────────┐
│    Deep     │ ◄── 永久保护门
│   Permanent │
└─────────────┘
```

## 配置参数

```yaml
memory_scheduler:
  zones:
    - transient
    - long_term
    - shallow_permanent
    - deep_permanent
  
  scheduling:
    auto_promotion: true
    promotion_interval: 3600  # seconds
    cleanup_interval: 600     # seconds
  
  gates:
    order: ["write", "stability", "promotion", "permanent_protection"]
    fail_fast: true
  
  monitoring:
    enabled: true
    metrics_interval: 60
    alert_threshold:
      transient_usage: 0.9
      long_term_usage: 0.85
```

## 性能要求

- 写入延迟: < 10ms (Transient)
- 提升延迟: < 100ms
- 调度吞吐量: > 1000 ops/s
- 内存开销: < 500MB
