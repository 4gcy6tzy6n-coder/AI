# Unit 定义

## 概述

Unit 是 Post Transformer AI 系统的基本处理单元，代表一个独立的认知任务或信息块。

## 核心属性

### 标识
- `unit_id`: 唯一标识符 (UUID)
- `parent_id`: 父 Unit ID (用于层级关系)
- `session_id`: 会话标识符
- `timestamp`: 创建时间戳

### 内容
- `content_type`: 内容类型 (text/code/query/result)
- `content`: 实际内容
- `metadata`: 附加元数据

### 状态
- `status`: 当前状态 (pending/processing/completed/failed)
- `priority`: 优先级 (1-10)
- `ttl`: 生存时间 (秒)

## Unit 类型

### 1. 输入 Unit (InputUnit)
接收用户输入或外部信号

```yaml
input_unit:
  source: user|api|system
  raw_input: string
  parsed_intent: object
```

### 2. 思考 Unit (ThinkingUnit)
表示一个推理步骤

```yaml
thinking_unit:
  reasoning_type: analysis|planning|deduction
  steps: list[Step]
  confidence: float
```

### 3. 检索 Unit (RetrievalUnit)
表示信息检索操作

```yaml
retrieval_unit:
  query: string
  sources: list[Source]
  results: list[Result]
  relevance_scores: list[float]
```

### 4. 记忆 Unit (MemoryUnit)
表示记忆存储操作

```yaml
memory_unit:
  operation: read|write|update|delete
  memory_zone: transient|long_term|shallow|deep
  content_hash: string
```

### 5. TSLA Unit (TSLAUnit)
可信度评估单元

```yaml
tsla_unit:
  action_type: accept|reject|review|escalate
  confidence_score: float
  risk_level: low|medium|high|critical
  audit_trail: list[AuditEntry]
```

## Unit 关系

### 层级关系
- 父子关系: 复杂 Unit 可包含子 Unit
- 兄弟关系: 同层级并行 Unit

### 依赖关系
- 顺序依赖: Unit B 需要 Unit A 的结果
- 数据依赖: Unit B 需要 Unit A 的输出数据

### 引用关系
- 记忆引用: Unit 引用已存储的记忆
- 证据引用: Unit 引用支持证据

## Unit 生命周期

1. **创建**: 实例化 Unit 对象
2. **验证**: 检查 Unit 完整性
3. **路由**: 分发到相应引擎
4. **处理**: 执行 Unit 任务
5. **评估**: TSLA 评估结果
6. **存储**: 决定记忆存储策略
7. **归档**: 完成或失败处理
