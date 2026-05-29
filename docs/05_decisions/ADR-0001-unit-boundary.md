# ADR-0001: Unit 边界定义

## 状态
- 状态: 已接受
- 日期: 2026-04-17
- 作者: Post Transformer AI Team

## 背景

在 Post Transformer AI 系统中，我们需要一个基本的处理单元来封装和管理认知任务。这个单元需要能够表示各种类型的处理步骤，并支持系统各组件间的通信。

## 决策

我们决定定义 **Unit** 作为系统的基本处理单元。

### Unit 核心属性

1. **标识属性**: unit_id, parent_id, session_id
2. **内容属性**: unit_type, content_type, content
3. **状态属性**: status, priority, ttl
4. **元数据**: metadata, timestamp

### Unit 类型

1. **InputUnit**: 用户输入
2. **ThinkingUnit**: 推理步骤
3. **RetrievalUnit**: 检索操作
4. **MemoryUnit**: 记忆操作
5. **TSLAUnit**: 可信度评估
6. **OutputUnit**: 系统输出

## 理由

### 为什么选择 Unit?

1. **统一抽象**: 所有处理步骤使用统一的数据结构
2. **可追踪性**: 通过 parent_id 支持层级关系追踪
3. **灵活性**: 支持多种 Unit 类型扩展
4. **序列化友好**: 便于存储和传输

### 替代方案考虑

#### 替代方案 1: 函数调用链
- **优点**: 简单直接
- **缺点**: 难以追踪状态，不支持持久化
- **结论**: 不适用

#### 替代方案 2: 事件流
- **优点**: 松耦合
- **缺点**: 复杂度高，难以保证顺序
- **结论**: 部分采用，作为内部通信机制

#### 替代方案 3: 工作流引擎
- **优点**: 成熟方案
- **缺点**: 过于重量级，灵活性不足
- **结论**: 不适用

## 影响

### 正面影响
- 统一的数据模型简化开发
- 支持完整的审计追踪
- 便于实现记忆系统

### 负面影响
- 引入一定的序列化开销
- 需要维护 Unit 类型注册表

## 实施

### 代码实现
```python
class Unit(BaseModel):
    unit_id: UUID
    unit_type: UnitType
    content: dict
    status: Status
    # ...
```

### 相关文档
- [Unit 定义](../01_theory_frozen/unit_definition.md)
- [Unit Schema](../03_api_contracts/unit_schema.yaml)

## 后续工作

1. 实现 Unit 验证器
2. 实现 Unit 序列化
3. 建立 Unit 类型注册表
