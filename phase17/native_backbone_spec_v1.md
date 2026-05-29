# Native Backbone Spec v1 - 原生主干规格 v1

## 目标

定义"新框架 AI"的可训练主干，不再依赖 API，而是拥有自己独立的模型结构。

## 核心设计原则

1. **参数负责能力骨架，不负责塞满全部知识**
2. **训练目标是把"缺口识别→检索→治理→晋升"内化成默认行为**
3. **五门迁移、TSLA 八动作和 Unit 边界应保持稳定，不在训练中漂移**

---

## 1. 输入单位 (Input Unit)

### 1.1 中文输入
- **基本单位**: 字 (character)
- **Unit 定义**: 1-4 个字组成的语义单元
- **编码方式**: 字级别 embedding + Unit 级别聚合

### 1.2 英文输入
- **基本单位**: word
- **Unit 定义**: 1-3 个 word 组成的语义单元
- **编码方式**: word embedding + Unit 级别聚合

### 1.3 统一编码空间
```
输入文本 → 分词/分字 → Unit 识别 → 统一编码向量 (dim=768)
```

---

## 2. 状态表示 (State Representation)

### 2.1 双状态架构

不再使用单一的 hidden-state，而是明确的**双状态**:

```python
class FrameState:
    """框架状态"""
    unit_state: Tensor[batch, seq_len, hidden_dim]  # Unit 级别状态
    memory_state: MemoryContext                      # 记忆上下文状态
    turn_state: TurnContext                          # 当前轮次状态
```

### 2.2 Unit State
- **维度**: [batch_size, max_units, hidden_dim=768]
- **作用**: 当前输入的 Unit 级别编码
- **更新**: 每轮对话更新

### 2.3 Memory State
- **结构**: 三层记忆索引 (瞬时/长期/深层永久)
- **作用**: 跨轮次保持的上下文
- **更新**: 由 Memory Writeback Head 决定

### 2.4 Turn State
- **内容**: 当前轮次的工作状态
  - 缺口识别结果
  - 检索计划
  - 治理动作
  - 输出策略

---

## 3. 前向主干 (Forward Backbone)

### 3.1 统一流程（可训练算子图）

```
输入 → 初步解析 → 思考 → 缺失感知 → 内部检索 → 外部检索 → 整合 → TSLA → 输出 → 记忆更新
```

### 3.2 各阶段定义

#### Stage 1: 输入编码 (Input Encoding)
```python
# 可学习
unit_encoder: UnitEncoder  # 将文本编码为 Unit 向量
```

#### Stage 2: 初步解析 (Initial Parsing)
```python
# 可学习
intent_head: IntentHead     # 意图识别
parse_encoder: ParseEncoder # 解析输入结构
```

#### Stage 3: 思考 (Thinking)
```python
# 可学习
thought_processor: ThoughtProcessor  # 内部思考过程
```

#### Stage 4: 缺失感知 (Gap Detection)
```python
# 可学习
gap_detector: GapDetector   # 识别知识缺口
necessity_scorer: NecessityScorer  # 检索必要性评分
```

#### Stage 5: 内部检索 (Internal Retrieval)
```python
# 可学习
memory_key_encoder: MemoryKeyEncoder  # 记忆键编码
retrieval_router: RetrievalRouter     # 检索路由

# 冻结规则
memory_index: MemoryIndex  # 记忆索引结构（先冻结）
```

#### Stage 6: 外部检索 (External Retrieval)
```python
# 冻结规则（外部接口）
external_retrieval: ExternalRetrievalInterface
```

#### Stage 7: 整合 (Integration)
```python
# 可学习
context_integrator: ContextIntegrator  # 上下文整合
knowledge_fusion: KnowledgeFusion      # 知识融合
```

#### Stage 8: TSLA (Think-Speak-Listen-Act)
```python
# 可学习
tsla_processor: TSLAProcessor  # TSLA 处理

# 冻结规则
tsla_actions: TSLAActionSet    # TSLA 八动作（冻结）
```

#### Stage 9: 输出 (Output)
```python
# 可学习
policy_head: PolicyHead              # 策略选择
governance_head: GovernanceHead      # 治理头
response_decoder: ResponseDecoder    # 响应解码
```

#### Stage 10: 记忆更新 (Memory Update)
```python
# 可学习
writeback_head: WritebackHead        # 写回头
memory_encoder: MemoryEncoder        # 记忆编码

# 冻结规则
memory_tier_policy: MemoryTierPolicy  # 记忆层级策略（冻结）
```

---

## 4. 可学习模块 vs 冻结规则模块

### 4.1 可学习模块（训练目标）

| 模块 | 功能 | 训练目标 |
|------|------|----------|
| UnitEncoder | 输入编码 | 准确编码 Unit 语义 |
| IntentHead | 意图识别 | 准确识别用户意图 |
| GapDetector | 缺口检测 | 准确识别何时需要检索 |
| NecessityScorer | 必要性评分 | 准确评估检索必要性 |
| MemoryKeyEncoder | 记忆键编码 | 生成有效的检索键 |
| RetrievalRouter | 检索路由 | 路由到正确的记忆层 |
| ContextIntegrator | 上下文整合 | 有效整合检索结果 |
| PolicyHead | 策略选择 | 准确选择响应策略 |
| GovernanceHead | 治理头 | 准确执行治理动作 |
| WritebackHead | 写回头 | 准确决定记忆写回 |
| ResponseDecoder | 响应解码 | 生成高质量响应 |

### 4.2 冻结规则模块（保持稳定）

| 模块 | 功能 | 冻结原因 |
|------|------|----------|
| TSLAActionSet | TSLA 八动作 | 框架核心行为定义 |
| MemoryTierPolicy | 记忆层级策略 | 三层记忆边界定义 |
| FiveGatePolicy | 五门迁移 | 晋升/隔离/回流规则 |
| PermanentLayerThreshold | 永久层门槛 | 高门槛原则 |

---

## 5. 模型结构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Native AI Framework                      │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                                 │
│  ├── UnitEncoder (可学习)                                    │
│  └── IntentHead (可学习)                                     │
├─────────────────────────────────────────────────────────────┤
│  Processing Layer                                            │
│  ├── ThoughtProcessor (可学习)                               │
│  ├── GapDetector (可学习)                                    │
│  └── NecessityScorer (可学习)                                │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Layer                                             │
│  ├── MemoryKeyEncoder (可学习)                               │
│  ├── RetrievalRouter (可学习)                                │
│  └── MemoryIndex (冻结规则)                                  │
├─────────────────────────────────────────────────────────────┤
│  Integration Layer                                           │
│  ├── ContextIntegrator (可学习)                              │
│  └── KnowledgeFusion (可学习)                                │
├─────────────────────────────────────────────────────────────┤
│  TSLA Layer (冻结规则框架 + 可学习处理器)                      │
│  ├── TSLAProcessor (可学习)                                  │
│  └── TSLAActionSet (冻结规则)                                │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                                │
│  ├── PolicyHead (可学习)                                     │
│  ├── GovernanceHead (可学习)                                 │
│  └── ResponseDecoder (可学习)                                │
├─────────────────────────────────────────────────────────────┤
│  Memory Layer                                                │
│  ├── WritebackHead (可学习)                                  │
│  ├── MemoryEncoder (可学习)                                  │
│  └── MemoryTierPolicy (冻结规则)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 训练接口定义

### 6.1 前向接口
```python
def forward(
    self,
    input_units: Tensor[batch, seq_len],
    memory_context: Optional[MemoryContext] = None,
    turn_context: Optional[TurnContext] = None
) -> FrameOutput:
    """前向传播"""
    pass
```

### 6.2 损失接口
```python
def compute_loss(
    self,
    outputs: FrameOutput,
    targets: FrameTargets
) -> Dict[str, Tensor]:
    """计算多任务损失"""
    return {
        "policy_loss": policy_loss,
        "retrieval_loss": retrieval_loss,
        "governance_loss": governance_loss,
        "memory_loss": memory_loss,
        "response_loss": response_loss,
    }
```

### 6.3 状态更新接口
```python
def update_memory(
    self,
    turn_output: TurnOutput,
    writeback_decision: WritebackDecision
) -> MemoryContext:
    """更新记忆状态"""
    pass
```

---

## 7. 交付物清单

- [x] native_backbone_spec_v1.md (本文档)
- [ ] unit_encoder_spec_v1.md
- [ ] memory_bus_spec_v1.md
- [ ] governance_head_spec_v1.md
- [ ] native_forward_graph_v1.md

---

## 8. 下一步

完成规格文档后，进入 **阶段 2：本地真实训练基础设施**。
