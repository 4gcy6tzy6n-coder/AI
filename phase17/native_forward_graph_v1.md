# Native Forward Graph v1 - 原生前向图 v1

## 目标

定义完整的可训练前向计算图，将统一流程文本转化为可执行的算子图。

## 完整前向流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Native AI Framework Forward Graph                    │
└─────────────────────────────────────────────────────────────────────────────┘

输入: 用户查询 (文本)
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: 输入编码 (Input Encoding)                                           │
│ 模块: UnitEncoder (可学习)                                                   │
│ 输出: unit_state [batch, num_units, hidden_dim]                             │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: 初步解析 (Initial Parsing)                                          │
│ 模块: IntentHead (可学习)                                                    │
│ 输出: intent_vector, parse_tree                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: 思考 (Thinking)                                                     │
│ 模块: ThoughtProcessor (可学习)                                              │
│ 输出: thought_state [batch, hidden_dim]                                     │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: 缺失感知 (Gap Detection)                                            │
│ 模块: GapDetector + NecessityScorer (可学习)                                 │
│ 输出: gap_detected (bool), necessity_score (float)                          │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
                    ┌─────────────────┐
                    │ gap_detected?   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │ True                              │ False
            ↓                                   ↓
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ Stage 5: 内部检索            │    │ (跳过检索)                  │
│ Memory Bus                   │    │ empty_retrieval             │
└─────────────────────────────┘    └─────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 6: 外部检索 (External Retrieval)                                       │
│ 模块: ExternalRetrievalInterface (冻结规则 - 外部API)                        │
│ 输出: external_results                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 7: 整合 (Integration)                                                  │
│ 模块: ContextIntegrator + KnowledgeFusion (可学习)                           │
│ 输出: integrated_context [batch, context_len, hidden_dim]                   │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 8: TSLA (Think-Speak-Listen-Act)                                       │
│ 模块: TSLAProcessor (可学习) + TSLAActionSet (冻结规则)                      │
│ 输出: tsla_state, selected_actions                                          │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 9: 治理 (Governance)                                                   │
│ 模块: GovernanceHead (可学习)                                                │
│ 输出: governance_decision (actions, gate, confidence)                       │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 10: 策略选择 (Policy Selection)                                        │
│ 模块: PolicyHead (可学习)                                                    │
│ 输出: response_strategy (DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW)│
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 11: 响应生成 (Response Generation)                                     │
│ 模块: ResponseDecoder (可学习)                                               │
│ 输出: response_text                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 12: 记忆更新 (Memory Update)                                           │
│ 模块: WritebackHead + MemoryEncoder (可学习)                                 │
│ 输出: writeback_decision, updated_memory_context                            │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
输出: 响应文本 + 更新后的记忆状态
```

## 详细算子定义

### Stage 1: UnitEncoder
```python
# 输入: input_text (str)
# 输出: unit_state [batch, num_units, 768]
unit_state = UnitEncoder(input_text)
```

### Stage 2: IntentHead
```python
# 输入: unit_state
# 输出: intent_vector, parse_tree
intent_vector = IntentHead(unit_state)
parse_tree = Parser(unit_state)
```

### Stage 3: ThoughtProcessor
```python
# 输入: unit_state, intent_vector
# 输出: thought_state
thought_state = ThoughtProcessor(unit_state, intent_vector)
```

### Stage 4: Gap Detection
```python
# 输入: thought_state, memory_context
# 输出: gap_detected, necessity_score
gap_detected = GapDetector(thought_state, memory_context)
necessity_score = NecessityScorer(thought_state, memory_context)
```

### Stage 5: Memory Retrieval
```python
# 输入: query=thought_state, necessity_score
# 输出: memory_results
if necessity_score > 0.5:
    memory_results = MemoryBus.retrieve(
        query=thought_state,
        top_k=5,
    )
else:
    memory_results = []
```

### Stage 6: External Retrieval
```python
# 输入: query, gap_detected
# 输出: external_results (冻结规则 - 可选)
if gap_detected and len(memory_results) < 3:
    external_results = ExternalRetrieval(query)
```

### Stage 7: Integration
```python
# 输入: unit_state, memory_results, external_results
# 输出: integrated_context
integrated_context = ContextIntegrator(
    unit_state,
    memory_results,
    external_results,
)
```

### Stage 8: TSLA
```python
# 输入: integrated_context
# 输出: tsla_state, selected_actions
tsla_state = TSLAProcessor(integrated_context)
selected_actions = TSLAActionSet.select(tsla_state)
```

### Stage 9: Governance
```python
# 输入: unit_state, memory_context, tsla_state
# 输出: governance_decision
governance_decision = GovernanceHead(
    unit_state,
    memory_context,
    tsla_state,
)
```

### Stage 10: Policy Selection
```python
# 输入: integrated_context, governance_decision
# 输出: response_strategy
response_strategy = PolicyHead(
    integrated_context,
    governance_decision,
)
```

### Stage 11: Response Generation
```python
# 输入: integrated_context, response_strategy
# 输出: response_text
response_text = ResponseDecoder(
    integrated_context,
    response_strategy,
)
```

### Stage 12: Memory Update
```python
# 输入: response_text, conversation_context
# 输出: writeback_decision
writeback_decision = WritebackHead(
    response_text,
    conversation_context,
)

if writeback_decision.should_writeback:
    MemoryBus.write(
        content=response_text,
        decision=writeback_decision,
    )
```

## 损失计算图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Loss Computation                                  │
└─────────────────────────────────────────────────────────────────────────────┘

预测输出
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Policy Loss                                                               │
│    目标: 正确选择响应策略 (DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW)│
│    损失: CrossEntropy(predicted_strategy, target_strategy)                  │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Retrieval Loss                                                            │
│    目标: 正确触发检索                                                        │
│    损失: BCE(retrieval_triggered, should_trigger)                           │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Governance Loss                                                           │
│    目标: 正确执行治理动作                                                    │
│    损失: MultiLabelBCE(predicted_actions, target_actions)                   │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Memory Loss                                                               │
│    目标: 正确决定记忆写回                                                    │
│    损失: BCE(should_writeback, target_writeback) + CrossEntropy(layer, target_layer)│
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Response Loss                                                             │
│    目标: 生成高质量响应                                                      │
│    损失: CrossEntropy(predicted_tokens, target_tokens)                      │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
总损失 = λ1*policy_loss + λ2*retrieval_loss + λ3*governance_loss + λ4*memory_loss + λ5*response_loss
```

## 可训练参数分布

```
总参数量估算 (Tiny版本 ~100M):

UnitEncoder:          20M  (20%)
IntentHead:            5M  (5%)
ThoughtProcessor:     15M  (15%)
GapDetector:           5M  (5%)
MemoryBus:            10M  (10%)
  - QueryEncoder:      3M
  - MemoryEncoder:     3M
  - LayerRouter:       2M
  - Stores:            2M
ContextIntegrator:    10M  (10%)
TSLAProcessor:         5M  (5%)
GovernanceHead:       10M  (10%)
PolicyHead:            5M  (5%)
ResponseDecoder:      15M  (15%)
─────────────────────────────────
Total:               ~100M
```

## 阶段 1 交付物完成

- [x] native_backbone_spec_v1.md
- [x] unit_encoder_spec_v1.md
- [x] memory_bus_spec_v1.md
- [x] governance_head_spec_v1.md
- [x] native_forward_graph_v1.md

---

## 下一步

进入 **阶段 2：本地真实训练基础设施**
