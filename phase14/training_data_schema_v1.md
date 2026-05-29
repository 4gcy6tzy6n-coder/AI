# Training Data Schema v1 - 训练数据模式 v1

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 14 - Controlled Training, Behavior Internalization & Governance-Aligned Learning

---

## 1. 概述

本文档定义 Phase 14 训练数据的完整 Schema，确保数据分层清晰、用途明确、不与真实用户数据混用。

### 1.1 核心原则

1. **五层分离** - D1-D5 严格分层，不混用
2. **治理优先** - 训练数据本身需经过治理审查
3. **行为内化** - 目标是把平台规则训成模型习惯
4. **知识外置** - 参数不负责吞掉全部知识

---

## 2. 数据分层架构

```
训练数据体系
├── D1: 高质量骨架数据 (30%)
│   └── 基础理解、表达、任务遵循
├── D2: 检索与治理决策数据 (25%)
│   └── 五类响应策略、检索触发
├── D3: 冲突/噪声/对抗数据 (20%)
│   └── 回流、隔离、拒绝装懂
├── D4: 多轮对话与记忆数据 (15%)
│   └── 长期行为、记忆判断
└── D5: 真实样本回放数据 (10%)
    └── 后期对齐、不用于早期训练
```

---

## 3. 通用 Schema 定义

### 3.1 基础字段

```json
{
  "id": "string",                    // 样本唯一ID
  "version": "1.0",                  // Schema版本
  "dataset_type": "D1|D2|D3|D4|D5",  // 数据层类型
  "created_at": "ISO8601",           // 创建时间
  "governance_status": {             // 治理状态
    "reviewed": true,
    "reviewer": "string",
    "approved_at": "ISO8601",
    "quality_score": 0.0-1.0
  }
}
```

### 3.2 输入字段

```json
{
  "input": {
    "user_query": "string",          // 用户输入
    "conversation_history": [        // 对话历史(可选)
      {
        "turn": 1,
        "role": "user|assistant",
        "content": "string"
      }
    ],
    "retrieved_context": [           // 检索结果(可选)
      {
        "content": "string",
        "source": "string",
        "confidence": 0.0-1.0
      }
    ],
    "metadata": {                    // 元数据
      "language": "zh|en|mixed",
      "domain": "string",
      "difficulty": "easy|medium|hard"
    }
  }
}
```

### 3.3 输出字段

```json
{
  "output": {
    "response_strategy": "DIRECT|RETRIEVAL_FIRST|CONSERVATIVE|DECLINE|REVIEW",
    "confidence_score": 0.0-1.0,
    "final_response": "string",
    "reasoning": "string",           // 推理过程
    "retrieval_triggered": true|false,
    "governance_actions": [          // 治理动作
      "TSLA|STRONG_REVIEW|ISOLATE|ARCHIVE|..."
    ],
    "memory_decision": {             // 记忆决策
      "should_writeback": true|false,
      "target_layer": "ephemeral|long_term|deep_permanent",
      "promotion_path": ["TSLA", "STRONG_REVIEW", "VALIDATION"]
    }
  }
}
```

---

## 4. 各层数据详细 Schema

### 4.1 D1: 高质量骨架数据

**用途**: 训基础理解、表达、任务遵循、意图识别

**Schema**:
```json
{
  "id": "D1_001",
  "dataset_type": "D1",
  "input": {
    "user_query": "string",
    "conversation_history": [],
    "metadata": {
      "language": "zh|en",
      "intent_type": "fact|opinion|action|clarification",
      "complexity": "simple|compound|complex"
    }
  },
  "output": {
    "response_strategy": "DIRECT",
    "confidence_score": 0.95,
    "final_response": "string",
    "persona_consistency": true,     // 是否符合AI人格
    "style_markers": {               // 风格标记
      "citation_used": false,
      "conservative_tone": false,
      "retrieval_referenced": false
    }
  },
  "quality_annotations": {
    "fluency": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "helpfulness": 0.0-1.0
  }
}
```

**样本示例**:
```json
{
  "id": "D1_001",
  "dataset_type": "D1",
  "input": {
    "user_query": "Python是什么编程语言？",
    "metadata": {
      "language": "zh",
      "intent_type": "fact",
      "complexity": "simple"
    }
  },
  "output": {
    "response_strategy": "DIRECT",
    "confidence_score": 0.95,
    "final_response": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁、易读的语法著称。",
    "persona_consistency": true,
    "style_markers": {
      "citation_used": false,
      "conservative_tone": false,
      "retrieval_referenced": false
    }
  }
}
```

---

### 4.2 D2: 检索与治理决策数据

**用途**: 训五类响应策略、检索触发、治理分流

**Schema**:
```json
{
  "id": "D2_001",
  "dataset_type": "D2",
  "input": {
    "user_query": "string",
    "conversation_history": [],
    "evidence_status": {             // 证据状态
      "has_sufficient_evidence": true|false,
      "evidence_confidence": 0.0-1.0,
      "evidence_sources": ["internal|external|memory"]
    },
    "risk_indicators": {             // 风险指标
      "sensitive_topic": true|false,
      "factual_claim": true|false,
      "high_stakes": true|false
    }
  },
  "output": {
    "response_strategy": "RETRIEVAL_FIRST|CONSERVATIVE|...",
    "confidence_score": 0.0-1.0,
    "retrieval_triggered": true|false,
    "retrieval_necessity_score": 0.0-1.0,
    "governance_actions": ["TSLA", "STRONG_REVIEW"],
    "final_response": "string",
    "reasoning": "为什么选这个策略"
  },
  "decision_ground_truth": {         // 决策标注
    "correct_strategy": "string",
    "explanation": "string"
  }
}
```

**样本类型**:

| 类型 | 描述 | 策略 |
|------|------|------|
| 可直接答 | 高置信度、无争议 | DIRECT |
| 需要检索 | 需验证、用户问"记得" | RETRIEVAL_FIRST |
| 高风险 | 敏感、可能有害 | REVIEW |
| 低证据 | 证据不足但流畅 | CONSERVATIVE |
| 易幻觉 | 模型可能瞎编 | DECLINE |

---

### 4.3 D3: 冲突/噪声/对抗数据

**用途**: 训回流、隔离、错误归档、拒绝装懂

**Schema**:
```json
{
  "id": "D3_001",
  "dataset_type": "D3",
  "input": {
    "user_query": "string",
    "conflict_type": "knowledge_conflict|evidence_conflict|internal_conflict",
    "noise_level": "low|medium|high",
    "adversarial_features": {
      "is_misleading": true|false,
      "contains_false_premise": true|false,
      "exploits_model_bias": true|false
    }
  },
  "output": {
    "response_strategy": "CONSERVATIVE|DECLINE|REVIEW",
    "confidence_score": 0.0-1.0,
    "final_response": "string",
    "governance_actions": ["ISOLATE", "ARCHIVE", "ROLLBACK"],
    "error_handling": {
      "detected_conflict": true|false,
      "escalation_path": "string",
      "fallback_strategy": "string"
    }
  },
  "adversarial_annotation": {
    "attack_type": "string",
    "vulnerability_exploited": "string",
    "correct_behavior": "string"
  }
}
```

**样本类型**:

| 类型 | 描述 | 预期行为 |
|------|------|----------|
| 冲突知识 | 矛盾的事实陈述 | 检测冲突，CONSERVATIVE |
| 伪证据 | 伪造的引用/来源 | 验证失败，DECLINE |
| 高置信错误 | 模型过度自信的错误 | 置信度校准 |
| 中英干扰 | 语言混合导致的混乱 | 保持策略一致 |
| 多义未拆分 | 歧义未解决 | 请求澄清 |

---

### 4.4 D4: 多轮对话与记忆数据

**用途**: 训 conversation orchestrator 的长期行为

**Schema**:
```json
{
  "id": "D4_001",
  "dataset_type": "D4",
  "input": {
    "conversation_id": "string",
    "current_turn": 5,
    "conversation_history": [
      {
        "turn": 1,
        "role": "user",
        "content": "string"
      },
      {
        "turn": 2,
        "role": "assistant",
        "content": "string",
        "strategy_used": "DIRECT"
      }
    ],
    "user_query": "string",
    "context_dependencies": [          // 上下文依赖
      {
        "refers_to_turn": 2,
        "dependency_type": "entity|topic|constraint"
      }
    ]
  },
  "output": {
    "response_strategy": "string",
    "final_response": "string",
    "memory_decision": {
      "should_writeback": true|false,
      "writeback_content": "string",
      "target_layer": "ephemeral|long_term|deep_permanent",
      "promotion_decision": "IMMEDIATE|DELAYED|REJECTED"
    },
    "context_maintenance": {
      "topic_consistency": 0.0-1.0,
      "persona_consistency": 0.0-1.0,
      "reference_accuracy": 0.0-1.0
    }
  },
  "memory_ground_truth": {
    "correct_writeback": true|false,
    "correct_layer": "string",
    "expected_retrieval": ["string"]
  }
}
```

**样本类型**:

| 类型 | 描述 | 记忆决策 |
|------|------|----------|
| 只留瞬时层 | 临时信息 | 不写回 |
| 进入长期受审 | 需验证的信息 | long_term + TSLA |
| 进入长期正常 | 已验证信息 | long_term |
| 隔离/错误归档 | 错误信息 | ISOLATE |
| 冻结晋升 | 高风险信息 | 不晋升 |

---

### 4.5 D5: 真实样本回放数据

**用途**: 后期对齐和校准，不用于早期大规模训练

**Schema**:
```json
{
  "id": "D5_001",
  "dataset_type": "D5",
  "source": {
    "phase": "12|13",
    "session_type": "shadow|pilot|test",
    "collection_date": "ISO8601",
    "failure_type": "retrieval_fail|governance_fail|memory_fail|user_complaint"
  },
  "input": {
    "user_query": "string",
    "conversation_history": [],
    "system_state": {
      "retrieval_results": [],
      "governance_decisions": [],
      "memory_state": {}
    }
  },
  "actual_output": {
    "response": "string",
    "strategy": "string",
    "user_feedback": "positive|negative|neutral"
  },
  "expected_output": {
    "response": "string",
    "strategy": "string",
    "improvement_notes": "string"
  },
  "alignment_priority": "high|medium|low",
  "usage_restriction": "eval_only|fine_tune_later|never_train"
}
```

**使用限制**:
- 早期训练: 仅用于评估
- 后期对齐: 可用于微调
- 敏感数据: 永不用于训练

---

## 5. 数据治理要求

### 5.1 准入检查

每批数据必须经过:

| 检查项 | 标准 | 工具 |
|--------|------|------|
| 格式验证 | 符合 Schema | JSON Schema Validator |
| 质量评分 | > 0.7 | 自动评分器 |
| 去重检查 | 相似度 < 0.9 | 语义去重 |
| 毒性检测 | 无毒 | 内容过滤器 |
| 治理审查 | 已批准 | 人工审核 |

### 5.2 版本控制

```
datasets/
├── v1.0.0/
│   ├── d1_seed_quality.jsonl
│   ├── d2_policy_governance.jsonl
│   ├── d3_conflict_noise.jsonl
│   ├── d4_multiturn_memory.jsonl
│   └── d5_real_case_replay.jsonl
├── v1.1.0/  # 增量更新
└── manifest.json  # 数据清单
```

### 5.3 使用追踪

```json
{
  "training_run_id": "run_001",
  "datasets_used": [
    {
      "dataset": "d1_seed_quality",
      "version": "v1.0.0",
      "samples_used": 10000,
      "weight": 0.30
    }
  ],
  "mixing_strategy": "stratified",
  "governance_approval": "approved"
}
```

---

## 6. 数据生成流水线

### 6.1 生成流程

```
原始数据
  │
  ▼
┌─────────────────┐
│  数据清洗        │
│  - 去重         │
│  - 过滤         │
│  - 格式化       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  质量评分        │
│  - 自动评分      │
│  - 人工抽检      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  治理审查        │
│  - TSLA初判     │
│  - 强审查       │
│  - 验证         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  分层存储        │
│  - D1-D5分类    │
│  - 版本标记      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  训练就绪        │
│  - 混合策略      │
│  - 加载器配置    │
└─────────────────┘
```

---

## 7. 附录

### 7.1 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本 |

### 7.2 相关文档

- training_metric_contract_v1.md
- train_policy_head.py
- train_retrieval_governance.py

---

**文档状态**: 生效中  
**负责人**: Phase 14 训练负责人
