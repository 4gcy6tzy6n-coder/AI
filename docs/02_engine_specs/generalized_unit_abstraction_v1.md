# 通用 Unit 抽象 v1

**版本**: v1.0  
**日期**: 2026-04-17  
**目标**: 将中文原型中的 Unit 机制抽象为更一般的上层表达

---

## 1. 抽象目标

### 1.1 当前状态

中文原型已固定：
- **最小感知起点**：字音 + 字形
- **最小长期治理对象**：汉字实体 = 字音 + 字形 + 基本义

### 1.2 抽象目标

明确区分：
1. **中文特有实现**：与汉字结构相关的具体机制
2. **通用治理机制**：可跨语言/跨领域复用的核心机制

将 Unit 提升到更抽象的一般对象：
- Concept Unit（概念单元）
- Relation Unit（关系单元）
- Rule Unit（规则单元）
- Task Pattern Unit（任务模式单元）

---

## 2. 通用 Unit 定义

### 2.1 核心抽象

```
┌─────────────────────────────────────────────────────────────┐
│                     通用 Unit 结构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Unit Identity（单元标识）                            │   │
│  │ - unique_id: 全局唯一标识符                          │   │
│  │ - unit_type: 单元类型（concept/relation/rule/task）  │   │
│  │ - version: 版本号                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Perceptual Core（感知核心）                          │   │
│  │ - 最小可感知特征集合                                 │   │
│  │ - 语言/领域相关的具体表示                            │   │
│  │ - 示例：汉字的字音+字形，英文的拼写+音标             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Semantic Core（语义核心）                            │   │
│  │ - 核心意义/功能定义                                  │   │
│  │ - 边界清晰性声明                                     │   │
│  │ - 使用条件约束                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Governance Metadata（治理元数据）                    │   │
│  │ - quality_score: 质量分数（Q/T/S/C/L）               │   │
│  │ - stability_cycles: 稳定周期数                       │   │
│  │ - source: 来源追踪                                   │   │
│  │ - verification_history: 验证历史                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Unit 类型体系

#### Concept Unit（概念单元）

```yaml
unit_type: concept

# 感知核心（语言相关）
perceptual_core:
  # 中文实现示例
  chinese:
    - glyph: 字形表示
    - pronunciation: 字音表示
  # 英文实现示例
  english:
    - spelling: 拼写
    - phonetic: 音标
  # 通用抽象
  universal:
    - surface_form: 表面形式
    - identifier: 唯一标识符

# 语义核心
semantic_core:
  primary_meaning: 主要意义
  boundary_conditions:
    - applicable_contexts: 适用上下文
    - excluded_contexts: 排除上下文
  relations:
    - synonyms: 同义关系
    - antonyms: 反义关系
    - hypernyms: 上位关系
    - hyponyms: 下位关系
```

#### Relation Unit（关系单元）

```yaml
unit_type: relation

# 感知核心
perceptual_core:
  # 关系标识符
  relation_marker: 关系标记（如："是"、"属于"、"导致"）
  
# 语义核心
semantic_core:
  relation_type: 关系类型
    - hierarchical: 层级关系（is-a, part-of）
    - causal: 因果关系（causes, leads-to）
    - temporal: 时序关系（before, after）
    - spatial: 空间关系（above, below）
    - functional: 功能关系（used-for, enables）
  
  arity: 关系元数（unary/binary/ternary/n-ary）
  
  constraints:
    - domain: 定义域（哪些类型的实体可以参与）
    - range: 值域（可以产生什么类型的结果）
    - cardinality: 基数约束（一对一、一对多等）
```

#### Rule Unit（规则单元）

```yaml
unit_type: rule

# 感知核心
perceptual_core:
  rule_pattern: 规则模式（可识别的规则结构）
  trigger_conditions: 触发条件

# 语义核心
semantic_core:
  rule_type:
    - inference: 推理规则（if-then）
    - transformation: 转换规则（rewrite rules）
    - constraint: 约束规则（must/must-not）
    - preference: 偏好规则（should/should-not）
  
  premises: 前提条件列表
  conclusions: 结论列表
  confidence: 规则置信度
  
  applicability:
    - scope: 适用范围
    - exceptions: 例外情况
    - priority: 优先级
```

#### Task Pattern Unit（任务模式单元）

```yaml
unit_type: task_pattern

# 感知核心
perceptual_core:
  task_signature: 任务签名（输入/输出模式）
  trigger_keywords: 触发关键词

# 语义核心
semantic_core:
  task_type:
    - classification: 分类任务
    - generation: 生成任务
    - reasoning: 推理任务
    - retrieval: 检索任务
    - verification: 验证任务
  
  input_schema: 输入模式定义
  output_schema: 输出模式定义
  
  procedure:
    - steps: 执行步骤
    - required_units: 需要的其他 Unit
    - success_criteria: 成功标准
```

---

## 3. 通用治理机制

### 3.1 跨 Unit 类型的通用治理

```
┌─────────────────────────────────────────────────────────────┐
│                   通用治理流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 输入识别（Input Recognition）                           │
│     - 识别 Unit 的感知核心                                  │
│     - 验证最小可感知特征完整性                              │
│     ↓                                                       │
│  2. 语义解析（Semantic Parsing）                            │
│     - 提取语义核心                                          │
│     - 建立边界条件                                          │
│     ↓                                                       │
│  3. 质量评估（Quality Assessment）                          │
│     - Q: 意义清晰度                                         │
│     - T: 透明度（可追溯性）                                 │
│     - S: 稳定性                                             │
│     - C: 一致性                                             │
│     - L: 合法性                                             │
│     ↓                                                       │
│  4. 冲突检测（Conflict Detection）                          │
│     - 与现有 Unit 比较                                      │
│     - 检测语义冲突                                          │
│     - 检测边界重叠                                          │
│     ↓                                                       │
│  5. 晋升决策（Promotion Decision）                          │
│     - 基于阈值和周期的决策                                  │
│     - 考虑来源可信度                                        │
│     - 生成验证要求                                          │
│     ↓                                                       │
│  6. 存储与索引（Storage & Indexing）                        │
│     - 按 Unit 类型分类存储                                  │
│     - 建立跨 Unit 关系索引                                  │
│     - 更新检索结构                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 跨类型关系管理

```yaml
# Unit 间关系类型
inter_unit_relations:
  # 组合关系
  composition:
    - whole_part: 整体-部分（Concept 包含 Sub-concepts）
    - sequence: 序列（Rule 的步骤序列）
  
  # 依赖关系
  dependency:
    - prerequisite: 前置依赖（Task 需要某些 Concept）
    - corequisite: 共现依赖（Concept 与 Relation 共现）
  
  # 演化关系
  evolution:
    - refinement: 细化（Rule 细化为更具体的 Rule）
    - generalization: 泛化（Concept 泛化为更抽象的 Concept）
    - substitution: 替代（旧 Unit 被新 Unit 替代）
```

---

## 4. 语言/领域适配层

### 4.1 适配层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    通用 Unit 层                              │
│              （概念/关系/规则/任务模式）                     │
├─────────────────────────────────────────────────────────────┤
│                    适配层接口                                │
│         （定义如何映射到具体语言/领域）                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 中文适配    │  │ 英文适配    │  │ 其他语言适配        │ │
│  │ - 汉字结构  │  │ - 词法结构  │  │ - ...               │ │
│  │ - 字音系统  │  │ - 句法结构  │  │                     │ │
│  │ - 语义网络  │  │ - 语义角色  │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    具体实现层                                │
│         （当前中文原型的具体实现）                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 中文特有实现

```yaml
# 中文适配层
chinese_adapter:
  # 特有感知特征
  perceptual_features:
    - stroke_order: 笔画顺序
    - radical_structure: 部首结构
    - tonal_patterns: 声调模式
    - character_composition: 字形组合规则
  
  # 特有语义特征
  semantic_features:
    - character_meaning_evolution: 字义演变
    - compound_word_formation: 复合词构成
    - classical_modern_mapping: 古今义映射
  
  # 特有治理规则
  governance_rules:
    - traditional_simplified_unification: 简繁统一规则
    - polyphone_disambiguation: 多音字消歧规则
    - variant_character_handling: 异体字处理规则
```

### 4.3 英文适配示例

```yaml
# 英文适配层
english_adapter:
  # 特有感知特征
  perceptual_features:
    - morpheme_structure: 词素结构
    - syllable_pattern: 音节模式
    - stress_pattern: 重音模式
  
  # 特有语义特征
  semantic_features:
    - word_sense_disambiguation: 词义消歧
    - phrasal_verb_patterns: 短语动词模式
    - collocation_constraints: 搭配约束
  
  # 特有治理规则
  governance_rules:
    - inflection_variation_handling: 屈折变化处理
    - derivation_pattern_recognition: 派生模式识别
```

---

## 5. 通用化路径

### 5.1 从中文原型到通用框架

```
阶段 1: 识别通用模式（当前阶段）
  - 从中文实现中提取治理机制
  - 识别可复用的流程和结构
  - 定义通用接口

阶段 2: 抽象核心概念
  - 定义通用 Unit 类型
  - 建立跨类型关系模型
  - 设计适配层接口

阶段 3: 多语言适配验证
  - 实现英文适配层
  - 验证通用机制有效性
  - 调整抽象层级

阶段 4: 领域扩展
  - 从 NLP 扩展到其他领域
  - 验证跨领域适用性
  - 完善通用框架
```

### 5.2 关键抽象原则

1. **感知核心最小化**：只保留跨语言通用的最小感知特征
2. **语义核心标准化**：使用形式化方式定义语义
3. **治理机制通用化**：治理流程与具体语言无关
4. **适配层隔离**：语言特有实现隔离在适配层

---

## 6. 当前中文原型的通用化映射

### 6.1 映射关系

| 中文原型概念 | 通用 Unit 类型 | 说明 |
|-------------|---------------|------|
| 汉字实体 | Concept Unit | 基本概念单元 |
| 字音-字形映射 | Relation Unit | 感知特征间关系 |
| 字义演变规则 | Rule Unit | 意义转换规则 |
| 构词模式 | Task Pattern Unit | 复合词生成任务 |
| 句法结构 | Relation Unit + Rule Unit | 结构关系 + 组合规则 |

### 6.2 治理机制映射

| 中文原型治理 | 通用治理机制 | 说明 |
|-------------|-------------|------|
| TSLA 阈值 | Quality Assessment | 通用质量评估框架 |
| 长期层晋升 | Promotion Decision | 通用晋升决策流程 |
| 冲突检测 | Conflict Detection | 通用冲突检测机制 |
| 参数晋升 | Parameter Promotion | 通用参数管理 |

---

## 7. 后续工作

### 7.1 短期目标（Phase 7-8）

1. **完善通用 Unit 定义**：细化各类 Unit 的字段和方法
2. **设计适配层接口**：定义语言适配的标准接口
3. **验证抽象有效性**：通过英文适配验证通用性

### 7.2 中期目标（Phase 9-10）

1. **实现多语言支持**：完成英文、其他语言适配
2. **扩展领域应用**：从 NLP 扩展到知识图谱、推理系统等
3. **优化治理机制**：基于多语言/多领域经验优化

### 7.3 长期目标

1. **形成标准规范**：建立通用 Unit 标准
2. **开源生态建设**：推动社区参与适配层开发
3. **跨系统互操作**：实现不同系统间的 Unit 交换

---

**文档版本**: v1.0  
**最后更新**: 2026-04-17  
**状态**: 通用化抽象初步完成，待后续验证和扩展
