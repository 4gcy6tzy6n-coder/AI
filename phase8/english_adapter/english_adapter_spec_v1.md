# English Adapter Specification v1

**版本**: v1.0  
**日期**: 2026-04-18  
**目标**: 定义英文输入如何接入通用治理主链

---

## 1. 设计原则

### 1.1 核心约束

1. **适配层隔离**: 英文特有逻辑必须限制在 adapter layer，不得污染 core governance layer
2. **通用 Unit 映射**: 所有英文输入必须可映射到 Concept/Relation/Rule/Task Pattern 四类通用 Unit
3. **治理主链复用**: 英文输入进入与中文完全一致的治理主链（解析→缺口识别→检索→整合→TSLA→分流→输出）
4. **双语一致性**: 同一概念的中英文表达应在治理层面达到一致结论

### 1.2 与中文原型的对应关系

| 中文原型 | 英文适配 |
|---------|---------|
| 汉字 (character) | 单词 (word) |
| 字音 + 字形 | 拼写 (spelling) + 音标 (phonetic) |
| 基本义 | 核心语义 (core meaning) |
| 字音-字形映射 | 拼写-发音映射 |
| 构词模式 | 词组/短语构成模式 |
| 句法结构 | 句法依赖关系 |

---

## 2. 英文最小输入对象定义

### 2.1 Word 作为最小感知起点

```yaml
english_word:
  # 标识信息
  word_id: "全局唯一标识"
  surface_form: "表面形式（大小写敏感）"
  canonical_form: "规范形式（小写）"
  
  # 感知核心（语言相关）
  perceptual_core:
    spelling:
      - original: "原始拼写"
      - normalized: "规范化拼写"
      - phonetic_representation: "音标表示（IPA）"
    
    pronunciation:
      - phonemes: ["音素序列"]
      - stress_pattern: "重音模式"
      - syllable_count: "音节数"
  
  # 词形变化（英文特有）
  inflectional_variants:
    - form: "复数/过去式/进行式等"
      type: "变化类型"
      rule_applied: "应用的规则"
  
  # 语义核心
  semantic_core:
    primary_senses:
      - sense_id: "义项标识"
        definition: "定义"
        pos: "词性（noun/verb/adj/adv等）"
        frequency_rank: "频率排名"
    
    boundary_conditions:
      applicable_contexts: ["适用上下文"]
      excluded_contexts: ["排除上下文"]
  
  # 治理元数据
  governance_metadata:
    quality_score:
      Q: 0.0-1.0  # 意义清晰度
      T: 0.0-1.0  # 透明度
      S: 0.0-1.0  # 稳定性
      C: 0.0-1.0  # 一致性
      L: 0.0-1.0  # 合法性
    
    stability_cycles: 0
    source: "来源追踪"
    verification_history: []
```

### 2.2 输入字段结构

```python
@dataclass
class EnglishWordInput:
    """英文单词输入结构"""
    
    # 基础信息
    text: str                          # 原始文本
    context: Optional[str] = None      # 上下文（用于消歧）
    
    # 预处理结果（由适配层填充）
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    
    # 语言检测
    detected_language: str = "en"
    language_confidence: float = 1.0
    
    # 适配层标记
    adapter_version: str = "v1.0"
    preprocessing_timestamp: str = ""
```

---

## 3. 英文解释层架构

### 3.1 三层解释结构

```
┌─────────────────────────────────────────────────────────────┐
│                   英文解释层架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 3: Sentence-Level Composition (预留，本阶段轻量)     │
│  ├── 句法解析                                               │
│  ├── 依存关系提取                                           │
│  └── 与 Task Pattern 的映射                                 │
│                          ↓                                  │
│  Layer 2: Phrase Explanation                                │
│  ├── 短语识别 (NP/VP/PP等)                                  │
│  ├── 短语语义组合                                           │
│  └── 与 Relation Unit 的映射                                │
│                          ↓                                  │
│  Layer 1: Word Explanation                                  │
│  ├── 词形还原                                               │
│  ├── 词义消歧                                               │
│  ├── 词性标注                                               │
│  └── 与 Concept Unit 的映射                                 │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Core Governance Layer                  │   │
│  │         （与中文完全一致的治理主链）                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Word Explanation Layer

#### 功能模块

1. **词形还原 (Lemmatization)**
   ```python
   def lemmatize_word(word: str, pos: str) -> str:
       """
       将词形变化还原为基本形式
       
       示例:
       - running -> run (verb)
       - better -> good (adj)
       - children -> child (noun)
       """
       pass
   ```

2. **词义消歧 (Word Sense Disambiguation)**
   ```python
   def disambiguate_sense(
       word: str,
       context: str,
       candidate_senses: List[Sense]
   ) -> Sense:
       """
       根据上下文选择正确义项
       
       示例:
       - "bank" in "river bank" -> 河岸
       - "bank" in "money bank" -> 银行
       """
       pass
   ```

3. **词性标注 (POS Tagging)**
   ```python
   def tag_pos(words: List[str]) -> List[str]:
       """
       标注词性
       
       使用标准: Penn Treebank POS tags
       """
       pass
   ```

### 3.3 Phrase Explanation Layer

#### 功能模块

1. **短语识别**
   ```python
   def identify_phrases(
       words: List[str],
       pos_tags: List[str]
   ) -> List[Phrase]:
       """
       识别名词短语、动词短语、介词短语等
       
       示例:
       - "the quick brown fox" -> NP
       - "jumped over" -> VP
       - "in the garden" -> PP
       """
       pass
   ```

2. **短语语义组合**
   ```python
   def compose_phrase_meaning(
       phrase: Phrase,
       word_senses: List[Sense]
   ) -> PhraseMeaning:
       """
       组合短语中各词的语义
       
       原则: 遵循组合性原则，但处理习语和固定搭配
       """
       pass
   ```

---

## 4. 英文适配规则

### 4.1 词形变化处理

```yaml
inflection_handling:
  noun:
    plural:
      - rule: "add -s"
        pattern: "[^sxz]$"
        example: "cat -> cats"
      - rule: "add -es"
        pattern: "[sxz]$"
        example: "box -> boxes"
      - rule: "-y to -ies"
        pattern: "[^aeiou]y$"
        example: "city -> cities"
      - rule: "irregular"
        pattern: "irregular"
        examples:
          - "child -> children"
          - "mouse -> mice"
  
  verb:
    third_person:
      - rule: "add -s"
        pattern: "[^sxyz]$"
        example: "walk -> walks"
    past_tense:
      - rule: "add -ed"
        pattern: "regular"
        example: "walk -> walked"
      - rule: "irregular"
        pattern: "irregular"
        examples:
          - "go -> went"
          - "eat -> ate"
    progressive:
      - rule: "add -ing"
        pattern: "regular"
        example: "walk -> walking"
  
  adjective:
    comparative:
      - rule: "add -er"
        pattern: "short_adj"
        example: "tall -> taller"
      - rule: "more + adj"
        pattern: "long_adj"
        example: "beautiful -> more beautiful"
    superlative:
      - rule: "add -est"
        pattern: "short_adj"
        example: "tall -> tallest"
      - rule: "most + adj"
        pattern: "long_adj"
        example: "beautiful -> most beautiful"
```

### 4.2 同形异义处理

```yaml
homonym_handling:
  detection:
    method: "context_based_disambiguation"
    min_context_window: 3  # 前后各3个词
  
  resolution_strategy:
    - priority: "frequency_based"
      description: "优先选择最常见义项"
    - priority: "collocation_based"
      description: "基于搭配模式选择"
    - priority: "domain_based"
      description: "基于领域上下文选择"
  
  examples:
    - word: "bank"
      senses:
        - "financial institution"
        - "river side"
        - "tilt/angle"
      disambiguation_cues:
        - "money" -> financial
        - "river" -> river side
        - "aircraft" -> tilt
```

### 4.3 短语组合规则

```yaml
phrase_composition:
  noun_phrase:
    structure: "(Det) + (Adj)* + N"
    head: "last noun"
    semantic_composition: "modifier_head"
  
  verb_phrase:
    structure: "V + (NP) + (PP)*"
    head: "verb"
    argument_structure: "subcategorization_frame"
  
  prepositional_phrase:
    structure: "P + NP"
    semantic_role: "determined_by_preposition"
    
  compound_words:
    types:
      - "closed_form: blackboard"
      - "hyphenated: mother-in-law"
      - "open_form: ice cream"
```

### 4.4 基本句法依赖触发

```yaml
syntactic_dependencies:
  # 主谓关系
  nsubj:
    pattern: "nominal subject"
    triggers: ["who", "what", "did what"]
  
  # 宾语关系
  dobj:
    pattern: "direct object"
    triggers: ["what", "whom"]
  
  # 修饰关系
  amod:
    pattern: "adjectival modifier"
    triggers: ["what kind", "which"]
  
  # 介词关系
  nmod:
    pattern: "nominal modifier"
    triggers: ["where", "when", "how"]
```

### 4.5 英文规则与 Task Pattern 映射

```yaml
task_pattern_mapping:
  # 定义任务
  definition_task:
    triggers: ["what is", "define", "explain"]
    pattern: "Task Pattern: classification/definition"
    expected_output: "concept explanation"
  
  # 比较任务
  comparison_task:
    triggers: ["difference between", "compare", "vs"]
    pattern: "Task Pattern: comparison"
    expected_output: "relation analysis"
  
  # 推理任务
  reasoning_task:
    triggers: ["why", "how come", "what if"]
    pattern: "Task Pattern: reasoning"
    expected_output: "rule application"
  
  # 验证任务
  verification_task:
    triggers: ["is it true", "verify", "check"]
    pattern: "Task Pattern: verification"
    expected_output: "truth assessment"
```

---

## 5. 通用 Unit 映射验证

### 5.1 Concept Unit 英文落地

```python
@dataclass
class EnglishConceptUnit:
    """英文 Concept Unit 实现"""
    
    unit_identity:
        unique_id: str
        unit_type: "concept"
        language: "en"
    
    perceptual_core:
        # 英文特有
        spelling_variants: List[str]  # 美式/英式拼写
        pronunciation_variants: List[str]  # 不同口音
        inflectional_forms: Dict[str, str]  # 各种词形
    
    semantic_core:
        word_senses: List[WordSense]
        domain_labels: List[str]
        register: List[str]  # formal/informal/slang等
    
    governance_metadata:
        # 与中文完全一致
        quality_score: QTSLC
        stability_cycles: int
        source: SourceInfo
```

### 5.2 Relation Unit 英文落地

```python
@dataclass
class EnglishRelationUnit:
    """英文 Relation Unit 实现"""
    
    unit_identity:
        unique_id: str
        unit_type: "relation"
        language: "en"
    
    perceptual_core:
        # 关系标记词
        relation_markers: List[str]
        # 句法模式
        syntactic_patterns: List[str]
    
    semantic_core:
        relation_type: RelationType
        arity: int
        domain_constraints: List[str]
        range_constraints: List[str]
    
    # 英文特有：介词映射
    preposition_mapping:
        spatial: ["in", "on", "at", "under", "over"]
        temporal: ["before", "after", "during", "since"]
        causal: ["because", "due to", "as a result"]
```

### 5.3 Rule Unit 英文落地

```python
@dataclass
class EnglishRuleUnit:
    """英文 Rule Unit 实现"""
    
    unit_identity:
        unique_id: str
        unit_type: "rule"
        language: "en"
    
    perceptual_core:
        # 规则触发词
        trigger_words: List[str]
        # 条件从句标记
        condition_markers: ["if", "when", "unless", "provided that"]
        # 结论从句标记
        conclusion_markers: ["then", "therefore", "thus"]
    
    semantic_core:
        rule_type: RuleType
        premises: List[Proposition]
        conclusions: List[Proposition]
        confidence: float
```

### 5.4 Task Pattern Unit 英文落地

```python
@dataclass
class EnglishTaskPatternUnit:
    """英文 Task Pattern Unit 实现"""
    
    unit_identity:
        unique_id: str
        unit_type: "task_pattern"
        language: "en"
    
    perceptual_core:
        # 任务触发关键词
        trigger_keywords: List[str]
        # 疑问词模式
        question_patterns: List[str]
        # 指令模式
        instruction_patterns: List[str]
    
    semantic_core:
        task_type: TaskType
        input_schema: InputSchema
        output_schema: OutputSchema
        required_units: List[str]
```

---

## 6. 适配层隔离性检查清单

### 6.1 必须属于 Adapter Layer 的内容

- [ ] 词形变化规则
- [ ] 拼写规范化（美式/英式）
- [ ] 音标表示与处理
- [ ] 词性标注规则
- [ ] 句法解析规则
- [ ] 介词用法映射
- [ ] 冠词处理
- [ ] 时态/语态标记

### 6.2 必须保留在 Core Governance Layer 的内容

- [ ] TSLA 评分机制
- [ ] 五门迁移逻辑
- [ ] 分层存储规则
- [ ] 晋升/降级决策
- [ ] 冲突检测算法
- [ ] 质量评估框架
- [ ] 稳定性追踪
- [ ] 验证历史管理

### 6.3 隔离验证方法

```python
def verify_adapter_isolation():
    """
    验证适配层隔离性
    
    检查项:
    1. Core governance 代码中无英文特有硬编码
    2. Adapter 通过标准接口与 Core 交互
    3. 双语输入产生一致的治理决策
    """
    # 测试用例
    test_cases = [
        {
            "zh": "学习",
            "en": "study",
            "expected_governance_path": "same"
        },
        {
            "zh": "因果关系",
            "en": "causal relation",
            "expected_governance_path": "same"
        }
    ]
    
    for case in test_cases:
        zh_result = process_chinese(case["zh"])
        en_result = process_english(case["en"])
        
        assert zh_result.governance_path == en_result.governance_path, \
            f"Governance path mismatch for {case}"
```

---

## 7. 双语基线测试集

### 7.1 测试集结构

```yaml
bilingual_test_set:
  # 概念对齐测试
  concept_alignment:
    - zh: "学习"
      en: "study"
      expected_unit_type: "concept"
      expected_quality_threshold: 0.7
    
    - zh: "苹果"
      en: "apple"
      expected_unit_type: "concept"
      expected_sense_count: 2  # 水果/公司
  
  # 关系对齐测试
  relation_alignment:
    - zh: "导致"
      en: "cause"
      expected_unit_type: "relation"
      expected_relation_type: "causal"
    
    - zh: "属于"
      en: "belong to"
      expected_unit_type: "relation"
      expected_relation_type: "hierarchical"
  
  # 规则对齐测试
  rule_alignment:
    - zh: "如果下雨，地面会湿"
      en: "If it rains, the ground will be wet"
      expected_unit_type: "rule"
      expected_rule_type: "inference"
  
  # 任务对齐测试
  task_alignment:
    - zh: "什么是机器学习？"
      en: "What is machine learning?"
      expected_unit_type: "task_pattern"
      expected_task_type: "definition"
```

### 7.2 验收标准

1. **功能验收**
   - [ ] 英文输入能够完成完整治理链路
   - [ ] 四类通用 Unit 在英文场景中全部可实例化
   - [ ] 中文逻辑与英文逻辑的差异主要体现在适配层

2. **一致性验收**
   - [ ] 英中双语在同一治理框架下均能通过基线测试
   - [ ] 同一概念的中英文表达在治理层面达到一致结论
   - [ ] Core governance layer 代码中无英文特有硬编码

3. **性能验收**
   - [ ] 英文处理延迟与中文相当（±20%）
   - [ ] 英文适配不显著增加内存占用（<10%）

---

## 8. 输出物清单

| 输出物 | 路径 | 状态 |
|--------|------|------|
| 本规格文档 | `phase8/english_adapter/english_adapter_spec_v1.md` | 已完成 |
| 英文 Unit 映射实现 | `phase8/english_adapter/english_unit_mapping.py` | 待开发 |
| 英文解释层实现 | `phase8/english_adapter/english_explanation_layer.py` | 待开发 |
| 双语治理测试 | `phase8/english_adapter/bilingual_governance_test.py` | 待开发 |
| 适配层验证报告 | `phase8/english_adapter/english_adapter_validation_report.md` | 待生成 |

---

**文档版本**: v1.0  
**最后更新**: 2026-04-18  
**状态**: 规格定义完成，待实现
