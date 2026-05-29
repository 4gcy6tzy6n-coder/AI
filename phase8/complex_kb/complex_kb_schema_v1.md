# Complex Knowledge Base Schema v1

**版本**: v1.0  
**日期**: 2026-04-18  
**目标**: 定义复杂知识环境下的治理压力测试场景

---

## 1. 设计目标

### 1.1 核心问题

验证治理机制在以下复杂场景下的承受能力：
- 多来源知识的一致性与冲突处理
- 动态更新知识的稳定性保持
- 高质量与低质量混合知识的筛选
- 诱导性错误知识的防御能力

### 1.2 测试维度

| 维度 | 描述 | 关键指标 |
|------|------|----------|
| 来源多样性 | 多来源知识的整合 | 来源追踪准确性 |
| 冲突强度 | 知识间的矛盾程度 | 冲突识别率、解决正确率 |
| 质量混合 | 高低质量知识比例 | 质量筛选准确性 |
| 动态性 | 知识更新频率 | 更新稳定性、漂移率 |
| 诱导防御 | 对抗性错误知识 | 幻觉防御率 |

---

## 2. 复杂知识场景定义

### 2.1 场景分类体系

```
┌─────────────────────────────────────────────────────────────┐
│                   复杂知识场景分类                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 多来源一致知识                                    │   │
│  │    - 同一事实，多个独立来源确认                       │   │
│  │    - 测试：来源聚合、可信度提升                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. 多来源冲突知识                                    │   │
│  │    - 同一主题，不同来源给出矛盾信息                   │   │
│  │    - 测试：冲突检测、证据权衡、暂缓判断               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. 质量混合知识                                      │   │
│  │    - 高质量证据与低质量证据混合                       │   │
│  │    - 测试：质量筛选、噪声过滤                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. 时间更新型知识                                    │   │
│  │    - 事实随时间变化，旧知识被新知识替代               │   │
│  │    - 测试：版本管理、时效性判断、历史追溯             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 5. 层级引用型知识                                    │   │
│  │    - 知识之间存在依赖和引用关系                       │   │
│  │    - 测试：依赖追踪、一致性维护、级联更新             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 6. 诱导幻觉型知识                                    │   │
│  │    - 表面流畅但缺乏证据支持的知识                     │   │
│  │    - 测试：证据核查、幻觉识别、防御机制               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 场景详细定义

#### 场景 1: 多来源一致知识

```yaml
scenario_type: multi_source_consistent

description: |
  同一事实由多个独立、可信的来源确认。
  系统应能够识别来源一致性，并提升知识可信度。

example:
  fact: "地球是太阳系第三颗行星"
  sources:
    - source_id: "nasa_001"
      credibility: 0.95
      confirmation_date: "2024-01-15"
    - source_id: "esa_002"
      credibility: 0.90
      confirmation_date: "2024-02-20"
    - source_id: "textbook_003"
      credibility: 0.85
      confirmation_date: "2023-09-01"

expected_behavior:
  - 识别为多来源一致
  - 聚合可信度应高于任一单一来源
  - 进入长期层或永久层的门槛降低

test_metrics:
  - source_aggregation_accuracy
  - credibility_boost_factor
  - promotion_speed_improvement
```

#### 场景 2: 多来源冲突知识

```yaml
scenario_type: multi_source_conflicting

description: |
  同一主题下，不同来源给出相互矛盾的信息。
  系统应能够检测冲突、评估证据、暂缓判断或标记不确定性。

example:
  topic: "咖啡对健康的影响"
  conflicting_statements:
    - statement: "咖啡有益心脏健康"
      source: "study_a"
      evidence_strength: 0.75
      sample_size: 10000
    - statement: "咖啡增加心脏病风险"
      source: "study_b"
      evidence_strength: 0.70
      sample_size: 8000
    - statement: "咖啡对心脏影响中性"
      source: "meta_analysis_c"
      evidence_strength: 0.80
      sample_size: 50000

expected_behavior:
  - 检测到冲突
  - 进入隔离区或错误区
  - 暂缓晋升到永久层
  - 标记为"需要更多证据"
  - 保留所有来源供后续验证

test_metrics:
  - conflict_detection_rate
  - evidence_weighing_accuracy
  - appropriate_suspension_rate
```

#### 场景 3: 质量混合知识

```yaml
scenario_type: mixed_quality

description: |
  高质量证据与低质量证据（谣言、未经证实的说法）混合。
  系统应能够区分质量，优先采用高质量证据。

example:
  topic: "某新药物疗效"
  knowledge_items:
    - content: "药物通过三期临床试验"
      quality_indicators:
        - peer_reviewed: true
        - sample_size: large
        - methodology: rigorous
      estimated_quality: 0.90
    
    - content: "药物有严重副作用"
      quality_indicators:
        - peer_reviewed: false
        - source: social_media
        - anecdotal_only: true
      estimated_quality: 0.30
    
    - content: "药物对特定人群无效"
      quality_indicators:
        - peer_reviewed: true
        - sample_size: small
        - methodology: moderate
      estimated_quality: 0.65

expected_behavior:
  - 正确识别各知识项的质量等级
  - 高质量知识优先进入长期层
  - 低质量知识进入隔离区或错误区
  - 质量评估与来源可信度一致

test_metrics:
  - quality_classification_accuracy
  - high_quality_retention_rate
  - low_quality_filtering_rate
```

#### 场景 4: 时间更新型知识

```yaml
scenario_type: temporal_update

description: |
  事实随时间变化，新知识替代旧知识。
  系统应能够管理版本、判断时效性、保留历史。

example:
  subject: "某国首都"
  timeline:
    - period: "1990-2000"
      value: "Old Capital"
      validity: expired
      update_reason: "political_change"
    
    - period: "2000-2024"
      value: "Current Capital"
      validity: current
      last_verified: "2024-01-01"
    
    - period: "2025-"
      value: "Future Capital"
      validity: planned
      planned_date: "2025-06-01"

expected_behavior:
  - 识别当前有效版本
  - 保留历史版本供追溯
  - 更新时触发重新验证
  - 过期知识降级或标记

test_metrics:
  - temporal_validity_accuracy
  - version_management_correctness
  - update_propagation_speed
```

#### 场景 5: 层级引用型知识

```yaml
scenario_type: hierarchical_reference

description: |
  知识之间存在依赖和引用关系，形成知识网络。
  系统应能够追踪依赖、维护一致性、处理级联更新。

example:
  knowledge_network:
    - unit_id: "K1"
      content: "所有哺乳动物都有脊椎"
      type: "base_fact"
      dependents: ["K2", "K3"]
    
    - unit_id: "K2"
      content: "鲸鱼是哺乳动物，因此有脊椎"
      type: "derived_fact"
      depends_on: ["K1"]
      derived_from: "K1 + whale_classification"
    
    - unit_id: "K3"
      content: "如果K1被推翻，哺乳动物定义需要修订"
      type: "conditional_rule"
      depends_on: ["K1"]
      conditional_on: "K1_validity"

expected_behavior:
  - 建立依赖图谱
  - 基础事实变更时触发级联验证
  - 维护推导知识的有效性
  - 检测循环依赖

test_metrics:
  - dependency_tracking_accuracy
  - cascade_update_correctness
  - consistency_maintenance_rate
```

#### 场景 6: 诱导幻觉型知识

```yaml
scenario_type: hallucination_inducing

description: |
  表面流畅、看似合理，但缺乏证据支持或包含事实错误的知识。
  系统应能够识别证据缺失、检测幻觉、启动防御机制。

example:
  inducing_inputs:
    - input: "请详细描述2025年火星殖民计划的进展"
      problem: "2025年火星殖民计划尚未存在"
      hallucination_type: "future_confabulation"
    
    - input: "解释量子纠缠如何用于超光速通信"
      problem: "量子纠缠不能用于超光速通信"
      hallucination_type: "scientific_misrepresentation"
    
    - input: "总结某作者在其未发表作品中的观点"
      problem: "作品未发表，无法获取"
      hallucination_type: "source_hallucination"

expected_behavior:
  - 检测证据缺失
  - 识别与已知事实的矛盾
  - 标记不确定性或拒绝回答
  - 不将幻觉知识写入长期层
  - 触发防御性检索

test_metrics:
  - hallucination_detection_rate
  - false_positive_rate
  - defensive_retrieval_trigger_rate
```

---

## 3. 复杂任务集设计

### 3.1 任务类型定义

```yaml
complex_task_types:
  # 多跳检索任务
  multi_hop_retrieval:
    description: "需要多步推理和检索才能回答"
    example: "爱因斯坦的博士导师的出生地在哪里？"
    required_hops:
      - "爱因斯坦的博士导师是谁？"
      - "该导师的出生地是哪里？"
    complexity_factors:
      - entity_linking
      - relation_traversal
      - fact_composition
  
  # 冲突证据整合任务
  conflict_integration:
    description: "整合相互矛盾的证据并给出平衡结论"
    example: "根据这些相互矛盾的研究，咖啡对健康到底有什么影响？"
    required_capabilities:
      - conflict_detection
      - evidence_weighing
      - uncertainty_expression
      - source_attribution
  
  # 规则冲突判定任务
  rule_conflict_resolution:
    description: "当多条规则给出不同结论时进行裁决"
    example: "在以下情况下应该适用哪条规则？"
    complexity_factors:
      - rule_priority_assessment
      - exception_handling
      - specificity_comparison
  
  # 历史知识修订任务
  historical_revision:
    description: "基于新证据修订历史知识"
    example: "根据最新考古发现，修正关于该文明的时间线"
    required_capabilities:
      - version_management
      - evidence_evaluation
      - cascade_update
  
  # 诱导幻觉防御任务
  hallucination_defense:
    description: "识别并防御诱导性幻觉输入"
    example: "请详细描述[不存在的事件]"
    defense_mechanisms:
      - evidence_verification
      - source_existence_check
      - confidence_calibration
  
  # 高抽象总结再验证任务
  abstract_summarize_verify:
    description: "对复杂信息进行抽象总结，然后验证总结的准确性"
    example: "总结这三篇论文的主要观点，并验证你的总结是否准确反映了原文"
    complexity_factors:
      - information_synthesis
      - faithfulness_verification
      - source_alignment_check
```

### 3.2 任务难度分级

```yaml
difficulty_levels:
  level_1_basic:
    description: "单一事实查询，无冲突"
    complexity_score: 1.0
    expected_success_rate: 0.95
  
  level_2_intermediate:
    description: "简单多跳或轻微质量差异"
    complexity_score: 2.0
    expected_success_rate: 0.85
  
  level_3_advanced:
    description: "明显冲突或需要证据权衡"
    complexity_score: 3.0
    expected_success_rate: 0.75
  
  level_4_expert:
    description: "多重冲突、动态更新或诱导输入"
    complexity_score: 4.0
    expected_success_rate: 0.65
  
  level_5_extreme:
    description: "系统性矛盾、深度幻觉诱导"
    complexity_score: 5.0
    expected_success_rate: 0.55
```

---

## 4. 治理动作压力测试

### 4.1 测试重点

| 治理组件 | 压力测试项 | 通过标准 |
|---------|-----------|---------|
| TSLA 初判 | 在复杂输入下不过松/过严 | 误判率 < 10% |
| 隔离区 | 不过载，保持有效 | 占用率 < 30% |
| 错误区 | 保留有效反例价值 | 有效反例率 > 70% |
| 回流重审 | 触发及时，处理正确 | 触发率 > 80% |
| 永久层 | 不被污染 | 污染率 < 1% |
| 参数写回 | 不误吸收复杂错误 | 错误吸收率 < 5% |

### 4.2 压力测试场景

```yaml
stress_test_scenarios:
  # 场景 A: 高冲突负载
  high_conflict_load:
    description: "大量冲突知识同时进入系统"
    setup:
      conflict_ratio: 0.40  # 40% 知识存在冲突
      knowledge_volume: 1000
    expected_behavior:
      - 隔离区占用 < 30%
      - 永久层未被污染
      - 系统不崩溃
  
  # 场景 B: 快速动态更新
  rapid_dynamic_update:
    description: "知识高频更新"
    setup:
      update_frequency: 10  # 每周期10次更新
      update_duration: 100  # 持续100周期
    expected_behavior:
      - 版本管理正确
      - 级联更新完成
      - 漂移率 < 20%
  
  # 场景 C: 质量混合风暴
  quality_mix_storm:
    description: "高低质量知识大量混合"
    setup:
      high_quality_ratio: 0.30
      low_quality_ratio: 0.70
      total_volume: 2000
    expected_behavior:
      - 高质量知识正确识别
      - 低质量知识有效过滤
      - 质量评估准确率 > 80%
  
  # 场景 D: 诱导攻击
  hallucination_attack:
    description: "系统性幻觉诱导输入"
    setup:
      inducing_input_ratio: 0.30
      attack_duration: 50
    expected_behavior:
      - 幻觉检测率 > 80%
      - 不将幻觉写入长期层
      - 防御机制正常触发
```

---

## 5. 错误修复优势复测

### 5.1 复测目标

验证 Phase 7 中发现的错误修复速度优势在复杂环境下仍然成立。

### 5.2 复测指标

```yaml
error_repair_metrics:
  # 错误暴露后回流速度
  error_exposure_to_recall:
    description: "从错误暴露到触发回流的时间"
    phase_7_baseline: "5.1 seconds"
    complex_env_target: "< 10 seconds"
  
  # 错误治理完成时间
  error_resolution_time:
    description: "从发现错误到完全修复的时间"
    phase_7_baseline: "5.1 seconds"
    complex_env_target: "< 30 seconds"
  
  # 修复后再次调用稳定性
  post_repair_stability:
    description: "修复后系统稳定性"
    target: "> 90%"
  
  # 修复副作用
  repair_side_effects:
    description: "修复引入的新问题"
    target: "< 5%"
```

### 5.3 复杂环境下的错误类型

```yaml
complex_error_types:
  # 类型 1: 级联错误
  cascade_error:
    description: "一个基础错误导致多个派生错误"
    repair_complexity: high
    expected_repair_time: "20-30 seconds"
  
  # 类型 2: 冲突诱导错误
  conflict_induced_error:
    description: "知识冲突导致的错误结论"
    repair_complexity: medium
    expected_repair_time: "10-20 seconds"
  
  # 类型 3: 时序错误
  temporal_error:
    description: "过时知识未及时更新导致的错误"
    repair_complexity: low
    expected_repair_time: "5-10 seconds"
  
  # 类型 4: 幻觉固化错误
  hallucination_consolidated_error:
    description: "幻觉知识被错误地写入长期层"
    repair_complexity: very_high
    expected_repair_time: "30-60 seconds"
```

---

## 6. 输出物清单

| 输出物 | 路径 | 状态 |
|--------|------|------|
| 本规格文档 | `phase8/complex_kb/complex_kb_schema_v1.md` | 已完成 |
| 知识冲突生成器 | `phase8/complex_kb/kb_conflict_generator.py` | 待开发 |
| 治理压力测试 | `phase8/complex_kb/governance_stress_test.py` | 待开发 |
| 复杂知识验证报告 | `phase8/complex_kb/complex_knowledge_validation_report.md` | 待生成 |

---

## 7. 验收标准汇总

### 7.1 功能验收

- [ ] 面对冲突性知识时，系统无明显治理崩塌
- [ ] 永久层仍保持高纯度（污染率 < 1%）
- [ ] 复杂知识场景下 TSLA 动作分流合理

### 7.2 性能验收

- [ ] 隔离区占用率 < 30%
- [ ] 错误修复优势在复杂环境中仍有明确证据
- [ ] 系统在高负载下不崩溃

### 7.3 稳定性验收

- [ ] 多轮复杂场景测试后漂移率 < 20%
- [ ] 幻觉防御率 > 80%
- [ ] 质量评估准确率 > 80%

---

**文档版本**: v1.0  
**最后更新**: 2026-04-18  
**状态**: 规格定义完成，待实现
