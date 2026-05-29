# English Error Taxonomy - 英文错误分类体系

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 9 - WP2: 英文能力精修

---

## 1. 分类目的

本文档系统化定义英文治理过程中可能出现的错误类型，为：
1. 错误追踪和统计提供标准
2. 优化方向提供指导
3. 回归测试提供用例
4. 双语一致性评估提供依据

---

## 2. 错误分类总览

```
English Error Taxonomy
├── 1. Relation Detection Errors (关系检测错误)
│   ├── 1.1 False Negatives (漏检)
│   ├── 1.2 False Positives (误检)
│   └── 1.3 Misclassification (错分)
├── 2. Rule Parsing Errors (规则解析错误)
│   ├── 2.1 Structure Recognition Failure (结构识别失败)
│   ├── 2.2 Condition-Conclusion Mismatch (条件结论不匹配)
│   └── 2.3 Complex Rule Breakdown (复合规则拆解失败)
├── 3. Task Pattern Errors (任务模式错误)
│   ├── 3.1 Trigger Misfire (触发错误)
│   ├── 3.2 Pattern Overgeneralization (过度泛化)
│   └── 3.3 Context Ignorance (上下文忽略)
├── 4. Explanation Layer Errors (解释层错误)
│   ├── 4.1 Word-Level Inconsistency (词级不一致)
│   ├── 4.2 Phrase-Level Conflict (短语级冲突)
│   └── 4.3 Sentence-Level Abstraction Error (句级抽象错误)
└── 5. Cross-Lingual Alignment Errors (跨语言对齐错误)
    ├── 5.1 Semantic Divergence (语义发散)
    ├── 5.2 Governance Action Mismatch (治理动作不匹配)
    └── 5.3 Quality Score Drift (质量分数漂移)
```

---

## 3. 详细错误定义

### 3.1 关系检测错误 (Relation Detection Errors)

#### 3.1.1 False Negatives (漏检)

**定义**: 文本中存在关系但系统未检测到

**示例**:
```
输入: "The cat sat on the mat."
期望检测: (cat, sat_on, mat)
实际结果: 未检测到关系

输入: "She gave him a book."
期望检测: (She, gave, him), (She, gave, book)
实际结果: 只检测到 (She, gave, book)
```

**子类型**:
- **FN-1**: 介词关系漏检 - 介词结构中的关系未识别
- **FN-2**: 从句关系漏检 - 从句中的关系未识别
- **FN-3**: 隐式关系漏检 - 隐含的关系未识别
- **FN-4**: 复合关系漏检 - 多重重叠关系只识别部分

**根因分析**:
- 模式库覆盖不足
- 句法分析深度不够
- 歧义消解策略保守

#### 3.1.2 False Positives (误检)

**定义**: 文本中不存在关系但系统错误检测到

**示例**:
```
输入: "I think that is correct."
错误检测: (I, think, that)  # "that" 不是实体

输入: "The meeting is on Monday."
错误检测: (meeting, on, Monday)  # "on" 表示时间而非空间关系
```

**子类型**:
- **FP-1**: 虚词误检 - 将虚词当作实体
- **FP-2**: 时间介词误检 - 时间介词被误判为空间关系
- **FP-3**: 从句标记误检 - 从句标记词被误判
- **FP-4**: 修饰语误检 - 修饰语被误判为关系

**根因分析**:
- 实体识别不准确
- 上下文理解不足
- 语义消歧能力弱

#### 3.1.3 Misclassification (错分)

**定义**: 检测到关系但关系类型判断错误

**示例**:
```
输入: "The company acquired the startup."
错误分类: (company, partner_with, startup)  # 应为 acquire
正确分类: (company, acquire, startup)

输入: "He is interested in music."
错误分类: (He, like, music)  # 应为 interested_in
正确分类: (He, interested_in, music)
```

**子类型**:
- **MC-1**: 关系强度错分 - 强弱关系混淆
- **MC-2**: 关系方向错分 - 关系方向判断错误
- **MC-3**: 关系类型混淆 - 不同类型关系混淆

---

### 3.2 规则解析错误 (Rule Parsing Errors)

#### 3.2.1 Structure Recognition Failure (结构识别失败)

**定义**: 无法识别文本中的条件-结论结构

**示例**:
```
输入: "If it rains, the ground will be wet."
错误: 未识别为规则结构
正确: Condition: "it rains", Conclusion: "ground will be wet"

输入: "You must finish homework before playing games."
错误: 识别为普通陈述
正确: Condition: "before playing games", Conclusion: "finish homework"
```

**子类型**:
- **SR-1**: 条件标记词缺失 - 缺少显式条件标记
- **SR-2**: 倒装结构 - 条件结论顺序颠倒
- **SR-3**: 省略结构 - 省略了条件或结论的部分

#### 3.2.2 Condition-Conclusion Mismatch (条件结论不匹配)

**定义**: 识别出结构但条件与结论对应错误

**示例**:
```
输入: "If you study hard, you will pass the exam."
错误: Condition: "you will pass", Conclusion: "study hard"
正确: Condition: "you study hard", Conclusion: "you will pass the exam"
```

#### 3.2.3 Complex Rule Breakdown (复合规则拆解失败)

**定义**: 无法处理多条件、多结论的复合规则

**示例**:
```
输入: "If A and B, then C or D."
错误: 识别为单一规则
正确: 应拆解为多个子规则
```

---

### 3.3 任务模式错误 (Task Pattern Errors)

#### 3.3.1 Trigger Misfire (触发错误)

**定义**: 任务模式被错误触发或未触发

**示例**:
```
输入: "How do I bake a cake?"
错误: 未触发 how_to 模式
正确: 应触发 how_to 模式

输入: "The weather is nice today."
错误: 错误触发 how_to 模式
正确: 不应触发任何任务模式
```

#### 3.3.2 Pattern Overgeneralization (过度泛化)

**定义**: 任务模式匹配过于宽泛

**示例**:
```
输入: "I need help with my homework."
错误: 触发 comparison 模式（因"help"与"compare"相似）
正确: 应触发 help_request 模式
```

#### 3.3.3 Context Ignorance (上下文忽略)

**定义**: 忽略上下文导致任务模式选择错误

**示例**:
```
输入: "Can you explain how photosynthesis works?"
错误: 只触发 explanation 模式
正确: 应同时触发 explanation + how_to 模式
```

---

### 3.4 解释层错误 (Explanation Layer Errors)

#### 3.4.1 Word-Level Inconsistency (词级不一致)

**定义**: 单词级别解释与整体语义不一致

**示例**:
```
输入: "The bank of the river."
错误: bank → 金融机构
正确: bank → 河岸
```

#### 3.4.2 Phrase-Level Conflict (短语级冲突)

**定义**: 短语解释与句子级结论冲突

**示例**:
```
输入: "Break a leg!"
短语解释: break → 破坏, leg → 腿
句子结论: 祝好运
冲突: 字面意思与习语意思冲突
```

#### 3.4.3 Sentence-Level Abstraction Error (句级抽象错误)

**定义**: 句子级抽象过度或不足

**示例**:
```
输入: "I have a dream that one day..."
错误: 抽象为 personal_statement
正确: 应抽象为 vision_statement
```

---

### 3.5 跨语言对齐错误 (Cross-Lingual Alignment Errors)

#### 3.5.1 Semantic Divergence (语义发散)

**定义**: 中英文相同语义但治理结果不同

**示例**:
```
中文: "他喜欢读书。"
英文: "He likes reading."
错误: 中文识别为 like_activity，英文识别为 preference
正确: 两者应识别为相同的 Unit 类型
```

#### 3.5.2 Governance Action Mismatch (治理动作不匹配)

**定义**: 相同输入在中英文场景下触发不同治理动作

**示例**:
```
场景: 低质量输入
中文处理: 降级到隔离区
英文处理: 直接拒绝
错误: 治理动作不一致
正确: 应执行相同的治理策略
```

#### 3.5.3 Quality Score Drift (质量分数漂移)

**定义**: 相同质量的内容在中英文场景下评分差异过大

**示例**:
```
输入: "这是一个测试。" / "This is a test."
中文质量分: 0.85
英文质量分: 0.65
错误: 差异 0.20 > 阈值 0.10
正确: 差异应 < 0.10
```

---

## 4. 错误统计与追踪

### 4.1 错误编码规范

```
[大类]-[子类]-[序号]

示例:
R-FN-001: 关系检测-漏检-介词关系漏检
R-FP-002: 关系检测-误检-时间介词误检
T-TM-001: 任务模式-触发错误-未触发
```

### 4.2 错误严重级别

| 级别 | 定义 | 示例 | 处理优先级 |
|------|------|------|-----------|
| P0 - Critical | 导致治理链路失败 | 核心关系漏检 | 立即修复 |
| P1 - High | 显著影响治理质量 | 关系类型错分 | 本周修复 |
| P2 - Medium | 影响用户体验 | 任务模式误触发 | 下周修复 |
| P3 - Low | 边缘场景问题 | 复杂规则拆解失败 | 排期修复 |

### 4.3 错误追踪模板

```yaml
error_id: "R-FN-001"
type: "False Negative"
subtype: "介词关系漏检"
severity: "P1"
input: "The cat sat on the mat."
expected: "(cat, sat_on, mat)"
actual: "None"
root_cause: "介词模式库覆盖不足"
fix_plan: "增加介词关系模式"
status: "open"
assignee: "developer"
created_at: "2026-04-18"
```

---

## 5. 优化方向映射

### 5.1 错误类型 → 优化策略

| 错误类型 | 优化策略 | 预期效果 |
|----------|----------|----------|
| FN-1 介词关系漏检 | 扩展介词模式库 | +15% 召回率 |
| FN-2 从句关系漏检 | 增强句法分析 | +10% 召回率 |
| FP-1 虚词误检 | 改进实体识别 | +10% 精确率 |
| MC-1 关系强度错分 | 引入语义特征 | +5% F1 |
| SR-1 条件标记词缺失 | 扩展规则模式 | +20% 规则识别 |
| TM-1 触发错误 | 优化触发阈值 | +10% 准确率 |
| SD-1 语义发散 | 双语对齐训练 | +15% 一致性 |

---

## 6. 回归测试用例

### 6.1 关系检测回归集

```python
relation_regression_cases = [
    {
        "id": "R-001",
        "input": "The cat sat on the mat.",
        "expected_relations": [("cat", "sat_on", "mat")],
        "tags": ["FN-1", "介词关系"]
    },
    {
        "id": "R-002",
        "input": "She gave him a book.",
        "expected_relations": [
            ("She", "gave", "him"),
            ("She", "gave", "book")
        ],
        "tags": ["FN-4", "复合关系"]
    },
    # ... 更多用例
]
```

### 6.2 双语对齐回归集

```python
bilingual_regression_cases = [
    {
        "id": "B-001",
        "zh": "他喜欢读书。",
        "en": "He likes reading.",
        "expected_alignment": True,
        "max_score_diff": 0.1
    },
    # ... 更多用例
]
```

---

## 7. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义5大类18小类错误 |

---

**文档状态**: 生效中  
**维护者**: Phase 9 WP2 负责人  
**更新频率**: 每周根据新发现错误更新
