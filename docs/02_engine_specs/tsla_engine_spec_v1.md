# TSLA 引擎规范 v1

## 概述

TSLA (Trustworthiness, Safety, Liability, Accountability) 引擎负责评估系统输出的可信度，并决定后续动作。

## 职责

1. 可信度评估
2. 风险分析
3. 动作决策
4. 审计追踪
5. 反馈学习

## 架构

```
┌─────────────────────────────────────┐
│          TSLA Engine                │
├─────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐          │
│  │ Scorer  │  │Hard Veto│          │
│  └────┬────┘  └────┬────┘          │
│       │            │               │
│       └─────┬──────┘               │
│             ▼                      │
│  ┌─────────────────────────┐       │
│  │    Action Router        │       │
│  └───────────┬─────────────┘       │
│              │                     │
│              ▼                     │
│  ┌─────────────────────────┐       │
│  │    Review Pipeline      │       │
│  └───────────┬─────────────┘       │
│              │                     │
│              ▼                     │
│  ┌─────────────────────────┐       │
│  │    Audit Trace          │       │
│  └─────────────────────────┘       │
└─────────────────────────────────────┘
```

## 组件规范

### 1. Scorer (评分器)

**输入**: Unit + Context
**输出**: TSLAScores (TSLA 分数)

**维度**:
- **Trustworthiness (可信度)**: 输出准确性、一致性
- **Safety (安全性)**: 有害内容检测、偏见检测
- **Liability (责任)**: 法律合规、伦理合规
- **Accountability (可追责)**: 可审计性、可追溯性

**接口**:
```python
class Scorer:
    def score(self, unit: Unit, context: Context) -> TSLAScores:
        """计算 TSLA 分数"""
        pass
    
    def score_trustworthiness(self, unit: Unit) -> float:
        """计算可信度分数"""
        pass
    
    def score_safety(self, unit: Unit) -> float:
        """计算安全性分数"""
        pass
```

### 2. Hard Veto (硬否决)

**输入**: Unit + TSLAScores
**输出**: VetoDecision (否决决策)

**否决条件**:
- 安全分数 < 0.1
- 检测到严重有害内容
- 法律合规失败
- 系统完整性威胁

**接口**:
```python
class HardVeto:
    def check(self, unit: Unit, scores: TSLAScores) -> VetoDecision:
        """检查是否否决"""
        pass
    
    def get_veto_reason(self, unit: Unit) -> str:
        """获取否决原因"""
        pass
```

### 3. Action Router (动作路由器)

**输入**: TSLAScores + VetoDecision
**输出**: TSLAAction (TSLA 动作)

**决策逻辑**:
根据分数矩阵选择动作 (ACCEPT, REJECT, REVIEW, ESCALATE, DECOMPOSE, AUGMENT)

**接口**:
```python
class ActionRouter:
    def route(
        self,
        scores: TSLAScores,
        veto: VetoDecision
    ) -> TSLAAction:
        """路由到相应动作"""
        pass
    
    def get_action_matrix(self) -> ActionMatrix:
        """获取动作决策矩阵"""
        pass
```

### 4. Review Pipeline (审查流水线)

**输入**: Unit (待审查)
**输出**: ReviewResult (审查结果)

**审查类型**:
- 自动审查: 规则引擎、模式匹配
- 人工审查: 专家审核、众包审核
- 混合审查: 自动 + 人工

**接口**:
```python
class ReviewPipeline:
    def submit(self, unit: Unit, review_type: ReviewType) -> ReviewTicket:
        """提交审查"""
        pass
    
    def get_result(self, ticket_id: str) -> ReviewResult:
        """获取审查结果"""
        pass
    
    def process_auto_review(self, unit: Unit) -> ReviewResult:
        """处理自动审查"""
        pass
```

### 5. Audit Trace (审计追踪)

**输入**: TSLAAction + Context
**输出**: AuditRecord (审计记录)

**记录内容**:
- 决策时间戳
- 输入内容
- 评分详情
- 决策原因
- 执行动作

**接口**:
```python
class AuditTrace:
    def record(self, action: TSLAAction, context: Context) -> AuditRecord:
        """记录审计日志"""
        pass
    
    def query(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: dict
    ) -> list[AuditRecord]:
        """查询审计记录"""
        pass
    
    def export(self, format: str = "json") -> bytes:
        """导出审计日志"""
        pass
```

## 数据模型

### TSLAScores
```python
class TSLAScores:
    trustworthiness: float  # 0-1
    safety: float          # 0-1
    liability: float       # 0-1
    accountability: float  # 0-1
    overall: float         # 加权平均
    
    details: dict          # 各维度详细分数
```

### TSLAAction
```python
class TSLAAction:
    action_type: ActionType  # ACCEPT, REJECT, etc.
    confidence: float
    reason: str
    metadata: dict
    requires_follow_up: bool
```

### AuditRecord
```python
class AuditRecord:
    record_id: str
    timestamp: datetime
    unit_id: str
    scores: TSLAScores
    action: TSLAAction
    context_hash: str
    signature: str  # 防篡改签名
```

## 配置参数

```yaml
tsla_engine:
  scorer:
    weights:
      trustworthiness: 0.35
      safety: 0.35
      liability: 0.15
      accountability: 0.15
  
  hard_veto:
    enabled: true
    safety_threshold: 0.1
    block_on_failure: true
  
  action_router:
    default_action: "REVIEW"
    auto_accept_threshold: 0.85
    auto_reject_threshold: 0.2
  
  review_pipeline:
    auto_review_enabled: true
    human_review_queue: "default"
    review_timeout_hours: 24
  
  audit:
    enabled: true
    storage: "deep_permanent"
    retention_days: 365
    encryption: true
```

## 性能要求

- 评分延迟: < 50ms
- 决策延迟: < 10ms
- 审计记录延迟: < 5ms
- 吞吐量: > 2000 evals/s
