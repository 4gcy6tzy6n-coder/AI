# TSLA 动作

## 概述

TSLA (Trustworthiness, Safety, Liability, Accountability) 是系统的可信度评估框架，决定如何处理每个 Unit 的结果。

## 动作类型

### 1. 接受 (ACCEPT)

**条件**:
- 可信度分数 ≥ 0.8
- 风险等级 = LOW
- 无冲突证据

**行为**:
- 直接输出结果
- 正常记忆存储
- 标准审计记录

**元数据**:
```python
accept_record = {
    "action": "ACCEPT",
    "confidence": float,
    "output_immediate": True,
    "storage_zone": "long_term"
}
```

### 2. 拒绝 (REJECT)

**条件**:
- 可信度分数 < 0.3
- 风险等级 = CRITICAL
- 检测到有害内容
- 严重事实错误

**行为**:
- 不输出结果
- 记录拒绝原因
- 触发警报 (如需要)
- 可选：返回错误信息

**元数据**:
```python
reject_record = {
    "action": "REJECT",
    "reason": str,
    "alert_level": "none"|"info"|"warning"|"critical",
    "user_message": Optional[str]
}
```

### 3. 审查 (REVIEW)

**条件**:
- 可信度分数 0.5-0.8
- 风险等级 = MEDIUM
- 存在不确定性
- 需要额外验证

**行为**:
- 标记待审查
- 启动审查流程
- 临时存储结果
- 通知审查者

**元数据**:
```python
review_record = {
    "action": "REVIEW",
    "review_type": "auto"|"human"|"hybrid",
    "priority": int,
    "deadline": datetime,
    "assigned_to": Optional[str]
}
```

### 4. 升级 (ESCALATE)

**条件**:
- 超出当前系统能力
- 需要高级推理
- 涉及敏感领域
- 检测到潜在风险

**行为**:
- 转发到高级引擎
- 增加计算资源
- 扩展检索范围
- 专家系统介入

**元数据**:
```python
escalate_record = {
    "action": "ESCALATE",
    "escalation_level": int,
    "target_engine": str,
    "additional_resources": list[str],
    "timeout_extension": int
}
```

### 5. 分解 (DECOMPOSE)

**条件**:
- 问题过于复杂
- 需要多步解决
- 可拆分为子任务

**行为**:
- 拆分为子 Unit
- 创建依赖图
- 顺序或并行执行
- 结果合并

**元数据**:
```python
decompose_record = {
    "action": "DECOMPOSE",
    "sub_units": list[Unit],
    "dependency_graph": Graph,
    "execution_mode": "sequential"|"parallel"|"hybrid"
}
```

### 6. 增强 (AUGMENT)

**条件**:
- 结果可改进
- 需要额外信息
- 置信度可提升

**行为**:
- 触发额外检索
- 扩展推理步骤
- 整合更多证据
- 重新评估

**元数据**:
```python
augment_record = {
    "action": "AUGMENT",
    "augmentation_type": "retrieval"|"reasoning"|"evidence",
    "additional_queries": list[str],
    "expected_improvement": float
}
```

## 动作决策矩阵

| 可信度 | 低风险 | 中风险 | 高风险 | 严重风险 |
|--------|--------|--------|--------|----------|
| ≥ 0.8  | ACCEPT | REVIEW | REVIEW | ESCALATE |
| 0.5-0.8| REVIEW | REVIEW | ESCALATE| REJECT  |
| 0.3-0.5| ESCALATE| ESCALATE| REJECT | REJECT  |
| < 0.3  | REJECT | REJECT | REJECT | REJECT  |

## 动作执行流程

```
评估输入
    │
    ▼
计算可信度分数
    │
    ▼
评估风险等级
    │
    ▼
查决策矩阵
    │
    ▼
选择动作
    │
    ├──► ACCEPT ──► 输出 + 存储
    │
    ├──► REJECT ──► 记录 + 警报
    │
    ├──► REVIEW ──► 审查队列
    │
    ├──► ESCALATE ──► 高级处理
    │
    ├──► DECOMPOSE ──► 子任务
    │
    └──► AUGMENT ──► 增强处理
```
