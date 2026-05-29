# 记忆门控机制

## 概述

记忆门控机制是确保记忆质量和系统可靠性的关键组件，控制信息在记忆层级间的流动。

## 门控类型

### 1. 写入门 (Write Gate)

**职责**: 控制是否可以写入记忆

**检查项**:
- 内容完整性检查
- 格式验证
- 重复检测
- 敏感信息过滤

**决策**:
```python
write_decision = {
    "allowed": bool,
    "target_zone": Zone,
    "conditions": list[str],
    "expires_at": datetime
}
```

### 2. 稳定性门 (Stability Gate)

**职责**: 验证记忆内容的稳定性

**检查项**:
- 一致性检查 (多次查询结果是否一致)
- 置信度阈值
- 时间衰减评估
- 冲突检测

**决策**:
```python
stability_decision = {
    "stable": bool,
    "stability_score": float,  # 0-1
    "promotion_allowed": bool,
    "review_required": bool
}
```

### 3. 提升门 (Promotion Gate)

**职责**: 控制记忆向更高层级的提升

**检查项**:
- 访问频率统计
- 重要性评分
- 关联强度
- 长期价值评估

**决策**:
```python
promotion_decision = {
    "promote": bool,
    "source_zone": Zone,
    "target_zone": Zone,
    "priority": int,
    "compression_allowed": bool
}
```

### 4. 回滚门 (Rollback Gate)

**职责**: 处理错误或有害记忆

**检查项**:
- 错误检测
- 有害内容识别
- 用户反馈
- 系统警报

**决策**:
```python
rollback_decision = {
    "rollback": bool,
    "scope": "unit"|"batch"|"cascade",
    "replacement": Optional[Memory],
    "audit_level": "normal"|"high"
}
```

### 5. 永久保护门 (Permanent Protection Gate)

**职责**: 保护深层永久记忆不被错误修改

**检查项**:
- 修改权限验证
- 影响范围评估
- 备份确认
- 多重签名 (关键修改)

**决策**:
```python
protection_decision = {
    "protected": bool,
    "required_approvals": int,
    "backup_created": bool,
    "modification_allowed": bool
}
```

## 门控流程

```
新记忆
  │
  ▼
┌─────────────┐
│  写入门     │ ──拒绝──► 丢弃
└──────┬──────┘
       │ 通过
       ▼
┌─────────────┐
│ 瞬态存储    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 稳定性门    │ ──不稳定──► 保持/标记
└──────┬──────┘
       │ 稳定
       ▼
┌─────────────┐
│ 提升门      │ ──不提升──► 长期存储
└──────┬──────┘
       │ 提升
       ▼
┌─────────────┐
│ 永久保护门  │ ──拒绝──► 浅层永久
└──────┬──────┘
       │ 通过
       ▼
┌─────────────┐
│ 深层永久    │
└─────────────┘
```

## 门控配置

### 阈值参数
```yaml
gates:
  write:
    min_content_length: 10
    max_content_length: 100000
    duplicate_threshold: 0.95
  
  stability:
    min_confidence: 0.7
    consistency_window: "7d"
    min_observations: 3
  
  promotion:
    min_access_count: 5
    min_importance: 0.6
    min_association_strength: 0.5
  
  rollback:
    auto_rollback_on_error: true
    user_feedback_threshold: 0.3
  
  permanent_protection:
    require_backup: true
    multi_sig_threshold: 0.8
```
