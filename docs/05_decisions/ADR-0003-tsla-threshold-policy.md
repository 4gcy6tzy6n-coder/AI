# ADR-0003: TSLA 阈值策略

## 状态
- 状态: 已接受
- 日期: 2026-04-17
- 作者: Post Transformer AI Team

## 背景

TSLA (Trustworthiness, Safety, Liability, Accountability) 系统需要根据评估分数决定如何处理每个输出。我们需要一套清晰的阈值策略来指导这些决策。

## 决策

我们决定采用 **双阈值 + 动作矩阵** 策略:

### 阈值定义

```python
AUTO_ACCEPT_THRESHOLD = 0.85
AUTO_REJECT_THRESHOLD = 0.30
```

### 动作决策矩阵

| 可信度 | 低风险 | 中风险 | 高风险 | 严重风险 |
|--------|--------|--------|--------|----------|
| ≥ 0.85 | ACCEPT | REVIEW | REVIEW | ESCALATE |
| 0.50-0.85| REVIEW | REVIEW | ESCALATE| REJECT  |
| 0.30-0.50| ESCALATE| ESCALATE| REJECT | REJECT  |
| < 0.30 | REJECT | REJECT | REJECT | REJECT  |

### 硬否决规则

以下情况直接拒绝，无视分数:
1. 安全分数 < 0.1
2. 检测到严重有害内容
3. 法律合规失败
4. 系统完整性威胁

## 理由

### 为什么双阈值?

1. **清晰性**: 明确的接受/拒绝边界
2. **灵活性**: 中间区域允许人工介入
3. **安全性**: 低分自动拒绝，高分自动接受

### 为什么动作矩阵?

1. **多维评估**: 同时考虑可信度和风险
2. **细粒度控制**: 不同情况不同处理
3. **可调整**: 矩阵参数可校准

## 替代方案考虑

#### 替代方案 1: 单阈值
- **优点**: 简单
- **缺点**: 缺乏灵活性，无法处理边界情况
- **结论**: 不适用

#### 替代方案 2: 连续评分
- **优点**: 精细
- **缺点**: 难以解释，难以调试
- **结论**: 不适用

#### 替代方案 3: 纯规则引擎
- **优点**: 可解释
- **缺点**: 难以覆盖所有情况
- **结论**: 部分采用，作为硬否决

## 影响

### 正面影响
- 决策透明可解释
- 支持人工介入
- 可校准优化

### 负面影响
- 需要维护决策矩阵
- 阈值需要定期校准
- 可能产生误判

## 实施

### 配置示例
```yaml
tsla:
  thresholds:
    auto_accept: 0.85
    auto_reject: 0.30
  
  action_matrix:
    high_trust_low_risk: ACCEPT
    high_trust_high_risk: ESCALATE
    low_trust_any: REJECT
  
  hard_veto:
    safety_min: 0.1
    block_harmful: true
    block_non_compliant: true
```

### 代码实现
```python
class TSLAActionRouter:
    def route(self, scores: TSLAScores) -> TSLAAction:
        # 硬否决检查
        if self.hard_veto.check(scores):
            return TSLAAction.REJECT
        
        # 查决策矩阵
        action = self.action_matrix.lookup(
            trust=scores.trustworthiness,
            risk=scores.risk_level
        )
        return action
```

### 相关文档
- [TSLA 动作](../01_theory_frozen/tsla_actions.md)
- [TSLA 引擎规范](../02_engine_specs/tsla_engine_spec_v1.md)
- [TSLA Schema](../03_api_contracts/tsla_result_schema.yaml)

## 后续工作

1. 实现校准流程
2. 建立反馈循环
3. 定期审查阈值有效性
