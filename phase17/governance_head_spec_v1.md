# Governance Head Spec v1 - 治理头规格 v1

## 目标

定义治理头的结构和行为，实现 TSLA 八动作的智能执行和五门迁移的决策。

## 核心设计

### 1. 治理头架构

```python
class GovernanceHead(nn.Module):
    """治理头 - 执行治理决策"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_actions: int = 8,  # TSLA 八动作
    ):
        # 上下文编码器
        self.context_encoder = ContextEncoder(hidden_dim)
        
        # 动作预测器
        self.action_predictor = ActionPredictor(hidden_dim, num_actions)
        
        # 强度评估器
        self.strength_estimator = StrengthEstimator(hidden_dim)
        
        # 五门决策器
        self.gate_decider = GateDecider(hidden_dim)
    
    def forward(
        self,
        unit_state: Tensor[batch, num_units, hidden_dim],
        memory_context: MemoryContext,
        tsla_state: TSLAState,
    ) -> GovernanceDecision:
        """
        前向传播
        
        Returns:
            GovernanceDecision 包含:
            - 治理动作
            - 动作强度
            - 五门决策
            - 置信度
        """
        # 1. 编码上下文
        context = self.context_encoder(
            unit_state, 
            memory_context,
            tsla_state,
        )
        
        # 2. 预测动作
        action_probs = self.action_predictor(context)
        
        # 3. 评估强度
        action_strength = self.strength_estimator(context)
        
        # 4. 五门决策
        gate_decision = self.gate_decider(context)
        
        return GovernanceDecision(
            actions=action_probs,
            strength=action_strength,
            gate=gate_decision,
            confidence=self._compute_confidence(action_probs),
        )
```

### 2. TSLA 八动作（冻结规则框架）

```python
class TSLAActionSet:
    """TSLA 八动作 - 冻结规则"""
    
    ACTIONS = {
        # Think 类
        'CLARIFY': {
            'code': 0,
            'description': '澄清 - 请求用户明确',
            'trigger': 'query_ambiguous',
        },
        'RETRIEVE': {
            'code': 1,
            'description': '检索 - 从记忆获取信息',
            'trigger': 'knowledge_gap_detected',
        },
        
        # Speak 类
        'ANSWER': {
            'code': 2,
            'description': '回答 - 直接给出答案',
            'trigger': 'confident_and_clear',
        },
        'CONSERVE': {
            'code': 3,
            'description': '保守 - 谨慎回答',
            'trigger': 'uncertainty_detected',
        },
        
        # Listen 类
        'CONFIRM': {
            'code': 4,
            'description': '确认 - 确认理解正确',
            'trigger': 'need_confirmation',
        },
        'PROBE': {
            'code': 5,
            'description': '探查 - 深入询问',
            'trigger': 'need_more_info',
        },
        
        # Act 类
        'DECLINE': {
            'code': 6,
            'description': '拒绝 - 无法回答',
            'trigger': 'cannot_answer_safely',
        },
        'REVIEW': {
            'code': 7,
            'description': '审查 - 需要人工审查',
            'trigger': 'high_risk_detected',
        },
    }
```

### 3. 动作预测器（可学习）

```python
class ActionPredictor(nn.Module):
    """动作预测器 - 预测应该执行哪些治理动作"""
    
    def __init__(self, hidden_dim: int, num_actions: int):
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),  # 融合多个上下文
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, num_actions),
            nn.Sigmoid(),  # 多标签分类
        )
    
    def forward(
        self,
        context: Tensor[hidden_dim],
    ) -> Tensor[num_actions]:
        """
        预测动作概率
        
        Returns:
            每个动作的概率 [0, 1]
        """
        encoded = self.encoder(context)
        probs = self.classifier(encoded)
        return probs
```

### 4. 五门迁移决策器

```python
class GateDecider(nn.Module):
    """五门决策器 - 决定知识/能力的迁移路径"""
    
    def __init__(self, hidden_dim: int):
        self.gate_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # 5个门
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        context: Tensor[hidden_dim],
    ) -> GateDecision:
        """
        五门决策
        
        Returns:
            GateDecision 包含:
            - 选择的门
            - 各门概率
            - 置信度
        """
        probs = self.gate_classifier(context)
        
        gate_names = ['PROMOTE', 'ISOLATE', 'RETURN', 'DEFER', 'REJECT']
        
        return GateDecision(
            selected_gate=gate_names[probs.argmax().item()],
            gate_probs={
                gate_names[i]: probs[i].item() 
                for i in range(5)
            },
            confidence=probs.max().item(),
        )
```

### 5. 五门迁移规则（冻结规则）

```python
class FiveGatePolicy:
    """五门迁移策略 - 冻结规则"""
    
    GATES = {
        'PROMOTE': {
            'description': '晋升 - 进入更高层级',
            'conditions': [
                'high_quality',
                'verified',
                'governance_approved',
            ],
            'next_step': 'enter_higher_tier',
        },
        'ISOLATE': {
            'description': '隔离 - 暂时隔离观察',
            'conditions': [
                'potential_conflict',
                'needs_verification',
            ],
            'next_step': 'enter_quarantine',
        },
        'RETURN': {
            'description': '回流 - 返回知识库链',
            'conditions': [
                'not_ready_for_promotion',
                'needs_refinement',
            ],
            'next_step': 'back_to_kb_pipeline',
        },
        'DEFER': {
            'description': '推迟 - 暂不处理',
            'conditions': [
                'insufficient_evidence',
                'low_priority',
            ],
            'next_step': 'wait_for_more_data',
        },
        'REJECT': {
            'description': '拒绝 - 不接受',
            'conditions': [
                'low_quality',
                'harmful',
                'unverifiable',
            ],
            'next_step': 'discard',
        },
    }
```

### 6. 治理决策流程

```
输入上下文
    ↓
编码上下文 (可学习)
    ↓
┌─────────────────────────────────────┐
│         并行分支                     │
├─────────────────────────────────────┤
│  动作预测 (可学习)                   │
│  ├── CLARIFY                        │
│  ├── RETRIEVE                       │
│  ├── ANSWER                         │
│  ├── CONSERVE                       │
│  ├── CONFIRM                        │
│  ├── PROBE                          │
│  ├── DECLINE                        │
│  └── REVIEW                         │
├─────────────────────────────────────┤
│  五门决策 (可学习)                   │
│  ├── PROMOTE                        │
│  ├── ISOLATE                        │
│  ├── RETURN                         │
│  ├── DEFER                          │
│  └── REJECT                         │
└─────────────────────────────────────┘
    ↓
应用冻结规则 (TSLA框架/五门策略)
    ↓
输出治理决策
```

### 7. 训练目标

```python
def compute_governance_loss(
    predictions: GovernanceDecision,
    targets: GovernanceTargets,
) -> Dict[str, Tensor]:
    """计算治理损失"""
    
    # 1. 动作预测损失 (多标签分类)
    action_loss = F.binary_cross_entropy(
        predictions.actions,
        targets.actions,
    )
    
    # 2. 五门决策损失 (单标签分类)
    gate_loss = F.cross_entropy(
        predictions.gate_probs,
        targets.gate,
    )
    
    # 3. 强度估计损失 (回归)
    strength_loss = F.mse_loss(
        predictions.strength,
        targets.strength,
    )
    
    return {
        'action_loss': action_loss,
        'gate_loss': gate_loss,
        'strength_loss': strength_loss,
        'total_loss': action_loss + gate_loss + strength_loss,
    }
```

### 8. 与主干的集成

```python
class NativeBackbone(nn.Module):
    """原生主干"""
    
    def __init__(self):
        self.unit_encoder = UnitEncoder()
        self.memory_bus = MemoryBus()
        self.governance_head = GovernanceHead()  # 治理头
        self.policy_head = PolicyHead()
        self.response_decoder = ResponseDecoder()
    
    def forward(self, input_units):
        # 1. 编码
        unit_state = self.unit_encoder(input_units)
        
        # 2. 检索
        memory_context = self.memory_bus.retrieve(unit_state)
        
        # 3. 治理决策
        tsla_state = self._compute_tsla_state(unit_state, memory_context)
        governance_decision = self.governance_head(
            unit_state,
            memory_context,
            tsla_state,
        )
        
        # 4. 策略选择
        policy = self.policy_head(unit_state, memory_context, governance_decision)
        
        # 5. 生成响应
        response = self.response_decoder(
            unit_state,
            memory_context,
            governance_decision,
            policy,
        )
        
        return response
```

---

## 交付物

- [x] governance_head_spec_v1.md (本文档)
- [ ] governance_head_impl_v1.py
