# Memory Bus Spec v1 - 记忆总线规格 v1

## 目标

定义三层记忆的统一访问接口和总线协议，实现记忆的高效存取和治理。

## 核心设计

### 1. 三层记忆架构

```
┌─────────────────────────────────────────┐
│         Deep Permanent Layer            │
│    (深层永久 - 核心知识/身份/原则)        │
│         晋升门槛: 极高                   │
├─────────────────────────────────────────┤
│         Long-term Layer                 │
│    (长期层 - 项目状态/技术栈/历史)        │
│         晋升门槛: 高                     │
├─────────────────────────────────────────┤
│         Ephemeral Layer                 │
│    (瞬时层 - 当前对话/临时上下文)         │
│         自动过期                         │
└─────────────────────────────────────────┘
```

### 2. 记忆单元结构

```python
@dataclass
class MemoryUnit:
    """记忆单元"""
    memory_id: str                    # 唯一ID
    content: str                      # 文本内容
    embedding: Tensor[hidden_dim]     # 向量表示
    layer: MemoryLayer                # 所属层级
    timestamp: float                  # 创建时间
    access_count: int                 # 访问次数
    importance_score: float           # 重要性评分
    source: str                       # 来源 (user/ai/retrieval/self_construct)
    confidence: float                 # 置信度
    relations: List[str]              # 关联记忆ID
```

### 3. 记忆总线接口

```python
class MemoryBus(nn.Module):
    """记忆总线 - 统一记忆访问接口"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        ephemeral_capacity: int = 100,
        longterm_capacity: int = 1000,
        permanent_capacity: int = 100,
    ):
        # 三层记忆存储
        self.ephemeral_store = EphemeralStore(capacity=ephemeral_capacity)
        self.longterm_store = LongTermStore(capacity=longterm_capacity)
        self.permanent_store = PermanentStore(capacity=permanent_capacity)
        
        # 检索编码器
        self.query_encoder = QueryEncoder(hidden_dim)
        self.memory_encoder = MemoryEncoder(hidden_dim)
        
        # 层级路由器
        self.layer_router = LayerRouter(hidden_dim)
    
    def retrieve(
        self,
        query: Tensor[hidden_dim],
        query_context: Dict[str, Any],
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        检索记忆
        
        流程:
        1. 编码查询
        2. 路由到相关层级
        3. 在各层检索
        4. 合并排序
        """
        # 1. 编码查询
        query_encoded = self.query_encoder(query)
        
        # 2. 路由决策
        layer_weights = self.layer_router(query_context)
        
        # 3. 分层检索
        results = []
        
        if layer_weights['permanent'] > 0.3:
            permanent_results = self.permanent_store.retrieve(
                query_encoded, 
                k=top_k
            )
            results.extend(permanent_results)
        
        if layer_weights['longterm'] > 0.3:
            longterm_results = self.longterm_store.retrieve(
                query_encoded, 
                k=top_k
            )
            results.extend(longterm_results)
        
        if layer_weights['ephemeral'] > 0.3:
            ephemeral_results = self.ephemeral_store.retrieve(
                query_encoded, 
                k=top_k
            )
            results.extend(ephemeral_results)
        
        # 4. 合并排序
        all_results = self._merge_and_rerank(results, query_encoded)
        
        return RetrievalResult(
            memories=all_results[:top_k],
            layer_distribution=layer_weights,
        )
    
    def write(
        self,
        content: str,
        writeback_decision: WritebackDecision,
        source: str,
    ) -> MemoryUnit:
        """
        写入记忆
        
        流程:
        1. 编码内容
        2. 根据决策确定层级
        3. 写入对应层级
        4. 更新索引
        """
        # 1. 编码
        embedding = self.memory_encoder(content)
        
        # 2. 确定层级
        target_layer = writeback_decision.target_layer
        
        # 3. 创建记忆单元
        memory_unit = MemoryUnit(
            memory_id=generate_id(),
            content=content,
            embedding=embedding,
            layer=target_layer,
            timestamp=time.time(),
            access_count=0,
            importance_score=writeback_decision.importance_score,
            source=source,
            confidence=writeback_decision.confidence,
            relations=[],
        )
        
        # 4. 写入对应层级
        if target_layer == MemoryLayer.EPHEMERAL:
            self.ephemeral_store.add(memory_unit)
        elif target_layer == MemoryLayer.LONG_TERM:
            self.longterm_store.add(memory_unit)
        elif target_layer == MemoryLayer.DEEP_PERMANENT:
            # 永久层需要额外验证
            if self._validate_permanent_entry(memory_unit):
                self.permanent_store.add(memory_unit)
        
        return memory_unit
    
    def promote(
        self,
        memory_id: str,
        promotion_decision: PromotionDecision,
    ) -> bool:
        """
        晋升记忆到更高层级
        
        流程:
        1. 验证晋升决策
        2. 检查目标层级容量
        3. 执行晋升
        4. 更新索引
        """
        if not promotion_decision.approved:
            return False
        
        # 查找记忆
        memory = self._find_memory(memory_id)
        if memory is None:
            return False
        
        # 检查门槛
        target_layer = promotion_decision.target_layer
        if not self._check_promotion_threshold(memory, target_layer):
            return False
        
        # 执行晋升
        old_layer = memory.layer
        memory.layer = target_layer
        
        # 从旧层级移除
        self._remove_from_layer(memory_id, old_layer)
        
        # 添加到新层级
        if target_layer == MemoryLayer.LONG_TERM:
            self.longterm_store.add(memory)
        elif target_layer == MemoryLayer.DEEP_PERMANENT:
            self.permanent_store.add(memory)
        
        return True
```

### 4. 层级路由器

```python
class LayerRouter(nn.Module):
    """层级路由器 - 决定查询应该访问哪些记忆层"""
    
    def __init__(self, hidden_dim: int):
        self.query_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # 3个层级
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        query_context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        路由决策
        
        Returns:
            {
                'ephemeral': 0.2,
                'longterm': 0.5,
                'permanent': 0.3,
            }
        """
        query_encoded = query_context['query_encoded']
        
        # 分类
        probs = self.query_classifier(query_encoded)
        
        return {
            'ephemeral': probs[0].item(),
            'longterm': probs[1].item(),
            'permanent': probs[2].item(),
        }
```

### 5. 晋升门槛策略（冻结规则）

```python
class PromotionThreshold:
    """晋升门槛 - 冻结规则"""
    
    # 到长期层的门槛
    TO_LONGTERM = {
        'min_importance': 0.7,
        'min_confidence': 0.8,
        'min_access_count': 3,
        'min_age_hours': 24,
    }
    
    # 到永久层的门槛
    TO_PERMANENT = {
        'min_importance': 0.9,
        'min_confidence': 0.95,
        'min_access_count': 10,
        'min_age_days': 7,
        'requires_verification': True,
        'requires_governance_approval': True,
    }
    
    @classmethod
    def check(cls, memory: MemoryUnit, target_layer: MemoryLayer) -> bool:
        """检查是否满足晋升门槛"""
        if target_layer == MemoryLayer.LONG_TERM:
            return (
                memory.importance_score >= cls.TO_LONGTERM['min_importance'] and
                memory.confidence >= cls.TO_LONGTERM['min_confidence'] and
                memory.access_count >= cls.TO_LONGTERM['min_access_count']
            )
        elif target_layer == MemoryLayer.DEEP_PERMANENT:
            return (
                memory.importance_score >= cls.TO_PERMANENT['min_importance'] and
                memory.confidence >= cls.TO_PERMANENT['min_confidence'] and
                memory.access_count >= cls.TO_PERMANENT['min_access_count']
            )
        return False
```

### 6. 记忆编码器

```python
class MemoryEncoder(nn.Module):
    """记忆编码器 - 将文本编码为记忆向量"""
    
    def __init__(self, hidden_dim: int):
        self.text_encoder = UnitEncoder()  # 复用 Unit Encoder
        self.memory_projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, content: str) -> Tensor[hidden_dim]:
        """编码记忆内容"""
        # 1. 文本编码
        unit_encoded = self.text_encoder.encode(content)
        
        # 2. 池化
        pooled = unit_encoded.mean(dim=1)  # [hidden_dim]
        
        # 3. 投影
        memory_embedding = self.memory_projection(pooled)
        
        return memory_embedding
```

### 7. 与主干的连接

```
┌─────────────────────────────────────────┐
│           Native Backbone               │
│                                         │
│  ┌─────────┐    ┌─────────────┐        │
│  │  Input  │───→│ Unit Encoder │        │
│  └─────────┘    └──────┬──────┘        │
│                        │                │
│                        ↓                │
│              ┌─────────────────┐        │
│              │   Memory Bus    │        │
│              │  ┌───────────┐  │        │
│              │  │ Ephemeral │  │        │
│              │  ├───────────┤  │        │
│              │  │ Long-term │  │        │
│              │  ├───────────┤  │        │
│              │  │ Permanent │  │        │
│              │  └───────────┘  │        │
│              └─────────────────┘        │
│                        │                │
│                        ↓                │
│              ┌─────────────────┐        │
│              │  Integration    │        │
│              └─────────────────┘        │
└─────────────────────────────────────────┘
```

---

## 交付物

- [x] memory_bus_spec_v1.md (本文档)
- [ ] memory_bus_impl_v1.py
