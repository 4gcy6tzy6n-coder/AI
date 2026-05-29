# Unit Encoder Spec v1 - Unit 编码器规格 v1

## 目标

定义 Unit 级别的输入编码器，将中英文文本统一编码到共享的语义空间。

## 核心设计

### 1. Unit 定义

#### 中文 Unit
- **粒度**: 1-4 个汉字
- **边界**: 基于语义完整性
- **示例**: 
  - "我叫" (2字)
  - "什么名字" (4字)
  - "项目" (2字)

#### 英文 Unit
- **粒度**: 1-3 个 word
- **边界**: 基于语义完整性
- **示例**:
  - "my name" (2 words)
  - "what is" (2 words)
  - "the project goal" (3 words)

### 2. 编码流程

```
输入文本
    ↓
分词/分字
    ↓
Unit 分割
    ↓
子单元编码 (字/word)
    ↓
Unit 聚合
    ↓
统一表示 [batch, num_units, hidden_dim]
```

### 3. 编码器结构

```python
class UnitEncoder(nn.Module):
    """Unit 编码器"""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        char_embed_dim: int = 256,
        unit_hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        # 字符/词级别嵌入
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)
        
        # 子单元编码器 (局部上下文)
        self.subunit_encoder = TransformerEncoder(
            d_model=char_embed_dim,
            nhead=4,
            num_layers=2,
        )
        
        # Unit 聚合器
        self.unit_aggregator = UnitAggregator(
            input_dim=char_embed_dim,
            output_dim=unit_hidden_dim,
        )
        
        # Unit 级别编码器 (全局上下文)
        self.unit_encoder = TransformerEncoder(
            d_model=unit_hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
        )
    
    def forward(
        self,
        input_ids: Tensor[batch, seq_len],
        unit_boundaries: Tensor[batch, num_units, 2],  # [start, end]
    ) -> Tensor[batch, num_units, hidden_dim]:
        """
        前向传播
        
        Args:
            input_ids: 字符/词 ID
            unit_boundaries: Unit 边界 [batch, num_units, (start, end)]
        
        Returns:
            unit_embeddings: Unit 级别表示
        """
        # 1. 字符级别嵌入
        char_embeds = self.char_embedding(input_ids)  # [batch, seq, char_dim]
        
        # 2. 子单元编码
        subunit_encoded = self.subunit_encoder(char_embeds)
        
        # 3. Unit 聚合
        unit_embeds = self.unit_aggregator(
            subunit_encoded, 
            unit_boundaries
        )  # [batch, num_units, hidden_dim]
        
        # 4. Unit 级别编码
        unit_encoded = self.unit_encoder(unit_embeds)
        
        return unit_encoded
```

### 4. Unit 聚合策略

```python
class UnitAggregator(nn.Module):
    """Unit 聚合器"""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.attention_pool = AttentionPooling(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(
        self,
        subunit_encoded: Tensor[batch, seq_len, input_dim],
        unit_boundaries: Tensor[batch, num_units, 2],
    ) -> Tensor[batch, num_units, output_dim]:
        """聚合子单元到 Unit 表示"""
        
        batch_size, num_units, _ = unit_boundaries.shape
        unit_reprs = []
        
        for b in range(batch_size):
            batch_units = []
            for u in range(num_units):
                start, end = unit_boundaries[b, u]
                unit_subunits = subunit_encoded[b, start:end]  # [unit_len, input_dim]
                
                # 注意力池化
                unit_repr = self.attention_pool(unit_subunits)  # [input_dim]
                batch_units.append(unit_repr)
            
            unit_reprs.append(torch.stack(batch_units))
        
        unit_embeds = torch.stack(unit_reprs)  # [batch, num_units, input_dim]
        
        # 投影到输出维度
        return self.projection(unit_embeds)
```

### 5. 与 Transformer 的区别

| 特性 | Transformer | Unit Encoder |
|------|-------------|--------------|
| 基本单位 | Token/Subword | Unit (语义单元) |
| 注意力范围 | 全局 | 分层 (子单元→Unit→全局) |
| 位置编码 | 绝对/相对 | Unit 级别相对位置 |
| 语义粒度 | 细粒度 | 语义单元粒度 |
| 跨语言 | 需多语言训练 | 统一 Unit 空间 |

### 6. 训练目标

```python
# 1. 重构损失 (自监督)
reconstruction_loss = mse_loss(
    decoded_units, 
    target_units
)

# 2. 语义相似度损失
similarity_loss = contrastive_loss(
    similar_units,
    dissimilar_units
)

# 3. 下游任务损失
downstream_loss = task_loss(
    unit_encoded,
    labels
)
```

### 7. 输出规格

- **形状**: [batch_size, max_units, hidden_dim=768]
- **范围**: 归一化到 [-1, 1]
- **掩码**: padding_mask [batch_size, max_units]

---

## 交付物

- [x] unit_encoder_spec_v1.md (本文档)
- [ ] unit_encoder_impl_v1.py
