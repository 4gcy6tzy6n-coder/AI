"""
Transformer Baseline Tiny

同参数量 Transformer 基线
用于与 Native Backbone Tiny V1 对比

架构：
Input → Transformer Encoder → 
  [GapDetector, PolicyHead, GovernanceHead, WritebackHead] → 
  ResponseDecoder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class TransformerTinyConfig:
    """Transformer Tiny 配置"""
    vocab_size: int = 10000
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_strategies: int = 5
    num_gap_types: int = 3
    num_governance_actions: int = 8
    num_writeback_types: int = 3
    dropout: float = 0.2
    max_seq_len: int = 150
    device: str = 'cpu'


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerUnitEncoder(nn.Module):
    """
    基于 Transformer 的 Unit Encoder
    
    标准 Transformer Encoder
    """
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            hidden_states: [batch, seq_len, hidden_dim]
            pooled: [batch, hidden_dim]
        """
        # Embedding + Positional Encoding
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoding
        hidden_states = self.transformer_encoder(x)
        hidden_states = self.norm(hidden_states)
        
        # 平均池化
        pooled = hidden_states.mean(dim=1)
        
        return hidden_states, pooled


class TransformerTinyGapDetector(nn.Module):
    """Transformer Tiny 版 Gap Detector"""
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_gap_types),
        )
    
    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


class TransformerTinyPolicyHead(nn.Module):
    """Transformer Tiny 版 Policy Head"""
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_gap_types, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_strategies),
        )
    
    def forward(self, pooled: torch.Tensor, gap_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pooled, gap_probs], dim=-1)
        return self.classifier(combined)


class TransformerTinyGovernanceHead(nn.Module):
    """Transformer Tiny 版 Governance Head"""
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_gap_types, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_governance_actions),
        )
    
    def forward(self, pooled: torch.Tensor, gap_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pooled, gap_probs], dim=-1)
        return self.classifier(combined)


class TransformerTinyWritebackHead(nn.Module):
    """Transformer Tiny 版 Writeback Head"""
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_gap_types, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_writeback_types),
        )
    
    def forward(self, pooled: torch.Tensor, gap_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pooled, gap_probs], dim=-1)
        return self.classifier(combined)


class TransformerTinyResponseDecoder(nn.Module):
    """Transformer Tiny 版 Response Decoder"""
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_strategies + config.num_gap_types, 128),
            nn.ReLU(),
            nn.Linear(128, config.vocab_size),
        )
    
    def forward(self, pooled: torch.Tensor, policy_probs: torch.Tensor, gap_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pooled, policy_probs, gap_probs], dim=-1)
        return self.decoder(combined)


class TransformerBaselineTiny(nn.Module):
    """
    Transformer Baseline Tiny
    
    同参数量 Transformer 基线
    """
    
    def __init__(self, config: TransformerTinyConfig):
        super().__init__()
        self.config = config
        
        # 1. Unit Encoder (Transformer-based)
        self.unit_encoder = TransformerUnitEncoder(config)
        
        # 2. Gap Detector
        self.gap_detector = TransformerTinyGapDetector(config)
        
        # 3. Policy Head
        self.policy_head = TransformerTinyPolicyHead(config)
        
        # 4. Governance Head
        self.governance_head = TransformerTinyGovernanceHead(config)
        
        # 5. Writeback Head
        self.writeback_head = TransformerTinyWritebackHead(config)
        
        # 6. Response Decoder
        self.response_decoder = TransformerTinyResponseDecoder(config)
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            包含各头输出的字典
        """
        batch_size = input_ids.size(0)
        
        # 1. Encode
        hidden_states, pooled = self.unit_encoder(input_ids)
        
        # 2. Gap Detection
        gap_logits = self.gap_detector(pooled)
        gap_probs = F.softmax(gap_logits, dim=-1)
        gap_type = gap_logits.argmax(dim=-1)
        
        # 3. Policy (conditioned on gap)
        policy_logits = self.policy_head(pooled, gap_probs)
        policy_probs = F.softmax(policy_logits, dim=-1)
        strategy = policy_logits.argmax(dim=-1)
        
        # 4. Governance (conditioned on gap)
        governance_logits = self.governance_head(pooled, gap_probs)
        governance_probs = torch.sigmoid(governance_logits)
        
        # 5. Writeback (conditioned on gap)
        writeback_logits = self.writeback_head(pooled, gap_probs)
        writeback_probs = F.softmax(writeback_logits, dim=-1)
        writeback_decision = writeback_logits.argmax(dim=-1)
        
        # 6. Response (conditioned on policy and gap)
        response_logits = self.response_decoder(pooled, policy_probs, gap_probs)
        
        return {
            'gap_logits': gap_logits,
            'gap_probs': gap_probs,
            'gap_type': gap_type,
            'policy_logits': policy_logits,
            'policy_probs': policy_probs,
            'strategy': strategy,
            'governance_logits': governance_logits,
            'governance_probs': governance_probs,
            'writeback_logits': writeback_logits,
            'writeback_probs': writeback_probs,
            'writeback_decision': writeback_decision,
            'response_logits': response_logits,
            'hidden_states': hidden_states,
            'pooled': pooled,
        }
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())


def test_transformer_tiny():
    """测试 Transformer Baseline Tiny"""
    print("=" * 70)
    print("Transformer Baseline Tiny 测试")
    print("=" * 70)
    
    config = TransformerTinyConfig()
    model = TransformerBaselineTiny(config)
    
    print(f"\n模型参数量: {model.count_parameters():,}")
    print(f"隐藏维度: {config.hidden_dim}")
    print(f"Transformer 层数: {config.num_layers}")
    print(f"注意力头数: {config.num_heads}")
    
    # 测试前向
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n输入: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print("\n输出结构:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n✓ Transformer Baseline Tiny 测试通过")
    
    return model


if __name__ == "__main__":
    model = test_transformer_tiny()
