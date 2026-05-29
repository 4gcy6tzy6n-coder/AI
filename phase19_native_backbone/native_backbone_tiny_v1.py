"""
Native Backbone Tiny V1

Tiny 版原生主干 - 非 Transformer 架构

核心设计：
1. 使用 RNN/GRU 替代 Transformer Attention
2. 保持框架核心组件：UnitEncoder, GapDetector, PolicyHead, etc.
3. 与同参数量 Transformer 进行对比

架构：
Input → UnitEncoder (GRU-based) → 
  [GapDetector, PolicyHead, GovernanceHead, WritebackHead] → 
  ResponseDecoder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NativeTinyConfig:
    """Native Backbone Tiny 配置"""
    vocab_size: int = 10000
    hidden_dim: int = 128  # 较小的隐藏维度
    num_layers: int = 2    # GRU 层数
    num_strategies: int = 5
    num_gap_types: int = 3
    num_governance_actions: int = 8
    num_writeback_types: int = 3
    dropout: float = 0.2
    max_seq_len: int = 150
    device: str = 'cpu'


class GRUUnitEncoder(nn.Module):
    """
    基于 GRU 的 Unit Encoder
    
    替代标准 Transformer Encoder
    使用双向 GRU 捕捉序列信息
    """
    
    def __init__(self, config: NativeTinyConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # 双向 GRU
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # 投影层：将双向输出投影到 hidden_dim
        self.projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
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
        # Embedding
        x = self.embedding(input_ids)  # [batch, seq, hidden]
        
        # GRU encoding
        gru_out, hidden = self.gru(x)  # [batch, seq, hidden*2], [layers*2, batch, hidden]
        
        # 投影
        hidden_states = self.projection(gru_out)  # [batch, seq, hidden]
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 平均池化
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        
        return hidden_states, pooled


class NativeTinyGapDetector(nn.Module):
    """Native Tiny 版 Gap Detector"""
    
    def __init__(self, config: NativeTinyConfig):
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


class NativeTinyPolicyHead(nn.Module):
    """Native Tiny 版 Policy Head"""
    
    def __init__(self, config: NativeTinyConfig):
        super().__init__()
        self.config = config
        
        # 拼接 gap 信息
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_gap_types, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_strategies),
        )
    
    def forward(self, pooled: torch.Tensor, gap_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([pooled, gap_probs], dim=-1)
        return self.classifier(combined)


class NativeTinyGovernanceHead(nn.Module):
    """Native Tiny 版 Governance Head"""
    
    def __init__(self, config: NativeTinyConfig):
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


class NativeTinyWritebackHead(nn.Module):
    """Native Tiny 版 Writeback Head"""
    
    def __init__(self, config: NativeTinyConfig):
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


class NativeTinyResponseDecoder(nn.Module):
    """Native Tiny 版 Response Decoder"""
    
    def __init__(self, config: NativeTinyConfig):
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


class NativeBackboneTinyV1(nn.Module):
    """
    Native Backbone Tiny V1
    
    完整的 tiny 版原生主干
    使用 GRU 替代 Transformer
    """
    
    def __init__(self, config: NativeTinyConfig):
        super().__init__()
        self.config = config
        
        # 1. Unit Encoder (GRU-based)
        self.unit_encoder = GRUUnitEncoder(config)
        
        # 2. Gap Detector
        self.gap_detector = NativeTinyGapDetector(config)
        
        # 3. Policy Head
        self.policy_head = NativeTinyPolicyHead(config)
        
        # 4. Governance Head
        self.governance_head = NativeTinyGovernanceHead(config)
        
        # 5. Writeback Head
        self.writeback_head = NativeTinyWritebackHead(config)
        
        # 6. Response Decoder
        self.response_decoder = NativeTinyResponseDecoder(config)
    
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


def test_native_tiny():
    """测试 Native Tiny Backbone"""
    print("=" * 70)
    print("Native Backbone Tiny V1 测试")
    print("=" * 70)
    
    config = NativeTinyConfig()
    model = NativeBackboneTinyV1(config)
    
    print(f"\n模型参数量: {model.count_parameters():,}")
    print(f"隐藏维度: {config.hidden_dim}")
    print(f"GRU 层数: {config.num_layers} (双向)")
    
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
    
    print("\n✓ Native Backbone Tiny V1 测试通过")
    
    return model


if __name__ == "__main__":
    model = test_native_tiny()
