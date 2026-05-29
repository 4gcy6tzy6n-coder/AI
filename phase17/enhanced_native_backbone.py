"""
Enhanced Native Backbone v1 - 增强版原生主干 v1

Phase 17 Stage 2.3: 记忆决策链接入
目标：GapDetector → Governance → Writeback 合理闭环

关键改进：
1. GapDetector: 识别缺口类型（无缺口/可检索补足/高风险缺口）
2. WritebackHead: 只判断到长期候选（不写回/瞬时保留/长期候选）
3. 避免训练目标失衡，保持策略准确率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EnhancedBackboneConfig:
    """增强版主干配置"""
    # 模型维度
    hidden_dim: int = 768
    vocab_size: int = 50000
    
    # 策略头
    num_strategies: int = 5  # DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW
    
    # 治理头
    num_governance_actions: int = 8  # TSLA八动作
    
    # GapDetector: 3类缺口判断
    # 0: 无缺口, 1: 可检索补足缺口, 2: 高风险缺口
    num_gap_types: int = 3
    
    # WritebackHead: 3类写回决策（不碰永久层）
    # 0: 不写回, 1: 瞬时保留, 2: 长期候选
    num_writeback_types: int = 3
    
    # Dropout
    dropout: float = 0.3
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GapDetector(nn.Module):
    """
    缺口检测器
    
    判断输入是否存在知识缺口，以及缺口类型：
    - 0: NO_GAP (无缺口) - 直接回答
    - 1: RETRIEVABLE_GAP (可检索补足) - 需要检索
    - 2: HIGH_RISK_GAP (高风险缺口) - 需要治理/拒绝
    """
    
    def __init__(self, hidden_dim: int, num_gap_types: int = 3, dropout: float = 0.3):
        super().__init__()
        self.gap_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_gap_types),
        )
        
        # 缺口严重程度估计（辅助信号）
        self.gap_severity = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_state: [batch, hidden_dim]
        
        Returns:
            gap_logits: [batch, num_gap_types] - 缺口类型logits
            gap_probs: [batch, num_gap_types] - 缺口类型概率
            gap_type: [batch] - 预测的缺口类型
            severity: [batch] - 缺口严重程度 [0, 1]
        """
        gap_logits = self.gap_classifier(hidden_state)
        gap_probs = F.softmax(gap_logits, dim=-1)
        gap_type = gap_logits.argmax(dim=-1)
        severity = self.gap_severity(hidden_state).squeeze(-1)
        
        return {
            'gap_logits': gap_logits,
            'gap_probs': gap_probs,
            'gap_type': gap_type,
            'gap_severity': severity,
        }


class WritebackHead(nn.Module):
    """
    记忆写回头（增强版）
    
    判断内容是否值得写回记忆，以及写回层级：
    - 0: NO_WRITEBACK (不写回) - 临时内容，不保留
    - 1: EPHEMERAL (瞬时保留) - 短期保留，快速淘汰
    - 2: LONG_TERM_CANDIDATE (长期候选) - 进入长期层候选队列
    
    注意：永久层晋升保持冻结规则，不由训练直接控制
    """
    
    def __init__(self, hidden_dim: int, num_writeback_types: int = 3, num_gap_types: int = 3, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gap_types = num_gap_types
        combined_dim = hidden_dim + num_gap_types  # 拼接gap概率
        
        # 主分类器
        self.writeback_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_writeback_types),
        )
        
        # 写回置信度
        self.confidence_estimator = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # 质量评分（用于判断是否值得长期保留）
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        hidden_state: torch.Tensor,
        gap_info: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_state: [batch, hidden_dim]
            gap_info: GapDetector的输出（可选）
        
        Returns:
            writeback_logits: [batch, num_writeback_types]
            writeback_probs: [batch, num_writeback_types]
            writeback_type: [batch] - 预测的写回类型
            confidence: [batch] - 写回置信度
            quality_score: [batch] - 内容质量评分
        """
        # 拼接gap信息（如果有）
        if gap_info is not None:
            gap_features = gap_info['gap_probs']  # [batch, num_gap_types]
            combined = torch.cat([hidden_state, gap_features], dim=-1)
        else:
            # 如果没有gap信息，补零
            batch_size = hidden_state.size(0)
            gap_placeholder = torch.zeros(batch_size, 3, device=hidden_state.device)
            combined = torch.cat([hidden_state, gap_placeholder], dim=-1)
        
        writeback_logits = self.writeback_classifier(combined)
        writeback_probs = F.softmax(writeback_logits, dim=-1)
        writeback_type = writeback_logits.argmax(dim=-1)
        confidence = self.confidence_estimator(combined).squeeze(-1)
        quality_score = self.quality_scorer(hidden_state).squeeze(-1)
        
        return {
            'writeback_logits': writeback_logits,
            'writeback_probs': writeback_probs,
            'writeback_type': writeback_type,
            'writeback_confidence': confidence,
            'quality_score': quality_score,
        }


class EnhancedNativeBackbone(nn.Module):
    """
    增强版原生主干
    
    架构: UnitEncoder → [PolicyHead, GapDetector, GovernanceHead, WritebackHead] → ResponseDecoder
    
    关键设计：
    1. GapDetector优先：先判断缺口，再决定策略
    2. WritebackHead受限：只到长期候选，永久层保持冻结规则
    3. 多任务联合训练：平衡各头学习，避免单一目标主导
    """
    
    def __init__(self, config: EnhancedBackboneConfig):
        super().__init__()
        self.config = config
        
        # 1. Unit Encoder
        self.unit_encoder = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        # 2. Policy Head（策略选择）
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_gap_types, 256),  # 拼接gap信息
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_strategies),
        )
        
        # 3. Gap Detector（缺口识别）- 新增
        self.gap_detector = GapDetector(
            hidden_dim=config.hidden_dim,
            num_gap_types=config.num_gap_types,
            dropout=config.dropout,
        )
        
        # 4. Governance Head（治理动作）
        self.governance_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_gap_types, 256),  # 拼接gap信息
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_governance_actions),
        )
        
        # 5. Writeback Head（记忆写回决策）- 增强版
        self.writeback_head = WritebackHead(
            hidden_dim=config.hidden_dim,
            num_writeback_types=config.num_writeback_types,
            num_gap_types=config.num_gap_types,
            dropout=config.dropout,
        )
        
        # 6. Response Decoder
        self.response_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_strategies + config.num_gap_types, 512),
            nn.ReLU(),
            nn.Linear(512, config.vocab_size),
        )
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            包含所有头输出的字典
        """
        # 1. Unit Encoding
        unit_state = self.unit_encoder(input_ids)  # [batch, seq, hidden]
        unit_pooled = unit_state.mean(dim=1)  # [batch, hidden]
        
        # 2. Gap Detection（优先）
        gap_outputs = self.gap_detector(unit_pooled)
        gap_probs = gap_outputs['gap_probs']  # [batch, num_gap_types]
        
        # 拼接gap信息用于其他头
        state_with_gap = torch.cat([unit_pooled, gap_probs], dim=-1)
        
        # 3. Policy Prediction（受gap信息影响）
        policy_logits = self.policy_head(state_with_gap)
        
        # 4. Governance Prediction（受gap信息影响）
        governance_logits = self.governance_head(state_with_gap)
        
        # 5. Writeback Decision（依赖gap信息）
        writeback_outputs = self.writeback_head(unit_pooled, gap_outputs)
        
        # 6. Response Prediction
        strategy_onehot = F.softmax(policy_logits, dim=-1)
        response_input = torch.cat([unit_pooled, strategy_onehot, gap_probs], dim=-1)
        response_logits = self.response_decoder(response_input)
        
        return {
            # 策略头
            'policy_logits': policy_logits,
            
            # Gap检测头
            'gap_logits': gap_outputs['gap_logits'],
            'gap_probs': gap_outputs['gap_probs'],
            'gap_type': gap_outputs['gap_type'],
            'gap_severity': gap_outputs['gap_severity'],
            
            # 治理头
            'governance_logits': governance_logits,
            
            # 写回头
            'writeback_logits': writeback_outputs['writeback_logits'],
            'writeback_probs': writeback_outputs['writeback_probs'],
            'writeback_type': writeback_outputs['writeback_type'],
            'writeback_confidence': writeback_outputs['writeback_confidence'],
            'quality_score': writeback_outputs['quality_score'],
            
            # 响应头
            'response_logits': response_logits.unsqueeze(1),  # [batch, 1, vocab]
        }


# 便捷函数
def create_enhanced_backbone(
    hidden_dim: int = 768,
    vocab_size: int = 50000,
    device: str = None,
) -> EnhancedNativeBackbone:
    """创建增强版原生主干"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = EnhancedBackboneConfig(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        device=device,
    )
    
    model = EnhancedNativeBackbone(config)
    return model.to(device)


# 测试
if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Native Backbone v1 - 测试")
    print("=" * 70)
    
    # 创建模型
    config = EnhancedBackboneConfig(
        hidden_dim=256,
        vocab_size=10000,
        device='cpu',
    )
    model = EnhancedNativeBackbone(config)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试输入
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # 前向传播
    outputs = model(input_ids)
    
    print("\n输出结构:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 解释输出
    print("\n输出解释:")
    print(f"  Gap类型预测: {outputs['gap_type'].tolist()}")
    print(f"    0=无缺口, 1=可检索补足, 2=高风险缺口")
    print(f"  Gap严重程度: {outputs['gap_severity'].tolist()}")
    print(f"  策略预测: {outputs['policy_logits'].argmax(dim=-1).tolist()}")
    print(f"  写回类型预测: {outputs['writeback_type'].tolist()}")
    print(f"    0=不写回, 1=瞬时保留, 2=长期候选")
    print(f"  写回置信度: {outputs['writeback_confidence'].tolist()}")
    print(f"  质量评分: {outputs['quality_score'].tolist()}")
    
    print("\n✓ 增强版原生主干工作正常")
