"""
Multi-head Loss Runner v1 - 多头损失运行器 v1

Phase 17 Stage 2: 本地真实训练基础设施
目标：支持 policy/governance/memory/response 多头联合训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MultiHeadLoss:
    """多头损失"""
    policy_loss: torch.Tensor
    governance_loss: torch.Tensor
    memory_loss: torch.Tensor
    response_loss: torch.Tensor
    total_loss: torch.Tensor
    
    # 各头损失值（用于日志）
    policy_loss_value: float
    governance_loss_value: float
    memory_loss_value: float
    response_loss_value: float


class MultiHeadLossRunner:
    """多头损失运行器"""
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        governance_weight: float = 1.0,
        memory_weight: float = 0.5,
        response_weight: float = 1.0,
    ):
        self.weights = {
            'policy': policy_weight,
            'governance': governance_weight,
            'memory': memory_weight,
            'response': response_weight,
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> MultiHeadLoss:
        """
        计算多头损失
        
        Args:
            predictions: 模型预测输出
                - policy_logits: [batch, num_strategies]
                - governance_logits: [batch, num_actions]
                - memory_logits: [batch, 2] (writeback: yes/no)
                - response_logits: [batch, seq_len, vocab_size]
            
            targets: 目标标签
                - policy_target: [batch]
                - governance_target: [batch, num_actions]
                - memory_target: [batch]
                - response_target: [batch, seq_len]
        
        Returns:
            MultiHeadLoss: 多头损失对象
        """
        # 1. Policy Loss (策略选择 - 多分类)
        policy_loss = self._compute_policy_loss(
            predictions.get('policy_logits'),
            targets.get('policy_target')
        )
        
        # 2. Governance Loss (治理动作 - 多标签分类)
        governance_loss = self._compute_governance_loss(
            predictions.get('governance_logits'),
            targets.get('governance_target')
        )
        
        # 3. Memory Loss (记忆写回 - 二分类)
        memory_loss = self._compute_memory_loss(
            predictions.get('memory_logits'),
            targets.get('memory_target')
        )
        
        # 4. Response Loss (响应生成 - 语言模型)
        response_loss = self._compute_response_loss(
            predictions.get('response_logits'),
            targets.get('response_target')
        )
        
        # 5. 加权总损失
        total_loss = (
            self.weights['policy'] * policy_loss +
            self.weights['governance'] * governance_loss +
            self.weights['memory'] * memory_loss +
            self.weights['response'] * response_loss
        )
        
        return MultiHeadLoss(
            policy_loss=policy_loss,
            governance_loss=governance_loss,
            memory_loss=memory_loss,
            response_loss=response_loss,
            total_loss=total_loss,
            policy_loss_value=policy_loss.item(),
            governance_loss_value=governance_loss.item(),
            memory_loss_value=memory_loss.item(),
            response_loss_value=response_loss.item(),
        )
    
    def _compute_policy_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算策略损失"""
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device)
        
        # CrossEntropy for multi-class classification
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def _compute_governance_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算治理损失"""
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device)
        
        # BCEWithLogitsLoss for multi-label classification
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return loss
    
    def _compute_memory_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算记忆损失"""
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device)
        
        # CrossEntropy for binary classification (writeback: yes/no)
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def _compute_response_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算响应损失"""
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device)
        
        # CrossEntropy for language modeling
        batch_size, seq_len, vocab_size = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=-100,  # padding token
        )
        return loss
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取损失权重"""
        return self.weights.copy()
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """设置损失权重"""
        self.weights.update(weights)


# 便捷函数
def create_multihead_loss_runner(
    policy_weight: float = 1.0,
    governance_weight: float = 1.0,
    memory_weight: float = 0.5,
    response_weight: float = 1.0,
) -> MultiHeadLossRunner:
    """创建多头损失运行器"""
    return MultiHeadLossRunner(
        policy_weight=policy_weight,
        governance_weight=governance_weight,
        memory_weight=memory_weight,
        response_weight=response_weight,
    )


# 测试
if __name__ == "__main__":
    print("="*70)
    print("Multi-head Loss Runner v1 - 测试")
    print("="*70)
    
    runner = create_multihead_loss_runner()
    
    # 模拟预测
    batch_size = 4
    predictions = {
        'policy_logits': torch.randn(batch_size, 5),  # 5种策略
        'governance_logits': torch.randn(batch_size, 8),  # 8种治理动作
        'memory_logits': torch.randn(batch_size, 2),  # 写回: yes/no
        'response_logits': torch.randn(batch_size, 20, 50000),  # seq_len=20, vocab=50000
    }
    
    # 模拟目标
    targets = {
        'policy_target': torch.randint(0, 5, (batch_size,)),
        'governance_target': torch.randint(0, 2, (batch_size, 8)).float(),
        'memory_target': torch.randint(0, 2, (batch_size,)),
        'response_target': torch.randint(0, 50000, (batch_size, 20)),
    }
    
    # 计算损失
    loss = runner.compute_loss(predictions, targets)
    
    print(f"\n损失计算结果:")
    print(f"  Policy Loss: {loss.policy_loss_value:.4f}")
    print(f"  Governance Loss: {loss.governance_loss_value:.4f}")
    print(f"  Memory Loss: {loss.memory_loss_value:.4f}")
    print(f"  Response Loss: {loss.response_loss_value:.4f}")
    print(f"  Total Loss: {loss.total_loss.item():.4f}")
    
    print("\n✓ 多头损失运行器工作正常")
