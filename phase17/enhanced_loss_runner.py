"""
Enhanced Multi-head Loss Runner v1 - 增强版多头损失运行器 v1

Phase 17 Stage 2.3: 支持 GapDetector 和 WritebackHead 的训练

关键改进：
1. 添加 Gap Loss - 缺口识别损失
2. 添加 Writeback Loss - 记忆写回决策损失
3. 平衡权重设计 - 避免单一目标主导
4. 支持4个新指标的追踪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnhancedMultiHeadLoss:
    """增强版多头损失"""
    # 各头损失
    policy_loss: torch.Tensor
    gap_loss: torch.Tensor
    governance_loss: torch.Tensor
    writeback_loss: torch.Tensor
    response_loss: torch.Tensor
    
    # 总损失
    total_loss: torch.Tensor
    
    # 各头损失值（用于日志）
    policy_loss_value: float
    gap_loss_value: float
    governance_loss_value: float
    writeback_loss_value: float
    response_loss_value: float
    
    # 辅助指标（用于监控训练平衡）
    gap_accuracy: float  # Gap检测准确率
    retrieval_false_positive: float  # 检索误触发率
    writeback_false_positive: float  # 写回误判率
    long_term_precision: float  # 长期候选精度


class EnhancedLossRunner:
    """
    增强版多头损失运行器
    
    权重设计原则：
    - policy: 1.0 (基础策略不能丢)
    - gap: 0.8 (缺口识别是框架核心)
    - governance: 1.0 (治理动作)
    - writeback: 0.5 (写回决策，较低权重避免过度写回)
    - response: 1.0 (响应质量)
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        gap_weight: float = 0.8,
        governance_weight: float = 1.0,
        writeback_weight: float = 0.5,
        response_weight: float = 1.0,
    ):
        self.weights = {
            'policy': policy_weight,
            'gap': gap_weight,
            'governance': governance_weight,
            'writeback': writeback_weight,
            'response': response_weight,
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> EnhancedMultiHeadLoss:
        """
        计算增强版多头损失
        
        Args:
            predictions: 模型预测输出
                - policy_logits: [batch, num_strategies]
                - gap_logits: [batch, num_gap_types]
                - governance_logits: [batch, num_actions]
                - writeback_logits: [batch, num_writeback_types]
                - response_logits: [batch, seq_len, vocab_size]
            
            targets: 目标标签
                - policy_target: [batch]
                - gap_target: [batch]
                - governance_target: [batch, num_actions]
                - writeback_target: [batch]
                - response_target: [batch, seq_len]
        
        Returns:
            EnhancedMultiHeadLoss: 增强版多头损失对象
        """
        # 1. Policy Loss (策略选择)
        policy_loss = self._compute_policy_loss(
            predictions.get('policy_logits'),
            targets.get('policy_target')
        )
        
        # 2. Gap Loss (缺口识别) - 新增
        gap_loss, gap_acc = self._compute_gap_loss(
            predictions.get('gap_logits'),
            targets.get('gap_target')
        )
        
        # 3. Governance Loss (治理动作)
        governance_loss = self._compute_governance_loss(
            predictions.get('governance_logits'),
            targets.get('governance_target')
        )
        
        # 4. Writeback Loss (写回决策) - 增强版
        writeback_loss, wb_fp, lt_prec = self._compute_writeback_loss(
            predictions.get('writeback_logits'),
            targets.get('writeback_target')
        )
        
        # 5. Response Loss (响应生成)
        response_loss = self._compute_response_loss(
            predictions.get('response_logits'),
            targets.get('response_target')
        )
        
        # 计算检索误触发率（基于gap预测）
        retrieval_fp = self._compute_retrieval_false_positive(
            predictions.get('gap_logits'),
            targets.get('gap_target')
        )
        
        # 加权总损失
        total_loss = (
            self.weights['policy'] * policy_loss +
            self.weights['gap'] * gap_loss +
            self.weights['governance'] * governance_loss +
            self.weights['writeback'] * writeback_loss +
            self.weights['response'] * response_loss
        )
        
        return EnhancedMultiHeadLoss(
            policy_loss=policy_loss,
            gap_loss=gap_loss,
            governance_loss=governance_loss,
            writeback_loss=writeback_loss,
            response_loss=response_loss,
            total_loss=total_loss,
            policy_loss_value=policy_loss.item(),
            gap_loss_value=gap_loss.item(),
            governance_loss_value=governance_loss.item(),
            writeback_loss_value=writeback_loss.item(),
            response_loss_value=response_loss.item(),
            gap_accuracy=gap_acc,
            retrieval_false_positive=retrieval_fp,
            writeback_false_positive=wb_fp,
            long_term_precision=lt_prec,
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
        
        return F.cross_entropy(logits, targets)
    
    def _compute_gap_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """
        计算Gap损失，同时返回准确率
        
        Gap类型：
        0: NO_GAP (无缺口)
        1: RETRIEVABLE_GAP (可检索补足)
        2: HIGH_RISK_GAP (高风险缺口)
        """
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device), 0.0
        
        loss = F.cross_entropy(logits, targets)
        
        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()
        
        return loss, accuracy
    
    def _compute_governance_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算治理损失"""
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device)
        
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    
    def _compute_writeback_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, float, float]:
        """
        计算Writeback损失，同时返回误判率和长期候选精度
        
        Writeback类型：
        0: NO_WRITEBACK (不写回)
        1: EPHEMERAL (瞬时保留)
        2: LONG_TERM_CANDIDATE (长期候选)
        """
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device), 0.0, 0.0
        
        loss = F.cross_entropy(logits, targets)
        
        # 计算指标
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # 写回误判率：不该写回(0)但被预测为写回(1或2)的比例
            should_not_write = (targets == 0)
            if should_not_write.sum() > 0:
                false_positives = ((preds != 0) & should_not_write).sum()
                wb_fp = (false_positives.float() / should_not_write.sum()).item()
            else:
                wb_fp = 0.0
            
            # 长期候选精度：预测为长期候选(2)且正确的比例
            predicted_long_term = (preds == 2)
            if predicted_long_term.sum() > 0:
                correct_long_term = ((preds == targets) & predicted_long_term).sum()
                lt_prec = (correct_long_term.float() / predicted_long_term.sum()).item()
            else:
                lt_prec = 0.0
        
        return loss, wb_fp, lt_prec
    
    def _compute_response_loss(
        self,
        logits: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算响应损失"""
        if logits is None or targets is None:
            device = logits.device if logits is not None else (targets.device if targets is not None else 'cpu')
            return torch.tensor(0.0, device=device)
        
        batch_size, seq_len, vocab_size = logits.shape
        return F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )
    
    def _compute_retrieval_false_positive(
        self,
        gap_logits: Optional[torch.Tensor],
        gap_targets: Optional[torch.Tensor],
    ) -> float:
        """
        计算检索误触发率
        
        误触发：实际无缺口(0)或高风险缺口(2)，但被预测为可检索补足(1)
        """
        if gap_logits is None or gap_targets is None:
            return 0.0
        
        with torch.no_grad():
            gap_preds = gap_logits.argmax(dim=-1)
            
            # 实际不需要检索的情况（无缺口或高风险）
            no_retrieval_needed = (gap_targets == 0) | (gap_targets == 2)
            
            if no_retrieval_needed.sum() > 0:
                false_retrieval = ((gap_preds == 1) & no_retrieval_needed).sum()
                return (false_retrieval.float() / no_retrieval_needed.sum()).item()
            else:
                return 0.0


# 便捷函数
def create_enhanced_loss_runner(
    policy_weight: float = 1.0,
    gap_weight: float = 0.8,
    governance_weight: float = 1.0,
    writeback_weight: float = 0.5,
    response_weight: float = 1.0,
) -> EnhancedLossRunner:
    """创建增强版多头损失运行器"""
    return EnhancedLossRunner(
        policy_weight=policy_weight,
        gap_weight=gap_weight,
        governance_weight=governance_weight,
        writeback_weight=writeback_weight,
        response_weight=response_weight,
    )


# 测试
if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Multi-head Loss Runner v1 - 测试")
    print("=" * 70)
    
    runner = create_enhanced_loss_runner()
    
    batch_size = 4
    
    # 模拟预测
    predictions = {
        'policy_logits': torch.randn(batch_size, 5),
        'gap_logits': torch.randn(batch_size, 3),
        'governance_logits': torch.randn(batch_size, 8),
        'writeback_logits': torch.randn(batch_size, 3),
        'response_logits': torch.randn(batch_size, 1, 50000),
    }
    
    # 模拟目标
    targets = {
        'policy_target': torch.randint(0, 5, (batch_size,)),
        'gap_target': torch.randint(0, 3, (batch_size,)),
        'governance_target': torch.randint(0, 2, (batch_size, 8)).float(),
        'writeback_target': torch.randint(0, 3, (batch_size,)),
        'response_target': torch.randint(0, 50000, (batch_size, 1)),
    }
    
    # 计算损失
    loss = runner.compute_loss(predictions, targets)
    
    print(f"\n损失计算结果:")
    print(f"  Policy Loss: {loss.policy_loss_value:.4f}")
    print(f"  Gap Loss: {loss.gap_loss_value:.4f}")
    print(f"  Governance Loss: {loss.governance_loss_value:.4f}")
    print(f"  Writeback Loss: {loss.writeback_loss_value:.4f}")
    print(f"  Response Loss: {loss.response_loss_value:.4f}")
    print(f"  Total Loss: {loss.total_loss.item():.4f}")
    
    print(f"\n关键指标:")
    print(f"  Gap Accuracy: {loss.gap_accuracy:.2%}")
    print(f"  Retrieval False Positive: {loss.retrieval_false_positive:.2%}")
    print(f"  Writeback False Positive: {loss.writeback_false_positive:.2%}")
    print(f"  Long-term Precision: {loss.long_term_precision:.2%}")
    
    print("\n✓ 增强版多头损失运行器工作正常")
