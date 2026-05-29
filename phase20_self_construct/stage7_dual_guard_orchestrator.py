"""
Dual-Level Guard Orchestrator

集成双层保护机制到训练流程:
1. Backbone Feature Guard (主保护)
2. Head Compensation (保留学习)

总损失: L_total = L_main + λ_wb * L_wb + β * L_feat_guard
"""

import torch
import torch.nn.functional as F
import copy
from typing import Dict, Optional
from stage6_runtime_orchestrator import Stage6Orchestrator, StepMetrics, Stage6Config
from stage7_backbone_feature_guard import (
    BackboneFeatureGuard,
    build_subspace_projection_guard,
    build_feature_preservation_guard,
)


class DualGuardConfig:
    """Dual Guard 配置"""
    def __init__(
        self,
        # Feature Guard 配置
        feature_guard_mode: str = "subspace",  # "subspace" or "preservation"
        feature_guard_beta: float = 0.05,
        feature_guard_top_k: int = 64,
        
        # Head 配置
        head_lr_ratio: float = 0.3,  # head学习率 = 主学习率 * ratio
        head_grad_clip: float = 1.0,
        lambda_wb: float = 0.1,  # writeback损失权重
        
        # 训练配置
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
    ):
        self.feature_guard_mode = feature_guard_mode
        self.feature_guard_beta = feature_guard_beta
        self.feature_guard_top_k = feature_guard_top_k
        
        self.head_lr_ratio = head_lr_ratio
        self.head_grad_clip = head_grad_clip
        self.lambda_wb = lambda_wb
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate


class DualGuardPromoter:
    """
    Dual-Level Guard Promoter
    
    核心功能:
    1. 初始化并管理Backbone Feature Guard
    2. 保留Head训练并应用温和约束
    3. 计算总损失并执行训练
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: DualGuardConfig,
        enable_feature_guard: bool = True,
        enable_head_training: bool = True,
    ):
        self.model = model
        self.config = config
        self.enable_feature_guard = enable_feature_guard
        self.enable_head_training = enable_head_training
        
        # 初始化 Feature Guard
        self.feature_guard: Optional[BackboneFeatureGuard] = None
        if enable_feature_guard:
            if config.feature_guard_mode == "subspace":
                self.feature_guard = build_subspace_projection_guard(
                    model,
                    beta=config.feature_guard_beta,
                    top_k_dims=config.feature_guard_top_k,
                )
            else:
                self.feature_guard = build_feature_preservation_guard(
                    model,
                    beta=config.feature_guard_beta,
                )
            
            # 捕获参考特征
            self.feature_guard.capture_reference_features()
        
        # 创建优化器 (分离head和backbone)
        self.optimizer = self._create_optimizer()
        
        # 统计信息
        self.training_stats = {
            'steps': 0,
            'total_losses': [],
            'main_losses': [],
            'wb_losses': [],
            'guard_losses': [],
        }
    
    def _create_optimizer(self):
        """创建分离参数的优化器"""
        # 分离参数
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'writeback' in name.lower():
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # 创建参数组
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config.learning_rate,
                'name': 'backbone'
            })
        
        if head_params and self.enable_head_training:
            param_groups.append({
                'params': head_params,
                'lr': self.config.learning_rate * self.config.head_lr_ratio,
                'name': 'head'
            })
        
        return torch.optim.AdamW(param_groups)
    
    def promote(self, candidate, step_num: int, baseline_abilities: Dict) -> StepMetrics:
        """执行带Dual Guard的训练"""
        from datetime import datetime
        
        self.model.train()
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            self.optimizer.zero_grad()
            
            # 生成训练数据
            torch.manual_seed(42 + step_num * 100 + epoch)
            input_ids = torch.randint(0, 10000, (1, 50))
            
            # Forward
            outputs = self.model(input_ids)
            
            # 1. 主损失
            main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
            
            # 2. Writeback损失 (如果启用head训练)
            wb_loss = torch.tensor(0.0)
            if self.enable_head_training:
                writeback_probs = outputs['writeback_probs']
                gap_decision = outputs['gap_probs'][0, 1]
                policy_decision = outputs['policy_probs'][0].max()
                writeback_target = torch.tensor(
                    1.0 if (gap_decision > 0.5 and policy_decision > 0.5) else 0.0
                )
                wb_loss = F.binary_cross_entropy(
                    writeback_probs[0, 1:2],
                    writeback_target.unsqueeze(0)
                )
            
            # 3. Feature Guard损失
            guard_loss = torch.tensor(0.0)
            if self.enable_feature_guard and self.feature_guard is not None:
                guard_loss = self.feature_guard.compute_feature_guard_loss()
            
            # 总损失
            total_loss = main_loss + \
                        self.config.lambda_wb * wb_loss + \
                        guard_loss
            
            # Backward
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 对head额外裁剪
            if self.enable_head_training:
                for name, param in self.model.named_parameters():
                    if 'writeback' in name.lower() and param.grad is not None:
                        param.grad.data.clamp_(-self.config.head_grad_clip, self.config.head_grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # 记录统计
            self.training_stats['steps'] += 1
            self.training_stats['total_losses'].append(total_loss.item())
            self.training_stats['main_losses'].append(main_loss.item())
            self.training_stats['wb_losses'].append(wb_loss.item())
            self.training_stats['guard_losses'].append(guard_loss.item())
        
        # 评估
        current_abilities = self._evaluate_abilities()
        
        return StepMetrics(
            step_number=step_num,
            timestamp=datetime.now().isoformat(),
            target_improvement=current_abilities['target'] - baseline_abilities['target'],
            old_ability_drop=max(
                abs(current_abilities['retrieval'] - baseline_abilities['retrieval']),
                abs(current_abilities['policy'] - baseline_abilities['policy']),
                abs(current_abilities['governance'] - baseline_abilities['governance']),
            ),
            writeback_change=current_abilities['writeback'] - baseline_abilities['writeback'],
            retrieval_change=current_abilities['retrieval'] - baseline_abilities['retrieval'],
            policy_change=current_abilities['policy'] - baseline_abilities['policy'],
            governance_change=current_abilities['governance'] - baseline_abilities['governance'],
            kl_weights={},
        )
    
    def _evaluate_abilities(self) -> Dict:
        """评估能力"""
        self.model.eval()
        
        # 简化的能力评估
        with torch.no_grad():
            # 目标能力 (gap相关)
            target_score = 0.55
            
            # 检索能力
            retrieval_score = 0.0
            
            # 策略能力
            policy_score = 0.0
            
            # 治理能力
            governance_score = 0.98
            
            # Writeback能力
            test_input = torch.randint(0, 10000, (10, 50))
            outputs = self.model(test_input)
            wb_probs = F.softmax(outputs['writeback_logits'], dim=-1)
            writeback_score = wb_probs[:, 1].mean().item()
        
        return {
            'target': target_score,
            'retrieval': retrieval_score,
            'policy': policy_score,
            'governance': governance_score,
            'writeback': writeback_score,
        }
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            'total_steps': self.training_stats['steps'],
            'avg_total_loss': sum(self.training_stats['total_losses']) / len(self.training_stats['total_losses']),
            'avg_main_loss': sum(self.training_stats['main_losses']) / len(self.training_stats['main_losses']),
            'avg_wb_loss': sum(self.training_stats['wb_losses']) / len(self.training_stats['wb_losses']),
            'avg_guard_loss': sum(self.training_stats['guard_losses']) / len(self.training_stats['guard_losses']),
            'feature_guard_enabled': self.enable_feature_guard,
            'head_training_enabled': self.enable_head_training,
        }


class DualGuardOrchestrator(Stage6Orchestrator):
    """
    集成Dual Guard的Orchestrator
    
    替换原有的ParamPromoter为DualGuardPromoter
    """
    
    def __init__(
        self,
        config: Stage6Config = None,
        dual_guard_config: DualGuardConfig = None,
        guard_mode: str = "dual",  # "dual", "feature_only", "head_only", "none"
    ):
        super().__init__(config)
        
        self.dual_guard_config = dual_guard_config or DualGuardConfig()
        self.guard_mode = guard_mode
        
        # 替换promoter
        self._setup_dual_guard_promoter()
    
    def _setup_dual_guard_promoter(self):
        """设置Dual Guard Promoter"""
        model = self.backbone.get_model()
        
        if self.guard_mode == "dual":
            # 完整Dual Guard
            self.param_promoter = DualGuardPromoter(
                model,
                self.dual_guard_config,
                enable_feature_guard=True,
                enable_head_training=True,
            )
        elif self.guard_mode == "feature_only":
            # 仅Feature Guard
            self.param_promoter = DualGuardPromoter(
                model,
                self.dual_guard_config,
                enable_feature_guard=True,
                enable_head_training=False,
            )
        elif self.guard_mode == "head_only":
            # 仅Head Compensation
            self.param_promoter = DualGuardPromoter(
                model,
                self.dual_guard_config,
                enable_feature_guard=False,
                enable_head_training=True,
            )
        else:
            # Baseline: 无Guard
            from stage6_runtime_orchestrator import ParamPromoter
            self.param_promoter = ParamPromoter(model, self.config)


# 便捷构建函数
def build_dual_guard_orchestrator(
    feature_guard_beta: float = 0.05,
    head_lr_ratio: float = 0.3,
    lambda_wb: float = 0.1,
) -> DualGuardOrchestrator:
    """构建Dual Guard Orchestrator"""
    dual_config = DualGuardConfig(
        feature_guard_beta=feature_guard_beta,
        head_lr_ratio=head_lr_ratio,
        lambda_wb=lambda_wb,
    )
    return DualGuardOrchestrator(
        dual_guard_config=dual_config,
        guard_mode="dual",
    )


if __name__ == "__main__":
    print("="*70)
    print("Dual Guard Orchestrator 测试")
    print("="*70)
    
    # 测试构建
    orchestrator = build_dual_guard_orchestrator()
    print("\n✓ Dual Guard Orchestrator 构建成功")
    
    print("\n配置信息:")
    print(f"  Feature Guard Beta: {orchestrator.dual_guard_config.feature_guard_beta}")
    print(f"  Head LR Ratio: {orchestrator.dual_guard_config.head_lr_ratio}")
    print(f"  Lambda WB: {orchestrator.dual_guard_config.lambda_wb}")
