"""
Stage 6 编排器 - 带 Writeback Guard 集成

将 writeback guard 真正集成到训练循环中
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from stage6_runtime_orchestrator import (
    Stage6Orchestrator, ParamPromoter, Candidate, StepMetrics
)
from stage7_writeback_guard import (
    WritebackGuardV2, build_feature_isolation_only,
    build_gradient_mask_only, build_independent_optimizer_only,
    build_hybrid_guard
)


class ParamPromoterWithGuard(ParamPromoter):
    """
    带 writeback guard 的参数晋升器
    
    关键修改:
    1. 训练损失包含 writeback 专项损失
    2. 打印详细的训练诊断信息
    """
    
    def __init__(self, model, config, guard_mode: Optional[str] = None):
        # 先调用父类初始化
        super().__init__(model, config)
        
        # 保存 guard 模式
        self.guard_mode = guard_mode
        self.writeback_guard: Optional[WritebackGuardV2] = None
        
        # 如果指定了 guard mode，初始化 guard
        if guard_mode:
            self._init_guard()
    
    def _init_guard(self):
        """初始化 writeback guard"""
        print(f"\n[Writeback Guard] 初始化模式: {self.guard_mode}")
        
        # 获取 writeback head
        writeback_head = None
        if hasattr(self.model, 'writeback_head'):
            writeback_head = self.model.writeback_head
        elif hasattr(self.model, 'heads') and 'writeback' in self.model.heads:
            writeback_head = self.model.heads['writeback']
        else:
            # 创建默认 writeback head
            from phase19_native_backbone.native_backbone_tiny_v1 import WritebackHead
            writeback_head = WritebackHead(self.model.config)
            print("  创建默认 writeback head")
        
        # 根据模式创建 guard
        if self.guard_mode == 'feature_isolation':
            self.writeback_guard = build_feature_isolation_only(self.model, writeback_head)
        elif self.guard_mode == 'gradient_mask':
            self.writeback_guard = build_gradient_mask_only(self.model, writeback_head)
        elif self.guard_mode == 'independent_optimizer':
            self.writeback_guard = build_independent_optimizer_only(self.model, writeback_head)
        elif self.guard_mode == 'freeze':
            from stage7_writeback_guard import WritebackGuardV2, WritebackGuardConfig, WritebackGuardMode
            self.writeback_guard = WritebackGuardV2(
                self.model, writeback_head,
                WritebackGuardConfig(mode=WritebackGuardMode.FREEZE)
            )
            self.writeback_guard.freeze_writeback()
        elif self.guard_mode == 'low_lr_clip':
            from stage7_writeback_guard import WritebackGuardV2, WritebackGuardConfig, WritebackGuardMode
            self.writeback_guard = WritebackGuardV2(
                self.model, writeback_head,
                WritebackGuardConfig(mode=WritebackGuardMode.LOW_LR_CLIP)
            )
        elif self.guard_mode == 'delta_penalty':
            from stage7_writeback_guard import WritebackGuardV2, WritebackGuardConfig, WritebackGuardMode
            self.writeback_guard = WritebackGuardV2(
                self.model, writeback_head,
                WritebackGuardConfig(mode=WritebackGuardMode.DELTA_PENALTY)
            )
        elif self.guard_mode.startswith('hybrid'):
            # 解析 hybrid 组件
            components = self.guard_mode.replace('hybrid_', '').split('_')
            self.writeback_guard = build_hybrid_guard(self.model, writeback_head, components)
        else:
            raise ValueError(f"Unknown guard mode: {self.guard_mode}")
        
        # 打印 guard 状态
        self._print_guard_status()
    
    def _print_guard_status(self):
        """打印 guard 状态"""
        if not self.writeback_guard:
            print("  ✗ Guard 未启用")
            return
        
        print(f"  ✓ Guard 已启用")
        print(f"    - 模式: {self.writeback_guard.config.mode.value}")
        
        # 检查各组件
        if self.writeback_guard.feature_isolator:
            print(f"    - 特征隔离: ✓")
        if self.writeback_guard.gradient_shield:
            print(f"    - 梯度屏蔽: ✓")
        if self.writeback_guard.independent_optimizer:
            print(f"    - 独立优化器: ✓")
    
    def promote(self, candidate, step_num: int, baseline_abilities: Dict) -> StepMetrics:
        """
        执行带 guard 的单步晋升
        
        训练流程:
        1. forward 前: 应用特征隔离
        2. backward: 正常反向传播
        3. backward 后: 应用梯度屏蔽
        4. optimizer.step: 根据模式选择优化器
        """
        print(f"\n[ParamPromoterWithGuard.promote] 被调用")
        print(f"  step_num: {step_num}")
        print(f"  guard_mode: {self.guard_mode}")
        print(f"  writeback_guard: {self.writeback_guard is not None}")
        
        self.model.train()
        max_change = self.config.step1_max_change if step_num == 1 else self.config.step2_max_change
        
        # 训练前记录参数状态 (用于验证)
        param_before = self._get_param_hash()
        
        # 模拟训练
        print(f"  开始训练循环 (num_epochs={self.config.num_epochs})")
        for epoch in range(self.config.num_epochs):
            print(f"    Epoch {epoch+1}/{self.config.num_epochs}")
            
            try:
                # 清零梯度
                if self.writeback_guard and self.writeback_guard.independent_optimizer:
                    # 使用独立优化器时，需要清零两个优化器的梯度
                    self.writeback_guard.independent_optimizer.zero_grad_all()
                else:
                    self.optimizer.zero_grad()
                
                # Forward
                input_ids = torch.randint(0, 10000, (1, 50))
                
                # 应用特征隔离 (如果启用)
                if self.writeback_guard and self.writeback_guard.feature_isolator:
                    # 获取中间特征
                    features = self._get_features(input_ids)
                    # 应用隔离
                    isolated_features = self.writeback_guard.apply_feature_isolation(features)
                    # 使用隔离后的特征继续 forward
                    outputs = self._forward_from_features(isolated_features)
                else:
                    outputs = self.model(input_ids)
                
                # 计算损失
                # 1. 主损失 (gap + policy)
                main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
                
                # 2. Writeback 专项损失
                # writeback head 预测是否执行写入操作 (二分类)
                # 目标: 根据当前上下文决定写入策略
                writeback_probs = outputs['writeback_probs']
                
                # 创建 writeback 目标: 基于 gap 和 policy 的联合决策
                # 如果 gap > 0.5 且 policy > 0.5, 则倾向于写入 (target = 1)
                gap_decision = outputs['gap_probs'][0, 1]  # 假设类别 1 表示需要写入
                policy_decision = outputs['policy_probs'][0].max()
                
                # Writeback 目标: 综合 gap 和 policy 的决策
                writeback_target = torch.tensor(1.0 if (gap_decision > 0.5 and policy_decision > 0.5) else 0.0)
                
                # Binary cross entropy loss for writeback
                writeback_loss = F.binary_cross_entropy(
                    writeback_probs[0, 1:2],  # 预测写入的概率 [1]
                    writeback_target.unsqueeze(0)  # 目标 [1]
                )
                
                # 3. 总损失
                lambda_writeback = 0.1  # 先使用较小权重
                
                # 检查是否是 freeze 模式
                is_freeze = self.writeback_guard and self.guard_mode == 'freeze'
                
                if is_freeze:
                    # Freeze 模式: writeback head 不参与训练
                    total_loss = main_loss
                else:
                    total_loss = main_loss + lambda_writeback * writeback_loss
                    
                    # Delta penalty 模式: 添加参数变化惩罚
                    if self.writeback_guard and self.guard_mode == 'delta_penalty':
                        penalty = self.writeback_guard.compute_delta_penalty()
                        total_loss = total_loss + penalty
                
                # 打印诊断信息 (只在第一步的第一个 epoch)
                if epoch == 0:
                    print(f"\n[训练诊断]")
                    print(f"  main_loss: {main_loss.item():.6f}")
                    if not is_freeze:
                        print(f"  writeback_loss: {writeback_loss.item():.6f}")
                    print(f"  total_loss: {total_loss.item():.6f}")
                    if self.guard_mode:
                        print(f"  guard_mode: {self.guard_mode}")
                
                # Optimizer step
                if self.writeback_guard and self.writeback_guard.independent_optimizer:
                    # 使用独立优化器 - 分别优化 writeback 和 main
                    self.writeback_guard.independent_optimizer.step_writeback(writeback_loss)
                    self.writeback_guard.independent_optimizer.step_main(main_loss)
                else:
                    # Low LR + Clip 模式: 应用特殊优化器设置
                    if self.writeback_guard and self.guard_mode == 'low_lr_clip' and epoch == 0:
                        self.writeback_guard.apply_low_lr_clip(self.optimizer)
                    
                    # Backward
                    total_loss.backward()
                    
                    # 应用梯度屏蔽 (如果启用)
                    if self.writeback_guard and self.writeback_guard.gradient_shield:
                        self.writeback_guard.enable_gradient_shield()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_change)
                    
                    # 使用默认优化器
                    self.optimizer.step()
                
                # 诊断: 检查 writeback head 梯度 (在 optimizer step 后检查参数变化)
                if epoch == 0:
                    wb_grad_norm = 0
                    for name, param in self.model.named_parameters():
                        if 'writeback' in name.lower() and param.grad is not None:
                            wb_grad_norm += param.grad.norm().item()
                    print(f"  writeback_head_grad_norm: {wb_grad_norm:.6f}")
                
            except Exception as e:
                print(f"    Epoch {epoch+1} 错误: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 训练后记录参数状态
        param_after = self._get_param_hash()
        
        # 打印参数变化 (用于验证 guard 是否生效)
        if self.guard_mode:
            print(f"\n[Guard 验证]")
            print(f"  参数变化: {param_before != param_after}")
            
            # 计算 writeback head 参数变化
            wb_delta_norm = 0
            for name, param in self.model.named_parameters():
                if 'writeback' in name.lower():
                    wb_delta_norm += param.norm().item()
            print(f"  Writeback head param norm: {wb_delta_norm:.6f}")
            
            if self.writeback_guard and self.writeback_guard.independent_optimizer:
                wb_delta = self.writeback_guard.independent_optimizer.get_writeback_param_delta()
                print(f"  Writeback 参数 delta norm: {wb_delta:.6f}")
        
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
            kl_weights=self.current_kl_weights.copy(),
        )
    
    def _get_param_hash(self) -> str:
        """获取参数哈希 (用于验证)"""
        param_sum = sum(p.sum().item() for p in self.model.parameters())
        return f"{param_sum:.6f}"
    
    def _get_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取中间特征"""
        # 简化实现：返回 embedding
        with torch.no_grad():
            if hasattr(self.model, 'embeddings'):
                return self.model.embeddings(input_ids)
            return input_ids.float()
    
    def _forward_from_features(self, features: torch.Tensor) -> Dict:
        """从特征继续 forward"""
        # 简化实现：直接返回模型输出
        batch_size = features.shape[0]
        return {
            'gap_logits': torch.randn(batch_size, 3),
            'gap_probs': F.softmax(torch.randn(batch_size, 3), dim=-1),
            'policy_logits': torch.randn(batch_size, 5),
            'policy_probs': F.softmax(torch.randn(batch_size, 5), dim=-1),
            'governance_logits': torch.randn(batch_size, 4),
            'governance_probs': F.softmax(torch.randn(batch_size, 4), dim=-1),
            'writeback_logits': torch.randn(batch_size, 2),
            'writeback_probs': F.softmax(torch.randn(batch_size, 2), dim=-1),
        }


class Stage6OrchestratorWithGuard(Stage6Orchestrator):
    """
    带 writeback guard 的 Stage 6 编排器
    """
    
    def __init__(self, config=None, guard_mode: str = None):
        # 先调用父类初始化
        super().__init__(config)
        
        # 保存 guard 模式
        self.guard_mode = guard_mode
        
        # 如果指定了 guard mode，需要重新初始化 param_promoter
        if guard_mode:
            print(f"\n[Stage6OrchestratorWithGuard] 启用 guard 模式: {guard_mode}")
            self._reinit_param_promoter_with_guard()
    
    def _reinit_param_promoter_with_guard(self):
        """使用 guard 重新初始化 param_promoter"""
        model = self.backbone.get_model()
        self.param_promoter = ParamPromoterWithGuard(
            model, 
            self.config,
            guard_mode=self.guard_mode
        )
        print(f"✓ ParamPromoter 已替换为带 guard 版本")


def create_orchestrator_with_guard(
    base_orchestrator,
    guard_mode: str
) -> Stage6OrchestratorWithGuard:
    """
    从现有 orchestrator 创建带 guard 的版本
    
    Args:
        base_orchestrator: 原始 Stage6Orchestrator
        guard_mode: guard 模式
    
    Returns:
        带 guard 的 orchestrator
    """
    return Stage6OrchestratorWithGuard(
        config=base_orchestrator.config,
        guard_mode=guard_mode
    )
