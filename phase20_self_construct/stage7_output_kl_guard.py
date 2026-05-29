"""
Output-level KL Guard

基于 Output KL 的 writeback 约束方案

核心改进:
1. 直接约束 writeback 输出 (logits/probs)，而非 backbone 特征
2. 使用 KL 散度衡量输出漂移
3. 保留 head 的补偿学习能力
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OutputKLGuardConfig:
    """Output KL Guard 配置"""
    beta: float = 0.1  # KL 损失权重
    num_anchor_samples: int = 30
    anchor_seed: int = 42
    temperature: float = 1.0  # KL 计算温度
    use_probs: bool = True  # 使用 probs (True) 还是 logits (False)


class OutputKLGuard:
    """
    Output-level KL Guard
    
    约束目标: writeback 输出分布不要偏离参考分布
    约束方式: KL(P_current || P_reference)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: OutputKLGuardConfig,
    ):
        self.model = model
        self.config = config
        
        # 参考输出 (训练前保存)
        self.ref_outputs: Optional[torch.Tensor] = None
        self.anchor_inputs: Optional[List[torch.Tensor]] = None
        
        # 初始化锚点样本
        self._setup_anchor_samples()
    
    def _setup_anchor_samples(self):
        """设置固定锚点样本"""
        torch.manual_seed(self.config.anchor_seed)
        device = next(self.model.parameters()).device
        self.anchor_inputs = [
            torch.randint(0, 10000, (1, 50)).to(device)
            for _ in range(self.config.num_anchor_samples)
        ]
    
    def capture_reference_outputs(self):
        """
        捕获训练前的参考输出
        
        在训练开始前调用
        """
        self.model.eval()
        
        ref_outputs_list = []
        with torch.no_grad():
            for input_ids in self.anchor_inputs:
                outputs = self.model(input_ids)
                writeback_logits = outputs['writeback_logits']
                
                if self.config.use_probs:
                    # 保存概率分布
                    probs = F.softmax(writeback_logits / self.config.temperature, dim=-1)
                    ref_outputs_list.append(probs.clone())
                else:
                    # 保存 logits
                    ref_outputs_list.append(writeback_logits.clone())
        
        self.ref_outputs = torch.cat(ref_outputs_list, dim=0)
        
        print(f"[OutputKLGuard] Reference outputs captured: shape={self.ref_outputs.shape}")
        print(f"[OutputKLGuard] Using {'probs' if self.config.use_probs else 'logits'}")
    
    def compute_kl_guard_loss(self) -> torch.Tensor:
        """
        计算 Output KL Guard 损失
        
        Returns:
            KL 散度损失 * beta
        """
        if self.ref_outputs is None:
            raise RuntimeError("Must call capture_reference_outputs() before training")
        
        # 获取当前输出 (允许梯度)
        current_outputs_list = []
        for input_ids in self.anchor_inputs:
            outputs = self.model(input_ids)
            writeback_logits = outputs['writeback_logits']
            
            if self.config.use_probs:
                probs = F.softmax(writeback_logits / self.config.temperature, dim=-1)
                current_outputs_list.append(probs)
            else:
                current_outputs_list.append(writeback_logits)
        
        current_outputs = torch.cat(current_outputs_list, dim=0)
        
        # 计算 KL 散度
        if self.config.use_probs:
            # KL(P_current || P_ref) = sum(P_current * log(P_current / P_ref))
            # 使用更稳定的计算方式
            eps = 1e-8
            current_safe = current_outputs.clamp(min=eps)
            ref_safe = self.ref_outputs.clamp(min=eps)
            
            kl_per_sample = (current_safe * (torch.log(current_safe) - torch.log(ref_safe))).sum(dim=-1)
            kl_loss = kl_per_sample.mean()
        else:
            # 使用 log_softmax 和 softmax 计算 KL
            current_log_probs = F.log_softmax(current_outputs / self.config.temperature, dim=-1)
            ref_probs = F.softmax(self.ref_outputs / self.config.temperature, dim=-1)
            
            kl_loss = F.kl_div(
                current_log_probs,
                ref_probs,
                reduction='batchmean'
            )
        
        return kl_loss * self.config.beta
    
    def compute_output_drift(self) -> float:
        """
        计算当前输出相对于参考的漂移
        
        Returns:
            平均 KL 散度 (不乘 beta)
        """
        if self.ref_outputs is None:
            return 0.0
        
        self.model.eval()
        with torch.no_grad():
            current_outputs_list = []
            for input_ids in self.anchor_inputs:
                outputs = self.model(input_ids)
                writeback_logits = outputs['writeback_logits']
                
                if self.config.use_probs:
                    probs = F.softmax(writeback_logits / self.config.temperature, dim=-1)
                    current_outputs_list.append(probs)
                else:
                    current_outputs_list.append(writeback_logits)
            
            current_outputs = torch.cat(current_outputs_list, dim=0)
            
            if self.config.use_probs:
                eps = 1e-8
                current_safe = current_outputs.clamp(min=eps)
                ref_safe = self.ref_outputs.clamp(min=eps)
                kl_per_sample = (current_safe * (torch.log(current_safe) - torch.log(ref_safe))).sum(dim=-1)
                drift = kl_per_sample.mean().item()
            else:
                current_log_probs = F.log_softmax(current_outputs / self.config.temperature, dim=-1)
                ref_probs = F.softmax(self.ref_outputs / self.config.temperature, dim=-1)
                drift = F.kl_div(current_log_probs, ref_probs, reduction='batchmean').item()
        
        return drift
    
    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'beta': self.config.beta,
            'num_anchor_samples': self.config.num_anchor_samples,
            'use_probs': self.config.use_probs,
            'reference_captured': self.ref_outputs is not None,
            'output_drift': self.compute_output_drift(),
        }


def build_output_kl_guard(
    model: torch.nn.Module,
    beta: float = 0.1,
    num_samples: int = 30,
    use_probs: bool = True,
) -> OutputKLGuard:
    """构建 Output KL Guard"""
    config = OutputKLGuardConfig(
        beta=beta,
        num_anchor_samples=num_samples,
        use_probs=use_probs,
    )
    guard = OutputKLGuard(model, config)
    guard.capture_reference_outputs()
    return guard


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Output KL Guard 测试")
    print("="*70)
    
    # 创建简单模型
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(10000, 128)
            self.encoder = torch.nn.Linear(128, 128)
            self.writeback_head = torch.nn.Linear(128, 2)
        
        def forward(self, input_ids):
            x = self.embeddings(input_ids).mean(dim=1)
            features = self.encoder(x)
            writeback_logits = self.writeback_head(features)
            return {
                'writeback_logits': writeback_logits,
                'gap_logits': torch.randn(1, 2),
                'policy_logits': torch.randn(1, 2),
                'gap_probs': torch.softmax(torch.randn(1, 2), dim=-1),
                'policy_probs': torch.softmax(torch.randn(1, 2), dim=-1),
                'writeback_probs': torch.softmax(writeback_logits, dim=-1),
                'backbone_features': features,
            }
    
    model = DummyModel()
    
    # 测试 Output KL Guard
    print("\n测试 Output KL Guard:")
    guard = build_output_kl_guard(model, beta=0.1, num_samples=10)
    
    # 计算初始 KL loss
    kl_loss = guard.compute_kl_guard_loss()
    print(f"  Initial KL Loss: {kl_loss.item():.6f}")
    
    # 模拟训练后
    print("\n模拟训练后...")
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    kl_loss_after = guard.compute_kl_guard_loss()
    drift = guard.compute_output_drift()
    print(f"  KL Loss after perturbation: {kl_loss_after.item():.6f}")
    print(f"  Output drift: {drift:.6f}")
    
    # 测试梯度
    model.train()
    guard.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    guard.optimizer.zero_grad()
    
    loss = guard.compute_kl_guard_loss()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"\n  Gradients computed: {has_grad}")
    
    print("\n✓ 测试完成")
