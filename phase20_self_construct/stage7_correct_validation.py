"""
正确的 Dual Guard 验证脚本

修复梯度计算逻辑:
- 分别计算 Guard Loss 和 Main Loss 的梯度
- 比较两者的相对大小
"""

import torch
import torch.nn.functional as F
import copy
from typing import Dict, List
from dataclasses import dataclass
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_runtime_orchestrator import Candidate
from stage7_dual_guard_orchestrator import DualGuardPromoter, DualGuardConfig
from stage7_backbone_feature_guard import build_subspace_projection_guard


@dataclass
class ValidationRecord:
    """验证记录"""
    step: int
    main_loss: float
    guard_loss: float
    guard_grad_norm: float  # 仅由 guard loss 产生的梯度
    main_grad_norm: float   # 仅由 main loss 产生的梯度
    total_grad_norm: float  # 总梯度
    grad_ratio: float       # guard_grad / main_grad
    wb_score: float
    wb_change: float
    
    def to_row(self) -> str:
        return (f"{self.step:<6} {self.main_loss:<12.6f} {self.guard_loss:<14.6f} "
                f"{self.guard_grad_norm:<14.6f} {self.main_grad_norm:<14.6f} "
                f"{self.grad_ratio:<10.4f} {self.wb_score:<12.4f} {self.wb_change:<12.4f}")


class CorrectValidator:
    """正确的验证器"""
    
    def __init__(self, model, feature_guard, optimizer, name: str):
        self.model = model
        self.feature_guard = feature_guard
        self.optimizer = optimizer
        self.name = name
        self.records: List[ValidationRecord] = []
        
        torch.manual_seed(42)
        self.anchor_inputs = [torch.randint(0, 10000, (1, 50)) for _ in range(30)]
        self.baseline_wb_score = None
    
    def set_baseline(self):
        """设置基线"""
        self.model.eval()
        with torch.no_grad():
            wb_scores = []
            for input_ids in self.anchor_inputs:
                outputs = self.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            self.baseline_wb_score = sum(wb_scores) / len(wb_scores)
        print(f"[{self.name}] 基线 WB Score: {self.baseline_wb_score:.4f}")
    
    def compute_grad_norm(self) -> float:
        """计算当前梯度范数"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        return total_norm ** 0.5
    
    def record_step(self, step: int):
        """记录单步状态 - 正确分离 Guard 和 Main 梯度"""
        self.model.train()
        
        # 1. 只计算 Guard Loss 的梯度
        self.optimizer.zero_grad()
        guard_loss = self.feature_guard.compute_feature_guard_loss()
        guard_loss.backward()
        guard_grad_norm = self.compute_grad_norm()
        
        # 2. 只计算 Main Loss 的梯度
        self.optimizer.zero_grad()
        torch.manual_seed(42 + step * 100)
        input_ids = torch.randint(0, 10000, (1, 50))
        outputs = self.model(input_ids)
        main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
        main_loss.backward()
        main_grad_norm = self.compute_grad_norm()
        
        # 3. 计算总梯度 (Guard + Main)
        self.optimizer.zero_grad()
        guard_loss = self.feature_guard.compute_feature_guard_loss()
        torch.manual_seed(42 + step * 100)
        input_ids = torch.randint(0, 10000, (1, 50))
        outputs = self.model(input_ids)
        main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
        total_loss = main_loss + guard_loss
        total_loss.backward()
        total_grad_norm = self.compute_grad_norm()
        
        # 4. 计算 Writeback Score
        self.model.eval()
        with torch.no_grad():
            wb_scores = []
            for input_ids in self.anchor_inputs:
                outputs = self.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            wb_score = sum(wb_scores) / len(wb_scores)
        
        # 计算梯度比例
        grad_ratio = guard_grad_norm / main_grad_norm if main_grad_norm > 0 else 0
        
        record = ValidationRecord(
            step=step,
            main_loss=main_loss.item(),
            guard_loss=guard_loss.item(),
            guard_grad_norm=guard_grad_norm,
            main_grad_norm=main_grad_norm,
            total_grad_norm=total_grad_norm,
            grad_ratio=grad_ratio,
            wb_score=wb_score,
            wb_change=wb_score - self.baseline_wb_score,
        )
        
        self.records.append(record)
        return record
    
    def train_and_record(self, num_steps: int, checkpoint_steps: List[int]):
        """训练并记录"""
        print(f"\n[{self.name}] 开始训练...")
        
        for step in range(num_steps):
            self.model.train()
            
            # 训练一步
            for epoch in range(3):
                self.optimizer.zero_grad()
                
                torch.manual_seed(42 + step * 100 + epoch)
                input_ids = torch.randint(0, 10000, (1, 50))
                
                outputs = self.model(input_ids)
                main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
                
                # Guard loss
                guard_loss = self.feature_guard.compute_feature_guard_loss()
                
                total_loss = main_loss + guard_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # 在检查点记录
            if (step + 1) in checkpoint_steps:
                self.record_step(step + 1)
                r = self.records[-1]
                print(f"  Step {step+1}: ratio={r.grad_ratio:.4f}, wb_change={r.wb_change:+.4f}")
    
    def print_results(self):
        """打印结果"""
        print(f"\n{'='*120}")
        print(f"{self.name} 验证结果")
        print(f"{'='*120}")
        print(f"{'Step':<6} {'Main Loss':<12} {'Guard Loss':<14} {'Guard Grad':<14} {'Main Grad':<14} "
              f"{'Ratio':<10} {'WB Score':<12} {'WB Change':<12}")
        print(f"{'-'*120}")
        for record in self.records:
            print(record.to_row())
        print(f"{'='*120}")
        
        # 统计
        first = self.records[0]
        last = self.records[-1]
        
        print(f"\n关键指标:")
        print(f"  Guard Loss: {first.guard_loss:.6f} -> {last.guard_loss:.6f}")
        print(f"  Guard Grad: {first.guard_grad_norm:.6f} -> {last.guard_grad_norm:.6f}")
        print(f"  Main Grad: {first.main_grad_norm:.6f} -> {last.main_grad_norm:.6f}")
        print(f"  Grad Ratio: {first.grad_ratio:.4f} -> {last.grad_ratio:.4f}")
        print(f"  WB Score: {first.wb_score:.4f} -> {last.wb_score:.4f}")
        print(f"  WB Change: {last.wb_change:+.4f}")
        
        # 判断是否成功
        ratio_maintained = last.grad_ratio >= 0.01  # 保持至少 1%
        wb_stable = abs(last.wb_change) < 0.1  # WB变化<10%
        
        print(f"\n验收:")
        print(f"  Grad Ratio >= 1%: {'✓' if ratio_maintained else '✗'} ({last.grad_ratio:.4f})")
        print(f"  WB Stable (<10%): {'✓' if wb_stable else '✗'} ({abs(last.wb_change):.4f})")
        
        all_pass = ratio_maintained and wb_stable
        print(f"\n  总体: {'✓ 通过' if all_pass else '✗ 未通过'}")
        
        return {
            'ratio_maintained': ratio_maintained,
            'wb_stable': wb_stable,
            'all_pass': all_pass,
            'final_wb_change': last.wb_change,
            'final_grad_ratio': last.grad_ratio,
        }


def run_correct_validation(num_steps: int = 100):
    """运行正确验证"""
    print("="*70)
    print("Dual Guard 正确验证 (修复梯度计算)")
    print("="*70)
    
    checkpoint_steps = [20, 40, 60, 80, 100]
    
    # 测试不同 beta 值
    beta_values = [0.05, 0.2, 0.5, 1.0]
    results = {}
    
    for beta in beta_values:
        print(f"\n{'='*70}")
        print(f"测试 beta={beta}")
        print(f"{'='*70}")
        
        torch.manual_seed(42)
        system = Stage6SystemOrchestrator(
            experiment_id=f'beta_{beta}_test',
            custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
        )
        system.establish_baseline()
        
        model = system.orchestrator.backbone.get_model()
        
        # 创建 Feature Guard
        feature_guard = build_subspace_projection_guard(model, beta=beta, top_k_dims=64)
        feature_guard.capture_reference_features()
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        validator = CorrectValidator(model, feature_guard, optimizer, f"beta={beta}")
        validator.set_baseline()
        validator.train_and_record(num_steps, checkpoint_steps)
        results[f'beta_{beta}'] = validator.print_results()
    
    # 对比总结
    print(f"\n{'='*70}")
    print("对比总结")
    print(f"{'='*70}")
    print(f"\n{'Beta':<10} {'Grad Ratio':<15} {'WB Change':<15} {'通过':<10}")
    print(f"{'-'*50}")
    
    for beta in beta_values:
        r = results[f'beta_{beta}']
        print(f"{beta:<10} {r['final_grad_ratio']:<15.4f} {r['final_wb_change']:<15.4f} "
              f"{'✓' if r['all_pass'] else '✗':<10}")
    
    # 找出最佳配置
    best_beta = None
    best_score = -1
    for beta in beta_values:
        r = results[f'beta_{beta}']
        if r['all_pass']:
            score = r['final_grad_ratio'] - abs(r['final_wb_change'])
            if score > best_score:
                best_score = score
                best_beta = beta
    
    print(f"\n{'='*70}")
    if best_beta:
        print(f"✓ 最佳配置: beta={best_beta}")
    else:
        print("✗ 所有配置均未通过，需要进一步调整")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    results = run_correct_validation(num_steps=100)
