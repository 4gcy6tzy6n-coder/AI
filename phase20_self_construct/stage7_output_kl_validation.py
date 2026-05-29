"""
Output KL Guard 验证脚本

对比 Feature Guard 和 Output KL Guard 的效果
"""

import torch
import torch.nn.functional as F
import copy
from typing import Dict, List
from dataclasses import dataclass
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_runtime_orchestrator import Candidate
from stage7_backbone_feature_guard import build_subspace_projection_guard
from stage7_output_kl_guard import build_output_kl_guard


@dataclass
class ValidationRecord:
    """验证记录"""
    step: int
    guard_loss: float
    guard_grad_norm: float
    main_grad_norm: float
    grad_ratio: float
    output_drift: float
    wb_score: float
    wb_change: float
    
    def to_row(self) -> str:
        return (f"{self.step:<6} {self.guard_loss:<14.6f} {self.guard_grad_norm:<14.6f} "
                f"{self.main_grad_norm:<14.6f} {self.grad_ratio:<10.4f} {self.output_drift:<12.6f} "
                f"{self.wb_score:<12.4f} {self.wb_change:<12.4f}")


class OutputKLValidator:
    """Output KL 验证器"""
    
    def __init__(self, model, guard, optimizer, name: str, guard_type: str):
        self.model = model
        self.guard = guard
        self.optimizer = optimizer
        self.name = name
        self.guard_type = guard_type  # 'feature' or 'output_kl'
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
        """记录单步状态"""
        self.model.train()
        
        # 1. 只计算 Guard Loss 的梯度
        self.optimizer.zero_grad()
        if self.guard_type == 'feature':
            guard_loss = self.guard.compute_feature_guard_loss()
        else:
            guard_loss = self.guard.compute_kl_guard_loss()
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
        
        # 3. 计算输出漂移
        if self.guard_type == 'feature':
            # Feature guard 没有直接的 output drift 方法，用 WB score 变化近似
            output_drift = 0.0
        else:
            output_drift = self.guard.compute_output_drift()
        
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
            guard_loss=guard_loss.item(),
            guard_grad_norm=guard_grad_norm,
            main_grad_norm=main_grad_norm,
            grad_ratio=grad_ratio,
            output_drift=output_drift,
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
                if self.guard_type == 'feature':
                    guard_loss = self.guard.compute_feature_guard_loss()
                else:
                    guard_loss = self.guard.compute_kl_guard_loss()
                
                total_loss = main_loss + guard_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # 在检查点记录
            if (step + 1) in checkpoint_steps:
                self.record_step(step + 1)
                r = self.records[-1]
                print(f"  Step {step+1}: ratio={r.grad_ratio:.4f}, drift={r.output_drift:.4f}, wb_change={r.wb_change:+.4f}")
    
    def print_results(self):
        """打印结果"""
        print(f"\n{'='*130}")
        print(f"{self.name} 验证结果")
        print(f"{'='*130}")
        print(f"{'Step':<6} {'Guard Loss':<14} {'Guard Grad':<14} {'Main Grad':<14} "
              f"{'Ratio':<10} {'Output Drift':<12} {'WB Score':<12} {'WB Change':<12}")
        print(f"{'-'*130}")
        for record in self.records:
            print(record.to_row())
        print(f"{'='*130}")
        
        # 统计
        first = self.records[0]
        last = self.records[-1]
        
        print(f"\n关键指标:")
        print(f"  Guard Loss: {first.guard_loss:.6f} -> {last.guard_loss:.6f}")
        print(f"  Guard Grad: {first.guard_grad_norm:.6f} -> {last.guard_grad_norm:.6f}")
        print(f"  Main Grad: {first.main_grad_norm:.6f} -> {last.main_grad_norm:.6f}")
        print(f"  Grad Ratio: {first.grad_ratio:.4f} -> {last.grad_ratio:.4f}")
        if self.guard_type == 'output_kl':
            print(f"  Output Drift: {first.output_drift:.6f} -> {last.output_drift:.6f}")
        print(f"  WB Score: {first.wb_score:.4f} -> {last.wb_score:.4f}")
        print(f"  WB Change: {last.wb_change:+.4f}")
        
        # 判断是否成功
        ratio_maintained = last.grad_ratio >= 0.005  # 保持至少 0.5%
        wb_stable = abs(last.wb_change) < 0.1  # WB变化<10%
        
        print(f"\n验收:")
        print(f"  Grad Ratio >= 0.5%: {'✓' if ratio_maintained else '✗'} ({last.grad_ratio:.4f})")
        print(f"  WB Stable (<10%): {'✓' if wb_stable else '✗'} ({abs(last.wb_change):.4f})")
        
        all_pass = ratio_maintained and wb_stable
        print(f"\n  总体: {'✓ 通过' if all_pass else '✗ 未通过'}")
        
        return {
            'ratio_maintained': ratio_maintained,
            'wb_stable': wb_stable,
            'all_pass': all_pass,
            'final_wb_change': last.wb_change,
            'final_grad_ratio': last.grad_ratio,
            'final_output_drift': last.output_drift if self.guard_type == 'output_kl' else 0,
        }


def run_output_kl_validation(num_steps: int = 100):
    """运行 Output KL 验证"""
    print("="*70)
    print("Output KL Guard vs Feature Guard 对比验证")
    print("="*70)
    
    checkpoint_steps = [20, 40, 60, 80, 100]
    
    # 测试 1: Feature Guard (beta=0.2)
    print(f"\n{'='*70}")
    print("测试 1: Feature Guard (beta=0.2)")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    system1 = Stage6SystemOrchestrator(
        experiment_id='feature_guard_test',
        custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
    )
    system1.establish_baseline()
    
    model1 = system1.orchestrator.backbone.get_model()
    feature_guard = build_subspace_projection_guard(model1, beta=0.2, top_k_dims=64)
    feature_guard.capture_reference_features()
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
    
    validator1 = OutputKLValidator(model1, feature_guard, optimizer1, "Feature Guard", "feature")
    validator1.set_baseline()
    validator1.train_and_record(num_steps, checkpoint_steps)
    results1 = validator1.print_results()
    
    # 测试 2: Output KL Guard (beta=0.1)
    print(f"\n{'='*70}")
    print("测试 2: Output KL Guard (beta=0.1)")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    system2 = Stage6SystemOrchestrator(
        experiment_id='output_kl_test',
        custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
    )
    system2.establish_baseline()
    
    model2 = system2.orchestrator.backbone.get_model()
    output_kl_guard = build_output_kl_guard(model2, beta=0.1, num_samples=30, use_probs=True)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    
    validator2 = OutputKLValidator(model2, output_kl_guard, optimizer2, "Output KL Guard", "output_kl")
    validator2.set_baseline()
    validator2.train_and_record(num_steps, checkpoint_steps)
    results2 = validator2.print_results()
    
    # 测试 3: Output KL Guard (beta=0.2)
    print(f"\n{'='*70}")
    print("测试 3: Output KL Guard (beta=0.2)")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    system3 = Stage6SystemOrchestrator(
        experiment_id='output_kl_test2',
        custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
    )
    system3.establish_baseline()
    
    model3 = system3.orchestrator.backbone.get_model()
    output_kl_guard2 = build_output_kl_guard(model3, beta=0.2, num_samples=30, use_probs=True)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=1e-4)
    
    validator3 = OutputKLValidator(model3, output_kl_guard2, optimizer3, "Output KL Guard (beta=0.2)", "output_kl")
    validator3.set_baseline()
    validator3.train_and_record(num_steps, checkpoint_steps)
    results3 = validator3.print_results()
    
    # 对比总结
    print(f"\n{'='*70}")
    print("对比总结")
    print(f"{'='*70}")
    print(f"\n{'方案':<30} {'Grad Ratio':<15} {'WB Change':<15} {'Output Drift':<15} {'通过':<10}")
    print(f"{'-'*85}")
    
    print(f"{'Feature Guard (beta=0.2)':<30} {results1['final_grad_ratio']:<15.4f} "
          f"{results1['final_wb_change']:<15.4f} {'N/A':<15} {'✓' if results1['all_pass'] else '✗':<10}")
    
    print(f"{'Output KL Guard (beta=0.1)':<30} {results2['final_grad_ratio']:<15.4f} "
          f"{results2['final_wb_change']:<15.4f} {results2['final_output_drift']:<15.6f} "
          f"{'✓' if results2['all_pass'] else '✗':<10}")
    
    print(f"{'Output KL Guard (beta=0.2)':<30} {results3['final_grad_ratio']:<15.4f} "
          f"{results3['final_wb_change']:<15.4f} {results3['final_output_drift']:<15.6f} "
          f"{'✓' if results3['all_pass'] else '✗':<10}")
    
    # 找出最佳方案
    best_config = None
    best_score = -1
    configs = [
        ('Feature Guard (beta=0.2)', results1),
        ('Output KL Guard (beta=0.1)', results2),
        ('Output KL Guard (beta=0.2)', results3),
    ]
    
    for name, r in configs:
        if r['all_pass']:
            score = r['final_grad_ratio'] - abs(r['final_wb_change'])
            if score > best_score:
                best_score = score
                best_config = name
    
    print(f"\n{'='*70}")
    if best_config:
        print(f"✓ 最佳配置: {best_config}")
    else:
        print("✗ 所有配置均未通过，需要进一步调整")
        print("\n建议:")
        print("  1. 进一步提高 Output KL Guard 的 beta")
        print("  2. 尝试混合方案 (Feature + Output KL)")
        print("  3. 调整学习率或训练步数")
    print(f"{'='*70}")
    
    return {
        'feature_guard': results1,
        'output_kl_01': results2,
        'output_kl_02': results3,
    }


if __name__ == "__main__":
    results = run_output_kl_validation(num_steps=100)
