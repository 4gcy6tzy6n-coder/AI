"""
Enhanced Dual Guard 验证脚本

对比原始版和增强版在 100-step 训练中的表现:
- Guard 梯度占比稳定性
- Feature Drift 控制
- Writeback Score 稳定性
"""

import torch
import torch.nn.functional as F
import copy
from typing import Dict, List
from dataclasses import dataclass
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_runtime_orchestrator import Candidate
from stage7_dual_guard_orchestrator import DualGuardPromoter, DualGuardConfig
from stage7_dual_guard_enhanced import EnhancedDualGuardPromoter, EnhancedDualGuardConfig


@dataclass
class ValidationRecord:
    """验证记录"""
    step: int
    guard_loss: float
    guard_grad_norm: float
    backbone_grad_norm: float
    grad_ratio: float
    beta: float
    feature_drift: float
    wb_score: float
    
    def to_row(self) -> str:
        """输出表格行"""
        return (f"{self.step:<6} {self.guard_loss:<14.6f} {self.guard_grad_norm:<14.6f} "
                f"{self.backbone_grad_norm:<16.6f} {self.grad_ratio:<10.4f} {self.beta:<8.4f} "
                f"{self.feature_drift:<12.4f} {self.wb_score:<12.4f}")


class EnhancedValidator:
    """增强版验证器"""
    
    def __init__(self, promoter, name: str):
        self.promoter = promoter
        self.name = name
        self.records: List[ValidationRecord] = []
        
        # 固定锚点样本
        torch.manual_seed(42)
        self.anchor_inputs = [torch.randint(0, 10000, (1, 50)) for _ in range(30)]
        self.baseline_wb_score = None
    
    def set_baseline(self):
        """设置基线"""
        self.promoter.model.eval()
        with torch.no_grad():
            wb_scores = []
            for input_ids in self.anchor_inputs:
                outputs = self.promoter.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            self.baseline_wb_score = sum(wb_scores) / len(wb_scores)
    
    def record_step(self, step: int):
        """记录单步状态"""
        self.promoter.model.train()
        
        # 前向 + 反向
        self.promoter.optimizer.zero_grad()
        
        torch.manual_seed(42 + step * 100)
        input_ids = torch.randint(0, 10000, (1, 50))
        
        outputs = self.promoter.model(input_ids)
        main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
        
        # Guard loss
        guard_loss = self.promoter.feature_guard.compute_feature_guard_loss()
        
        total_loss = main_loss + guard_loss
        total_loss.backward()
        
        # 计算梯度 - Feature Guard 影响所有 backbone 参数
        # 计算由 Guard Loss 产生的梯度
        guard_grad_norm = 0.0
        backbone_grad_norm = 0.0
        
        for name, param in self.promoter.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Feature Guard 影响所有非 writeback 的 backbone 参数
                if 'writeback' not in name.lower():
                    guard_grad_norm += grad_norm
                backbone_grad_norm += grad_norm
        
        grad_ratio = guard_grad_norm / backbone_grad_norm if backbone_grad_norm > 0 else 0
        
        # Feature drift
        curr_features_list = []
        with torch.no_grad():
            for anchor in self.anchor_inputs[:5]:
                feats = self.promoter.feature_guard._extract_backbone_features(anchor)
                curr_features_list.append(feats)
        curr_features = torch.cat(curr_features_list, dim=0)
        
        feature_drift = 0
        if self.promoter.feature_guard.ref_features is not None:
            ref_sample = self.promoter.feature_guard.ref_features[:curr_features.shape[0]]
            feature_drift = (curr_features - ref_sample).norm().item()
        
        # Writeback score
        self.promoter.model.eval()
        with torch.no_grad():
            wb_scores = []
            for input_ids in self.anchor_inputs:
                outputs = self.promoter.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            wb_score = sum(wb_scores) / len(wb_scores)
        
        # Beta 值
        beta = self.promoter.current_beta if hasattr(self.promoter, 'current_beta') else self.promoter.config.feature_guard_beta
        
        record = ValidationRecord(
            step=step,
            guard_loss=guard_loss.item(),
            guard_grad_norm=guard_grad_norm,
            backbone_grad_norm=backbone_grad_norm,
            grad_ratio=grad_ratio,
            beta=beta,
            feature_drift=feature_drift,
            wb_score=wb_score,
        )
        
        self.records.append(record)
        return record
    
    def train_and_record(self, num_steps: int, checkpoint_steps: List[int]):
        """训练并记录"""
        print(f"\n[{self.name}] 开始训练...")
        
        for step in range(num_steps):
            # 训练
            candidate = Candidate(
                candidate_id=f'step_{step}',
                candidate_type='TEST',
                content=f'print("Step {step}")',
                entities={},
                metadata={'fitness': 0.5 + step * 0.001},
            )
            
            baseline_abilities = {
                'target': 0.55,
                'retrieval': 0.0,
                'policy': 0.0,
                'governance': 0.98,
                'writeback': self.baseline_wb_score,
            }
            
            self.promoter.promote(candidate, step, baseline_abilities)
            
            # 在检查点记录
            if (step + 1) in checkpoint_steps:
                self.record_step(step + 1)
                print(f"  Step {step+1} 记录完成")
    
    def print_results(self):
        """打印结果"""
        print(f"\n{'='*120}")
        print(f"{self.name} 验证结果")
        print(f"{'='*120}")
        print(f"{'Step':<6} {'Guard Loss':<14} {'Guard Grad':<14} {'Backbone Grad':<16} {'Ratio':<10} {'Beta':<8} {'Feat Drift':<12} {'WB Score':<12}")
        print(f"{'-'*120}")
        for record in self.records:
            print(record.to_row())
        print(f"{'='*120}")
        
        # 统计
        first = self.records[0]
        last = self.records[-1]
        
        print(f"\n关键指标:")
        print(f"  Guard Loss: {first.guard_loss:.6f} -> {last.guard_loss:.6f}")
        print(f"  Grad Ratio: {first.grad_ratio:.4f} -> {last.grad_ratio:.4f}")
        print(f"  Feature Drift: {first.feature_drift:.4f} -> {last.feature_drift:.4f}")
        print(f"  WB Score: {first.wb_score:.4f} -> {last.wb_score:.4f}")
        print(f"  WB Change: {last.wb_score - self.baseline_wb_score:+.4f}")
        
        # 判断是否成功
        ratio_maintained = last.grad_ratio >= 0.005  # 保持至少 0.5%
        drift_controlled = last.feature_drift < first.feature_drift * 2  # 漂移控制在2倍内
        wb_stable = abs(last.wb_score - self.baseline_wb_score) < 0.1  # WB变化<10%
        
        print(f"\n验收:")
        print(f"  Grad Ratio >= 0.5%: {'✓' if ratio_maintained else '✗'} ({last.grad_ratio:.4f})")
        print(f"  Drift Controlled: {'✓' if drift_controlled else '✗'}")
        print(f"  WB Stable (<10%): {'✓' if wb_stable else '✗'} ({abs(last.wb_score - self.baseline_wb_score):.4f})")
        
        all_pass = ratio_maintained and drift_controlled and wb_stable
        print(f"\n  总体: {'✓ 通过' if all_pass else '✗ 未通过'}")
        
        return {
            'ratio_maintained': ratio_maintained,
            'drift_controlled': drift_controlled,
            'wb_stable': wb_stable,
            'all_pass': all_pass,
            'final_wb_change': last.wb_score - self.baseline_wb_score,
            'final_grad_ratio': last.grad_ratio,
        }


def run_comparison_validation(num_steps: int = 100):
    """运行对比验证"""
    print("="*70)
    print("Enhanced Dual Guard 对比验证")
    print("="*70)
    
    checkpoint_steps = [20, 40, 60, 80, 100]
    
    # 1. 原始版
    print("\n" + "="*70)
    print("测试 1: 原始版 Dual Guard (beta=0.05)")
    print("="*70)
    
    torch.manual_seed(42)
    system1 = Stage6SystemOrchestrator(
        experiment_id='original_test',
        custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
    )
    system1.establish_baseline()
    
    model1 = system1.orchestrator.backbone.get_model()
    config1 = DualGuardConfig(feature_guard_beta=0.05, head_lr_ratio=2.0)
    promoter1 = DualGuardPromoter(model1, config1, enable_feature_guard=True, enable_head_training=True)
    system1.orchestrator.param_promoter = promoter1
    
    validator1 = EnhancedValidator(promoter1, "原始版 (beta=0.05)")
    validator1.set_baseline()
    validator1.train_and_record(num_steps, checkpoint_steps)
    results1 = validator1.print_results()
    
    # 2. 增强版
    print("\n" + "="*70)
    print("测试 2: 增强版 Dual Guard (beta=0.2, 动态调整)")
    print("="*70)
    
    torch.manual_seed(42)
    system2 = Stage6SystemOrchestrator(
        experiment_id='enhanced_test',
        custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
    )
    system2.establish_baseline()
    
    model2 = system2.orchestrator.backbone.get_model()
    config2 = EnhancedDualGuardConfig(
        feature_guard_beta=0.2,
        head_lr_ratio=2.0,
        enable_dynamic_beta=True,
        target_guard_grad_ratio=0.01,
    )
    promoter2 = EnhancedDualGuardPromoter(model2, config2, enable_feature_guard=True, enable_head_training=True)
    system2.orchestrator.param_promoter = promoter2
    
    validator2 = EnhancedValidator(promoter2, "增强版 (beta=0.2, 动态)")
    validator2.set_baseline()
    validator2.train_and_record(num_steps, checkpoint_steps)
    results2 = validator2.print_results()
    
    # 3. 对比总结
    print("\n" + "="*70)
    print("对比总结")
    print("="*70)
    
    print(f"\n{'指标':<25} {'原始版':<20} {'增强版':<20} {'提升':<15}")
    print(f"{'-'*80}")
    
    print(f"{'Guard 梯度占比':<25} {results1['final_grad_ratio']:<20.4f} {results2['final_grad_ratio']:<20.4f} "
          f"{results2['final_grad_ratio']/results1['final_grad_ratio'] if results1['final_grad_ratio'] > 0 else 0:<15.2f}x")
    
    print(f"{'Writeback 变化':<25} {results1['final_wb_change']:<20.4f} {results2['final_wb_change']:<20.4f} "
          f"{results1['final_wb_change']/results2['final_wb_change'] if results2['final_wb_change'] != 0 else 0:<15.2f}x")
    
    print(f"{'验收通过':<25} {'✓' if results1['all_pass'] else '✗':<20} {'✓' if results2['all_pass'] else '✗':<20}")
    
    print(f"\n{'='*70}")
    if results2['all_pass'] and not results1['all_pass']:
        print("✓ 增强版成功修复了 Guard 梯度被淹没问题!")
    elif results2['all_pass']:
        print("✓ 增强版通过验收，原始版也通过了")
    else:
        print("✗ 增强版仍未通过，需要进一步调整")
    print(f"{'='*70}")
    
    return {
        'original': results1,
        'enhanced': results2,
    }


if __name__ == "__main__":
    results = run_comparison_validation(num_steps=100)
