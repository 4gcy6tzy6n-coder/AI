"""
Feature Guard 诊断脚本

目标: 回答3个问题
1. 参考特征是否一直固定且可访问
2. Feature Guard loss 是否在每个阶段都真实进入总 loss
3. 它的梯度和约束强度是否在训练后期退化为接近0

诊断维度:
- 参考特征状态 (hash/norm/drift)
- Guard loss 参与情况
- 梯度分布
- Anchor样本一致性
- 中间checkpoint趋势
"""

import torch
import torch.nn.functional as F
import copy
import hashlib
from typing import Dict, List
from dataclasses import dataclass
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_runtime_orchestrator import Candidate
from stage7_dual_guard_orchestrator import DualGuardPromoter, DualGuardConfig


@dataclass
class DiagnosisRecord:
    """诊断记录"""
    step: int
    main_loss: float
    wb_loss: float
    guard_loss: float
    total_loss: float
    ref_feature_norm: float
    curr_feature_norm: float
    feature_drift: float
    output_kl: float
    wb_score: float
    guard_grad_norm: float
    backbone_grad_norm: float
    
    def to_row(self) -> str:
        """输出表格行"""
        ratio = self.guard_grad_norm / self.backbone_grad_norm if self.backbone_grad_norm > 0 else 0
        return (f"{self.step:<6} {self.main_loss:<12.6f} {self.wb_loss:<12.6f} {self.guard_loss:<14.6f} "
                f"{self.total_loss:<12.6f} {self.feature_drift:<12.6f} {self.output_kl:<10.6f} "
                f"{self.wb_score:<12.4f} {self.guard_grad_norm:<14.6f} {self.backbone_grad_norm:<16.6f} "
                f"{ratio:<10.4f}")


class FeatureGuardDiagnoser:
    """Feature Guard 诊断器"""
    
    def __init__(self, promoter: DualGuardPromoter):
        self.promoter = promoter
        self.feature_guard = promoter.feature_guard
        self.model = promoter.model
        
        # 诊断记录
        self.records: List[DiagnosisRecord] = []
        
        # 锚点样本 (固定)
        torch.manual_seed(42)
        self.anchor_inputs = [
            torch.randint(0, 10000, (1, 50))
            for _ in range(30)
        ]
        
        # 基线输出缓存
        self.baseline_outputs = None
        self.baseline_wb_score = None
    
    def set_baseline(self):
        """设置基线"""
        self.model.eval()
        with torch.no_grad():
            # 缓存基线输出
            self.baseline_outputs = []
            wb_scores = []
            
            for input_ids in self.anchor_inputs:
                outputs = self.model(input_ids)
                self.baseline_outputs.append(outputs['writeback_logits'].clone())
                
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            
            self.baseline_wb_score = sum(wb_scores) / len(wb_scores)
        
        print(f"[诊断] 基线已设置: wb_score={self.baseline_wb_score:.4f}")
    
    def diagnose_step(self, step: int) -> DiagnosisRecord:
        """诊断单步状态"""
        self.model.train()
        
        # 1. 前向传播并记录各项loss
        torch.manual_seed(42 + step * 100)
        input_ids = torch.randint(0, 10000, (1, 50))
        
        outputs = self.model(input_ids)
        
        # Main loss
        main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
        
        # Writeback loss
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
        
        # Guard loss
        guard_loss = self.feature_guard.compute_feature_guard_loss()
        
        # Total loss
        total_loss = main_loss + 0.1 * wb_loss + guard_loss
        
        # 2. 参考特征状态
        ref_norm = self.feature_guard.ref_features.norm().item() if self.feature_guard.ref_features is not None else 0
        
        # 当前特征
        curr_features_list = []
        with torch.no_grad():
            for anchor_input in self.anchor_inputs[:5]:  # 采样部分
                feats = self.feature_guard._extract_backbone_features(anchor_input)
                curr_features_list.append(feats)
        curr_features = torch.cat(curr_features_list, dim=0)
        curr_norm = curr_features.norm().item()
        
        # 特征漂移 (简化计算)
        feature_drift = 0
        if self.feature_guard.ref_features is not None:
            ref_sample = self.feature_guard.ref_features[:curr_features.shape[0]]
            feature_drift = (curr_features - ref_sample).norm().item()
        
        # 3. 计算梯度
        self.promoter.optimizer.zero_grad()
        total_loss.backward()
        
        # Guard-related grad norm
        guard_grad_norm = 0
        backbone_grad_norm = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'writeback' in name.lower():
                    guard_grad_norm += grad_norm
                else:
                    backbone_grad_norm += grad_norm
        
        # 4. 计算输出KL和writeback score
        self.model.eval()
        with torch.no_grad():
            # Output KL
            total_kl = 0
            for i, input_ids in enumerate(self.anchor_inputs):
                outputs = self.model(input_ids)
                curr_logits = outputs['writeback_logits']
                base_logits = self.baseline_outputs[i]
                
                kl = F.kl_div(
                    F.log_softmax(curr_logits, dim=-1),
                    F.softmax(base_logits, dim=-1),
                    reduction='batchmean'
                ).item()
                total_kl += kl
            output_kl = total_kl / len(self.anchor_inputs)
            
            # Writeback score
            wb_scores = []
            for input_ids in self.anchor_inputs:
                outputs = self.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            wb_score = sum(wb_scores) / len(wb_scores)
        
        record = DiagnosisRecord(
            step=step,
            main_loss=main_loss.item(),
            wb_loss=wb_loss.item(),
            guard_loss=guard_loss.item(),
            total_loss=total_loss.item(),
            ref_feature_norm=ref_norm,
            curr_feature_norm=curr_norm,
            feature_drift=feature_drift,
            output_kl=output_kl,
            wb_score=wb_score,
            guard_grad_norm=guard_grad_norm,
            backbone_grad_norm=backbone_grad_norm,
        )
        
        return record
    
    def print_header(self):
        """打印表头"""
        print("\n" + "="*150)
        print("Feature Guard 诊断表")
        print("="*150)
        print(f"{'Step':<6} {'Main Loss':<12} {'WB Loss':<12} {'Guard Loss':<14} {'Total Loss':<12} "
              f"{'Feat Drift':<12} {'Output KL':<10} {'WB Score':<12} {'Guard Grad':<14} {'Backbone Grad':<16} {'Ratio':<10}")
        print("-"*150)
    
    def print_records(self):
        """打印所有记录"""
        self.print_header()
        for record in self.records:
            print(record.to_row())
        print("="*150)


def run_diagnosis(num_steps: int = 100, checkpoint_steps: List[int] = None):
    """运行完整诊断"""
    if checkpoint_steps is None:
        checkpoint_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("="*70)
    print("Feature Guard 完整诊断")
    print("="*70)
    
    # 1. 创建系统
    print("\n[1/4] 初始化系统...")
    torch.manual_seed(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id='fg_diagnosis',
        custom_config=config,
    )
    system.establish_baseline()
    
    # 2. 设置 Dual Guard
    print("\n[2/4] 设置 Dual Guard...")
    model = system.orchestrator.backbone.get_model()
    
    dual_config = DualGuardConfig(
        feature_guard_beta=0.05,
        head_lr_ratio=2.0,
        lambda_wb=0.1,
    )
    
    promoter = DualGuardPromoter(
        model,
        dual_config,
        enable_feature_guard=True,
        enable_head_training=True,
    )
    
    system.orchestrator.param_promoter = promoter
    
    # 3. 创建诊断器
    print("\n[3/4] 创建诊断器...")
    diagnoser = FeatureGuardDiagnoser(promoter)
    diagnoser.set_baseline()
    
    # 4. 运行训练并诊断
    print(f"\n[4/4] 运行 {num_steps} 步训练诊断...")
    
    baseline = system.protocol.baseline.scores
    baseline_abilities = {
        'target': baseline.target_score,
        'retrieval': baseline.retrieval_score,
        'policy': baseline.policy_score,
        'governance': baseline.governance_score,
        'writeback': baseline.writeback_score,
    }
    
    for step in range(num_steps):
        # 训练
        candidate = Candidate(
            candidate_id=f'step_{step}',
            candidate_type='TEST',
            content=f'print("Step {step}")',
            entities={},
            metadata={'fitness': 0.5 + step * 0.001},
        )
        
        # 执行一步训练
        for epoch in range(3):
            promoter.optimizer.zero_grad()
            
            torch.manual_seed(42 + step * 100 + epoch)
            input_ids = torch.randint(0, 10000, (1, 50))
            
            outputs = model(input_ids)
            main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
            
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
            
            guard_loss = promoter.feature_guard.compute_feature_guard_loss()
            total_loss = main_loss + 0.1 * wb_loss + guard_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            promoter.optimizer.step()
        
        # 在检查点诊断
        if (step + 1) in checkpoint_steps:
            record = diagnoser.diagnose_step(step + 1)
            diagnoser.records.append(record)
            print(f"  Step {step+1} 诊断完成")
    
    # 5. 输出诊断结果
    print("\n[5/5] 诊断结果")
    diagnoser.print_records()
    
    # 6. 关键发现
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    # 分析趋势
    first_record = diagnoser.records[0]
    last_record = diagnoser.records[-1]
    
    print(f"\n1. Guard Loss 趋势:")
    print(f"   初始: {first_record.guard_loss:.6f}")
    print(f"   最终: {last_record.guard_loss:.6f}")
    print(f"   变化: {last_record.guard_loss - first_record.guard_loss:+.6f}")
    
    if last_record.guard_loss < first_record.guard_loss * 0.1:
        print("   ⚠ Guard Loss 严重退化!")
    elif last_record.guard_loss < 0.001:
        print("   ⚠ Guard Loss 接近0，可能失效!")
    else:
        print("   ✓ Guard Loss 保持有效")
    
    print(f"\n2. 梯度比例趋势:")
    first_ratio = first_record.guard_grad_norm / first_record.backbone_grad_norm if first_record.backbone_grad_norm > 0 else 0
    last_ratio = last_record.guard_grad_norm / last_record.backbone_grad_norm if last_record.backbone_grad_norm > 0 else 0
    print(f"   初始 Guard/Backbone 梯度比: {first_ratio:.4f}")
    print(f"   最终 Guard/Backbone 梯度比: {last_ratio:.4f}")
    
    if last_ratio < first_ratio * 0.1:
        print("   ⚠ Guard 梯度被主任务淹没!")
    else:
        print("   ✓ Guard 梯度保持有效")
    
    print(f"\n3. Feature Drift 趋势:")
    print(f"   初始: {first_record.feature_drift:.6f}")
    print(f"   最终: {last_record.feature_drift:.6f}")
    print(f"   增长: {last_record.feature_drift - first_record.feature_drift:+.6f}")
    
    if last_record.feature_drift > first_record.feature_drift * 10:
        print("   ⚠ Feature Drift 严重增长!")
    else:
        print("   ✓ Feature Drift 可控")
    
    print(f"\n4. Writeback Score 趋势:")
    print(f"   基线: {diagnoser.baseline_wb_score:.4f}")
    print(f"   最终: {last_record.wb_score:.4f}")
    print(f"   变化: {last_record.wb_score - diagnoser.baseline_wb_score:+.4f}")
    
    # 根因判断
    print("\n" + "="*70)
    print("根因判断")
    print("="*70)
    
    if last_record.guard_loss < 0.001:
        print("最可能根因: Guard Loss 退化至接近0")
        print("建议: 检查 beta 权重或 loss 计算逻辑")
    elif last_ratio < 0.01:
        print("最可能根因: Guard 梯度被主任务梯度淹没")
        print("建议: 增加 beta 权重或调整梯度裁剪策略")
    elif last_record.feature_drift > 1.0:
        print("最可能根因: Feature Drift 超出约束范围")
        print("建议: 考虑动态 reference 或 EMA 更新")
    else:
        print("未发现明显根因，建议检查其他维度")
    
    return diagnoser.records


if __name__ == "__main__":
    records = run_diagnosis(num_steps=100)
    
    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
