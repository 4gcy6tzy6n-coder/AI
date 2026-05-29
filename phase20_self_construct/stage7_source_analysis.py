"""
Writeback 变化来源拆解实验

4组实验:
A: 冻结head, 训练backbone (freeze - 已有)
B: 冻结backbone, 只训练writeback head
C: 冻结backbone + 冻结head (纯控制组)
D: 训练backbone + head (baseline - 完整对照)

记录3类中间量:
1. head参数漂移
2. backbone特征漂移
3. 输出漂移
"""

import torch
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard


class SourceAnalysisPromoter:
    """专门用于来源分析的promoter"""
    
    def __init__(self, base_promoter, freeze_backbone: bool = False, freeze_head: bool = False):
        self.base_promoter = base_promoter
        self.freeze_backbone = freeze_backbone
        self.freeze_head = freeze_head
        self.model = base_promoter.model
        self.config = base_promoter.config
        self.optimizer = base_promoter.optimizer
        
        # 冻结设置
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_head:
            self._freeze_writeback_head()
    
    def _freeze_backbone(self):
        """冻结backbone参数"""
        for name, param in self.model.named_parameters():
            if 'writeback' not in name.lower():
                param.requires_grad = False
        print("  [SourceAnalysis] Backbone 已冻结")
    
    def _freeze_writeback_head(self):
        """冻结writeback head参数"""
        for name, param in self.model.named_parameters():
            if 'writeback' in name.lower():
                param.requires_grad = False
        print("  [SourceAnalysis] Writeback head 已冻结")
    
    def _unfreeze_all(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def promote(self, candidate, step_num: int, baseline_abilities: Dict):
        """执行训练并记录详细指标"""
        from stage6_runtime_orchestrator import StepMetrics
        from datetime import datetime
        
        self.model.train()
        max_change = self.config.step1_max_change if step_num == 1 else self.config.step2_max_change
        
        # 记录训练前状态
        wb_params_before = {}
        for name, param in self.model.named_parameters():
            if 'writeback' in name.lower():
                wb_params_before[name] = param.clone().detach()
        
        # 记录训练前特征 (使用固定输入)
        test_input = torch.randint(0, 10000, (1, 50))
        with torch.no_grad():
            features_before = self._extract_features(test_input)
            outputs_before = self.model(test_input)
        
        # 训练循环
        metrics = {
            'wb_grad_norms': [],
            'main_losses': [],
            'wb_losses': [],
        }
        
        for epoch in range(self.config.num_epochs):
            self.optimizer.zero_grad()
            
            # Forward
            input_ids = torch.randint(0, 10000, (1, 50))
            outputs = self.model(input_ids)
            
            # 计算损失
            main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
            
            # 检查writeback head是否需要训练
            wb_loss = None
            if not self.freeze_head:
                writeback_probs = outputs['writeback_probs']
                gap_decision = outputs['gap_probs'][0, 1]
                policy_decision = outputs['policy_probs'][0].max()
                writeback_target = torch.tensor(1.0 if (gap_decision > 0.5 and policy_decision > 0.5) else 0.0)
                wb_loss = F.binary_cross_entropy(writeback_probs[0, 1:2], writeback_target.unsqueeze(0))
                total_loss = main_loss + 0.1 * wb_loss
            else:
                total_loss = main_loss
            
            # Backward (只在有需要梯度的参数时)
            if any(p.requires_grad for p in self.model.parameters()):
                total_loss.backward()
                
                # 记录writeback梯度
                wb_grad_norm = 0
                for name, param in self.model.named_parameters():
                    if 'writeback' in name.lower() and param.grad is not None:
                        wb_grad_norm += param.grad.norm().item()
                metrics['wb_grad_norms'].append(wb_grad_norm)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_change)
                
                # Optimizer step
                self.optimizer.step()
            else:
                metrics['wb_grad_norms'].append(0.0)
            
            metrics['main_losses'].append(main_loss.item())
            if wb_loss is not None:
                metrics['wb_losses'].append(wb_loss.item())
        
        # 记录训练后状态
        wb_params_after = {}
        for name, param in self.model.named_parameters():
            if 'writeback' in name.lower():
                wb_params_after[name] = param.clone().detach()
        
        # 记录训练后特征和输出
        with torch.no_grad():
            features_after = self._extract_features(test_input)
            outputs_after = self.model(test_input)
        
        # 计算各类漂移
        drift_analysis = self._compute_drifts(
            wb_params_before, wb_params_after,
            features_before, features_after,
            outputs_before, outputs_after
        )
        
        # 评估
        current_abilities = self._evaluate_abilities()
        
        result = StepMetrics(
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
        
        return result, drift_analysis, metrics
    
    def _extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """提取backbone特征"""
        with torch.no_grad():
            if hasattr(self.model, 'embeddings'):
                return self.model.embeddings(input_ids)
            return input_ids.float()
    
    def _compute_drifts(self, wb_before, wb_after, feat_before, feat_after, out_before, out_after) -> Dict:
        """计算各类漂移"""
        drifts = {}
        
        # 1. 参数漂移
        param_drift = 0
        for name in wb_before:
            delta = (wb_after[name] - wb_before[name]).norm().item()
            param_drift += delta
        drifts['param_drift'] = param_drift
        
        # 2. 特征漂移
        feat_drift = (feat_after - feat_before).norm().item()
        feat_cosine = F.cosine_similarity(feat_after.flatten(), feat_before.flatten(), dim=0).item()
        drifts['feature_l2_drift'] = feat_drift
        drifts['feature_cosine'] = feat_cosine
        
        # 3. 输出漂移
        wb_logits_before = out_before['writeback_logits']
        wb_logits_after = out_after['writeback_logits']
        output_kl = F.kl_div(
            F.log_softmax(wb_logits_after, dim=-1),
            F.softmax(wb_logits_before, dim=-1),
            reduction='batchmean'
        ).item()
        drifts['output_kl'] = output_kl
        
        return drifts
    
    def _evaluate_abilities(self) -> Dict:
        """评估能力"""
        return self.base_promoter._evaluate_abilities()


def run_source_experiment(
    exp_name: str,
    freeze_backbone: bool,
    freeze_head: bool,
    num_steps: int = 10
) -> Dict:
    """运行来源拆解实验"""
    print(f"\n{'='*70}")
    print(f"来源拆解实验: {exp_name}")
    print(f"  Freeze backbone: {freeze_backbone}")
    print(f"  Freeze head: {freeze_head}")
    print(f"  Steps: {num_steps}")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id=f'source_{exp_name}_{num_steps}',
        custom_config=config,
    )
    
    # 先建立基线 (这会初始化 param_promoter)
    system.establish_baseline()
    
    # 替换promoter为分析版本
    original_promoter = system.orchestrator.param_promoter
    if original_promoter is None:
        # 如果还没有初始化，手动创建
        from stage6_runtime_orchestrator import ParamPromoter
        model = system.orchestrator.backbone.get_model()
        original_promoter = ParamPromoter(model, system.orchestrator.config)
        system.orchestrator.param_promoter = original_promoter
    
    analysis_promoter = SourceAnalysisPromoter(
        original_promoter,
        freeze_backbone=freeze_backbone,
        freeze_head=freeze_head
    )
    system.orchestrator.param_promoter = analysis_promoter
    
    # 运行训练
    all_drifts = []
    all_metrics = []
    
    # 获取基线能力
    baseline_scores = system.protocol.baseline.scores
    baseline_abilities = {
        'target': baseline_scores.target_score,
        'retrieval': baseline_scores.retrieval_score,
        'policy': baseline_scores.policy_score,
        'governance': baseline_scores.governance_score,
        'writeback': baseline_scores.writeback_score,
    }
    
    for step in range(num_steps):
        result, drift, metrics = analysis_promoter.promote(
            None, step + 1,
            baseline_abilities
        )
        all_drifts.append(drift)
        all_metrics.append(metrics)
    
    # 最终评估
    eval_result = system.protocol.evaluate_with_protocol(f'source_{exp_name}', num_samples=30)
    baseline = system.protocol.baseline.scores
    
    # 汇总结果
    summary = {
        'exp_name': exp_name,
        'freeze_backbone': freeze_backbone,
        'freeze_head': freeze_head,
        'num_steps': num_steps,
        'target_gain': eval_result.scores.target_score - baseline.target_score,
        'writeback_change': abs(eval_result.scores.writeback_score - baseline.writeback_score),
        'avg_param_drift': sum(d['param_drift'] for d in all_drifts) / len(all_drifts),
        'avg_feature_drift': sum(d['feature_l2_drift'] for d in all_drifts) / len(all_drifts),
        'avg_feature_cosine': sum(d['feature_cosine'] for d in all_drifts) / len(all_drifts),
        'avg_output_kl': sum(d['output_kl'] for d in all_drifts) / len(all_drifts),
        'avg_wb_grad_norm': sum(sum(m['wb_grad_norms']) / len(m['wb_grad_norms']) 
                               for m in all_metrics) / len(all_metrics),
    }
    
    print(f"\n[结果汇总]")
    print(f"  Target gain: {summary['target_gain']:+.2%}")
    print(f"  Writeback change: {summary['writeback_change']:+.2%}")
    print(f"  Param drift: {summary['avg_param_drift']:.6f}")
    print(f"  Feature drift: {summary['avg_feature_drift']:.6f}")
    print(f"  Feature cosine: {summary['avg_feature_cosine']:.6f}")
    print(f"  Output KL: {summary['avg_output_kl']:.6f}")
    print(f"  WB grad norm: {summary['avg_wb_grad_norm']:.6f}")
    
    return summary


def main():
    """主实验"""
    print("="*70)
    print("Writeback 变化来源拆解")
    print("="*70)
    
    results = []
    
    # 4组实验
    experiments = [
        ('A_freeze_head', False, True),      # 冻结head, 训练backbone
        ('B_freeze_backbone', True, False),  # 冻结backbone, 训练head
        ('C_freeze_both', True, True),       # 全冻结
        ('D_train_both', False, False),      # 全训练
    ]
    
    for exp_name, freeze_bb, freeze_head in experiments:
        result = run_source_experiment(exp_name, freeze_bb, freeze_head, num_steps=10)
        results.append(result)
    
    # 对比分析
    print("\n" + "="*70)
    print("来源分析对比")
    print("="*70)
    
    print(f"\n{'实验':<20} {'WB Change':<12} {'Param':<10} {'Feature':<10} {'Cosine':<10} {'Output KL':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['exp_name']:<20} {r['writeback_change']:>10.2%}  "
              f"{r['avg_param_drift']:>8.4f}  {r['avg_feature_drift']:>8.4f}  "
              f"{r['avg_feature_cosine']:>8.4f}  {r['avg_output_kl']:>8.4f}")
    
    # 关键结论
    print("\n" + "="*70)
    print("关键结论")
    print("="*70)
    
    # 找到各组
    exp_a = next(r for r in results if r['exp_name'] == 'A_freeze_head')
    exp_b = next(r for r in results if r['exp_name'] == 'B_freeze_backbone')
    exp_c = next(r for r in results if r['exp_name'] == 'C_freeze_both')
    exp_d = next(r for r in results if r['exp_name'] == 'D_train_both')
    
    print("\n1. 参数漂移来源:")
    print(f"   A (freeze head): param_drift={exp_a['avg_param_drift']:.6f}")
    print(f"   B (freeze backbone): param_drift={exp_b['avg_param_drift']:.6f}")
    print(f"   → 参数漂移主要来自: {'head' if exp_b['avg_param_drift'] > exp_a['avg_param_drift'] else 'backbone'}")
    
    print("\n2. 特征漂移来源:")
    print(f"   A (freeze head): feature_drift={exp_a['avg_feature_drift']:.6f}")
    print(f"   B (freeze backbone): feature_drift={exp_b['avg_feature_drift']:.6f}")
    print(f"   → 特征漂移主要来自: {'backbone训练' if exp_a['avg_feature_drift'] > exp_b['avg_feature_drift'] else 'head训练'}")
    
    print("\n3. Writeback指标漂移来源:")
    print(f"   A (freeze head): writeback_change={exp_a['writeback_change']:.2%}")
    print(f"   B (freeze backbone): writeback_change={exp_b['writeback_change']:.2%}")
    print(f"   C (freeze both): writeback_change={exp_c['writeback_change']:.2%}")
    print(f"   → 指标漂移主因: ", end="")
    
    if exp_c['writeback_change'] < 0.01:
        if exp_a['writeback_change'] > exp_b['writeback_change']:
            print("backbone特征变化")
        else:
            print("head参数变化")
    else:
        print("其他因素(评估噪声/数据变化)")
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*70)
    print("来源拆解完成")
    print("="*70)
