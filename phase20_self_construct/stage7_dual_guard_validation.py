"""
Dual Guard 验证实验

4组50-step对照实验:
1. baseline: 无Guard
2. feature_guard_only: 仅Backbone Feature Guard
3. head_comp_only: 仅Head Compensation
4. dual_guard: 完整双层保护

评估指标:
- writeback score change
- output KL
- target gain
- old ability drop
"""

import torch
import torch.nn.functional as F
import copy
import hashlib
from typing import Dict, List
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage7_dual_guard_orchestrator import (
    DualGuardOrchestrator,
    DualGuardConfig,
    DualGuardPromoter,
)


def compute_model_hash(model) -> str:
    """计算模型参数哈希"""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().tobytes())
    combined = b''.join(params)
    return hashlib.md5(combined).hexdigest()[:16]


class FixedSampleEvaluator:
    """固定样本评估器"""
    
    def __init__(self, num_samples: int = 30, seed: int = 42):
        torch.manual_seed(seed)
        self.fixed_samples = [
            torch.randint(0, 10000, (1, 50))
            for _ in range(num_samples)
        ]
    
    def evaluate(self, model) -> Dict:
        """评估writeback能力"""
        model.eval()
        scores = []
        all_logits = []
        
        with torch.no_grad():
            for input_ids in self.fixed_samples:
                outputs = model(input_ids)
                wb_logits = outputs['writeback_logits']
                wb_probs = F.softmax(wb_logits, dim=-1)
                
                score = wb_probs[0, 1].item() if wb_probs.shape[1] > 1 else wb_probs[0, 0].item()
                scores.append(score)
                all_logits.append(wb_logits.clone())
        
        return {
            'writeback_score': sum(scores) / len(scores),
            'all_logits': all_logits,
        }


def run_dual_guard_experiment(
    exp_name: str,
    guard_mode: str,
    base_state: Dict,
    base_system: Stage6SystemOrchestrator,
    num_steps: int = 50,
) -> Dict:
    """
    运行单组Dual Guard实验
    
    Args:
        exp_name: 实验名称
        guard_mode: "none", "feature_only", "head_only", "dual"
        base_state: 基线模型状态
        base_system: 基线系统 (用于评估)
        num_steps: 训练步数
    """
    print(f"\n{'='*70}")
    print(f"实验: {exp_name}")
    print(f"  Guard Mode: {guard_mode}")
    print(f"  Steps: {num_steps}")
    print(f"{'='*70}")
    
    # 重置模型到基线状态
    model = base_system.orchestrator.backbone.get_model()
    model.load_state_dict(base_state)
    
    # 重置所有参数的requires_grad
    for param in model.parameters():
        param.requires_grad = True
    
    # 验证初始哈希
    initial_hash = compute_model_hash(model)
    print(f"\n[初始模型哈希] {initial_hash}")
    
    # 创建评估器
    evaluator = FixedSampleEvaluator()
    
    # 训练前评估
    print("\n[训练前评估]")
    before_eval = evaluator.evaluate(model)
    print(f"  Writeback score: {before_eval['writeback_score']:.4f}")
    
    # 根据guard_mode设置训练
    if guard_mode == "none":
        # Baseline: 正常训练
        print("\n[训练模式] Baseline (无Guard)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        enable_wb_loss = True
        enable_guard = False
        
    elif guard_mode == "feature_only":
        # 仅Feature Guard
        print("\n[训练模式] Feature Guard Only")
        from stage7_backbone_feature_guard import build_subspace_projection_guard
        
        feature_guard = build_subspace_projection_guard(model, beta=0.05, top_k_dims=64)
        feature_guard.capture_reference_features()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        enable_wb_loss = False  # 不训练head
        enable_guard = True
        
    elif guard_mode == "head_only":
        # 仅Head Compensation
        print("\n[训练模式] Head Compensation Only")
        
        # 分离优化器
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'writeback' in name.lower():
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4},
            {'params': head_params, 'lr': 3e-5},  # 较小学习率
        ])
        enable_wb_loss = True
        enable_guard = False
        
    else:  # dual
        # 完整Dual Guard
        print("\n[训练模式] Dual Guard")
        
        dual_config = DualGuardConfig(
            feature_guard_beta=0.05,
            head_lr_ratio=0.3,
            lambda_wb=0.1,
        )
        
        promoter = DualGuardPromoter(
            model,
            dual_config,
            enable_feature_guard=True,
            enable_head_training=True,
        )
        
        # 使用promoter的训练逻辑
        optimizer = promoter.optimizer
        enable_wb_loss = True
        enable_guard = True
        feature_guard = promoter.feature_guard if enable_guard else None
    
    # 训练循环
    print(f"\n[训练 {num_steps} steps]")
    
    for step in range(num_steps):
        model.train()
        
        for epoch in range(3):  # 每step 3个epoch
            optimizer.zero_grad()
            
            # 生成训练数据
            torch.manual_seed(42 + step * 100 + epoch)
            input_ids = torch.randint(0, 10000, (1, 50))
            
            # Forward
            outputs = model(input_ids)
            
            # 主损失
            main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
            
            # Writeback损失
            wb_loss = torch.tensor(0.0)
            if enable_wb_loss:
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
            
            # Feature Guard损失
            guard_loss = torch.tensor(0.0)
            if enable_guard and guard_mode == "feature_only":
                guard_loss = feature_guard.compute_feature_guard_loss()
            elif enable_guard and guard_mode == "dual":
                guard_loss = promoter.feature_guard.compute_feature_guard_loss()
            
            # 总损失
            if guard_mode == "dual":
                total_loss = main_loss + 0.1 * wb_loss + guard_loss
            elif guard_mode == "head_only":
                total_loss = main_loss + 0.1 * wb_loss
            elif guard_mode == "feature_only":
                total_loss = main_loss + guard_loss
            else:
                total_loss = main_loss + 0.1 * wb_loss
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps} 完成")
    
    # 训练后评估
    print("\n[训练后评估]")
    after_eval = evaluator.evaluate(model)
    print(f"  Writeback score: {after_eval['writeback_score']:.4f}")
    
    # 计算变化
    score_change = after_eval['writeback_score'] - before_eval['writeback_score']
    
    # 计算输出KL
    total_kl = 0
    for before_logits, after_logits in zip(before_eval['all_logits'], after_eval['all_logits']):
        kl = F.kl_div(
            F.log_softmax(after_logits, dim=-1),
            F.softmax(before_logits, dim=-1),
            reduction='batchmean'
        ).item()
        total_kl += kl
    avg_kl = total_kl / len(before_eval['all_logits'])
    
    # 系统级评估
    eval_result = base_system.protocol.evaluate_with_protocol(f'{exp_name}_{num_steps}', num_samples=30)
    baseline = base_system.protocol.baseline.scores
    
    result = {
        'exp_name': exp_name,
        'guard_mode': guard_mode,
        'num_steps': num_steps,
        'initial_hash': initial_hash,
        'before_score': before_eval['writeback_score'],
        'after_score': after_eval['writeback_score'],
        'score_change': score_change,
        'output_kl': avg_kl,
        'target_gain': eval_result.scores.target_score - baseline.target_score,
        'old_ability_drop': max(
            abs(eval_result.scores.retrieval_score - baseline.retrieval_score),
            abs(eval_result.scores.policy_score - baseline.policy_score),
        ),
        'writeback_change': abs(eval_result.scores.writeback_score - baseline.writeback_score),
    }
    
    print(f"\n[结果汇总]")
    print(f"  Score change: {score_change:+.4f}")
    print(f"  Output KL: {avg_kl:.6f}")
    print(f"  Target gain: {result['target_gain']:+.2%}")
    print(f"  Old ability drop: {result['old_ability_drop']:.2%}")
    print(f"  Writeback change: {result['writeback_change']:+.2%}")
    
    return result


def main():
    """主验证实验"""
    print("="*70)
    print("Dual Guard 验证实验")
    print("="*70)
    
    # Step 1: 创建统一基线
    print("\n" + "="*70)
    print("Step 1: 创建统一基线模型")
    print("="*70)
    
    torch.manual_seed(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    base_system = Stage6SystemOrchestrator(
        experiment_id='dual_guard_validation',
        custom_config=config,
    )
    base_system.establish_baseline()
    
    # 保存基线状态
    base_state = base_system.orchestrator.backbone.get_model().state_dict()
    print(f"\n基线模型已创建")
    print(f"  Hash: {compute_model_hash(base_system.orchestrator.backbone.get_model())}")
    
    # Step 2: 运行4组实验
    print("\n" + "="*70)
    print("Step 2: 运行4组对照实验 (50 steps)")
    print("="*70)
    
    experiments = [
        ('baseline', 'none'),
        ('feature_guard_only', 'feature_only'),
        ('head_comp_only', 'head_only'),
        ('dual_guard', 'dual'),
    ]
    
    results = []
    
    for exp_name, guard_mode in experiments:
        result = run_dual_guard_experiment(
            exp_name,
            guard_mode,
            base_state,
            base_system,
            num_steps=50,
        )
        results.append(result)
    
    # Step 3: 结果对比
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    
    print(f"\n{'实验':<20} {'Mode':<15} {'Score Δ':<10} {'Output KL':<10} {'Target':<10} {'Old Ability':<12} {'Writeback':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['exp_name']:<20} {r['guard_mode']:<15} "
              f"{r['score_change']:>+8.4f}  {r['output_kl']:>8.6f}  "
              f"{r['target_gain']:>+8.2%}  {r['old_ability_drop']:>10.2%}  {r['writeback_change']:>+8.2%}")
    
    # Step 4: 关键结论
    print("\n" + "="*70)
    print("关键结论")
    print("="*70)
    
    baseline = next(r for r in results if r['guard_mode'] == 'none')
    feature_only = next(r for r in results if r['guard_mode'] == 'feature_only')
    head_only = next(r for r in results if r['guard_mode'] == 'head_only')
    dual = next(r for r in results if r['guard_mode'] == 'dual')
    
    print("\n1. Feature Guard效果验证:")
    print(f"   Baseline score change: {baseline['score_change']:+.4f}")
    print(f"   Feature guard score change: {feature_only['score_change']:+.4f}")
    if abs(feature_only['score_change']) < abs(baseline['score_change']):
        print("   ✓ Feature guard减少了负漂移")
    else:
        print("   ✗ Feature guard未达预期")
    
    print("\n2. Head Compensation效果:")
    print(f"   Head only score change: {head_only['score_change']:+.4f}")
    if head_only['score_change'] > 0:
        print("   ✓ Head训练产生正向补偿")
    
    print("\n3. Dual Guard综合效果:")
    print(f"   Dual guard score change: {dual['score_change']:+.4f}")
    print(f"   vs Head only: {dual['score_change'] - head_only['score_change']:+.4f}")
    if dual['score_change'] > head_only['score_change']:
        print("   ✓ Dual guard优于单一head补偿")
    
    print("\n4. 主任务保持:")
    print(f"   Dual guard target gain: {dual['target_gain']:+.2%}")
    print(f"   Dual guard old ability drop: {dual['old_ability_drop']:.2%}")
    if dual['target_gain'] > 0.1 and dual['old_ability_drop'] < 0.15:
        print("   ✓ 同时保住主任务指标")
    
    # 最终判断
    print("\n" + "="*70)
    print("最终判断")
    print("="*70)
    
    success = True
    
    if abs(feature_only['score_change']) >= abs(baseline['score_change']):
        print("✗ Feature guard未有效减少负漂移")
        success = False
    
    if dual['score_change'] <= head_only['score_change']:
        print("✗ Dual guard未优于单一机制")
        success = False
    
    if dual['target_gain'] < 0.1 or dual['old_ability_drop'] > 0.15:
        print("✗ Dual guard破坏主任务指标")
        success = False
    
    if success:
        print("\n✓ Dual Guard验证成功!")
        print("  建议作为Stage 7新主候选")
    else:
        print("\n⚠ Dual Guard需要调整参数")
        print("  建议调整beta/lr_ratio后重试")
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*70)
    print("验证完成")
    print("="*70)
