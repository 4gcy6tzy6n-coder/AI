"""
C 策略 (independent_optimizer) 多步验证

对比 baseline vs C-only
rollback off, 同一 seed
"""

import torch
import copy
from typing import Dict, List
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard


def run_multi_step_experiment(
    strategy_name: str,
    guard_mode: str,
    num_steps: int,
    seed: int = 42
) -> Dict:
    """运行多步实验"""
    print(f"\n{'='*70}")
    print(f"多步实验: {strategy_name}")
    print(f"  Guard: {guard_mode}")
    print(f"  Steps: {num_steps}")
    print(f"  Seed: {seed}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False  # 禁用回滚
    
    system = Stage6SystemOrchestrator(
        experiment_id=f'multi_step_{strategy_name}_{num_steps}',
        custom_config=config,
    )
    
    if guard_mode:
        system.orchestrator = Stage6OrchestratorWithGuard(
            config=system.orchestrator.config,
            guard_mode=guard_mode
        )
    
    system.establish_baseline()
    
    # 获取模型
    model = system.orchestrator.backbone.get_model()
    
    # 记录初始状态
    wb_params_initial = {}
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            wb_params_initial[name] = param.clone()
    
    # 运行多步训练
    metrics_history = []
    for step in range(num_steps):
        result = system.orchestrator.run_single_step(f'step {step}')
        metrics_history.append({
            'step': step + 1,
            'target_gain': result.target_improvement,
            'old_ability_drop': result.old_ability_drop,
            'writeback_change': result.writeback_change,
        })
    
    # 记录最终状态
    wb_params_final = {}
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            wb_params_final[name] = param.clone()
    
    # 计算累计变化
    wb_delta_norm = 0
    for name in wb_params_initial:
        delta = (wb_params_final[name] - wb_params_initial[name]).norm().item()
        wb_delta_norm += delta
    
    # 最终评估
    eval_result = system.protocol.evaluate_with_protocol(f'{strategy_name}_{num_steps}steps', num_samples=30)
    baseline = system.protocol.baseline.scores
    
    final_metrics = {
        'target_gain': eval_result.scores.target_score - baseline.target_score,
        'old_ability_drop': max(
            abs(eval_result.scores.retrieval_score - baseline.retrieval_score),
            abs(eval_result.scores.policy_score - baseline.policy_score),
        ),
        'writeback_change': abs(eval_result.scores.writeback_score - baseline.writeback_score),
        'wb_delta_norm': wb_delta_norm,
        'metrics_history': metrics_history,
    }
    
    print(f"\n[累计结果 - {num_steps} steps]")
    print(f"  Writeback head delta norm: {wb_delta_norm:.6f}")
    print(f"  Target gain: {final_metrics['target_gain']:+.2%}")
    print(f"  Old ability drop: {final_metrics['old_ability_drop']:.2%}")
    print(f"  Writeback change: {final_metrics['writeback_change']:+.2%}")
    
    return final_metrics


def compare_strategies():
    """对比 baseline 和 C 策略"""
    print("="*70)
    print("C 策略多步验证")
    print("="*70)
    
    results = {}
    
    # 10 steps
    print("\n" + "="*70)
    print("10 Steps 对比")
    print("="*70)
    
    results['baseline_10'] = run_multi_step_experiment('baseline', None, 10)
    results['C_10'] = run_multi_step_experiment('C_independent_optimizer', 'independent_optimizer', 10)
    
    # 50 steps
    print("\n" + "="*70)
    print("50 Steps 对比")
    print("="*70)
    
    results['baseline_50'] = run_multi_step_experiment('baseline', None, 50)
    results['C_50'] = run_multi_step_experiment('C_independent_optimizer', 'independent_optimizer', 50)
    
    # 汇总对比
    print("\n" + "="*70)
    print("汇总对比")
    print("="*70)
    
    print(f"\n{'策略':<25} {'Steps':<8} {'WB Delta':<12} {'Target':<10} {'Old Ability':<12} {'Writeback':<10}")
    print("-" * 80)
    
    for key, r in results.items():
        strategy = 'baseline' if 'baseline' in key else 'C_only'
        steps = '10' if '10' in key else '50'
        print(f"{strategy:<25} {steps:<8} {r['wb_delta_norm']:>10.6f}  "
              f"{r['target_gain']:>+8.2%}  {r['old_ability_drop']:>10.2%}  {r['writeback_change']:>+8.2%}")
    
    # 关键发现
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    for steps in [10, 50]:
        baseline_key = f'baseline_{steps}'
        c_key = f'C_{steps}'
        
        baseline_wb = results[baseline_key]['wb_delta_norm']
        c_wb = results[c_key]['wb_delta_norm']
        
        print(f"\n{steps} steps:")
        print(f"  Baseline WB delta: {baseline_wb:.6f}")
        print(f"  C strategy WB delta: {c_wb:.6f}")
        print(f"  差异: {abs(c_wb - baseline_wb):.6f}")
        
        if c_wb > baseline_wb:
            print(f"  ✓ C 策略 writeback head 更新更多 (可能保护不足)")
        elif c_wb < baseline_wb:
            print(f"  ✓ C 策略 writeback head 更新更少 (保护生效)")
        else:
            print(f"  ⚠ 两者相同")
    
    return results


if __name__ == "__main__":
    results = compare_strategies()
    
    print("\n" + "="*70)
    print("多步验证完成")
    print("="*70)
