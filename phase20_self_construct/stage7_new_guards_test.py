"""
新保护机制最小实验

对比 4 组策略:
- baseline
- freeze_writeback
- low_lr_clip
- delta_penalty

每组跑 10 steps 和 50 steps
"""

import torch
import copy
from typing import Dict
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard


def run_experiment(
    strategy_name: str,
    guard_mode: str,
    num_steps: int,
    seed: int = 42
) -> Dict:
    """运行单组实验"""
    print(f"\n{'='*70}")
    print(f"实验: {strategy_name} | Steps: {num_steps}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id=f'{strategy_name}_{num_steps}',
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
    
    # 记录初始 writeback 参数
    wb_params_initial = {}
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            wb_params_initial[name] = param.clone()
    
    # 运行训练
    for step in range(num_steps):
        result = system.orchestrator.run_single_step(f'step {step}')
    
    # 记录最终 writeback 参数
    wb_params_final = {}
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            wb_params_final[name] = param.clone()
    
    # 计算 writeback 参数变化
    wb_delta_norm = 0
    for name in wb_params_initial:
        delta = (wb_params_final[name] - wb_params_initial[name]).norm().item()
        wb_delta_norm += delta
    
    # 最终评估
    eval_result = system.protocol.evaluate_with_protocol(f'{strategy_name}_{num_steps}', num_samples=30)
    baseline = system.protocol.baseline.scores
    
    result = {
        'strategy': strategy_name,
        'steps': num_steps,
        'wb_delta_norm': wb_delta_norm,
        'target_gain': eval_result.scores.target_score - baseline.target_score,
        'old_ability_drop': max(
            abs(eval_result.scores.retrieval_score - baseline.retrieval_score),
            abs(eval_result.scores.policy_score - baseline.policy_score),
        ),
        'writeback_change': abs(eval_result.scores.writeback_score - baseline.writeback_score),
    }
    
    print(f"\n[结果]")
    print(f"  WB delta norm: {result['wb_delta_norm']:.6f}")
    print(f"  Target gain: {result['target_gain']:+.2%}")
    print(f"  Old ability drop: {result['old_ability_drop']:.2%}")
    print(f"  Writeback change: {result['writeback_change']:+.2%}")
    
    return result


def main():
    """主实验"""
    print("="*70)
    print("新保护机制最小实验")
    print("="*70)
    
    strategies = [
        ('baseline', None),
        ('freeze', 'freeze'),
        ('low_lr_clip', 'low_lr_clip'),
        ('delta_penalty', 'delta_penalty'),
    ]
    
    all_results = []
    
    # 10 steps
    print("\n" + "="*70)
    print("10 Steps 实验")
    print("="*70)
    
    for name, guard_mode in strategies:
        result = run_experiment(name, guard_mode, 10)
        all_results.append(result)
    
    # 50 steps
    print("\n" + "="*70)
    print("50 Steps 实验")
    print("="*70)
    
    for name, guard_mode in strategies:
        result = run_experiment(name, guard_mode, 50)
        all_results.append(result)
    
    # 汇总表
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    
    print(f"\n{'策略':<20} {'Steps':<8} {'WB Delta':<12} {'Target':<10} {'Old Ability':<12} {'Writeback':<10}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r['strategy']:<20} {r['steps']:<8} {r['wb_delta_norm']:>10.6f}  "
              f"{r['target_gain']:>+8.2%}  {r['old_ability_drop']:>10.2%}  {r['writeback_change']:>+8.2%}")
    
    # 关键发现
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    # 按 steps 分组分析
    for steps in [10, 50]:
        print(f"\n{steps} Steps:")
        step_results = [r for r in all_results if r['steps'] == steps]
        baseline = next(r for r in step_results if r['strategy'] == 'baseline')
        
        for r in step_results:
            if r['strategy'] != 'baseline':
                wb_diff = r['wb_delta_norm'] - baseline['wb_delta_norm']
                target_diff = r['target_gain'] - baseline['target_gain']
                writeback_diff = r['writeback_change'] - baseline['writeback_change']
                
                print(f"  {r['strategy']}:")
                print(f"    WB delta: {r['wb_delta_norm']:.6f} (vs baseline: {wb_diff:+.6f})")
                print(f"    Writeback change: {r['writeback_change']:+.2%} (vs baseline: {writeback_diff:+.2%})")
                
                if r['wb_delta_norm'] < baseline['wb_delta_norm'] * 0.5:
                    print(f"    ✓ WB 更新明显减少 (保护生效)")
                elif r['wb_delta_norm'] > baseline['wb_delta_norm'] * 1.5:
                    print(f"    ✗ WB 更新增加 (保护不足)")
                else:
                    print(f"    ~ WB 更新相似")
    
    # 最佳策略推荐
    print("\n" + "="*70)
    print("策略评估")
    print("="*70)
    
    for steps in [10, 50]:
        print(f"\n{steps} Steps 最佳策略:")
        step_results = [r for r in all_results if r['steps'] == steps]
        
        # 按 writeback_change 排序 (越小越好)
        sorted_by_wb = sorted(step_results, key=lambda x: x['writeback_change'])
        best_for_wb = sorted_by_wb[0]
        
        # 按 target_gain 排序 (越大越好)
        sorted_by_target = sorted(step_results, key=lambda x: -x['target_gain'])
        best_for_target = sorted_by_target[0]
        
        print(f"  最佳 writeback 保护: {best_for_wb['strategy']} ({best_for_wb['writeback_change']:+.2%})")
        print(f"  最佳 target gain: {best_for_target['strategy']} ({best_for_target['target_gain']:+.2%})")
    
    return all_results


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*70)
    print("实验完成")
    print("="*70)
