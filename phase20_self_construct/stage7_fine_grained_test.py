"""
Stage 7 精细化验证实验

观测粒度从"整模型"细化到:
- writeback head 参数变化
- protected vs unprotected 参数变化
- rollback-off 多步训练
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import copy
from datetime import datetime
from typing import Dict, List, Tuple

from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import create_orchestrator_with_guard


def get_writeback_head_params(model) -> List[torch.nn.Parameter]:
    """获取 writeback head 的参数"""
    params = []
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            params.append(param)
    return params


def get_protected_params(model, guard) -> List[torch.nn.Parameter]:
    """获取受 guard 保护的参数"""
    if guard and guard.independent_optimizer:
        # 从独立优化器获取 writeback 参数
        return guard.independent_optimizer.writeback_optimizer.param_groups[0]['params']
    return []


def compute_param_delta(before: Dict[str, torch.Tensor], 
                        after: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """计算参数变化"""
    deltas = {}
    for name in before:
        if name in after:
            delta = (after[name] - before[name]).norm().item()
            deltas[name] = delta
    return deltas


def run_fine_grained_experiment(
    strategy_name: str,
    guard_mode: str,
    num_steps: int = 10,
    disable_rollback: bool = True
) -> Dict:
    """
    运行精细化实验
    
    Args:
        strategy_name: 策略名称
        guard_mode: guard 模式
        num_steps: 训练步数
        disable_rollback: 是否禁用回滚
    
    Returns:
        详细的实验结果
    """
    print(f"\n{'='*70}")
    print(f"精细化实验: {strategy_name}")
    print(f"  Guard: {guard_mode}")
    print(f"  Steps: {num_steps}")
    print(f"  Rollback: {'off' if disable_rollback else 'on'}")
    print(f"{'='*70}")
    
    # 设置相同种子
    torch.manual_seed(42)
    
    # 创建配置
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    if disable_rollback:
        config['auto_rollback'] = False
    
    # 创建系统
    system = Stage6SystemOrchestrator(
        experiment_id=f"fine_{strategy_name}_{datetime.now().strftime('%H%M%S')}",
        custom_config=config,
    )
    
    # 如果指定了 guard_mode，替换 orchestrator
    if guard_mode:
        from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard
        system.orchestrator = Stage6OrchestratorWithGuard(
            config=system.orchestrator.config,
            guard_mode=guard_mode
        )
    
    # 建立基线
    system.establish_baseline()
    
    # 获取模型
    model = system.orchestrator.backbone.get_model()
    
    # 记录训练前状态
    params_before = {name: p.clone() for name, p in model.named_parameters()}
    wb_params_before = get_writeback_head_params(model)
    wb_hash_before = sum(p.sum().item() for p in wb_params_before) if wb_params_before else 0
    
    # 获取 guard 和保护参数
    guard = None
    protected_params = []
    if hasattr(system.orchestrator, 'param_promoter') and \
       hasattr(system.orchestrator.param_promoter, 'writeback_guard'):
        guard = system.orchestrator.param_promoter.writeback_guard
        if guard:
            protected_params = get_protected_params(model, guard)
    
    protected_hash_before = sum(p.sum().item() for p in protected_params) if protected_params else 0
    
    print(f"\n[训练前]")
    print(f"  Writeback head 参数数量: {len(wb_params_before)}")
    print(f"  Writeback head hash: {wb_hash_before:.6f}")
    print(f"  受保护参数数量: {len(protected_params)}")
    print(f"  受保护参数 hash: {protected_hash_before:.6f}")
    
    # 运行训练
    print(f"\n[训练 {num_steps} 步]")
    step_metrics = []
    
    for i in range(num_steps):
        try:
            result = system.orchestrator.run_single_step(f"step {i}")
            step_metrics.append({
                'step': i,
                'target_improvement': result.target_improvement if hasattr(result, 'target_improvement') else 0,
                'writeback_change': result.writeback_change if hasattr(result, 'writeback_change') else 0,
            })
            if i % 5 == 0:
                print(f"  Step {i}: target={result.target_improvement:+.2%}, writeback={result.writeback_change:+.2%}")
        except Exception as e:
            print(f"  Step {i} 错误: {e}")
            break
    
    # 记录训练后状态
    params_after = {name: p.clone() for name, p in model.named_parameters()}
    wb_params_after = get_writeback_head_params(model)
    wb_hash_after = sum(p.sum().item() for p in wb_params_after) if wb_params_after else 0
    protected_hash_after = sum(p.sum().item() for p in protected_params) if protected_params else 0
    
    # 计算变化
    wb_delta = wb_hash_after - wb_hash_before
    protected_delta = protected_hash_after - protected_hash_before
    
    # 计算所有参数的变化 norm
    total_delta_norm = 0
    wb_delta_norm = 0
    protected_delta_norm = 0
    
    # 获取 protected 参数名列表 (用于匹配)
    protected_param_names = set()
    if guard and guard.independent_optimizer:
        protected_param_set = set(guard.independent_optimizer.writeback_optimizer.param_groups[0]['params'])
        for name, p in model.named_parameters():
            if p in protected_param_set:
                protected_param_names.add(name)
    
    for name in params_before:
        delta = (params_after[name] - params_before[name]).norm().item()
        total_delta_norm += delta
        
        if 'writeback' in name.lower():
            wb_delta_norm += delta
        
        # 检查是否是受保护参数
        if name in protected_param_names:
            protected_delta_norm += delta
    
    print(f"\n[训练后]")
    print(f"  Writeback head hash: {wb_hash_after:.6f}")
    print(f"  Writeback head delta: {wb_delta:.6f}")
    print(f"  Writeback head delta norm: {wb_delta_norm:.6f}")
    print(f"  受保护参数 delta: {protected_delta:.6f}")
    print(f"  受保护参数 delta norm: {protected_delta_norm:.6f}")
    print(f"  总参数 delta norm: {total_delta_norm:.6f}")
    
    # 评估
    eval_result = system.protocol.evaluate_with_protocol(f"fine_{strategy_name}", num_samples=30)
    baseline = system.protocol.baseline.scores
    
    target_gain = eval_result.scores.target_score - baseline.target_score
    old_ability_drop = max(
        abs(eval_result.scores.retrieval_score - baseline.retrieval_score),
        abs(eval_result.scores.policy_score - baseline.policy_score),
    )
    writeback_change = abs(eval_result.scores.writeback_score - baseline.writeback_score)
    
    print(f"\n[评估结果]")
    print(f"  Target gain: {target_gain:+.2%}")
    print(f"  Old ability drop: {old_ability_drop:.2%}")
    print(f"  Writeback change: {writeback_change:+.2%}")
    
    return {
        'strategy': strategy_name,
        'guard_mode': guard_mode,
        'num_steps': num_steps,
        'rollback_off': disable_rollback,
        'guard_created': guard is not None,
        'wb_params_count': len(wb_params_before),
        'protected_params_count': len(protected_params),
        'wb_delta': wb_delta,
        'wb_delta_norm': wb_delta_norm,
        'protected_delta': protected_delta,
        'protected_delta_norm': protected_delta_norm,
        'total_delta_norm': total_delta_norm,
        'target_gain': target_gain,
        'old_ability_drop': old_ability_drop,
        'writeback_change': writeback_change,
        'step_metrics': step_metrics,
    }


def run_experiment_matrix():
    """
    运行实验矩阵
    
    baseline, rollback off, 10 steps
    B only, rollback off, 10 steps
    C only, rollback off, 10 steps
    hybrid, rollback off, 10 steps
    """
    print("="*70)
    print("Stage 7 精细化验证 - 实验矩阵")
    print("="*70)
    print("\n配置:")
    print("  - Rollback: OFF (避免恢复干扰)")
    print("  - Steps: 10 (足够观察累积效果)")
    print("  - Seed: 42 (确保可复现)")
    print("  - 观测粒度: writeback head / protected params / total")
    
    experiments = [
        ('baseline', None),
        ('B_gradient_mask', 'gradient_mask'),
        ('C_independent_optimizer', 'independent_optimizer'),
        ('D_hybrid', 'hybrid_feature_isolation_gradient_mask'),
    ]
    
    results = []
    
    for name, mode in experiments:
        result = run_fine_grained_experiment(
            strategy_name=name,
            guard_mode=mode,
            num_steps=10,
            disable_rollback=True
        )
        results.append(result)
    
    # 汇总分析
    print("\n" + "="*70)
    print("实验矩阵结果汇总")
    print("="*70)
    
    # 表头
    print(f"\n{'策略':<25} {'Guard':<20} {'WB Delta':<12} {'Protected':<12} {'Total':<12} {'Target':<10} {'Writeback':<10}")
    print("-" * 110)
    
    baseline_wb_delta = None
    
    for r in results:
        guard_short = r['guard_mode'][:18] if r['guard_mode'] else 'None'
        print(f"{r['strategy']:<25} {guard_short:<20} "
              f"{r['wb_delta_norm']:>10.6f}  "
              f"{r['protected_delta_norm']:>10.6f}  "
              f"{r['total_delta_norm']:>10.6f}  "
              f"{r['target_gain']:>+8.2%}  "
              f"{r['writeback_change']:>+8.2%}")
        
        if r['strategy'] == 'baseline':
            baseline_wb_delta = r['wb_delta_norm']
    
    # 关键发现
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    if baseline_wb_delta is not None:
        for r in results:
            if r['strategy'] != 'baseline':
                diff = abs(r['wb_delta_norm'] - baseline_wb_delta)
                if diff > 0.0001:
                    print(f"✓ {r['strategy']}: writeback head 变化与 baseline 不同")
                    print(f"    baseline: {baseline_wb_delta:.6f}")
                    print(f"    {r['strategy']}: {r['wb_delta_norm']:.6f}")
                    print(f"    差异: {diff:.6f}")
                else:
                    print(f"✗ {r['strategy']}: writeback head 变化与 baseline 相同")
                    print(f"    差异: {diff:.6f} (可能 guard 未生效或效果太弱)")
    
    # 检查 protected params
    print("\n受保护参数分析:")
    for r in results:
        if r['protected_params_count'] > 0:
            print(f"  {r['strategy']}: {r['protected_params_count']} 个受保护参数")
            print(f"    变化: {r['protected_delta_norm']:.6f}")
    
    return results


if __name__ == "__main__":
    results = run_experiment_matrix()
    
    print("\n" + "="*70)
    print("精细化验证完成")
    print("="*70)
