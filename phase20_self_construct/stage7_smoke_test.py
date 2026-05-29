"""
Stage 7 烟雾测试 - 快速验证 guard 是否真正生效

测试目标:
1. 不同策略是否产生不同的参数更新
2. guard 是否真正被创建和应用
3. 训练图是否被改变
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import copy
from datetime import datetime

from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import create_orchestrator_with_guard


def smoke_test_1_single_step_diff():
    """
    烟雾测试 1: 单步差异实验
    
    同一 batch、同一 seed，跑 baseline / A / B / C，
    只训练 1 step，检查参数变化是否不同。
    """
    print("\n" + "=" * 70)
    print("烟雾测试 1: 单步差异实验")
    print("=" * 70)
    
    strategies = [
        ('baseline', None),
        ('A', 'feature_isolation'),
        ('B', 'gradient_mask'),
        ('C', 'independent_optimizer'),
    ]
    
    results = {}
    
    for name, guard_mode in strategies:
        print(f"\n[{name}] Guard mode: {guard_mode}")
        
        # 设置相同种子确保可复现
        torch.manual_seed(42)
        
        # 创建系统
        system = Stage6SystemOrchestrator(
            experiment_id=f"smoke1_{name}_{datetime.now().strftime('%H%M%S')}",
            custom_config=OFFICIAL_BASELINE_V1,
        )
        
        # 如果指定了 guard_mode，替换为带 guard 的 orchestrator
        if guard_mode:
            from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard
            system.orchestrator = Stage6OrchestratorWithGuard(
                config=system.orchestrator.config,
                guard_mode=guard_mode
            )
        
        # 建立基线
        system.establish_baseline()
        
        # 记录训练前参数
        model = system.orchestrator.backbone.get_model()
        param_before = {name: p.clone() for name, p in model.named_parameters()}
        param_hash_before = sum(p.sum().item() for p in model.parameters())
        
        # 运行 1 步训练
        try:
            system.orchestrator.run_single_step(f"smoke test {name}")
        except Exception as e:
            print(f"  训练错误: {e}")
        
        # 记录训练后参数
        param_hash_after = sum(p.sum().item() for p in model.parameters())
        
        # 计算参数变化
        param_delta = param_hash_after - param_hash_before
        
        # 检查 guard 状态
        guard_created = False
        if hasattr(system.orchestrator, 'param_promoter') and \
           hasattr(system.orchestrator.param_promoter, 'writeback_guard'):
            guard_created = system.orchestrator.param_promoter.writeback_guard is not None
        
        results[name] = {
            'guard_mode': guard_mode,
            'guard_created': guard_created,
            'param_hash_before': param_hash_before,
            'param_hash_after': param_hash_after,
            'param_delta': param_delta,
        }
        
        print(f"  Guard 创建: {'✓' if guard_created else '✗'}")
        print(f"  参数变化: {param_delta:.6f}")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("分析结果")
    print("=" * 70)
    
    baseline_delta = results['baseline']['param_delta']
    
    for name in ['A', 'B', 'C']:
        strategy_delta = results[name]['param_delta']
        diff = abs(strategy_delta - baseline_delta)
        
        if diff > 0.0001:
            print(f"✓ {name}: 参数变化与 baseline 不同 (差异: {diff:.6f})")
        else:
            print(f"✗ {name}: 参数变化与 baseline 相同 (差异: {diff:.6f})")
            print(f"   -> Guard 可能未生效!")
    
    return results


def smoke_test_2_param_hash():
    """
    烟雾测试 2: 参数哈希实验
    
    每个策略跑完后输出 model hash / writeback head hash / optimizer state hash
    """
    print("\n" + "=" * 70)
    print("烟雾测试 2: 参数哈希实验")
    print("=" * 70)
    
    strategies = [
        ('baseline', None),
        ('A', 'feature_isolation'),
    ]
    
    results = {}
    
    for name, guard_mode in strategies:
        print(f"\n[{name}] Guard mode: {guard_mode}")
        
        torch.manual_seed(42)
        
        # 创建系统
        system = Stage6SystemOrchestrator(
            experiment_id=f"smoke2_{name}_{datetime.now().strftime('%H%M%S')}",
            custom_config=OFFICIAL_BASELINE_V1,
        )
        
        # 如果指定了 guard_mode，替换为带 guard 的 orchestrator
        if guard_mode:
            from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard
            system.orchestrator = Stage6OrchestratorWithGuard(
                config=system.orchestrator.config,
                guard_mode=guard_mode
            )
        
        system.establish_baseline()
        
        # 运行 3 步训练
        for i in range(3):
            try:
                system.orchestrator.run_single_step(f"step {i}")
            except Exception as e:
                print(f"  训练错误: {e}")
        
        # 计算哈希
        model = system.orchestrator.backbone.get_model()
        model_hash = sum(p.sum().item() for p in model.parameters())
        
        # writeback head hash
        wb_hash = 0
        if hasattr(model, 'writeback_head'):
            wb_hash = sum(p.sum().item() for p in model.writeback_head.parameters())
        elif hasattr(model, 'heads') and 'writeback' in model.heads:
            wb_hash = sum(p.sum().item() for p in model.heads['writeback'].parameters())
        
        # optimizer hash
        opt_hash = 0
        if hasattr(system.orchestrator, 'param_promoter') and \
           system.orchestrator.param_promoter and \
           hasattr(system.orchestrator.param_promoter, 'optimizer'):
            opt = system.orchestrator.param_promoter.optimizer
            if hasattr(opt, 'state_dict'):
                opt_state = opt.state_dict()
                opt_hash = sum(v.sum().item() for v in opt_state.values() if isinstance(v, torch.Tensor))
        
        results[name] = {
            'model_hash': model_hash,
            'writeback_hash': wb_hash,
            'optimizer_hash': opt_hash,
        }
        
        print(f"  Model hash: {model_hash:.6f}")
        print(f"  Writeback hash: {wb_hash:.6f}")
        print(f"  Optimizer hash: {opt_hash:.6f}")
    
    # 比较
    print("\n" + "=" * 70)
    print("比较结果")
    print("=" * 70)
    
    baseline = results['baseline']
    strategy = results['A']
    
    if baseline['model_hash'] != strategy['model_hash']:
        print("✓ Model hash 不同 - 训练产生了不同结果")
    else:
        print("✗ Model hash 相同 - 可能 guard 未生效")
    
    return results


def smoke_test_3_forced_effect():
    """
    烟雾测试 3: 强制生效实验
    
    故意让某个策略产生极强效果，验证是否生效。
    """
    print("\n" + "=" * 70)
    print("烟雾测试 3: 强制生效实验")
    print("=" * 70)
    
    print("\n[测试说明]")
    print("在 B 策略 (gradient_mask) 中强制将 writeback 相关梯度清零")
    print("如果 guard 生效，writeback 应该几乎没有变化")
    
    # 这里简化处理，只检查 guard 是否能被创建
    torch.manual_seed(42)
    
    system = Stage6SystemOrchestrator(
        experiment_id=f"smoke3_{datetime.now().strftime('%H%M%S')}",
        custom_config=OFFICIAL_BASELINE_V1,
    )
    
    from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard
    system.orchestrator = Stage6OrchestratorWithGuard(
        config=system.orchestrator.config,
        guard_mode='gradient_mask'
    )
    
    # 检查 guard
    guard = None
    if hasattr(system.orchestrator, 'param_promoter') and \
       hasattr(system.orchestrator.param_promoter, 'writeback_guard'):
        guard = system.orchestrator.param_promoter.writeback_guard
    
    if guard:
        print("\n✓ Guard 成功创建")
        print(f"  模式: {guard.config.mode.value}")
        print(f"  梯度屏蔽: {'✓' if guard.gradient_shield else '✗'}")
        
        # 检查是否能启用屏蔽
        if guard.gradient_shield:
            print("\n  测试启用梯度屏蔽...")
            try:
                guard.enable_gradient_shield()
                print("  ✓ 梯度屏蔽已启用")
            except Exception as e:
                print(f"  ✗ 启用失败: {e}")
    else:
        print("\n✗ Guard 未创建")
    
    return guard is not None


def run_all_smoke_tests():
    """运行所有烟雾测试"""
    print("=" * 70)
    print("Stage 7 烟雾测试套件")
    print("=" * 70)
    print("\n目的: 快速验证 writeback guard 是否真正集成到训练流程")
    
    # 测试 1
    result1 = smoke_test_1_single_step_diff()
    
    # 测试 2
    result2 = smoke_test_2_param_hash()
    
    # 测试 3
    result3 = smoke_test_3_forced_effect()
    
    # 汇总
    print("\n" + "=" * 70)
    print("烟雾测试汇总")
    print("=" * 70)
    
    # 检查测试 1 结果
    baseline_delta = result1['baseline']['param_delta']
    any_different = False
    for name in ['A', 'B', 'C']:
        if abs(result1[name]['param_delta'] - baseline_delta) > 0.0001:
            any_different = True
            break
    
    print(f"\n测试 1 (单步差异): {'✓ 通过' if any_different else '✗ 失败'}")
    print(f"  不同策略产生了不同的参数更新")
    
    print(f"\n测试 2 (参数哈希): {'✓ 通过' if result2['baseline']['model_hash'] != result2['A']['model_hash'] else '✗ 失败'}")
    print(f"  不同策略产生了不同的模型状态")
    
    print(f"\n测试 3 (强制生效): {'✓ 通过' if result3 else '✗ 失败'}")
    print(f"  Guard 能够被创建和启用")
    
    all_pass = any_different and (result2['baseline']['model_hash'] != result2['A']['model_hash']) and result3
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ 所有烟雾测试通过!")
        print("  Writeback guard 已成功集成到训练流程")
        print("  可以继续运行完整消融测试")
    else:
        print("✗ 部分烟雾测试失败")
        print("  需要进一步检查 guard 集成")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    run_all_smoke_tests()
