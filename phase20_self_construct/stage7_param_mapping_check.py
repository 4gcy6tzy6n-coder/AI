"""
检查 protected params 与 writeback head 参数的映射关系
"""

import torch
import copy
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard


def check_param_mapping():
    """检查参数映射"""
    print("="*70)
    print("Protected Params 映射检查")
    print("="*70)
    
    torch.manual_seed(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id='param_check',
        custom_config=config,
    )
    
    system.orchestrator = Stage6OrchestratorWithGuard(
        config=system.orchestrator.config,
        guard_mode='independent_optimizer'
    )
    
    system.establish_baseline()
    
    # 获取模型和 guard
    model = system.orchestrator.backbone.get_model()
    guard = system.orchestrator.param_promoter.writeback_guard
    
    print("\n" + "="*70)
    print("1. 模型中所有 writeback 相关参数")
    print("="*70)
    writeback_params_in_model = []
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            writeback_params_in_model.append((name, param))
            print(f"  {name}: shape={param.shape}, norm={param.norm().item():.6f}")
    
    print(f"\n  总计: {len(writeback_params_in_model)} 个参数")
    
    print("\n" + "="*70)
    print("2. Guard 中 protected_params (从独立优化器获取)")
    print("="*70)
    if guard and guard.independent_optimizer:
        protected_params = guard.independent_optimizer.writeback_optimizer.param_groups[0]['params']
        print(f"  受保护参数数量: {len(protected_params)}")
        
        for i, param in enumerate(protected_params):
            # 查找参数名
            param_name = None
            for name, p in model.named_parameters():
                if p is param:
                    param_name = name
                    break
            
            if param_name:
                print(f"  [{i}] {param_name}: shape={param.shape}, norm={param.norm().item():.6f}")
            else:
                print(f"  [{i}] <未命名>: shape={param.shape}, norm={param.norm().item():.6f}")
    else:
        print("  ✗ 没有独立优化器")
    
    print("\n" + "="*70)
    print("3. 对比检查")
    print("="*70)
    if guard and guard.independent_optimizer:
        # 检查 protected_params 是否覆盖了所有 writeback 参数
        writeback_param_set = set(p for _, p in writeback_params_in_model)
        protected_param_set = set(protected_params)
        
        missing_in_protected = writeback_param_set - protected_param_set
        extra_in_protected = protected_param_set - writeback_param_set
        
        if missing_in_protected:
            print(f"  ✗ 模型中有 {len(missing_in_protected)} 个 writeback 参数未被保护:")
            for name, param in writeback_params_in_model:
                if param in missing_in_protected:
                    print(f"    - {name}")
        else:
            print("  ✓ 所有模型 writeback 参数都被保护了")
        
        if extra_in_protected:
            print(f"  ⚠ Protected 中有 {len(extra_in_protected)} 个非 writeback 参数")
        else:
            print("  ✓ Protected 中没有多余的参数")
    
    print("\n" + "="*70)
    print("4. 运行单步训练后再次检查")
    print("="*70)
    
    # 记录训练前状态
    wb_params_before = {}
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            wb_params_before[name] = param.clone()
    
    # 运行单步
    result = system.orchestrator.run_single_step('test query')
    
    # 检查训练后变化
    print("\n  Writeback 参数变化:")
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            before = wb_params_before[name]
            delta = (param - before).norm().item()
            print(f"    {name}: delta={delta:.6f}")
    
    print("\n" + "="*70)
    print("检查完成")
    print("="*70)


if __name__ == "__main__":
    check_param_mapping()
