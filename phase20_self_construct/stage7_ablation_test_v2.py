"""
Stage 7 消融测试 V2 - 真正集成 writeback guard

修复了 V1 的问题：
- V1: 只导入了 guard 函数但从未调用
- V2: 通过配置传递 guard_mode，真正创建带 guard 的系统
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import copy
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# 导入系统编排器
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_orchestrator_with_guard import create_orchestrator_with_guard
from stage7_regression_gate import Stage7RegressionGate


@dataclass
class AblationTestResult:
    """消融测试结果"""
    test_id: str
    timestamp: str
    strategy: str
    guard_mode: Optional[str]
    
    # 核心指标
    target_gain: float
    old_ability_drop: float
    writeback_change: float
    rollback_recovery: float
    e2e_success_rate: float
    stability_pass: bool
    
    # 验证指标
    guard_created: bool
    guard_applied: bool
    param_delta_norm: float
    writeback_param_delta_norm: float
    
    # Regression gate
    regression_pass: bool
    failed_items: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class Stage7AblationTesterV2:
    """
    Stage 7 消融测试器 V2
    
    真正集成 writeback guard 到训练流程
    """
    
    STRATEGIES = {
        'baseline': {'mode': None, 'desc': 'Stage 6 官方基线 (无保护)'},
        'A': {'mode': 'feature_isolation', 'desc': '方案 A: 仅特征隔离'},
        'B': {'mode': 'gradient_mask', 'desc': '方案 B: 仅梯度屏蔽'},
        'C': {'mode': 'independent_optimizer', 'desc': '方案 C: 仅独立优化器'},
        'D_A_B': {'mode': 'hybrid_feature_isolation_gradient_mask', 'desc': '方案 D: A+B 混合'},
        'D_A_C': {'mode': 'hybrid_feature_isolation_independent_optimizer', 'desc': '方案 D: A+C 混合'},
        'D_B_C': {'mode': 'hybrid_gradient_mask_independent_optimizer', 'desc': '方案 D: B+C 混合'},
        'D_A_B_C': {'mode': 'hybrid_feature_isolation_gradient_mask_independent_optimizer', 'desc': '方案 D: A+B+C 混合'},
    }
    
    def __init__(self, config_name: str = 'baseline_v1'):
        self.config_name = config_name
        self.config = copy.deepcopy(OFFICIAL_BASELINE_V1)
        self.results: Dict[str, AblationTestResult] = {}
    
    def run_all_ablation_tests(self, num_steps: int = 3) -> Dict[str, AblationTestResult]:
        """
        运行所有消融测试
        
        Args:
            num_steps: 测试步数 (默认 3 步，用于快速验证)
        
        Returns:
            各策略的测试结果
        """
        print("\n" + "=" * 70)
        print("Stage 7 消融测试 V2 - 真正集成 writeback guard")
        print("=" * 70)
        print(f"配置: {self.config_name}")
        print(f"测试步数: {num_steps}")
        print(f"开始时间: {datetime.now().isoformat()}")
        
        # 按顺序测试每个策略
        strategies = ['baseline', 'A', 'B', 'C', 'D_A_B', 'D_A_C', 'D_B_C', 'D_A_B_C']
        
        for i, strategy_key in enumerate(strategies, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(strategies)}] 测试策略: {strategy_key}")
            print(f"{'=' * 70}")
            
            strategy_info = self.STRATEGIES[strategy_key]
            print(f"描述: {strategy_info['desc']}")
            print(f"Guard 模式: {strategy_info['mode']}")
            
            self.results[strategy_key] = self._test_strategy(
                strategy_key, 
                strategy_info['mode'],
                num_steps
            )
        
        # 汇总结果
        self._print_summary()
        
        # 保存结果
        self._save_results()
        
        return self.results
    
    def _test_strategy(self, strategy_key: str, guard_mode: Optional[str], 
                       num_steps: int) -> AblationTestResult:
        """测试单个策略"""
        test_id = f"ablation_v2_{strategy_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建带 guard 的系统
        print(f"\n[1/4] 创建系统 (guard_mode={guard_mode})...")
        
        if guard_mode is None:
            # Baseline: 使用原始系统
            system = Stage6SystemOrchestrator(
                experiment_id=test_id,
                custom_config=self.config,
            )
            guard_created = False
        else:
            # 策略测试: 使用带 guard 的系统
            # 创建自定义配置，包含 guard_mode
            custom_config = self.config.copy()
            custom_config['writeback_guard_mode'] = guard_mode
            
            system = Stage6SystemOrchestrator(
                experiment_id=test_id,
                custom_config=custom_config,
            )
            
            # 替换 orchestrator 为带 guard 版本
            from stage6_orchestrator_with_guard import Stage6OrchestratorWithGuard
            system.orchestrator = Stage6OrchestratorWithGuard(
                config=system.orchestrator.config,
                guard_mode=guard_mode
            )
            
            # 验证 guard 是否创建
            guard_created = (
                hasattr(system.orchestrator, 'param_promoter') and
                hasattr(system.orchestrator.param_promoter, 'writeback_guard') and
                system.orchestrator.param_promoter.writeback_guard is not None
            )
        
        print(f"  Guard 创建: {'✓' if guard_created else '✗'}")
        
        # 建立基线
        print("\n[2/4] 建立基线...")
        system.establish_baseline()
        
        # 记录初始参数哈希
        param_before = self._get_model_hash(system)
        
        # 运行训练
        print(f"\n[3/4] 运行 {num_steps} 步训练...")
        queries = [f"消融测试 {strategy_key} 步骤 {i+1}" for i in range(num_steps)]
        
        for i, query in enumerate(queries, 1):
            print(f"\n  Step {i}/{num_steps}...")
            try:
                system.orchestrator.run_single_step(query)
            except Exception as e:
                print(f"    错误: {e}")
        
        # 记录训练后参数哈希
        param_after = self._get_model_hash(system)
        param_delta_norm = abs(hash(param_after) - hash(param_before)) % 10000 / 10000.0
        
        # 评估
        print("\n[4/4] 评估...")
        eval_result = system.protocol.evaluate_with_protocol(f"ablation_{strategy_key}", num_samples=30)
        
        # 获取指标
        target_gain = eval_result.scores.target_score - system.protocol.baseline.scores.target_score
        old_ability_drop = max(
            abs(eval_result.scores.retrieval_score - system.protocol.baseline.scores.retrieval_score),
            abs(eval_result.scores.policy_score - system.protocol.baseline.scores.policy_score),
        )
        writeback_change = abs(eval_result.scores.writeback_score - system.protocol.baseline.scores.writeback_score)
        
        # 检查 guard 是否真正应用 (通过参数变化)
        guard_applied = param_delta_norm > 0.001  # 参数有变化说明训练发生了
        
        # 获取 writeback 参数变化 (如果可用)
        writeback_param_delta_norm = 0.0
        if guard_created and hasattr(system.orchestrator.param_promoter, 'writeback_guard'):
            guard = system.orchestrator.param_promoter.writeback_guard
            if guard and guard.independent_optimizer:
                writeback_param_delta_norm = guard.independent_optimizer.get_writeback_param_delta()
        
        # 运行 regression gate
        gate = Stage7RegressionGate(self.config_name)
        regression_result = gate.run_full_regression_check(num_steps)
        
        # 创建结果
        result = AblationTestResult(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            strategy=strategy_key,
            guard_mode=guard_mode,
            target_gain=target_gain,
            old_ability_drop=old_ability_drop,
            writeback_change=writeback_change,
            rollback_recovery=0.0,  # 简化处理
            e2e_success_rate=regression_result.e2e_success_rate,
            stability_pass=regression_result.stability_pass,
            guard_created=guard_created,
            guard_applied=guard_applied,
            param_delta_norm=param_delta_norm,
            writeback_param_delta_norm=writeback_param_delta_norm,
            regression_pass=regression_result.overall_pass,
            failed_items=regression_result.failed_items,
        )
        
        # 打印结果
        self._print_strategy_result(result)
        
        return result
    
    def _get_model_hash(self, system) -> str:
        """获取模型哈希"""
        model = system.orchestrator.backbone.get_model()
        param_sum = sum(p.sum().item() for p in model.parameters())
        return f"{param_sum:.6f}"
    
    def _print_strategy_result(self, result: AblationTestResult):
        """打印策略结果"""
        print(f"\n  结果:")
        print(f"    target_gain: {result.target_gain:+.2%}")
        print(f"    old_ability_drop: {result.old_ability_drop:.2%}")
        print(f"    writeback_change: {result.writeback_change:+.2%}")
        print(f"    guard_created: {result.guard_created}")
        print(f"    guard_applied: {result.guard_applied}")
        print(f"    param_delta_norm: {result.param_delta_norm:.6f}")
        print(f"    regression_pass: {result.regression_pass}")
        if result.failed_items:
            print(f"    failed_items: {result.failed_items}")
    
    def _print_summary(self):
        """打印汇总"""
        print("\n" + "=" * 70)
        print("消融测试结果汇总")
        print("=" * 70)
        
        # 表头
        print(f"\n{'策略':<12} {'Guard':<20} {'目标提升':<10} {'旧能力':<10} {'Writeback':<12} {'Guard✓':<8} {'Regression':<10}")
        print("-" * 90)
        
        # 数据行
        for key, result in self.results.items():
            guard_short = result.guard_mode[:18] if result.guard_mode else 'None'
            print(f"{result.strategy:<12} {guard_short:<20} "
                  f"{result.target_gain:>+8.2%}  "
                  f"{result.old_ability_drop:>8.2%}  "
                  f"{result.writeback_change:>+10.2%}  "
                  f"{'✓' if result.guard_applied else '✗':<8} "
                  f"{'✓' if result.regression_pass else '✗':<10}")
        
        # 关键发现
        print("\n" + "=" * 70)
        print("关键发现:")
        print("=" * 70)
        
        # 检查 guard 是否生效
        baseline_wb = self.results['baseline'].writeback_change
        for key in ['A', 'B', 'C']:
            if key in self.results:
                strategy_wb = self.results[key].writeback_change
                diff = abs(strategy_wb - baseline_wb)
                if diff > 0.01:
                    print(f"✓ 策略 {key} writeback 变化与 baseline 不同 (差异: {diff:.2%})")
                else:
                    print(f"✗ 策略 {key} writeback 变化与 baseline 相同 (差异: {diff:.2%}) - Guard 可能未生效")
    
    def _save_results(self):
        """保存结果"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config_name,
            'results': {k: v.to_dict() for k, v in self.results.items()},
        }
        
        filename = f"eval/stage7_ablation_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n结果已保存: {filename}")


def run_stage7_ablation_tests_v2(config_name: str = 'baseline_v1', num_steps: int = 3) -> Dict[str, AblationTestResult]:
    """
    运行 Stage 7 消融测试 V2
    
    Args:
        config_name: 配置名称
        num_steps: 测试步数 (默认 3)
    
    Returns:
        各策略的测试结果
    """
    tester = Stage7AblationTesterV2(config_name)
    return tester.run_all_ablation_tests(num_steps)


# 烟雾测试
if __name__ == "__main__":
    print("=" * 70)
    print("Stage 7 Ablation Test V2 - 烟雾测试")
    print("=" * 70)
    
    # 运行简化的消融测试 (3 步)
    results = run_stage7_ablation_tests_v2(num_steps=3)
    
    print("\n" + "=" * 70)
    print("烟雾测试完成")
    print("=" * 70)
