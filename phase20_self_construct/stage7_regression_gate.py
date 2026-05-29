"""
Stage 7 回归护栏

确保 Stage 7 的优化不破坏 Stage 6 已验证的主链
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from stage6_system_orchestrator import Stage6SystemOrchestrator
from stage6_optimization_configs import get_config


@dataclass
class RegressionCheckResult:
    """回归检查结果"""
    
    # 检查项
    target_gain_pass: bool = False
    old_ability_pass: bool = False
    writeback_pass: bool = False
    rollback_pass: bool = False
    e2e_success_pass: bool = False
    stability_pass: bool = False
    
    # 详细数值
    target_gain: float = 0.0
    old_ability_drop: float = 0.0
    writeback_change: float = 0.0
    rollback_recovery: float = 0.0
    e2e_success_rate: float = 0.0
    stability_result: str = ""
    
    # 总体结果
    overall_pass: bool = False
    
    # 失败项列表
    failed_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'checks': {
                'target_gain': {'value': self.target_gain, 'pass': self.target_gain_pass},
                'old_ability_drop': {'value': self.old_ability_drop, 'pass': self.old_ability_pass},
                'writeback_change': {'value': self.writeback_change, 'pass': self.writeback_pass},
                'rollback_recovery': {'value': self.rollback_recovery, 'pass': self.rollback_pass},
                'e2e_success': {'value': self.e2e_success_rate, 'pass': self.e2e_success_pass},
                'stability': {'result': self.stability_result, 'pass': self.stability_pass},
            },
            'overall_pass': self.overall_pass,
            'failed_items': self.failed_items,
        }


class Stage7RegressionGate:
    """
    Stage 7 回归护栏
    
    每次改动都要检查 Stage 6 的四个必须项：
    1. target gain > 10%
    2. old ability drop < 15%
    3. E2E success > 90%
    4. 100-step stability pass
    
    以及 Stage 7 的两个优化目标：
    5. writeback change < 5%
    6. rollback recovery > 90%
    """
    
    # Stage 6 必须项阈值 (不可突破)
    STAGE6_REQUIREMENTS = {
        'target_gain_min': 0.10,      # > 10%
        'old_ability_max': 0.15,      # < 15%
        'e2e_success_min': 0.90,      # > 90%
    }
    
    # Stage 7 优化目标 (当前优化方向)
    STAGE7_TARGETS = {
        'writeback_max': 0.05,        # < 5%
        'rollback_min': 0.90,         # > 90%
    }
    
    def __init__(self, config_name: str = 'baseline_v1'):
        self.config_name = config_name
        self.config = get_config(config_name)
        self.check_id = f"regression_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def run_full_regression_check(self, num_steps: int = 10) -> RegressionCheckResult:
        """
        运行完整回归检查
        
        这是 Stage 7 每次改动的必做检查
        """
        print("\n" + "="*70)
        print("Stage 7 回归护栏检查")
        print("="*70)
        print(f"检查ID: {self.check_id}")
        print(f"配置: {self.config_name}")
        print(f"检查时间: {datetime.now().isoformat()}")
        
        result = RegressionCheckResult()
        
        # 1. 检查核心指标 (target gain, old ability, writeback)
        print("\n[1/3] 核心指标检查...")
        self._check_core_metrics(result, num_steps)
        
        # 2. 检查端到端成功率
        print("\n[2/3] 端到端成功率检查...")
        self._check_e2e_success(result)
        
        # 3. 检查稳定性
        print("\n[3/3] 稳定性检查...")
        self._check_stability(result)
        
        # 评估总体结果
        self._evaluate_overall(result)
        
        return result
    
    def _check_core_metrics(self, result: RegressionCheckResult, num_steps: int):
        """检查核心指标"""
        # 创建系统
        system = Stage6SystemOrchestrator(
            experiment_id=self.check_id,
            custom_config=self.config,
        )
        
        # 建立基线
        system.establish_baseline()
        baseline_target = system.protocol.baseline.scores.target_score
        
        # 运行测试步数
        queries = [f"回归检查查询 {i+1}" for i in range(num_steps)]
        for i, query in enumerate(queries, 1):
            trace = system._execute_step(i, query)
            if trace.error_message:
                print(f"  Step {i}: 错误 - {trace.error_message}")
        
        # 最终评估
        final_result = system.protocol.evaluate_with_protocol("regression", num_samples=50)
        final_target = system.protocol.evaluator.evaluate_all(num_samples=50).scores.target_score
        
        # 计算指标
        result.target_gain = final_target - baseline_target
        result.old_ability_drop = final_result.old_ability_drop
        result.writeback_change = final_result.writeback_change
        
        # 检查 Stage 6 必须项
        result.target_gain_pass = result.target_gain >= self.STAGE6_REQUIREMENTS['target_gain_min']
        result.old_ability_pass = result.old_ability_drop <= self.STAGE6_REQUIREMENTS['old_ability_max']
        
        # 检查 Stage 7 目标 (当前是优化方向，不阻断)
        result.writeback_pass = abs(result.writeback_change) <= self.STAGE7_TARGETS['writeback_max']
        
        print(f"  目标提升: {result.target_gain:+.2%} {'✓' if result.target_gain_pass else '✗'}")
        print(f"  旧能力掉落: {result.old_ability_drop:.2%} {'✓' if result.old_ability_pass else '✗'}")
        print(f"  writeback 变化: {result.writeback_change:+.2%} {'✓' if result.writeback_pass else '○'}")
    
    def _check_e2e_success(self, result: RegressionCheckResult):
        """检查端到端成功率"""
        # 简化版 E2E 测试
        test_queries = [
            "简单查询测试 1",
            "简单查询测试 2",
            "简单查询测试 3",
            "简单查询测试 4",
            "简单查询测试 5",
        ]
        
        success_count = 0
        system = Stage6SystemOrchestrator(
            experiment_id=f"{self.check_id}_e2e",
            custom_config=self.config,
        )
        
        for query in test_queries:
            try:
                trace = system._execute_step(1, query)
                if not trace.error_message:
                    success_count += 1
            except Exception as e:
                pass
        
        result.e2e_success_rate = success_count / len(test_queries)
        result.e2e_success_pass = result.e2e_success_rate >= self.STAGE6_REQUIREMENTS['e2e_success_min']
        
        print(f"  成功率: {result.e2e_success_rate:.0%} ({success_count}/{len(test_queries)}) {'✓' if result.e2e_success_pass else '✗'}")
    
    def _check_stability(self, result: RegressionCheckResult):
        """检查稳定性 (简化版)"""
        # 运行 20 步作为稳定性快速检查
        system = Stage6SystemOrchestrator(
            experiment_id=f"{self.check_id}_stability",
            custom_config=self.config,
        )
        
        error_count = 0
        queries = [f"稳定性测试 {i}" for i in range(20)]
        
        for i, query in enumerate(queries, 1):
            try:
                trace = system._execute_step(i, query)
                if trace.error_message:
                    error_count += 1
            except Exception as e:
                error_count += 1
        
        # 通过率 > 95% 认为稳定
        stability_rate = 1 - (error_count / 20)
        result.stability_pass = stability_rate >= 0.95
        result.stability_result = f"{stability_rate:.0%} 通过" if result.stability_pass else f"{stability_rate:.0%} 不稳定"
        
        print(f"  稳定性: {result.stability_result} {'✓' if result.stability_pass else '✗'}")
    
    def _evaluate_overall(self, result: RegressionCheckResult):
        """评估总体结果"""
        print("\n" + "="*70)
        print("回归检查结果汇总")
        print("="*70)
        
        # Stage 6 必须项 (不可突破)
        stage6_checks = [
            ('目标提升 > 10%', result.target_gain_pass, result.target_gain, '+'),
            ('旧能力掉落 < 15%', result.old_ability_pass, result.old_ability_drop, ''),
            ('端到端成功率 > 90%', result.e2e_success_pass, result.e2e_success_rate, ''),
            ('稳定性通过', result.stability_pass, result.stability_result, ''),
        ]
        
        print("\nStage 6 必须项 (不可突破):")
        for name, passed, value, prefix in stage6_checks:
            status = '✓' if passed else '✗'
            val_str = f"{prefix}{value:.2%}" if isinstance(value, float) else str(value)
            print(f"  {status} {name}: {val_str}")
            if not passed:
                result.failed_items.append(name)
        
        # Stage 7 优化目标
        stage7_checks = [
            ('writeback 变化 < 5%', result.writeback_pass, result.writeback_change, '+'),
        ]
        
        print("\nStage 7 优化目标:")
        for name, passed, value, prefix in stage7_checks:
            status = '✓' if passed else '○'
            val_str = f"{prefix}{value:.2%}" if isinstance(value, float) else str(value)
            print(f"  {status} {name}: {val_str}")
        
        # 总体评估
        # Stage 6 必须项全部通过才算通过
        result.overall_pass = (
            result.target_gain_pass and
            result.old_ability_pass and
            result.e2e_success_pass and
            result.stability_pass
        )
        
        print("\n" + "-"*70)
        if result.overall_pass:
            print("✅ 回归检查通过 - Stage 6 主链未破坏")
        else:
            print("❌ 回归检查未通过 - Stage 6 主链被破坏!")
            print(f"   失败项: {', '.join(result.failed_items)}")
        print("="*70)
    
    def check_before_commit(self, result: RegressionCheckResult) -> bool:
        """
        提交前检查
        
        Returns:
            True if can commit, False otherwise
        """
        if not result.overall_pass:
            print("\n⚠️  警告: Stage 6 必须项未通过，不能提交!")
            return False
        
        print("\n✅ Stage 6 必须项全部通过，可以提交")
        
        if not result.writeback_pass:
            print("   但 writeback 优化目标未达成，建议继续优化")
        
        return True


def run_regression_gate(config_name: str = 'baseline_v1') -> RegressionCheckResult:
    """
    运行回归护栏检查
    
    这是 Stage 7 的标准入口
    """
    gate = Stage7RegressionGate(config_name=config_name)
    result = gate.run_full_regression_check(num_steps=10)
    gate.check_before_commit(result)
    return result


if __name__ == "__main__":
    # 运行回归检查
    result = run_regression_gate('baseline_v1')
