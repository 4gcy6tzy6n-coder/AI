"""
Stage 6 最小消融测试

三组消融:
1. 仅开 writeback isolation
2. 仅开 full rollback snapshot
3. 两者同时开

只看四个指标:
- target gain
- old ability drop
- writeback change
- rollback recovery

目标: 确定是谁破坏了 target gain
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
from typing import Dict
from datetime import datetime

from stage6_runtime_orchestrator import Stage6Orchestrator, Stage6Config
from stage6_real_evaluator import Stage6RealEvaluator
from stage6_writeback_isolation_fix import apply_writeback_isolation_fix
from stage6_full_rollback_snapshot import FullRollbackManager


# 基线配置 (修复前)
BASELINE_CONFIG = {
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}


class AblationTestRunner:
    """消融测试运行器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or BASELINE_CONFIG
        self.results = {}
    
    def test_baseline(self) -> Dict:
        """测试 0: 基线 (无任何修复)"""
        print("\n" + "="*70)
        print("消融测试 0: 基线 (无修复)")
        print("="*70)
        
        orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        model = orchestrator.backbone.get_model()
        evaluator = Stage6RealEvaluator(model)
        
        # 设置基线
        baseline_result = evaluator.evaluate_all(num_samples=50)
        evaluator.set_baseline(baseline_result.scores)
        
        print(f"基线: target={baseline_result.scores.target_score:.2%}")
        
        # 执行 3 步晋升
        results = []
        for i in range(3):
            torch.manual_seed(42 + i)
            orchestrator.run_single_step(f"查询 {i+1}")
            post_eval = evaluator.evaluate_all(num_samples=30)
            results.append({
                'target_improvement': post_eval.target_improvement,
                'old_ability_drop': post_eval.old_ability_drop,
                'writeback_change': post_eval.writeback_change,
            })
            print(f"  步骤 {i+1}: target={post_eval.target_improvement:+.2%}, writeback={post_eval.writeback_change:+.2%}")
        
        final = results[-1]
        return {
            'name': 'baseline',
            'target_gain': final['target_improvement'],
            'old_drop': final['old_ability_drop'],
            'writeback_change': final['writeback_change'],
            'rollback_recovery': None,  # 基线不测 rollback
        }
    
    def test_writeback_only(self) -> Dict:
        """测试 1: 仅 writeback isolation"""
        print("\n" + "="*70)
        print("消融测试 1: 仅 writeback isolation")
        print("="*70)
        
        orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        model = orchestrator.backbone.get_model()
        
        # 应用 writeback 隔离
        wrapper = apply_writeback_isolation_fix(model)
        
        evaluator = Stage6RealEvaluator(model)
        
        # 设置基线
        baseline_result = evaluator.evaluate_all(num_samples=50)
        evaluator.set_baseline(baseline_result.scores)
        
        print(f"基线: target={baseline_result.scores.target_score:.2%}")
        
        # 执行 3 步晋升 (带 writeback 冻结)
        results = []
        for i in range(3):
            wrapper.freeze_writeback()
            torch.manual_seed(42 + i)
            orchestrator.run_single_step(f"查询 {i+1}")
            wrapper.unfreeze_writeback()
            
            post_eval = evaluator.evaluate_all(num_samples=30)
            results.append({
                'target_improvement': post_eval.target_improvement,
                'old_ability_drop': post_eval.old_ability_drop,
                'writeback_change': post_eval.writeback_change,
            })
            print(f"  步骤 {i+1}: target={post_eval.target_improvement:+.2%}, writeback={post_eval.writeback_change:+.2%}")
        
        final = results[-1]
        return {
            'name': 'writeback_only',
            'target_gain': final['target_improvement'],
            'old_drop': final['old_ability_drop'],
            'writeback_change': final['writeback_change'],
            'rollback_recovery': None,
        }
    
    def test_rollback_only(self) -> Dict:
        """测试 2: 仅 full rollback snapshot"""
        print("\n" + "="*70)
        print("消融测试 2: 仅 full rollback snapshot")
        print("="*70)
        
        orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        model = orchestrator.backbone.get_model()
        evaluator = Stage6RealEvaluator(model)
        
        # 延迟初始化 rollback manager
        rollback_manager = None
        
        # 设置基线
        baseline_result = evaluator.evaluate_all(num_samples=50)
        evaluator.set_baseline(baseline_result.scores)
        
        print(f"基线: target={baseline_result.scores.target_score:.2%}")
        
        # 创建完整快照
        optimizer = orchestrator.param_promoter.optimizer if orchestrator.param_promoter else None
        rollback_manager = FullRollbackManager(model, optimizer)
        snapshot_id = rollback_manager.create_snapshot("baseline", {})
        
        # 执行 3 步晋升
        results = []
        for i in range(3):
            torch.manual_seed(42 + i)
            orchestrator.run_single_step(f"查询 {i+1}")
            post_eval = evaluator.evaluate_all(num_samples=30)
            results.append({
                'target_improvement': post_eval.target_improvement,
                'old_ability_drop': post_eval.old_ability_drop,
                'writeback_change': post_eval.writeback_change,
            })
            print(f"  步骤 {i+1}: target={post_eval.target_improvement:+.2%}, writeback={post_eval.writeback_change:+.2%}")
        
        # 测试 rollback 恢复
        print("\n  测试 rollback 恢复...")
        current_eval = evaluator.evaluate_all(num_samples=30)
        
        success = rollback_manager.restore_snapshot(snapshot_id)
        recovered_eval = evaluator.evaluate_all(num_samples=30)
        
        # 计算恢复率
        baseline_scores = baseline_result.scores
        recovery_rates = {}
        for key in ['target_score', 'writeback_score']:
            baseline_val = getattr(baseline_scores, key)
            current_val = getattr(current_eval.scores, key)
            recovered_val = getattr(recovered_eval.scores, key)
            
            if baseline_val > 0:
                pre_deviation = abs(current_val - baseline_val) / baseline_val
                post_deviation = abs(recovered_val - baseline_val) / baseline_val
                recovery_rate = max(0, (pre_deviation - post_deviation) / pre_deviation * 100) if pre_deviation > 0 else 100
                recovery_rates[key] = recovery_rate
        
        avg_recovery = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0
        print(f"  恢复率: {avg_recovery:.1f}%")
        
        final = results[-1]
        return {
            'name': 'rollback_only',
            'target_gain': final['target_improvement'],
            'old_drop': final['old_ability_drop'],
            'writeback_change': final['writeback_change'],
            'rollback_recovery': avg_recovery,
        }
    
    def test_both(self) -> Dict:
        """测试 3: writeback + rollback 同时开"""
        print("\n" + "="*70)
        print("消融测试 3: writeback + rollback 同时开")
        print("="*70)
        
        orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        model = orchestrator.backbone.get_model()
        
        # 应用 writeback 隔离
        wrapper = apply_writeback_isolation_fix(model)
        
        evaluator = Stage6RealEvaluator(model)
        
        # 延迟初始化 rollback manager
        rollback_manager = None
        
        # 设置基线
        baseline_result = evaluator.evaluate_all(num_samples=50)
        evaluator.set_baseline(baseline_result.scores)
        
        print(f"基线: target={baseline_result.scores.target_score:.2%}")
        
        # 延迟创建完整快照
        optimizer = orchestrator.param_promoter.optimizer if orchestrator.param_promoter else None
        rollback_manager = FullRollbackManager(model, optimizer)
        snapshot_id = rollback_manager.create_snapshot("baseline", {})
        
        # 执行 3 步晋升 (带 writeback 冻结)
        results = []
        for i in range(3):
            wrapper.freeze_writeback()
            torch.manual_seed(42 + i)
            orchestrator.run_single_step(f"查询 {i+1}")
            wrapper.unfreeze_writeback()
            
            post_eval = evaluator.evaluate_all(num_samples=30)
            results.append({
                'target_improvement': post_eval.target_improvement,
                'old_ability_drop': post_eval.old_ability_drop,
                'writeback_change': post_eval.writeback_change,
            })
            print(f"  步骤 {i+1}: target={post_eval.target_improvement:+.2%}, writeback={post_eval.writeback_change:+.2%}")
        
        # 测试 rollback 恢复
        print("\n  测试 rollback 恢复...")
        current_eval = evaluator.evaluate_all(num_samples=30)
        
        success = rollback_manager.restore_snapshot(snapshot_id)
        recovered_eval = evaluator.evaluate_all(num_samples=30)
        
        # 计算恢复率
        baseline_scores = baseline_result.scores
        recovery_rates = {}
        for key in ['target_score', 'writeback_score']:
            baseline_val = getattr(baseline_scores, key)
            current_val = getattr(current_eval.scores, key)
            recovered_val = getattr(recovered_eval.scores, key)
            
            if baseline_val > 0:
                pre_deviation = abs(current_val - baseline_val) / baseline_val
                post_deviation = abs(recovered_val - baseline_val) / baseline_val
                recovery_rate = max(0, (pre_deviation - post_deviation) / pre_deviation * 100) if pre_deviation > 0 else 100
                recovery_rates[key] = recovery_rate
        
        avg_recovery = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0
        print(f"  恢复率: {avg_recovery:.1f}%")
        
        final = results[-1]
        return {
            'name': 'both',
            'target_gain': final['target_improvement'],
            'old_drop': final['old_ability_drop'],
            'writeback_change': final['writeback_change'],
            'rollback_recovery': avg_recovery,
        }
    
    def run_all_tests(self) -> Dict:
        """运行所有消融测试"""
        print("="*70)
        print("Stage 6 最小消融测试")
        print("="*70)
        print(f"开始时间: {datetime.now().isoformat()}")
        
        # 运行四组测试
        self.results['baseline'] = self.test_baseline()
        self.results['writeback_only'] = self.test_writeback_only()
        self.results['rollback_only'] = self.test_rollback_only()
        self.results['both'] = self.test_both()
        
        return self.results
    
    def generate_report(self) -> Dict:
        """生成诊断报告"""
        print("\n" + "="*70)
        print("消融测试结果汇总")
        print("="*70)
        
        # 打印对比表
        print(f"\n{'配置':<20} {'Target':<12} {'Old Drop':<12} {'Writeback':<12} {'Rollback':<12}")
        print("-" * 70)
        
        for name, result in self.results.items():
            target = f"{result['target_gain']:+.2%}"
            old = f"{result['old_drop']:.2%}"
            wb = f"{result['writeback_change']:+.2%}"
            rb = f"{result['rollback_recovery']:.1f}%" if result['rollback_recovery'] is not None else "N/A"
            print(f"{name:<20} {target:<12} {old:<12} {wb:<12} {rb:<12}")
        
        # 分析谁破坏了 target gain
        baseline_target = self.results['baseline']['target_gain']
        writeback_target = self.results['writeback_only']['target_gain']
        rollback_target = self.results['rollback_only']['target_gain']
        both_target = self.results['both']['target_gain']
        
        print("\n" + "="*70)
        print("诊断分析")
        print("="*70)
        
        # 判断破坏者
        writeback_damage = baseline_target - writeback_target
        rollback_damage = baseline_target - rollback_target
        both_damage = baseline_target - both_target
        
        print(f"\nTarget Gain 损失分析:")
        print(f"  基线: {baseline_target:+.2%}")
        print(f"  writeback_only: {writeback_target:+.2%} (损失: {writeback_damage:+.2%})")
        print(f"  rollback_only: {rollback_target:+.2%} (损失: {rollback_damage:+.2%})")
        print(f"  both: {both_target:+.2%} (损失: {both_damage:+.2%})")
        
        # 确定主要破坏者
        if writeback_damage > 0.02 and rollback_damage > 0.02:
            culprit = "两者都有显著影响"
        elif writeback_damage > 0.02:
            culprit = "writeback isolation"
        elif rollback_damage > 0.02:
            culprit = "rollback snapshot"
        else:
            culprit = "无明显破坏者 (可能是随机性)"
        
        print(f"\n主要破坏者: {culprit}")
        
        # 建议
        print(f"\n建议:")
        if writeback_damage > 0.02:
            print("  - writeback isolation 拖累了 target gain，建议移除")
        if rollback_damage > 0.02:
            print("  - rollback snapshot 拖累了 target gain，建议移除")
        if writeback_damage <= 0.02 and rollback_damage <= 0.02:
            print("  - 消融测试未显示明显破坏者，之前的结果可能是随机波动")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'analysis': {
                'baseline_target': baseline_target,
                'writeback_damage': writeback_damage,
                'rollback_damage': rollback_damage,
                'both_damage': both_damage,
                'culprit': culprit,
            },
            'recommendation': self._generate_recommendation(culprit, writeback_damage, rollback_damage),
        }
        
        return report
    
    def _generate_recommendation(self, culprit: str, wb_damage: float, rb_damage: float) -> str:
        """生成建议"""
        if wb_damage > 0.02 and rb_damage > 0.02:
            return "移除所有修复，使用基线进入第三阶段，接受 writeback/rollback 缺陷"
        elif wb_damage > 0.02:
            return "移除 writeback isolation，保留 rollback snapshot"
        elif rb_damage > 0.02:
            return "移除 rollback snapshot，保留 writeback isolation"
        else:
            return "消融测试无结论，建议重复测试或接受基线"


def main():
    """主函数"""
    runner = AblationTestRunner()
    results = runner.run_all_tests()
    report = runner.generate_report()
    
    # 导出报告
    with open("eval/stage6_ablation_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("消融测试完成")
    print("="*70)
    print(f"报告已导出: eval/stage6_ablation_report.json")
    print(f"\n最终建议: {report['recommendation']}")


if __name__ == "__main__":
    main()
