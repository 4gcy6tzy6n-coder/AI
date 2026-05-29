"""
Stage 6 Architecture Fix Validation

架构修复后验证 - 只测 4 项关键指标:
1. Target capability
2. Old capability retention
3. Writeback drop
4. Rollback recovery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
from typing import Dict, List, Optional
from datetime import datetime

from stage6_runtime_orchestrator import Stage6Orchestrator, Stage6Config
from stage6_real_evaluator import Stage6RealEvaluator, EvaluationResult
from stage6_writeback_isolation_fix import WritebackIsolationWrapper, apply_writeback_isolation_fix
from stage6_full_rollback_snapshot import FullRollbackManager


# ==================== 架构修复配置 ====================

ARCH_FIX_CONFIG = {
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

# 验收标准
ACCEPTANCE_CRITERIA = {
    'target_improvement': 0.10,      # > +10%
    'old_ability_drop': 0.15,        # < 15%
    'writeback_change': 0.05,        # < 5%
    'rollback_recovery': 0.90,       # > 90%
}


class Stage6ArchFixValidator:
    """架构修复验证器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or ARCH_FIX_CONFIG
        self.orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        
        # 获取模型
        self.model = self.orchestrator.backbone.get_model()
        
        # 应用 writeback 隔离
        self.writeback_wrapper = apply_writeback_isolation_fix(self.model)
        
        # 创建评估器
        self.evaluator = Stage6RealEvaluator(self.model)
        
        # rollback 管理器延迟初始化
        self.rollback_manager = None
        
        # 基线
        self.baseline_result: Optional[EvaluationResult] = None
        self.baseline_snapshot_id: Optional[str] = None
        
    def setup_baseline(self):
        """设置基线 - 使用完整快照"""
        print("\n" + "="*70)
        print("设置基线 (架构修复版)")
        print("="*70)
        
        # 评估基线
        self.baseline_result = self.evaluator.evaluate_all(num_samples=100)
        self.evaluator.set_baseline(self.baseline_result.scores)
        
        # 延迟初始化 rollback 管理器
        if self.rollback_manager is None:
            # 获取优化器 (param_promoter 现在应该已初始化)
            optimizer = None
            if self.orchestrator.param_promoter:
                optimizer = self.orchestrator.param_promoter.optimizer
            self.rollback_manager = FullRollbackManager(self.model, optimizer)
        
        # 创建完整快照
        self.baseline_snapshot_id = self.rollback_manager.create_snapshot(
            "baseline",
            {
                'description': 'Baseline with arch fixes',
                'target_score': self.baseline_result.scores.target_score,
                'writeback_score': self.baseline_result.scores.writeback_score,
            }
        )
        
        print(f"\n基线已设置:")
        print(f"  目标能力: {self.baseline_result.scores.target_score:.2%}")
        print(f"  检索能力: {self.baseline_result.scores.retrieval_score:.2%}")
        print(f"  writeback: {self.baseline_result.scores.writeback_score:.2%}")
        print(f"  快照ID: {self.baseline_snapshot_id}")
    
    def run_promotion_step(self, step_num: int, context: str, protect_writeback: bool = True) -> Dict:
        """执行晋升步骤"""
        print(f"\n--- 第 {step_num} 步 ---")
        
        # 晋升前评估
        print("  晋升前评估...")
        pre_eval = self.evaluator.evaluate_all(num_samples=50)
        
        # 冻结 writeback (架构修复)
        if protect_writeback:
            self.writeback_wrapper.freeze_writeback()
            print("  Writeback 已冻结")
        
        # 执行晋升
        print("  执行晋升...")
        torch.manual_seed(42 + step_num)
        promotion_result = self.orchestrator.run_single_step(context)
        
        # 解冻 writeback
        if protect_writeback:
            self.writeback_wrapper.unfreeze_writeback()
        
        # 晋升后评估
        print("  晋升后评估...")
        post_eval = self.evaluator.evaluate_all(num_samples=50)
        
        return {
            'step': step_num,
            'pre_eval': pre_eval,
            'post_eval': post_eval,
            'target_improvement': post_eval.target_improvement,
            'old_ability_drop': post_eval.old_ability_drop,
            'writeback_change': post_eval.writeback_change,
            'rollback_triggered': promotion_result.rollback_performed if promotion_result.steps else False,
        }
    
    def test_5_step(self) -> Dict:
        """测试 5 步连续晋升"""
        print("\n" + "="*70)
        print("测试 1: 5 步连续晋升 (架构修复)")
        print("="*70)
        
        self.setup_baseline()
        
        results = []
        for i in range(5):
            result = self.run_promotion_step(i + 1, f"查询 {i+1}")
            results.append(result)
            
            print(f"  目标提升: {result['target_improvement']:+.2%}")
            print(f"  旧能力掉落: {result['old_ability_drop']:.2%}")
            print(f"  writeback: {result['writeback_change']:+.2%}")
        
        # 计算指标
        final_result = results[-1]
        max_old_drop = max(r['old_ability_drop'] for r in results)
        max_writeback_change = max(abs(r['writeback_change']) for r in results)
        
        return {
            'target_improvement': final_result['target_improvement'],
            'max_old_ability_drop': max_old_drop,
            'max_writeback_change': max_writeback_change,
            'pass_target': final_result['target_improvement'] > ACCEPTANCE_CRITERIA['target_improvement'],
            'pass_old': max_old_drop < ACCEPTANCE_CRITERIA['old_ability_drop'],
            'pass_writeback': max_writeback_change < ACCEPTANCE_CRITERIA['writeback_change'],
            'results': results,
        }
    
    def test_rollback_recovery(self) -> Dict:
        """测试回滚恢复 (使用完整快照)"""
        print("\n" + "="*70)
        print("测试 2: 回滚恢复 (完整快照)")
        print("="*70)
        
        self.setup_baseline()
        baseline_scores = copy.deepcopy(self.baseline_result.scores)
        
        print(f"\n基线状态:")
        print(f"  target: {baseline_scores.target_score:.2%}")
        print(f"  writeback: {baseline_scores.writeback_score:.2%}")
        
        # 执行 3 步晋升
        print("\n执行 3 步晋升...")
        for i in range(3):
            self.run_promotion_step(i + 1, f"查询 {i+1}")
        
        # 检查晋升后状态
        print("\n晋升后评估...")
        current_eval = self.evaluator.evaluate_all(num_samples=50)
        print(f"  target: {current_eval.scores.target_score:.2%}")
        print(f"  writeback: {current_eval.scores.writeback_score:.2%}")
        
        # 使用完整快照恢复
        print("\n执行完整快照恢复...")
        success = self.rollback_manager.restore_snapshot(self.baseline_snapshot_id)
        
        if not success:
            return {
                'recovery_rate': 0.0,
                'pass': False,
                'error': 'Snapshot restore failed',
            }
        
        # 验证恢复
        print("\n恢复后评估...")
        recovered_eval = self.evaluator.evaluate_all(num_samples=50)
        print(f"  target: {recovered_eval.scores.target_score:.2%}")
        print(f"  writeback: {recovered_eval.scores.writeback_score:.2%}")
        
        # 计算恢复率
        recovery_rates = {}
        for key in ['target_score', 'retrieval_score', 'policy_score', 'governance_score', 'writeback_score']:
            baseline_val = getattr(baseline_scores, key)
            current_val = getattr(current_eval.scores, key)
            recovered_val = getattr(recovered_eval.scores, key)
            
            if baseline_val > 0:
                pre_deviation = abs(current_val - baseline_val) / baseline_val
                post_deviation = abs(recovered_val - baseline_val) / baseline_val
                recovery_rate = max(0, (pre_deviation - post_deviation) / pre_deviation * 100) if pre_deviation > 0 else 100
                recovery_rates[key] = recovery_rate
        
        avg_recovery = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0
        
        print(f"\n恢复率:")
        for key, rate in recovery_rates.items():
            print(f"  {key}: {rate:.1f}%")
        print(f"  平均: {avg_recovery:.1f}%")
        
        # 验证完全恢复
        full_recovery = all(rate > 95 for rate in recovery_rates.values())
        
        return {
            'recovery_rates': recovery_rates,
            'avg_recovery_rate': avg_recovery,
            'full_recovery': full_recovery,
            'pass': avg_recovery > ACCEPTANCE_CRITERIA['rollback_recovery'] * 100,
        }
    
    def run_full_validation(self) -> Dict:
        """运行完整验证"""
        print("="*70)
        print("Stage 6 架构修复验证")
        print("="*70)
        print(f"修复内容:")
        print(f"  - Writeback 隔离 (独立 head)")
        print(f"  - 完整 Rollback 快照")
        print(f"开始时间: {datetime.now().isoformat()}")
        
        # 测试 1: 5 步连续晋升
        test_5_result = self.test_5_step()
        
        # 测试 2: 回滚恢复
        test_rollback_result = self.test_rollback_recovery()
        
        # 汇总结果
        summary = {
            'target_improvement': test_5_result['target_improvement'],
            'max_old_ability_drop': test_5_result['max_old_ability_drop'],
            'max_writeback_change': test_5_result['max_writeback_change'],
            'rollback_recovery_rate': test_rollback_result['avg_recovery_rate'],
            'pass_target': test_5_result['pass_target'],
            'pass_old': test_5_result['pass_old'],
            'pass_writeback': test_5_result['pass_writeback'],
            'pass_rollback': test_rollback_result['pass'],
            'overall_pass': (
                test_5_result['pass_target'] and
                test_5_result['pass_old'] and
                test_5_result['pass_writeback'] and
                test_rollback_result['pass']
            ),
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'config': ARCH_FIX_CONFIG,
            'acceptance_criteria': ACCEPTANCE_CRITERIA,
            'test_5_step': test_5_result,
            'test_rollback': test_rollback_result,
            'summary': summary,
        }


def main():
    """主函数"""
    validator = Stage6ArchFixValidator()
    report = validator.run_full_validation()
    
    # 导出报告
    with open("eval/stage6_arch_fix_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 打印结果
    print("\n" + "="*70)
    print("架构修复验证完成")
    print("="*70)
    print(f"报告已导出: eval/stage6_arch_fix_report.json")
    
    print(f"\n关键指标:")
    print(f"  目标提升: {report['summary']['target_improvement']:+.2%} (目标: >{ACCEPTANCE_CRITERIA['target_improvement']:.0%}) {'✓' if report['summary']['pass_target'] else '✗'}")
    print(f"  旧能力掉落: {report['summary']['max_old_ability_drop']:.2%} (目标: <{ACCEPTANCE_CRITERIA['old_ability_drop']:.0%}) {'✓' if report['summary']['pass_old'] else '✗'}")
    print(f"  writeback 变化: {report['summary']['max_writeback_change']:.2%} (目标: <{ACCEPTANCE_CRITERIA['writeback_change']:.0%}) {'✓' if report['summary']['pass_writeback'] else '✗'}")
    print(f"  回滚恢复率: {report['summary']['rollback_recovery_rate']:.1f}% (目标: >{ACCEPTANCE_CRITERIA['rollback_recovery']*100:.0f}%) {'✓' if report['summary']['pass_rollback'] else '✗'}")
    
    print(f"\n总体结果: {'✓ 阶段二验收通过' if report['summary']['overall_pass'] else '✗ 阶段二验收失败'}")
    
    if report['summary']['overall_pass']:
        print("\n🎉 所有验收项通过！Stage 6 第二阶段正式完成。")
        print("可以进入第三阶段: 系统级评估")
    else:
        print("\n⚠️ 仍有验收项未通过。")
        print("需要进一步架构修复或调整验收标准。")


if __name__ == "__main__":
    main()
