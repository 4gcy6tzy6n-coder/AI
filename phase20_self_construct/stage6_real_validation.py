"""
Stage 6 Real Validation

使用真实评估器的长期成长验证

与 stage6_long_term_validation.py 的区别:
- 使用 Stage6RealEvaluator 进行真实能力评估
- 基于标准测试集，结果可信
- 符合 stage6_eval_metric_contract.md 定义的验收标准
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from stage6_runtime_orchestrator import Stage6Orchestrator, Stage6Config
from stage6_real_evaluator import Stage6RealEvaluator, CapabilityScores, EvaluationResult


# ==================== 配置 ====================

REAL_VALIDATION_CONFIG = {
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.38,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.35,
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
    'step1_max_drop': 0.10,      # 单步最大掉落 < 10%
    'step5_max_drop': 0.15,      # 5步后 < 15%
    'step10_max_drop': 0.20,     # 10步后 < 20%
    'writeback_max_change': 0.05, # writeback < 5%
    'target_improvement_5step': 0.10,  # 5步目标提升 > 10%
    'target_improvement_10step': 0.15, # 10步目标提升 > 15%
    'rollback_recovery_rate': 0.90,    # 回滚恢复率 > 90%
}


# ==================== 数据类 ====================

@dataclass
class StepValidationResult:
    """单步验证结果"""
    step_number: int
    timestamp: str
    evaluation: EvaluationResult
    rollback_triggered: bool
    
    def to_dict(self) -> Dict:
        return {
            'step_number': self.step_number,
            'timestamp': self.timestamp,
            'evaluation': self.evaluation.to_dict(),
            'rollback_triggered': self.rollback_triggered,
        }


@dataclass
class RealValidationReport:
    """真实验证报告"""
    timestamp: str
    num_steps: int
    step_results: List[StepValidationResult]
    baseline_scores: CapabilityScores
    final_scores: CapabilityScores
    
    # 关键指标
    total_target_improvement: float
    max_old_ability_drop: float
    max_writeback_change: float
    max_single_step_drop: float
    rollback_count: int
    
    # 验收结果
    criteria_check: Dict[str, bool]
    overall_pass: bool
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'num_steps': self.num_steps,
            'step_results': [s.to_dict() for s in self.step_results],
            'baseline_scores': self.baseline_scores.to_dict(),
            'final_scores': self.final_scores.to_dict(),
            'total_target_improvement': self.total_target_improvement,
            'max_old_ability_drop': self.max_old_ability_drop,
            'max_writeback_change': self.max_writeback_change,
            'max_single_step_drop': self.max_single_step_drop,
            'rollback_count': self.rollback_count,
            'criteria_check': self.criteria_check,
            'overall_pass': self.overall_pass,
        }


# ==================== 真实验证器 ====================

class Stage6RealValidator:
    """使用真实评估器的 Stage 6 验证器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or REAL_VALIDATION_CONFIG
        self.orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        
        # 创建真实评估器
        model = self.orchestrator.backbone.get_model()
        self.evaluator = Stage6RealEvaluator(model)
        
        # 基线
        self.baseline_result: Optional[EvaluationResult] = None
        self.step_results: List[StepValidationResult] = []
        
    def setup_baseline(self):
        """设置基线"""
        print("\n" + "="*70)
        print("设置基线评估")
        print("="*70)
        
        self.baseline_result = self.evaluator.evaluate_all(num_samples=100)
        self.evaluator.set_baseline(self.baseline_result.scores)
        
        print(f"\n基线已设置:")
        print(f"  目标能力: {self.baseline_result.scores.target_score:.2%}")
        print(f"  检索能力: {self.baseline_result.scores.retrieval_score:.2%}")
        print(f"  策略能力: {self.baseline_result.scores.policy_score:.2%}")
        print(f"  治理能力: {self.baseline_result.scores.governance_score:.2%}")
        print(f"  writeback: {self.baseline_result.scores.writeback_score:.2%}")
    
    def run_step_with_evaluation(self, step_num: int, context: str) -> StepValidationResult:
        """执行单步并评估"""
        print(f"\n--- 第 {step_num} 步 ---")
        
        # 1. 执行晋升前评估
        print("  晋升前评估...")
        pre_eval = self.evaluator.evaluate_all(num_samples=50)
        
        # 2. 执行晋升
        print("  执行晋升...")
        torch.manual_seed(42 + step_num)
        promotion_result = self.orchestrator.run_single_step(context)
        
        # 3. 执行晋升后评估
        print("  晋升后评估...")
        post_eval = self.evaluator.evaluate_all(num_samples=50)
        
        # 4. 检查是否触发回滚
        rollback_triggered = promotion_result.rollback_performed if promotion_result.steps else False
        
        # 5. 记录结果
        step_result = StepValidationResult(
            step_number=step_num,
            timestamp=datetime.now().isoformat(),
            evaluation=post_eval,
            rollback_triggered=rollback_triggered,
        )
        self.step_results.append(step_result)
        
        # 6. 打印结果
        print(f"  目标提升: {post_eval.target_improvement:+.2%}")
        print(f"  旧能力掉落: {post_eval.old_ability_drop:.2%}")
        print(f"  writeback: {post_eval.writeback_change:+.2%}")
        if rollback_triggered:
            print(f"  ⚠️ 已触发回滚")
        
        return step_result
    
    def run_n_step_validation(self, n: int, contexts: List[str] = None) -> RealValidationReport:
        """运行 N 步真实验证"""
        print(f"\n{'='*70}")
        print(f"Stage 6 真实验证 - {n} 步连续晋升")
        print(f"{'='*70}")
        print(f"开始时间: {datetime.now().isoformat()}")
        print(f"评估方式: 真实评估器 (标准测试集)")
        
        # 设置基线
        self.setup_baseline()
        
        # 生成上下文
        if contexts is None:
            contexts = [f"查询 {i+1}: 主题相关查询内容" for i in range(n)]
        
        # 执行 N 步
        for i in range(n):
            self.run_step_with_evaluation(i + 1, contexts[i])
        
        # 生成报告
        return self.generate_report(n)
    
    def generate_report(self, n: int) -> RealValidationReport:
        """生成验证报告"""
        if not self.step_results or not self.baseline_result:
            raise ValueError("No validation data available")
        
        # 计算关键指标
        final_result = self.step_results[-1]
        
        total_target_improvement = final_result.evaluation.scores.target_score - self.baseline_result.scores.target_score
        
        old_drops = [s.evaluation.old_ability_drop for s in self.step_results]
        max_old_ability_drop = max(old_drops) if old_drops else 0
        max_single_step_drop = max(old_drops) if old_drops else 0
        
        writeback_changes = [s.evaluation.writeback_change for s in self.step_results]
        max_writeback_change = max(writeback_changes) if writeback_changes else 0
        
        rollback_count = sum(1 for s in self.step_results if s.rollback_triggered)
        
        # 验收检查
        criteria_check = {
            'step1_max_drop': max_single_step_drop < ACCEPTANCE_CRITERIA['step1_max_drop'],
            'step5_max_drop': max_old_ability_drop < ACCEPTANCE_CRITERIA['step5_max_drop'] if n >= 5 else True,
            'step10_max_drop': max_old_ability_drop < ACCEPTANCE_CRITERIA['step10_max_drop'] if n >= 10 else True,
            'writeback_max_change': max_writeback_change < ACCEPTANCE_CRITERIA['writeback_max_change'],
            'target_improvement': total_target_improvement > (ACCEPTANCE_CRITERIA['target_improvement_5step'] if n <= 5 else ACCEPTANCE_CRITERIA['target_improvement_10step']),
        }
        
        overall_pass = all(criteria_check.values())
        
        report = RealValidationReport(
            timestamp=datetime.now().isoformat(),
            num_steps=n,
            step_results=self.step_results,
            baseline_scores=self.baseline_result.scores,
            final_scores=final_result.evaluation.scores,
            total_target_improvement=total_target_improvement,
            max_old_ability_drop=max_old_ability_drop,
            max_writeback_change=max_writeback_change,
            max_single_step_drop=max_single_step_drop,
            rollback_count=rollback_count,
            criteria_check=criteria_check,
            overall_pass=overall_pass,
        )
        
        return report
    
    def test_rollback_recovery(self) -> Dict:
        """测试回滚恢复能力"""
        print(f"\n{'='*70}")
        print("回滚恢复测试")
        print(f"{'='*70}")
        
        # 保存基线
        self.setup_baseline()
        baseline_scores = copy.deepcopy(self.baseline_result.scores)
        
        # 执行几步（可能触发回滚）
        for i in range(3):
            self.run_step_with_evaluation(i + 1, f"查询 {i+1}")
        
        # 检查当前状态
        current_eval = self.evaluator.evaluate_all(num_samples=50)
        
        # 手动回滚
        model = self.orchestrator.backbone.get_model()
        rollback_success = self.orchestrator.rollback_hook.rollback(model)
        
        # 检查回滚后状态
        recovered_eval = self.evaluator.evaluate_all(num_samples=50)
        
        # 计算恢复率
        recovery_rates = {}
        for key in ['target_score', 'retrieval_score', 'policy_score', 'governance_score', 'writeback_score']:
            baseline_val = getattr(baseline_scores, key)
            current_val = getattr(current_eval.scores, key)
            recovered_val = getattr(recovered_eval.scores, key)
            
            if baseline_val > 0:
                pre_deviation = abs(current_val - baseline_val) / baseline_val
                post_deviation = abs(recovered_val - baseline_val) / baseline_val
                recovery_rates[key] = max(0, (pre_deviation - post_deviation) / pre_deviation * 100) if pre_deviation > 0 else 100
        
        avg_recovery = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0
        
        print(f"\n回滚恢复率:")
        for key, rate in recovery_rates.items():
            print(f"  {key}: {rate:.1f}%")
        print(f"  平均: {avg_recovery:.1f}%")
        
        return {
            'rollback_success': rollback_success,
            'recovery_rates': recovery_rates,
            'avg_recovery_rate': avg_recovery,
            'pass': avg_recovery > ACCEPTANCE_CRITERIA['rollback_recovery_rate'] * 100,
        }


# ==================== 主函数 ====================

def main():
    """主函数 - 真实验证"""
    print("="*70)
    print("Stage 6 Real Validation - 真实能力评估验证")
    print("="*70)
    print(f"开始时间: {datetime.now().isoformat()}")
    print(f"配置: 6B (base_kl=0.30/0.38, replay=0.35)")
    
    # 1. 5 步验证
    print("\n" + "="*70)
    print("验证 1: 5 步连续晋升 (真实评估)")
    print("="*70)
    
    validator_5 = Stage6RealValidator()
    report_5 = validator_5.run_n_step_validation(5)
    
    print(f"\n5 步验证结果:")
    print(f"  总体通过: {'✓' if report_5.overall_pass else '✗'}")
    print(f"  目标提升: {report_5.total_target_improvement:+.2%}")
    print(f"  最大旧能力掉落: {report_5.max_old_ability_drop:.2%}")
    print(f"  最大 writeback 变化: {report_5.max_writeback_change:.2%}")
    print(f"  回滚次数: {report_5.rollback_count}")
    
    # 2. 10 步验证
    print("\n" + "="*70)
    print("验证 2: 10 步连续晋升 (真实评估)")
    print("="*70)
    
    validator_10 = Stage6RealValidator()
    report_10 = validator_10.run_n_step_validation(10)
    
    print(f"\n10 步验证结果:")
    print(f"  总体通过: {'✓' if report_10.overall_pass else '✗'}")
    print(f"  目标提升: {report_10.total_target_improvement:+.2%}")
    print(f"  最大旧能力掉落: {report_10.max_old_ability_drop:.2%}")
    print(f"  最大 writeback 变化: {report_10.max_writeback_change:.2%}")
    print(f"  回滚次数: {report_10.rollback_count}")
    
    # 3. 回滚恢复测试
    print("\n" + "="*70)
    print("验证 3: 回滚恢复能力")
    print("="*70)
    
    validator_rollback = Stage6RealValidator()
    rollback_result = validator_rollback.test_rollback_recovery()
    
    # 4. 导出报告
    full_report = {
        'timestamp': datetime.now().isoformat(),
        'config': REAL_VALIDATION_CONFIG,
        'acceptance_criteria': ACCEPTANCE_CRITERIA,
        '5_step_report': report_5.to_dict(),
        '10_step_report': report_10.to_dict(),
        'rollback_test': rollback_result,
        'summary': {
            '5_step_pass': report_5.overall_pass,
            '10_step_pass': report_10.overall_pass,
            'rollback_pass': rollback_result['pass'],
            'overall_pass': report_5.overall_pass and report_10.overall_pass and rollback_result['pass'],
        }
    }
    
    with open("eval/stage6_real_validation_report.json", 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print("\n" + "="*70)
    print("真实验证完成")
    print("="*70)
    print(f"报告已导出: eval/stage6_real_validation_report.json")
    print(f"\n验收结果:")
    print(f"  5 步验证: {'✓ 通过' if report_5.overall_pass else '✗ 失败'}")
    print(f"  10 步验证: {'✓ 通过' if report_10.overall_pass else '✗ 失败'}")
    print(f"  回滚恢复: {'✓ 通过' if rollback_result['pass'] else '✗ 失败'}")
    print(f"  总体: {'✓ 阶段二验收通过' if full_report['summary']['overall_pass'] else '✗ 阶段二验收失败'}")


if __name__ == "__main__":
    main()
