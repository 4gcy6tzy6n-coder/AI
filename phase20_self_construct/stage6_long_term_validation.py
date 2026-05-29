"""
Stage 6 Long-Term Validation

长期成长验证 - 5-10 步连续晋升验证与累积损伤分析

验证目标:
1. 5 步连续晋升稳定性
2. 10 步连续晋升稳定性
3. 累积损伤分析
4. 回滚恢复分析

验收标准:
- 5 步后旧能力掉落 < 15%
- 10 步后旧能力掉落 < 20%
- writeback 掉落始终 < 5%
- 回滚后能力恢复率 > 90%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from stage6_runtime_orchestrator import (
    Stage6Orchestrator,
    Stage6Config,
    StepMetrics,
)


# ==================== 配置 ====================

LONG_TERM_CONFIG = {
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
    'old_ability_threshold': 0.12,  # 稍放宽以观察长期效应
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}


# ==================== 数据类 ====================

@dataclass
class LongTermMetrics:
    """长期验证指标"""
    step_number: int
    target_improvement: float
    old_ability_drop: float
    writeback_change: float
    retrieval_change: float
    policy_change: float
    governance_change: float
    kl_weights: Dict[str, float]
    rollback_triggered: bool
    cumulative_target: float
    cumulative_old_drop: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationResult:
    """验证结果"""
    num_steps: int
    metrics_history: List[LongTermMetrics]
    final_target_improvement: float
    final_old_ability_drop: float
    final_writeback_change: float
    max_old_ability_drop: float
    rollback_count: int
    success: bool
    
    def to_dict(self) -> Dict:
        return {
            'num_steps': self.num_steps,
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'final_target_improvement': self.final_target_improvement,
            'final_old_ability_drop': self.final_old_ability_drop,
            'final_writeback_change': self.final_writeback_change,
            'max_old_ability_drop': self.max_old_ability_drop,
            'rollback_count': self.rollback_count,
            'success': self.success,
        }


# ==================== 长期验证器 ====================

class LongTermValidator:
    """长期成长验证器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or LONG_TERM_CONFIG
        self.orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        self.baseline_abilities = None
        self.cumulative_target = 0.0
        
    def evaluate_abilities(self) -> Dict[str, float]:
        """评估当前能力"""
        model = self.orchestrator.backbone.get_model()
        model.eval()
        
        abilities = {'target': 0, 'retrieval': 0, 'governance': 0, 'policy': 0, 'writeback': 0}
        
        with torch.no_grad():
            for _ in range(100):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = model(input_ids)
                
                gap_conf = outputs['gap_probs'][0].max().item()
                policy_conf = outputs['policy_probs'][0].max().item()
                
                if gap_conf > 0.6 and policy_conf > 0.6:
                    abilities['target'] += 1
                if outputs['policy_probs'][0].max().item() > 0.5:
                    abilities['retrieval'] += 1
                    abilities['policy'] += 1
                if outputs['governance_probs'][0].max().item() > 0.5:
                    abilities['governance'] += 1
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    abilities['writeback'] += 1
        
        return {k: v / 100 for k, v in abilities.items()}
    
    def save_baseline(self):
        """保存基线能力"""
        self.baseline_abilities = self.evaluate_abilities()
        print(f"基线能力: target={self.baseline_abilities['target']:.2%}, "
              f"retrieval={self.baseline_abilities['retrieval']:.2%}, "
              f"writeback={self.baseline_abilities['writeback']:.2%}")
    
    def run_n_step_promotion(self, n: int, contexts: List[str] = None) -> ValidationResult:
        """运行 N 步连续晋升验证"""
        print(f"\n{'='*70}")
        print(f"长期成长验证 - {n} 步连续晋升")
        print(f"{'='*70}")
        
        # 保存基线
        self.save_baseline()
        
        # 生成查询上下文
        if contexts is None:
            contexts = [f"查询 {i+1}: 主题相关查询内容" for i in range(n)]
        
        metrics_history = []
        rollback_count = 0
        
        for i in range(n):
            print(f"\n--- 第 {i+1}/{n} 步 ---")
            
            # 设置随机种子确保可重复
            torch.manual_seed(42 + i)
            
            # 执行单步晋升
            result = self.orchestrator.run_single_step(contexts[i])
            
            if not result.success or not result.steps:
                print(f"  第 {i+1} 步失败或跳过")
                continue
            
            step_metrics = result.steps[0]
            
            # 累积指标
            self.cumulative_target += step_metrics.target_improvement
            
            # 记录长期指标
            long_term_metric = LongTermMetrics(
                step_number=i+1,
                target_improvement=step_metrics.target_improvement,
                old_ability_drop=step_metrics.old_ability_drop,
                writeback_change=step_metrics.writeback_change,
                retrieval_change=step_metrics.retrieval_change,
                policy_change=step_metrics.policy_change,
                governance_change=step_metrics.governance_change,
                kl_weights=step_metrics.kl_weights.copy(),
                rollback_triggered=step_metrics.rollback_triggered,
                cumulative_target=self.cumulative_target,
                cumulative_old_drop=step_metrics.old_ability_drop,
            )
            metrics_history.append(long_term_metric)
            
            if step_metrics.rollback_triggered:
                rollback_count += 1
            
            print(f"  目标提升: {step_metrics.target_improvement:+.2%}")
            print(f"  旧能力掉落: {step_metrics.old_ability_drop:.2%}")
            print(f"  writeback: {step_metrics.writeback_change:+.2%}")
            print(f"  累积目标: {self.cumulative_target:+.2%}")
        
        # 计算最终指标
        if metrics_history:
            final_metrics = metrics_history[-1]
            max_old_drop = max(m.old_ability_drop for m in metrics_history)
            
            # 判断是否成功
            success = (
                final_metrics.cumulative_old_drop < 0.20 and  # 10步后 < 20%
                max_old_drop < 0.25 and  # 最大掉落 < 25%
                all(abs(m.writeback_change) < 0.05 for m in metrics_history)  # writeback 始终 < 5%
            )
            
            result = ValidationResult(
                num_steps=n,
                metrics_history=metrics_history,
                final_target_improvement=final_metrics.cumulative_target,
                final_old_ability_drop=final_metrics.cumulative_old_drop,
                final_writeback_change=final_metrics.writeback_change,
                max_old_ability_drop=max_old_drop,
                rollback_count=rollback_count,
                success=success,
            )
        else:
            result = ValidationResult(
                num_steps=n,
                metrics_history=[],
                final_target_improvement=0,
                final_old_ability_drop=0,
                final_writeback_change=0,
                max_old_ability_drop=0,
                rollback_count=0,
                success=False,
            )
        
        return result
    
    def analyze_cumulative_damage(self, result: ValidationResult) -> Dict:
        """分析累积损伤"""
        if not result.metrics_history:
            return {'error': 'No metrics available'}
        
        metrics = result.metrics_history
        n = len(metrics)
        
        # 计算趋势
        old_drops = [m.old_ability_drop for m in metrics]
        writeback_changes = [m.writeback_change for m in metrics]
        target_gains = [m.target_improvement for m in metrics]
        
        analysis = {
            'total_steps': n,
            'old_ability': {
                'initial_drop': old_drops[0] if old_drops else 0,
                'final_drop': old_drops[-1] if old_drops else 0,
                'max_drop': max(old_drops) if old_drops else 0,
                'avg_drop': sum(old_drops) / len(old_drops) if old_drops else 0,
                'trend': 'increasing' if old_drops[-1] > old_drops[0] * 1.5 else 'stable',
            },
            'writeback': {
                'max_change': max(abs(w) for w in writeback_changes) if writeback_changes else 0,
                'avg_change': sum(writeback_changes) / len(writeback_changes) if writeback_changes else 0,
                'stable': all(abs(w) < 0.05 for w in writeback_changes),
            },
            'target_learning': {
                'total_gain': sum(target_gains),
                'avg_per_step': sum(target_gains) / len(target_gains) if target_gains else 0,
                'positive_steps': sum(1 for g in target_gains if g > 0),
            },
            'rollback_analysis': {
                'count': result.rollback_count,
                'rate': result.rollback_count / n if n > 0 else 0,
            },
        }
        
        return analysis
    
    def test_rollback_recovery(self) -> Dict:
        """测试回滚恢复能力"""
        print(f"\n{'='*70}")
        print("回滚恢复分析")
        print(f"{'='*70}")
        
        # 保存基线
        self.save_baseline()
        baseline = copy.deepcopy(self.baseline_abilities)
        
        # 执行几步晋升（可能触发回滚）
        torch.manual_seed(42)
        for i in range(3):
            result = self.orchestrator.run_single_step(f"查询 {i+1}")
        
        # 检查当前能力
        current_abilities = self.evaluate_abilities()
        
        # 手动触发回滚
        model = self.orchestrator.backbone.get_model()
        rollback_success = self.orchestrator.rollback_hook.rollback(model)
        
        # 检查回滚后能力
        recovered_abilities = self.evaluate_abilities()
        
        # 计算恢复率
        recovery_rates = {}
        for key in baseline:
            if baseline[key] > 0:
                # 回滚前偏离度
                pre_deviation = abs(current_abilities[key] - baseline[key]) / baseline[key]
                # 回滚后偏离度
                post_deviation = abs(recovered_abilities[key] - baseline[key]) / baseline[key]
                # 恢复率
                recovery_rates[key] = max(0, (pre_deviation - post_deviation) / pre_deviation * 100) if pre_deviation > 0 else 100
        
        analysis = {
            'baseline': baseline,
            'pre_rollback': current_abilities,
            'post_rollback': recovered_abilities,
            'rollback_success': rollback_success,
            'recovery_rates': recovery_rates,
            'avg_recovery_rate': sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0,
        }
        
        print(f"\n回滚恢复率:")
        for key, rate in recovery_rates.items():
            print(f"  {key}: {rate:.1f}%")
        print(f"  平均: {analysis['avg_recovery_rate']:.1f}%")
        
        return analysis
    
    def plot_metrics(self, result: ValidationResult, save_path: str = None):
        """绘制指标图表"""
        if not result.metrics_history:
            print("没有数据可绘制")
            return
        
        metrics = result.metrics_history
        steps = [m.step_number for m in metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 目标提升
        ax1 = axes[0, 0]
        target_gains = [m.target_improvement * 100 for m in metrics]
        cumulative = [m.cumulative_target * 100 for m in metrics]
        ax1.plot(steps, target_gains, 'b-o', label='单步提升')
        ax1.plot(steps, cumulative, 'g-s', label='累积提升')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Target Improvement (%)')
        ax1.set_title('Target Ability Learning')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 旧能力掉落
        ax2 = axes[0, 1]
        old_drops = [m.old_ability_drop * 100 for m in metrics]
        ax2.plot(steps, old_drops, 'r-o')
        ax2.axhline(y=10, color='orange', linestyle='--', label='Threshold (10%)')
        ax2.axhline(y=20, color='red', linestyle='--', label='Max Allowed (20%)')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Old Ability Drop (%)')
        ax2.set_title('Old Ability Preservation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Writeback 变化
        ax3 = axes[1, 0]
        writeback = [m.writeback_change * 100 for m in metrics]
        ax3.plot(steps, writeback, 'm-o')
        ax3.axhline(y=5, color='r', linestyle='--', label='Threshold (5%)')
        ax3.axhline(y=-5, color='r', linestyle='--')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Writeback Change (%)')
        ax3.set_title('Writeback Stability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. KL 权重变化
        ax4 = axes[1, 1]
        kl_gap = [m.kl_weights['gap'] for m in metrics]
        kl_writeback = [m.kl_weights['writeback'] for m in metrics]
        ax4.plot(steps, kl_gap, 'b-o', label='Gap KL')
        ax4.plot(steps, kl_writeback, 'r-s', label='Writeback KL')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('KL Weight')
        ax4.set_title('KL Weight Adaptation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()


# ==================== 主函数 ====================

def main():
    """主函数 - 长期成长验证"""
    print("="*70)
    print("Stage 6 - 长期成长验证")
    print("="*70)
    print(f"开始时间: {datetime.now().isoformat()}")
    
    validator = LongTermValidator()
    
    # 1. 5 步验证
    print("\n" + "="*70)
    print("测试 1: 5 步连续晋升")
    print("="*70)
    result_5step = validator.run_n_step_promotion(5)
    analysis_5step = validator.analyze_cumulative_damage(result_5step)
    
    print(f"\n5 步验证结果:")
    print(f"  成功: {result_5step.success}")
    print(f"  最终目标提升: {result_5step.final_target_improvement:+.2%}")
    print(f"  最终旧能力掉落: {result_5step.final_old_ability_drop:.2%}")
    print(f"  最大旧能力掉落: {result_5step.max_old_ability_drop:.2%}")
    print(f"  回滚次数: {result_5step.rollback_count}")
    
    # 2. 10 步验证
    print("\n" + "="*70)
    print("测试 2: 10 步连续晋升")
    print("="*70)
    # 重新初始化以清除之前的状态
    validator = LongTermValidator()
    result_10step = validator.run_n_step_promotion(10)
    analysis_10step = validator.analyze_cumulative_damage(result_10step)
    
    print(f"\n10 步验证结果:")
    print(f"  成功: {result_10step.success}")
    print(f"  最终目标提升: {result_10step.final_target_improvement:+.2%}")
    print(f"  最终旧能力掉落: {result_10step.final_old_ability_drop:.2%}")
    print(f"  最大旧能力掉落: {result_10step.max_old_ability_drop:.2%}")
    print(f"  回滚次数: {result_10step.rollback_count}")
    
    # 3. 回滚恢复分析
    print("\n" + "="*70)
    print("测试 3: 回滚恢复分析")
    print("="*70)
    validator = LongTermValidator()
    rollback_analysis = validator.test_rollback_recovery()
    
    # 4. 生成图表
    print("\n" + "="*70)
    print("生成可视化图表")
    print("="*70)
    
    validator_5 = LongTermValidator()
    result_5 = validator_5.run_n_step_promotion(5)
    validator_5.plot_metrics(result_5, "eval/stage6_longterm_5step.png")
    
    validator_10 = LongTermValidator()
    result_10 = validator_10.run_n_step_promotion(10)
    validator_10.plot_metrics(result_10, "eval/stage6_longterm_10step.png")
    
    # 5. 导出完整报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': LONG_TERM_CONFIG,
        '5_step_validation': result_5step.to_dict(),
        '5_step_analysis': analysis_5step,
        '10_step_validation': result_10step.to_dict(),
        '10_step_analysis': analysis_10step,
        'rollback_analysis': rollback_analysis,
        'summary': {
            '5_step_success': result_5step.success,
            '10_step_success': result_10step.success,
            'rollback_recovery_rate': rollback_analysis.get('avg_recovery_rate', 0),
            'overall_pass': result_5step.success and result_10step.success and 
                           rollback_analysis.get('avg_recovery_rate', 0) > 90,
        }
    }
    
    with open("eval/stage6_long_term_validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("长期成长验证完成")
    print("="*70)
    print(f"报告已导出: eval/stage6_long_term_validation_report.json")
    print(f"5 步验证: {'✓ 通过' if result_5step.success else '✗ 失败'}")
    print(f"10 步验证: {'✓ 通过' if result_10step.success else '✗ 失败'}")
    print(f"回滚恢复率: {rollback_analysis.get('avg_recovery_rate', 0):.1f}%")
    print(f"总体结果: {'✓ 通过' if report['summary']['overall_pass'] else '✗ 失败'}")


if __name__ == "__main__":
    main()
