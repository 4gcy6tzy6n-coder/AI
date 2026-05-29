"""
Stage 6 Target Gain Verification

专门验证目标提升指标的完整测试

使用:
- 官方基线 V1.0
- 统一评估协议
- 完整测试流程 (非简化)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from typing import Dict, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_optimization_configs import get_config


@dataclass
class TargetGainVerificationReport:
    """目标提升验证报告"""
    experiment_id: str
    timestamp: str
    config_version: str
    
    # 基线指标
    baseline_target_score: float = 0.0
    
    # 测试后指标
    final_target_score: float = 0.0
    target_gain_percent: float = 0.0
    
    # 其他关键指标
    old_ability_drop: float = 0.0
    writeback_change: float = 0.0
    
    # 验证结果
    target_gain_pass: bool = False
    old_ability_pass: bool = False
    writeback_pass: bool = False
    overall_pass: bool = False
    
    # 详细记录
    step_records: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'config_version': self.config_version,
            'baseline_target_score': self.baseline_target_score,
            'final_target_score': self.final_target_score,
            'target_gain_percent': self.target_gain_percent,
            'old_ability_drop': self.old_ability_drop,
            'writeback_change': self.writeback_change,
            'verification': {
                'target_gain_pass': self.target_gain_pass,
                'old_ability_pass': self.old_ability_pass,
                'writeback_pass': self.writeback_pass,
                'overall_pass': self.overall_pass,
            },
            'step_records': self.step_records,
        }


class TargetGainVerifier:
    """目标提升验证器"""
    
    TARGET_GAIN_THRESHOLD = 0.10  # 10%
    OLD_DROP_THRESHOLD = 0.15     # 15%
    WRITEBACK_THRESHOLD = 0.05    # 5%
    
    def __init__(self, config_name: str = 'baseline_v1'):
        self.config_name = config_name
        self.config = get_config(config_name)
        self.experiment_id = f"target_gain_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.report = TargetGainVerificationReport(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            config_version=self.config.get('version', 'unknown'),
        )
    
    def run_verification(self, num_steps: int = 10) -> TargetGainVerificationReport:
        """
        运行目标提升验证
        
        Args:
            num_steps: 测试步数 (默认10步，足够验证目标提升)
        """
        print("\n" + "="*70)
        print("Stage 6 - 目标提升验证")
        print("="*70)
        print(f"实验ID: {self.experiment_id}")
        print(f"配置: {self.config_name} (v{self.config.get('version', 'N/A')})")
        print(f"测试步数: {num_steps}")
        print(f"开始时间: {self.report.timestamp}")
        
        # 创建系统
        system = Stage6SystemOrchestrator(
            experiment_id=self.experiment_id,
            custom_config=self.config,
        )
        
        # 建立基线并记录
        print("\n" + "-"*70)
        print("[1/3] 建立基线")
        print("-"*70)
        system.establish_baseline()
        
        # 记录基线目标能力
        self.report.baseline_target_score = system.protocol.baseline.scores.target_score
        print(f"基线目标能力: {self.report.baseline_target_score:.2%}")
        
        # 运行测试步数
        print("\n" + "-"*70)
        print(f"[2/3] 运行 {num_steps} 步测试")
        print("-"*70)
        
        queries = [f"目标提升验证查询 {i+1}: 系统学习测试" for i in range(num_steps)]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Step {i}/{num_steps} ---")
            
            # 执行单步
            trace = system._execute_step(i, query)
            
            # 记录
            self.report.step_records.append({
                'step': i,
                'target_improvement': trace.target_improvement,
                'old_ability_drop': trace.old_ability_drop,
                'writeback_change': trace.writeback_change,
                'error': trace.error_message,
            })
            
            print(f"  target: {trace.target_improvement:+.2%}, "
                  f"old_drop: {trace.old_ability_drop:.2%}, "
                  f"writeback: {trace.writeback_change:+.2%}")
        
        # 最终评估
        print("\n" + "-"*70)
        print("[3/3] 最终评估")
        print("-"*70)
        
        final_result = system.protocol.evaluate_with_protocol("final", num_samples=50)
        
        self.report.final_target_score = system.protocol.evaluator.evaluate_all(num_samples=50).scores.target_score
        self.report.target_gain_percent = final_result.target_improvement
        self.report.old_ability_drop = final_result.old_ability_drop
        self.report.writeback_change = final_result.writeback_change
        
        # 验证结果
        self._evaluate()
        
        return self.report
    
    def _evaluate(self):
        """评估验证结果"""
        print("\n" + "="*70)
        print("验证结果评估")
        print("="*70)
        
        # 目标提升
        self.report.target_gain_pass = self.report.target_gain_percent >= self.TARGET_GAIN_THRESHOLD
        print(f"\n目标提升:")
        print(f"  基线: {self.report.baseline_target_score:.2%}")
        print(f"  最终: {self.report.final_target_score:.2%}")
        print(f"  提升: {self.report.target_gain_percent:+.2%}")
        print(f"  阈值: > {self.TARGET_GAIN_THRESHOLD:.0%}")
        print(f"  结果: {'✓ 通过' if self.report.target_gain_pass else '✗ 未通过'}")
        
        # 旧能力保护
        self.report.old_ability_pass = self.report.old_ability_drop <= self.OLD_DROP_THRESHOLD
        print(f"\n旧能力保护:")
        print(f"  掉落: {self.report.old_ability_drop:.2%}")
        print(f"  阈值: < {self.OLD_DROP_THRESHOLD:.0%}")
        print(f"  结果: {'✓ 通过' if self.report.old_ability_pass else '✗ 未通过'}")
        
        # writeback 保护
        self.report.writeback_pass = abs(self.report.writeback_change) <= self.WRITEBACK_THRESHOLD
        print(f"\nwriteback 保护:")
        print(f"  变化: {self.report.writeback_change:+.2%}")
        print(f"  阈值: < {self.WRITEBACK_THRESHOLD:.0%}")
        print(f"  结果: {'✓ 通过' if self.report.writeback_pass else '✗ 未通过'}")
        
        # 总体
        self.report.overall_pass = (
            self.report.target_gain_pass and
            self.report.old_ability_pass and
            self.report.writeback_pass
        )
        
        print(f"\n总体结果:")
        print(f"  {'✓ 验证通过' if self.report.overall_pass else '✗ 验证未通过'}")
        
        print("\n" + "="*70)
    
    def export_report(self, filepath: str = None):
        """导出报告"""
        filepath = filepath or f"eval/{self.experiment_id}_report.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        
        print(f"\n✓ 报告已导出: {filepath}")
        return filepath
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*70)
        print("目标提升验证摘要")
        print("="*70)
        
        print(f"\n配置: {self.config_name} (v{self.config.get('version', 'N/A')})")
        print(f"测试步数: {len(self.report.step_records)}")
        
        print(f"\n关键指标:")
        print(f"  目标提升: {self.report.target_gain_percent:+.2%} {'✓' if self.report.target_gain_pass else '✗'}")
        print(f"  旧能力掉落: {self.report.old_ability_drop:.2%} {'✓' if self.report.old_ability_pass else '✗'}")
        print(f"  writeback 变化: {self.report.writeback_change:+.2%} {'✓' if self.report.writeback_pass else '✗'}")
        
        print(f"\n结论:")
        if self.report.overall_pass:
            print("  ✅ 目标提升验证通过！")
            print("  Stage 6 可以宣告完成")
        else:
            print("  ⚠️ 目标提升验证未通过")
            if not self.report.target_gain_pass:
                print(f"  - 目标提升 {self.report.target_gain_percent:+.2%} < {self.TARGET_GAIN_THRESHOLD:.0%}")
            if not self.report.old_ability_pass:
                print(f"  - 旧能力掉落 {self.report.old_ability_drop:.2%} > {self.OLD_DROP_THRESHOLD:.0%}")
            if not self.report.writeback_pass:
                print(f"  - writeback 变化 {self.report.writeback_change:+.2%} > {self.WRITEBACK_THRESHOLD:.0%}")
        
        print("\n" + "="*70)


def verify_target_gain(config_name: str = 'baseline_v1', num_steps: int = 10) -> TargetGainVerificationReport:
    """
    验证目标提升
    
    这是 Stage 6 最终验收的核心验证
    """
    verifier = TargetGainVerifier(config_name=config_name)
    report = verifier.run_verification(num_steps=num_steps)
    verifier.print_summary()
    verifier.export_report()
    return report


def compare_configs_for_target_gain(config_names: List[str], num_steps: int = 10) -> Dict[str, TargetGainVerificationReport]:
    """比较多个配置的目标提升"""
    print("\n" + "="*70)
    print("配置比较 - 目标提升")
    print("="*70)
    
    results = {}
    
    for config_name in config_names:
        print(f"\n\n测试配置: {config_name}")
        print("-"*70)
        
        report = verify_target_gain(config_name, num_steps)
        results[config_name] = report
    
    # 比较结果
    print("\n" + "="*70)
    print("配置比较结果")
    print("="*70)
    
    print(f"\n{'配置':<20} {'目标提升':<12} {'旧能力':<12} {'writeback':<12} {'总体':<10}")
    print("-"*70)
    
    for name, report in results.items():
        target = f"{report.target_gain_percent:+.1%}"
        old = f"{report.old_ability_drop:.1%}"
        wb = f"{report.writeback_change:+.1%}"
        overall = "通过" if report.overall_pass else "未通过"
        print(f"{name:<20} {target:<12} {old:<12} {wb:<12} {overall:<10}")
    
    # 推荐最佳配置
    passing_configs = [name for name, report in results.items() if report.overall_pass]
    if passing_configs:
        print(f"\n✅ 通过验证的配置: {', '.join(passing_configs)}")
    else:
        print("\n⚠️ 所有配置均未通过验证")
        # 找最接近的
        best = max(results.items(), key=lambda x: x[1].target_gain_percent)
        print(f"最接近目标的配置: {best[0]} ({best[1].target_gain_percent:+.1%})")
    
    return results


if __name__ == "__main__":
    # 验证官方基线
    print("\n" + "="*70)
    print("Stage 6 最终验证 - 官方基线 V1.0")
    print("="*70)
    
    report = verify_target_gain(config_name='baseline_v1', num_steps=10)
    
    # 如果基线未通过，测试 optimized_v2
    if not report.overall_pass:
        print("\n\n官方基线未通过，测试优化配置...")
        print("="*70)
        
        results = compare_configs_for_target_gain(
            ['baseline_v1', 'writeback_c', 'optimized_v2'],
            num_steps=10
        )
