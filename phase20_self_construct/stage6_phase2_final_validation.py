"""
Stage 6 Phase 2 Final Validation

第二阶段最终验收 - 使用统一评估协议

这是 Stage 6 第二阶段唯一的官方验收入口
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from datetime import datetime

from stage6_runtime_orchestrator import Stage6Orchestrator, Stage6Config
from stage6_evaluation_protocol import UnifiedEvaluationProtocol
from stage6_writeback_isolation_fix import apply_writeback_isolation_fix
from stage6_full_rollback_snapshot import FullRollbackManager


# 官方配置
OFFICIAL_CONFIG = {
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
    'target_improvement': 0.10,  # > +10%
    'old_ability_drop': 0.15,    # < 15%
    'writeback_change': 0.05,    # < 5%
    'rollback_recovery': 90.0,   # > 90%
}


class Stage6Phase2FinalValidator:
    """第二阶段最终验证器"""
    
    def __init__(self, use_writeback_fix: bool = True, use_rollback_fix: bool = True):
        self.config = OFFICIAL_CONFIG
        self.use_writeback_fix = use_writeback_fix
        self.use_rollback_fix = use_rollback_fix
        
        # 创建编排器
        self.orchestrator = Stage6Orchestrator(Stage6Config(**self.config))
        self.model = self.orchestrator.backbone.get_model()
        
        # 应用 writeback 隔离 (在基线之前)
        self.writeback_wrapper = None
        if use_writeback_fix:
            self.writeback_wrapper = apply_writeback_isolation_fix(self.model)
            print("✓ Writeback 隔离已应用")
        
        # 创建统一评估协议
        self.protocol = UnifiedEvaluationProtocol(self.model)
        
        # 创建 rollback 管理器
        self.rollback_manager = None
        if use_rollback_fix:
            optimizer = self.orchestrator.param_promoter.optimizer if self.orchestrator.param_promoter else None
            self.rollback_manager = FullRollbackManager(self.model, optimizer)
            print("✓ Rollback 快照已启用")
    
    def run_full_validation(self) -> dict:
        """运行完整验证"""
        print("="*70)
        print("Stage 6 第二阶段 - 最终验收")
        print("="*70)
        print(f"配置: writeback_fix={self.use_writeback_fix}, rollback_fix={self.use_rollback_fix}")
        print(f"开始时间: {datetime.now().isoformat()}")
        
        # 1. 建立统一基线
        print("\n" + "="*70)
        print("步骤 1: 建立统一基线")
        print("="*70)
        self.protocol.establish_baseline("stage6_phase2_final")
        
        # 如果使用 rollback fix，同时创建完整快照
        if self.rollback_manager:
            snapshot_id = self.rollback_manager.create_snapshot(
                "baseline",
                {'description': 'Phase 2 final validation baseline'}
            )
            print(f"✓ Rollback 快照已创建: {snapshot_id}")
        
        # 2. 执行 5 步连续晋升
        print("\n" + "="*70)
        print("步骤 2: 5 步连续晋升")
        print("="*70)
        
        step_results = []
        for i in range(5):
            print(f"\n--- 第 {i+1} 步 ---")
            
            # 晋升前评估
            pre_result = self.protocol.evaluate_with_protocol(f"step_{i+1}_pre", num_samples=30)
            
            # 冻结 writeback (如果启用)
            if self.writeback_wrapper:
                self.writeback_wrapper.freeze_writeback()
                print("  Writeback 已冻结")
            
            # 执行晋升
            torch.manual_seed(42 + i)
            self.orchestrator.run_single_step(f"查询 {i+1}")
            
            # 解冻 writeback
            if self.writeback_wrapper:
                self.writeback_wrapper.unfreeze_writeback()
            
            # 晋升后评估
            post_result = self.protocol.evaluate_with_protocol(f"step_{i+1}_post", num_samples=30)
            
            step_results.append({
                'step': i + 1,
                'target_improvement': post_result.target_improvement,
                'old_ability_drop': post_result.old_ability_drop,
                'writeback_change': post_result.writeback_change,
            })
        
        # 3. 计算指标
        print("\n" + "="*70)
        print("步骤 3: 计算关键指标")
        print("="*70)
        
        final_result = step_results[-1]
        max_old_drop = max(r['old_ability_drop'] for r in step_results)
        max_writeback_change = max(abs(r['writeback_change']) for r in step_results)
        
        print(f"\n关键指标:")
        print(f"  目标提升: {final_result['target_improvement']:+.2%}")
        print(f"  最大旧能力掉落: {max_old_drop:.2%}")
        print(f"  最大 writeback 变化: {max_writeback_change:.2%}")
        
        # 4. 测试 rollback 恢复
        print("\n" + "="*70)
        print("步骤 4: 测试 Rollback 恢复")
        print("="*70)
        
        rollback_recovery_rate = 0.0
        
        if self.rollback_manager:
            # 使用完整快照恢复
            print("\n使用完整快照恢复...")
            success = self.rollback_manager.restore_snapshot("baseline")
            
            if success:
                # 评估恢复后状态
                recovered_result = self.protocol.evaluate_with_protocol("post_rollback", num_samples=30)
                
                # 计算恢复率
                baseline_scores = self.protocol.baseline.scores
                recovery_rates = {}
                
                for key in ['target_score', 'retrieval_score', 'policy_score', 'governance_score', 'writeback_score']:
                    baseline_val = getattr(baseline_scores, key)
                    current_val = getattr(recovered_result.scores, key)
                    
                    # 简化的恢复率计算: 越接近基线越好
                    if baseline_val > 0:
                        deviation = abs(current_val - baseline_val) / baseline_val
                        recovery_rate = max(0, 100 - deviation * 100)
                        recovery_rates[key] = recovery_rate
                
                rollback_recovery_rate = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0
                
                print(f"\n恢复率:")
                for key, rate in recovery_rates.items():
                    print(f"  {key}: {rate:.1f}%")
                print(f"  平均: {rollback_recovery_rate:.1f}%")
            else:
                print("✗ 快照恢复失败")
        else:
            # 使用协议的内置恢复
            print("\n使用协议内置恢复...")
            success = self.protocol.restore_to_baseline()
            
            if success:
                recovered_result = self.protocol.evaluate_with_protocol("post_rollback", num_samples=30)
                
                # 简化计算
                baseline_scores = self.protocol.baseline.scores
                recovery_rates = {}
                
                for key in ['target_score', 'retrieval_score', 'policy_score', 'governance_score', 'writeback_score']:
                    baseline_val = getattr(baseline_scores, key)
                    current_val = getattr(recovered_result.scores, key)
                    
                    if baseline_val > 0:
                        deviation = abs(current_val - baseline_val) / baseline_val
                        recovery_rate = max(0, 100 - deviation * 100)
                        recovery_rates[key] = recovery_rate
                
                rollback_recovery_rate = sum(recovery_rates.values()) / len(recovery_rates) if recovery_rates else 0
                
                print(f"\n恢复率:")
                for key, rate in recovery_rates.items():
                    print(f"  {key}: {rate:.1f}%")
                print(f"  平均: {rollback_recovery_rate:.1f}%")
        
        # 5. 验收判断
        print("\n" + "="*70)
        print("步骤 5: 验收判断")
        print("="*70)
        
        criteria_check = {
            'target_improvement': final_result['target_improvement'] > ACCEPTANCE_CRITERIA['target_improvement'],
            'old_ability_drop': max_old_drop < ACCEPTANCE_CRITERIA['old_ability_drop'],
            'writeback_change': max_writeback_change < ACCEPTANCE_CRITERIA['writeback_change'],
            'rollback_recovery': rollback_recovery_rate > ACCEPTANCE_CRITERIA['rollback_recovery'],
        }
        
        print(f"\n验收标准:")
        print(f"  目标提升 > {ACCEPTANCE_CRITERIA['target_improvement']:.0%}: {final_result['target_improvement']:+.2%} {'✓' if criteria_check['target_improvement'] else '✗'}")
        print(f"  旧能力掉落 < {ACCEPTANCE_CRITERIA['old_ability_drop']:.0%}: {max_old_drop:.2%} {'✓' if criteria_check['old_ability_drop'] else '✗'}")
        print(f"  writeback 变化 < {ACCEPTANCE_CRITERIA['writeback_change']:.0%}: {max_writeback_change:.2%} {'✓' if criteria_check['writeback_change'] else '✗'}")
        print(f"  rollback 恢复 > {ACCEPTANCE_CRITERIA['rollback_recovery']:.0f}%: {rollback_recovery_rate:.1f}% {'✓' if criteria_check['rollback_recovery'] else '✗'}")
        
        overall_pass = all(criteria_check.values())
        
        print(f"\n  总体结果: {'✓ 验收通过' if overall_pass else '✗ 验收失败'}")
        
        # 6. 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'use_writeback_fix': self.use_writeback_fix,
                'use_rollback_fix': self.use_rollback_fix,
                'model_config': self.config,
            },
            'acceptance_criteria': ACCEPTANCE_CRITERIA,
            'baseline': self.protocol.baseline.to_dict() if self.protocol.baseline else None,
            'step_results': step_results,
            'final_metrics': {
                'target_improvement': final_result['target_improvement'],
                'max_old_ability_drop': max_old_drop,
                'max_writeback_change': max_writeback_change,
                'rollback_recovery_rate': rollback_recovery_rate,
            },
            'acceptance': {
                'criteria_check': criteria_check,
                'overall_pass': overall_pass,
            },
        }
        
        return report


def main():
    """主函数"""
    # 使用架构修复运行最终验收
    validator = Stage6Phase2FinalValidator(
        use_writeback_fix=True,
        use_rollback_fix=True
    )
    
    report = validator.run_full_validation()
    
    # 导出报告
    with open("eval/stage6_phase2_final_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("第二阶段最终验收完成")
    print("="*70)
    print(f"报告已导出: eval/stage6_phase2_final_report.json")
    
    if report['acceptance']['overall_pass']:
        print("\n🎉 Stage 6 第二阶段验收通过！")
        print("可以进入第三阶段: 系统级评估")
    else:
        print("\n⚠️ Stage 6 第二阶段验收未通过")
        print("需要进一步修复或调整标准")


if __name__ == "__main__":
    main()
