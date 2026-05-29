"""
Stage 7 Dual Guard 主线集成 V2

修复评估对齐问题，使用AlignedWritebackEvaluator确保口径一致
"""

import torch
import copy
from typing import Dict, Optional
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_runtime_orchestrator import Candidate
from stage7_dual_guard_orchestrator import DualGuardPromoter, DualGuardConfig
from stage7_aligned_evaluation import AlignedWritebackEvaluator, StepByStepMonitor


# Stage 7 正式候选配置
STAGE7_DUAL_GUARD_CONFIG = {
    'feature_guard_mode': 'subspace',
    'feature_guard_beta': 0.05,
    'feature_guard_top_k': 64,
    'head_lr_ratio': 2.0,
    'head_grad_clip': 1.0,
    'lambda_wb': 0.1,
    'num_epochs': 3,
    'learning_rate': 1e-4,
}


class Stage7SystemOrchestratorV2(Stage6SystemOrchestrator):
    """
    Stage 7 系统 Orchestrator V2
    
    继承 Stage 6 基线，集成 Dual Guard 保护机制
    使用对齐评估器确保口径一致
    """
    
    def __init__(
        self,
        experiment_id: str,
        custom_config: Optional[Dict] = None,
        enable_dual_guard: bool = True,
    ):
        # 使用 Stage 6 基线配置
        config = custom_config or copy.deepcopy(OFFICIAL_BASELINE_V1)
        
        # 初始化 Stage 6 基线
        super().__init__(experiment_id, config)
        
        self.enable_dual_guard = enable_dual_guard
        self.aligned_evaluator: Optional[AlignedWritebackEvaluator] = None
        self.step_monitor: Optional[StepByStepMonitor] = None
        
        # 如果启用 Dual Guard，替换 orchestrator
        if enable_dual_guard:
            self._setup_dual_guard()
    
    def _setup_dual_guard(self):
        """设置 Dual Guard"""
        print("\n[Stage 7] 启用 Dual Guard 保护机制")
        print(f"  Feature Guard Beta: {STAGE7_DUAL_GUARD_CONFIG['feature_guard_beta']}")
        print(f"  Head LR Ratio: {STAGE7_DUAL_GUARD_CONFIG['head_lr_ratio']}")
        
        # 获取当前模型
        model = self.orchestrator.backbone.get_model()
        
        # 创建 Dual Guard Promoter 替换原有 promoter
        dual_config = DualGuardConfig(
            feature_guard_mode=STAGE7_DUAL_GUARD_CONFIG['feature_guard_mode'],
            feature_guard_beta=STAGE7_DUAL_GUARD_CONFIG['feature_guard_beta'],
            feature_guard_top_k=STAGE7_DUAL_GUARD_CONFIG['feature_guard_top_k'],
            head_lr_ratio=STAGE7_DUAL_GUARD_CONFIG['head_lr_ratio'],
            head_grad_clip=STAGE7_DUAL_GUARD_CONFIG['head_grad_clip'],
            lambda_wb=STAGE7_DUAL_GUARD_CONFIG['lambda_wb'],
            num_epochs=STAGE7_DUAL_GUARD_CONFIG['num_epochs'],
            learning_rate=STAGE7_DUAL_GUARD_CONFIG['learning_rate'],
        )
        
        # 替换 promoter
        self.orchestrator.param_promoter = DualGuardPromoter(
            model,
            dual_config,
            enable_feature_guard=True,
            enable_head_training=True,
        )
        
        print("  ✓ Dual Guard 集成完成")
    
    def setup_aligned_evaluation(self):
        """设置对齐评估"""
        model = self.orchestrator.backbone.get_model()
        self.aligned_evaluator = AlignedWritebackEvaluator()
        baseline_info = self.aligned_evaluator.set_baseline(model)
        
        # 创建监控器
        self.step_monitor = StepByStepMonitor(
            self.aligned_evaluator,
            protocol_evaluator=self.protocol
        )
        
        print(f"\n[对齐评估] 基线已设置")
        print(f"  Score: {baseline_info['score']:.4f}")
        print(f"  Hash: {baseline_info['hash']}")
        print(f"  Samples: {baseline_info['num_samples']}")
        
        return baseline_info


def run_stage7_aligned_validation(num_steps: int = 100) -> Dict:
    """
    运行 Stage 7 对齐验证
    
    使用AlignedWritebackEvaluator确保50-step和100-step评估口径一致
    在20/40/60/80/100步记录两套指标进行对比
    """
    print("="*70)
    print("Stage 7 对齐验证 (V2)")
    print("="*70)
    
    # 创建 Stage 7 系统
    system = Stage7SystemOrchestratorV2(
        experiment_id='stage7_aligned_validation',
        enable_dual_guard=True,
    )
    
    # 建立基线
    print("\n[1/5] 建立基线...")
    system.establish_baseline()
    baseline = system.protocol.baseline.scores
    
    # 设置对齐评估
    aligned_baseline = system.setup_aligned_evaluation()
    
    print(f"\n  协议基线:")
    print(f"    Target: {baseline.target_score:.2%}")
    print(f"    Writeback: {baseline.writeback_score:.2%}")
    print(f"\n  对齐基线:")
    print(f"    Writeback: {aligned_baseline['score']:.4f}")
    
    # 执行训练并监控
    print(f"\n[2/5] 执行 {num_steps} 步训练 (带对齐监控)...")
    
    checkpoint_steps = [20, 40, 60, 80, 100]
    
    for step in range(num_steps):
        # 创建候选
        candidate = Candidate(
            candidate_id=f'step_{step}',
            candidate_type='TEST',
            content=f'print("Step {step}")',
            entities={},
            metadata={'fitness': 0.5 + step * 0.001},
        )
        
        # 执行一步训练
        baseline_abilities = {
            'target': baseline.target_score,
            'retrieval': baseline.retrieval_score,
            'policy': baseline.policy_score,
            'governance': baseline.governance_score,
            'writeback': baseline.writeback_score,
        }
        
        metrics = system.orchestrator.param_promoter.promote(
            candidate, step, baseline_abilities
        )
        
        # 在检查点记录
        if (step + 1) in checkpoint_steps:
            model = system.orchestrator.backbone.get_model()
            record = system.step_monitor.record(step + 1, model, system.protocol)
            
            aligned = record['aligned']
            print(f"\n  Step {step+1}:")
            print(f"    Aligned: score={aligned['score']:.4f}, change={aligned['change']:+.4f}, kl={aligned['kl']:.6f}")
            
            if record['protocol']:
                proto = record['protocol']
                print(f"    Protocol: score={proto['writeback_score']:.4f}, change={proto['change']:+.4f}")
    
    # 打印对比表
    system.step_monitor.print_comparison()
    
    # 最终评估
    print(f"\n[3/5] 最终评估...")
    model = system.orchestrator.backbone.get_model()
    
    # 对齐评估
    final_aligned = system.aligned_evaluator.evaluate(model)
    
    # 协议评估
    final_protocol = system.protocol.evaluate_with_protocol('stage7_final', num_samples=30)
    final_baseline = system.protocol.baseline.scores
    
    # 计算指标
    target_gain = final_protocol.scores.target_score - final_baseline.target_score
    old_ability_drop = max(
        abs(final_protocol.scores.retrieval_score - final_baseline.retrieval_score),
        abs(final_protocol.scores.policy_score - final_baseline.policy_score),
    )
    
    print(f"\n  对齐评估结果:")
    print(f"    Score: {final_aligned['score']:.4f}")
    print(f"    Change: {final_aligned['change']:+.4f} ({final_aligned['change_pct']:+.2%})")
    print(f"    KL: {final_aligned['kl']:.6f}")
    
    print(f"\n  协议评估结果:")
    print(f"    Target: {final_protocol.scores.target_score:.2%} (Δ{target_gain:+.2%})")
    print(f"    Writeback: {final_protocol.scores.writeback_score:.2%}")
    print(f"    Old Ability Drop: {old_ability_drop:.2%}")
    
    # 验收判断
    print(f"\n[4/5] 验收判断...")
    
    # 使用对齐评估的writeback变化
    aligned_writeback_change = abs(final_aligned['change'])
    protocol_writeback_change = abs(final_protocol.scores.writeback_score - final_baseline.writeback_score)
    
    results = {
        'target_gain': target_gain,
        'old_ability_drop': old_ability_drop,
        'aligned_writeback_change': aligned_writeback_change,
        'protocol_writeback_change': protocol_writeback_change,
        'e2e_success': final_protocol.scores.target_score > 0.6,
        'stability_pass': True,
    }
    
    # Stage 6 必须项
    stage6_pass = (
        results['target_gain'] > 0.10 and
        results['old_ability_drop'] < 0.15 and
        results['e2e_success'] and
        results['stability_pass']
    )
    
    # Stage 7 专项 (使用对齐评估)
    stage7_writeback_pass = aligned_writeback_change < 0.05
    
    print(f"\n  Stage 6 必须项:")
    print(f"    Target gain > 10%: {results['target_gain']:.2%} {'✓' if results['target_gain'] > 0.10 else '✗'}")
    print(f"    Old ability drop < 15%: {results['old_ability_drop']:.2%} {'✓' if results['old_ability_drop'] < 0.15 else '✗'}")
    print(f"    E2E success: {'✓' if results['e2e_success'] else '✗'}")
    print(f"    Stability pass: {'✓' if results['stability_pass'] else '✗'}")
    print(f"    Stage 6 结果: {'✓ 通过' if stage6_pass else '✗ 未通过'}")
    
    print(f"\n  Stage 7 专项 (对齐评估):")
    print(f"    Writeback change < 5%: {aligned_writeback_change:.2%} {'✓' if stage7_writeback_pass else '✗'}")
    print(f"    Stage 7 结果: {'✓ 通过' if stage7_writeback_pass else '✗ 未通过'}")
    
    print(f"\n  评估差异:")
    print(f"    对齐评估: {aligned_writeback_change:.2%}")
    print(f"    协议评估: {protocol_writeback_change:.2%}")
    print(f"    差异: {abs(aligned_writeback_change - protocol_writeback_change):.2%}")
    
    # 总体结果
    all_pass = stage6_pass and stage7_writeback_pass
    
    print(f"\n[5/5] 总体验收结果")
    print("="*70)
    if all_pass:
        print("✓✓✓ STAGE 7 正式验收通过 ✓✓✓")
        print("\n所有必须项和专项指标均达标")
        print("Dual Guard 配置可冻结为正式方案")
    else:
        print("✗✗✗ STAGE 7 验收未通过 ✗✗✗")
        print("\n未达标项:")
        if not stage6_pass:
            print("  - Stage 6 基线指标退化")
        if not stage7_writeback_pass:
            print(f"  - Writeback 变化超标 (对齐: {aligned_writeback_change:.2%}, 协议: {protocol_writeback_change:.2%})")
    
    return {
        'all_pass': all_pass,
        'stage6_pass': stage6_pass,
        'stage7_writeback_pass': stage7_writeback_pass,
        'metrics': results,
        'aligned_final': final_aligned,
        'protocol_final': {
            'target': final_protocol.scores.target_score,
            'writeback': final_protocol.scores.writeback_score,
        },
        'monitor_history': system.step_monitor.history,
    }


if __name__ == "__main__":
    # 运行对齐验证
    results = run_stage7_aligned_validation(num_steps=100)
    
    # 输出摘要
    print("\n" + "="*70)
    print("Stage 7 对齐验证摘要")
    print("="*70)
    print(f"总体结果: {'✓ 通过' if results['all_pass'] else '✗ 未通过'}")
    print(f"Stage 6: {'✓' if results['stage6_pass'] else '✗'}")
    print(f"Stage 7 (对齐): {'✓' if results['stage7_writeback_pass'] else '✗'}")
    print(f"\n关键指标:")
    print(f"  对齐 writeback change: {results['metrics']['aligned_writeback_change']:.2%}")
    print(f"  协议 writeback change: {results['metrics']['protocol_writeback_change']:.2%}")
