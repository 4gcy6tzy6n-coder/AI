"""
Stage 6 Evaluation Protocol

统一评估协议 - 修复 baseline 一致性问题

核心原则:
1. 统一 baseline checkpoint 来源
2. 统一 pre/post 对照顺序
3. 统一计算入口
4. 消除评估状态残留
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from stage6_real_evaluator import Stage6RealEvaluator, CapabilityScores, EvaluationResult


@dataclass
class UnifiedBaseline:
    """统一基线"""
    timestamp: str
    checkpoint_id: str
    scores: CapabilityScores
    model_state: Dict
    rng_states: Dict
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'checkpoint_id': self.checkpoint_id,
            'scores': self.scores.to_dict(),
        }


class UnifiedEvaluationProtocol:
    """
    统一评估协议
    
    确保所有评估使用一致的 baseline 和计算方式
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.evaluator = Stage6RealEvaluator(model, device)
        self.baseline: Optional[UnifiedBaseline] = None
        self.evaluation_history: List[Dict] = []
        
    def establish_baseline(self, checkpoint_id: str = "official_baseline") -> UnifiedBaseline:
        """
        建立统一基线
        
        这是整个第二阶段唯一的官方基线来源
        """
        print("\n" + "="*70)
        print("建立统一基线")
        print("="*70)
        
        # 1. 保存模型状态
        print("[1/3] 保存模型状态...")
        model_state = copy.deepcopy(self.model.state_dict())
        
        # 2. 保存随机数状态
        print("[2/3] 保存随机数状态...")
        import random
        import numpy as np
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        
        # 3. 执行评估
        print("[3/3] 执行基线评估...")
        self.model.eval()
        with torch.no_grad():
            eval_result = self.evaluator.evaluate_all(num_samples=100)
        
        # 4. 构建统一基线
        self.baseline = UnifiedBaseline(
            timestamp=datetime.now().isoformat(),
            checkpoint_id=checkpoint_id,
            scores=eval_result.scores,
            model_state=model_state,
            rng_states=rng_states,
        )
        
        # 5. 设置评估器基线
        self.evaluator.set_baseline(eval_result.scores)
        
        print(f"\n✓ 基线已建立: {checkpoint_id}")
        print(f"  时间戳: {self.baseline.timestamp}")
        print(f"  目标能力: {self.baseline.scores.target_score:.2%}")
        print(f"  检索能力: {self.baseline.scores.retrieval_score:.2%}")
        print(f"  治理能力: {self.baseline.scores.governance_score:.2%}")
        print(f"  writeback: {self.baseline.scores.writeback_score:.2%}")
        
        return self.baseline
    
    def evaluate_with_protocol(self, step_name: str, num_samples: int = 50) -> EvaluationResult:
        """
        使用统一协议执行评估
        
        Args:
            step_name: 步骤名称 (用于记录)
            num_samples: 评估样本数
            
        Returns:
            评估结果
        """
        if self.baseline is None:
            raise ValueError("必须先建立基线")
        
        print(f"\n执行评估: {step_name}")
        print("-" * 50)
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 清除任何可能的梯度
        self.model.zero_grad(set_to_none=True)
        
        # 执行评估
        with torch.no_grad():
            result = self.evaluator.evaluate_all(num_samples=num_samples)
        
        # 记录历史
        record = {
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'result': result.to_dict(),
        }
        self.evaluation_history.append(record)
        
        print(f"  目标提升: {result.target_improvement:+.2%}")
        print(f"  旧能力掉落: {result.old_ability_drop:.2%}")
        print(f"  writeback: {result.writeback_change:+.2%}")
        
        return result
    
    def calculate_old_ability_drop(self, current_scores: CapabilityScores) -> float:
        """
        计算旧能力掉落 (统一计算入口)
        
        公式: max(|current - baseline| / baseline) across all old abilities
        """
        if self.baseline is None:
            raise ValueError("必须先建立基线")
        
        baseline = self.baseline.scores
        
        # 计算各项能力的相对变化
        changes = []
        
        if baseline.retrieval_score > 0:
            retrieval_change = abs(current_scores.retrieval_score - baseline.retrieval_score) / baseline.retrieval_score
            changes.append(retrieval_change)
        
        if baseline.policy_score > 0:
            policy_change = abs(current_scores.policy_score - baseline.policy_score) / baseline.policy_score
            changes.append(policy_change)
        
        if baseline.governance_score > 0:
            governance_change = abs(current_scores.governance_score - baseline.governance_score) / baseline.governance_score
            changes.append(governance_change)
        
        return max(changes) if changes else 0.0
    
    def verify_baseline_consistency(self) -> bool:
        """
        验证基线一致性
        
        检查当前模型状态是否与基线一致
        """
        if self.baseline is None:
            print("错误: 无基线")
            return False
        
        print("\n验证基线一致性...")
        
        # 检查模型参数
        current_state = self.model.state_dict()
        baseline_state = self.baseline.model_state
        
        match = True
        for key in baseline_state.keys():
            if key in current_state:
                if not torch.allclose(current_state[key], baseline_state[key], atol=1e-6):
                    print(f"  ✗ 参数不匹配: {key}")
                    match = False
        
        if match:
            print("  ✓ 模型状态与基线一致")
        else:
            print("  ✗ 模型状态与基线不一致")
        
        return match
    
    def restore_to_baseline(self) -> bool:
        """
        恢复到基线状态
        
        用于 rollback 测试
        """
        if self.baseline is None:
            print("错误: 无基线可恢复")
            return False
        
        print("\n恢复到基线状态...")
        
        try:
            # 恢复模型状态
            self.model.load_state_dict(self.baseline.model_state)
            
            # 恢复随机数状态
            import random
            import numpy as np
            random.setstate(self.baseline.rng_states['python'])
            np.random.set_state(self.baseline.rng_states['numpy'])
            torch.set_rng_state(self.baseline.rng_states['torch'])
            
            print("  ✓ 已恢复到基线状态")
            return True
            
        except Exception as e:
            print(f"  ✗ 恢复失败: {e}")
            return False
    
    def export_report(self, filepath: str):
        """导出评估报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline': self.baseline.to_dict() if self.baseline else None,
            'evaluation_history': self.evaluation_history,
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n评估报告已导出: {filepath}")


# ==================== 便捷函数 ====================

def run_official_evaluation(model, steps: int = 5) -> Dict:
    """
    运行官方评估流程
    
    这是第二阶段唯一的官方验收入口
    """
    print("="*70)
    print("Stage 6 第二阶段 - 官方评估流程")
    print("="*70)
    
    protocol = UnifiedEvaluationProtocol(model)
    
    # 1. 建立基线
    protocol.establish_baseline("stage6_phase2_official")
    
    # 2. 执行 N 步并评估
    results = []
    for i in range(steps):
        result = protocol.evaluate_with_protocol(f"step_{i+1}")
        results.append({
            'step': i + 1,
            'target_improvement': result.target_improvement,
            'old_ability_drop': result.old_ability_drop,
            'writeback_change': result.writeback_change,
        })
    
    # 3. 计算最终指标
    final = results[-1]
    max_old_drop = max(r['old_ability_drop'] for r in results)
    max_writeback_change = max(abs(r['writeback_change']) for r in results)
    
    # 4. 测试 rollback 恢复
    print("\n" + "="*70)
    print("测试 Rollback 恢复")
    print("="*70)
    
    # 保存当前状态
    pre_rollback_scores = protocol.evaluator.evaluate_all(num_samples=50).scores
    
    # 恢复到基线
    success = protocol.restore_to_baseline()
    
    # 评估恢复后状态
    post_rollback_scores = protocol.evaluator.evaluate_all(num_samples=50).scores
    
    # 计算恢复率
    recovery_rates = {}
    for key in ['target_score', 'retrieval_score', 'policy_score', 'governance_score', 'writeback_score']:
        baseline_val = getattr(protocol.baseline.scores, key)
        current_val = getattr(pre_rollback_scores, key)
        recovered_val = getattr(post_rollback_scores, key)
        
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
    
    # 5. 汇总报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline': protocol.baseline.to_dict(),
        'step_results': results,
        'final_metrics': {
            'target_improvement': final['target_improvement'],
            'max_old_ability_drop': max_old_drop,
            'max_writeback_change': max_writeback_change,
            'rollback_recovery_rate': avg_recovery,
        },
        'rollback_recovery': recovery_rates,
    }
    
    # 6. 验收判断
    criteria = {
        'target_improvement': final['target_improvement'] > 0.10,
        'old_ability_drop': max_old_drop < 0.15,
        'writeback_change': max_writeback_change < 0.05,
        'rollback_recovery': avg_recovery > 90,
    }
    
    print("\n" + "="*70)
    print("验收结果")
    print("="*70)
    print(f"  目标提升: {final['target_improvement']:+.2%} {'✓' if criteria['target_improvement'] else '✗'}")
    print(f"  旧能力掉落: {max_old_drop:.2%} {'✓' if criteria['old_ability_drop'] else '✗'}")
    print(f"  writeback: {max_writeback_change:.2%} {'✓' if criteria['writeback_change'] else '✗'}")
    print(f"  rollback: {avg_recovery:.1f}% {'✓' if criteria['rollback_recovery'] else '✗'}")
    
    overall_pass = all(criteria.values())
    print(f"\n  总体: {'✓ 通过' if overall_pass else '✗ 失败'}")
    
    report['acceptance'] = {
        'criteria': criteria,
        'overall_pass': overall_pass,
    }
    
    return report


# ==================== 测试 ====================

def test_protocol():
    """测试统一协议"""
    print("\n" + "="*70)
    print("测试统一评估协议")
    print("="*70)
    
    from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
    
    config = NativeTinyConfig()
    model = NativeBackboneTinyV1(config)
    
    protocol = UnifiedEvaluationProtocol(model)
    
    # 建立基线
    baseline = protocol.establish_baseline("test_baseline")
    
    # 验证一致性
    consistent = protocol.verify_baseline_consistency()
    assert consistent, "基线一致性验证失败"
    
    # 执行评估
    result1 = protocol.evaluate_with_protocol("test_step_1")
    
    # 恢复到基线
    success = protocol.restore_to_baseline()
    assert success, "恢复基线失败"
    
    # 验证一致性
    consistent = protocol.verify_baseline_consistency()
    assert consistent, "恢复后一致性验证失败"
    
    # 再次评估 (应该与基线相同)
    result2 = protocol.evaluate_with_protocol("test_step_2_post_restore")
    
    print("\n✓ 协议测试通过")


if __name__ == "__main__":
    test_protocol()
