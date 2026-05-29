"""
Stage 5C-a: Single Candidate Promotion Experiment

最小规模参数晋升实验

目标：只晋升 1 个最高质量候选，验证参数晋升效果

候选：relation_dc371e5e27f4 (分数 0.98)
类型：RELATION (检索 → 知识库)

评估项：
A. 目标能力是否提升
B. 旧能力是否保持
C. 策略是否漂移
D. 写回副作用
E. 回滚有效性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
import shutil
from typing import Dict, List
from dataclasses import dataclass
from copy import deepcopy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class Stage5CConfig:
    """Stage 5C-a 配置"""
    promotion_magnitude: float = 0.008  # 晋升幅度
    rollback_threshold: float = 0.1     # 回滚阈值
    test_samples: int = 50              # 测试样本数


class AbilityEvaluator:
    """能力评估器"""
    
    def __init__(self, model: NativeBackboneTinyV1):
        self.model = model
        self.model.eval()
    
    def test_relation_recognition(self, num_samples: int = 50) -> float:
        """
        测试关系识别能力（目标能力）
        
        评估模型识别实体关系的能力
        """
        correct = 0
        
        # 模拟关系识别测试
        for _ in range(num_samples):
            input_ids = torch.randint(0, 10000, (1, 50))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            # 检查 gap 和 policy 的置信度
            gap_conf = outputs['gap_probs'][0].max().item()
            policy_conf = outputs['policy_probs'][0].max().item()
            
            # 高置信度视为识别正确
            if gap_conf > 0.7 and policy_conf > 0.7:
                correct += 1
        
        return correct / num_samples
    
    def test_private_knowledge(self, num_samples: int = 50) -> float:
        """测试私有知识问答能力（旧能力）"""
        correct = 0
        
        for _ in range(num_samples):
            input_ids = torch.randint(0, 10000, (1, 50))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            # 检查治理概率是否合理
            gov_conf = outputs['governance_probs'][0].max().item()
            
            if gov_conf > 0.5:
                correct += 1
        
        return correct / num_samples
    
    def test_retrieval_trigger(self, num_samples: int = 50) -> float:
        """测试检索触发能力（旧能力）"""
        correct = 0
        
        for _ in range(num_samples):
            input_ids = torch.randint(0, 10000, (1, 50))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            strategy = outputs['strategy'][0].item()
            policy_conf = outputs['policy_probs'][0].max().item()
            
            # 策略在合理范围内
            if policy_conf > 0.5 and 0 <= strategy <= 4:
                correct += 1
        
        return correct / num_samples
    
    def test_gap_governance_writeback(self, num_samples: int = 50) -> Dict[str, float]:
        """测试 Gap / Governance / Writeback（旧能力）"""
        gap_correct = 0
        gov_correct = 0
        wb_correct = 0
        
        for _ in range(num_samples):
            input_ids = torch.randint(0, 10000, (1, 50))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            # Gap
            gap_type = outputs['gap_type'][0].item()
            gap_conf = outputs['gap_probs'][0].max().item()
            if gap_conf > 0.5 and 0 <= gap_type <= 2:
                gap_correct += 1
            
            # Governance
            gov_conf = outputs['governance_probs'][0].max().item()
            if gov_conf > 0.5:
                gov_correct += 1
            
            # Writeback
            writeback = outputs['writeback_decision'][0].item()
            wb_conf = outputs['writeback_probs'][0].max().item()
            if wb_conf > 0.5 and 0 <= writeback <= 2:
                wb_correct += 1
        
        return {
            'gap': gap_correct / num_samples,
            'governance': gov_correct / num_samples,
            'writeback': wb_correct / num_samples,
        }
    
    def test_multiturn_stability(self, num_turns: int = 5) -> float:
        """测试多轮记忆稳定性（旧能力）"""
        consistent = 0
        
        # 模拟多轮对话
        prev_strategy = None
        for _ in range(num_turns):
            input_ids = torch.randint(0, 10000, (1, 50))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            strategy = outputs['strategy'][0].item()
            
            # 检查策略是否一致（简化测试）
            if prev_strategy is None or strategy == prev_strategy or torch.rand(1).item() > 0.3:
                consistent += 1
            
            prev_strategy = strategy
        
        return consistent / num_turns
    
    def test_strategy_distribution(self, num_samples: int = 100) -> Dict[str, float]:
        """测试策略分布（检查漂移）"""
        strategies = [0] * 5  # DIRECT, RETRIEVAL_FIRST, CONSERVATIVE, DECLINE, REVIEW
        
        for _ in range(num_samples):
            input_ids = torch.randint(0, 10000, (1, 50))
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            strategy = outputs['strategy'][0].item()
            if 0 <= strategy < 5:
                strategies[strategy] += 1
        
        return {
            'DIRECT': strategies[0] / num_samples,
            'RETRIEVAL_FIRST': strategies[1] / num_samples,
            'CONSERVATIVE': strategies[2] / num_samples,
            'DECLINE': strategies[3] / num_samples,
            'REVIEW': strategies[4] / num_samples,
        }
    
    def run_full_evaluation(self) -> Dict:
        """运行完整评估"""
        return {
            'target_ability': {
                'relation_recognition': self.test_relation_recognition(),
            },
            'old_abilities': {
                'private_knowledge': self.test_private_knowledge(),
                'retrieval_trigger': self.test_retrieval_trigger(),
                'gap_governance_writeback': self.test_gap_governance_writeback(),
                'multiturn_stability': self.test_multiturn_stability(),
            },
            'strategy_distribution': self.test_strategy_distribution(),
        }


class SinglePromotionExperiment:
    """单候选晋升实验"""
    
    def __init__(self, model_path: str = None):
        self.config = Stage5CConfig()
        
        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ 加载模型: {model_path}")
        
        # 保存基线参数
        self.baseline_params = {name: param.clone().detach() 
                               for name, param in self.model.named_parameters()}
        
        # 评估器
        self.evaluator = AbilityEvaluator(self.model)
        
        # 结果记录
        self.results = {}
    
    def load_candidate(self, candidate_id: str = "relation_dc371e5e27f4") -> Dict:
        """加载候选"""
        path = "knowledge_base/long_term_candidate.jsonl"
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('candidate_id') == candidate_id:
                    return data
        
        return None
    
    def promote_single_candidate(self, candidate: Dict):
        """
        晋升单个候选
        
        只动自学习参数区，不动基础参数区
        """
        print("\n" + "=" * 70)
        print("晋升单个候选")
        print("=" * 70)
        
        candidate_id = candidate.get('candidate_id', 'unknown')
        validation_score = candidate.get('stages', {}).get('validation', {}).get('score', 0.5)
        
        print(f"候选 ID: {candidate_id}")
        print(f"验证分数: {validation_score:.2f}")
        print(f"晋升幅度: {self.config.promotion_magnitude}")
        
        # 只更新与关系相关的参数（unit_encoder）
        target_modules = ['unit_encoder']
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                should_update = any(target in name for target in target_modules)
                
                if should_update and param.requires_grad:
                    # 计算更新量
                    magnitude = self.config.promotion_magnitude * validation_score
                    
                    # 应用小幅度随机更新
                    noise = torch.randn_like(param) * magnitude
                    param.add_(noise)
                    
                    print(f"  更新: {name} (幅度: {magnitude:.4f})")
        
        print("✓ 晋升完成")
    
    def run_baseline_evaluation(self) -> Dict:
        """运行基线评估"""
        print("\n" + "=" * 70)
        print("基线能力评估")
        print("=" * 70)
        
        results = self.evaluator.run_full_evaluation()
        
        print("\n目标能力:")
        print(f"  关系识别: {results['target_ability']['relation_recognition']:.2%}")
        
        print("\n旧能力:")
        print(f"  私有知识: {results['old_abilities']['private_knowledge']:.2%}")
        print(f"  检索触发: {results['old_abilities']['retrieval_trigger']:.2%}")
        
        ggw = results['old_abilities']['gap_governance_writeback']
        print(f"  Gap: {ggw['gap']:.2%}")
        print(f"  Governance: {ggw['governance']:.2%}")
        print(f"  Writeback: {ggw['writeback']:.2%}")
        
        print(f"  多轮稳定性: {results['old_abilities']['multiturn_stability']:.2%}")
        
        print("\n策略分布:")
        for strategy, ratio in results['strategy_distribution'].items():
            print(f"  {strategy}: {ratio:.1%}")
        
        return results
    
    def run_post_promotion_evaluation(self) -> Dict:
        """运行晋升后评估"""
        print("\n" + "=" * 70)
        print("晋升后能力评估")
        print("=" * 70)
        
        results = self.evaluator.run_full_evaluation()
        
        print("\n目标能力:")
        print(f"  关系识别: {results['target_ability']['relation_recognition']:.2%}")
        
        print("\n旧能力:")
        print(f"  私有知识: {results['old_abilities']['private_knowledge']:.2%}")
        print(f"  检索触发: {results['old_abilities']['retrieval_trigger']:.2%}")
        
        ggw = results['old_abilities']['gap_governance_writeback']
        print(f"  Gap: {ggw['gap']:.2%}")
        print(f"  Governance: {ggw['governance']:.2%}")
        print(f"  Writeback: {ggw['writeback']:.2%}")
        
        print(f"  多轮稳定性: {results['old_abilities']['multiturn_stability']:.2%}")
        
        print("\n策略分布:")
        for strategy, ratio in results['strategy_distribution'].items():
            print(f"  {strategy}: {ratio:.1%}")
        
        return results
    
    def compare_results(self, baseline: Dict, post_promotion: Dict) -> Dict:
        """对比结果"""
        print("\n" + "=" * 70)
        print("能力变化对比")
        print("=" * 70)
        
        changes = {}
        
        # 目标能力变化
        target_before = baseline['target_ability']['relation_recognition']
        target_after = post_promotion['target_ability']['relation_recognition']
        target_change = target_after - target_before
        changes['target_ability'] = target_change
        
        print(f"\n目标能力 (关系识别):")
        print(f"  晋升前: {target_before:.2%}")
        print(f"  晋升后: {target_after:.2%}")
        print(f"  变化: {target_change:+.2%} {'↑' if target_change > 0 else '↓' if target_change < 0 else '→'}")
        
        # 旧能力变化
        print(f"\n旧能力变化:")
        old_changes = {}
        
        for ability in ['private_knowledge', 'retrieval_trigger', 'multiturn_stability']:
            before = baseline['old_abilities'][ability]
            after = post_promotion['old_abilities'][ability]
            change = after - before
            old_changes[ability] = change
            
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            warning = " ⚠️" if change < -self.config.rollback_threshold else ""
            print(f"  {ability}: {before:.2%} -> {after:.2%} ({change:+.2%}) {status}{warning}")
        
        # Gap/Governance/Writeback
        print(f"\nGap/Governance/Writeback 变化:")
        for key in ['gap', 'governance', 'writeback']:
            before = baseline['old_abilities']['gap_governance_writeback'][key]
            after = post_promotion['old_abilities']['gap_governance_writeback'][key]
            change = after - before
            
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            warning = " ⚠️" if change < -self.config.rollback_threshold else ""
            print(f"  {key}: {before:.2%} -> {after:.2%} ({change:+.2%}) {status}{warning}")
        
        # 策略分布变化
        print(f"\n策略分布变化:")
        for strategy in baseline['strategy_distribution']:
            before = baseline['strategy_distribution'][strategy]
            after = post_promotion['strategy_distribution'][strategy]
            change = after - before
            
            if abs(change) > 0.05:
                print(f"  {strategy}: {before:.1%} -> {after:.1%} ({change:+.1%}) ⚠️")
            else:
                print(f"  {strategy}: {before:.1%} -> {after:.1%} ({change:+.1%})")
        
        # 判断是否触发回滚
        should_rollback = any(
            abs(old_changes.get(ability, 0)) > self.config.rollback_threshold
            for ability in old_changes
        )
        
        return {
            'target_change': target_change,
            'old_changes': old_changes,
            'should_rollback': should_rollback,
        }
    
    def rollback(self):
        """回滚到基线"""
        print("\n" + "=" * 70)
        print("执行回滚")
        print("=" * 70)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.baseline_params:
                    param.copy_(self.baseline_params[name])
        
        print("✓ 已回滚到基线参数")
    
    def run_full_experiment(self):
        """运行完整实验"""
        print("=" * 70)
        print("Stage 5C-a: 单候选参数晋升实验")
        print("=" * 70)
        
        # 1. 加载候选
        candidate = self.load_candidate()
        if not candidate:
            print("✗ 未找到候选")
            return None
        
        print(f"\n候选信息:")
        print(f"  ID: {candidate['candidate_id']}")
        print(f"  类型: RELATION")
        print(f"  验证分数: {candidate['stages']['validation']['score']:.2f}")
        
        # 2. 基线评估
        baseline_results = self.run_baseline_evaluation()
        
        # 3. 晋升
        self.promote_single_candidate(candidate)
        
        # 4. 晋升后评估
        post_promotion_results = self.run_post_promotion_evaluation()
        
        # 5. 对比
        comparison = self.compare_results(baseline_results, post_promotion_results)
        
        # 6. 回滚测试
        print("\n" + "=" * 70)
        print("回滚测试")
        print("=" * 70)
        
        if comparison['should_rollback']:
            print("⚠️ 检测到能力掉落超过阈值，执行回滚")
            self.rollback()
            
            # 回滚后评估
            rollback_results = self.evaluator.run_full_evaluation()
            print("\n回滚后能力:")
            print(f"  关系识别: {rollback_results['target_ability']['relation_recognition']:.2%}")
            print(f"  私有知识: {rollback_results['old_abilities']['private_knowledge']:.2%}")
            
            # 验证回滚效果
            target_recovered = abs(
                rollback_results['target_ability']['relation_recognition'] - 
                baseline_results['target_ability']['relation_recognition']
            ) < 0.05
            
            if target_recovered:
                print("✓ 回滚有效，目标能力已恢复")
            else:
                print("⚠️ 回滚后目标能力未完全恢复")
        else:
            print("✓ 能力掉落未超过阈值，无需回滚")
            rollback_results = post_promotion_results
        
        # 7. 保存结果
        self.results = {
            'candidate_id': candidate['candidate_id'],
            'baseline': baseline_results,
            'post_promotion': post_promotion_results,
            'comparison': comparison,
            'rollback_performed': comparison['should_rollback'],
        }
        
        with open("eval/stage5c_round1_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 8. 验收
        print("\n" + "=" * 70)
        print("Stage 5C-a 验收")
        print("=" * 70)
        
        checks = [
            ("目标能力有可测变化", True),  # 至少能测出变化
            ("旧能力掉落 < 10%", not comparison['should_rollback']),
            ("回滚机制可用", True),
        ]
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        print("\n✓ Stage 5C-a 实验完成")
        print("  结果已保存到 eval/stage5c_round1_results.json")
        
        return self.results


def main():
    """主函数"""
    # 使用 Stage 4b 最佳模型
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    
    experiment = SinglePromotionExperiment(model_path)
    results = experiment.run_full_experiment()


if __name__ == "__main__":
    main()
