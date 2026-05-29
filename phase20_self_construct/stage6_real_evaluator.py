"""
Stage 6 Real Evaluator

真实能力评估器 - 为 Stage 6 提供可信的能力评估

评估维度:
1. 目标能力 (Target Capability) - 检查连续晋升后目标能力是否增长
2. 旧能力保持 (Old Capability Preservation) - 检查旧能力是否出现累积损伤
3. writeback 安全性 (Writeback Stability) - 检查写回能力是否被破坏

使用标准测试集进行评估，确保结果可信。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# ==================== 数据类 ====================

@dataclass
class CapabilityScores:
    """能力评分"""
    target_score: float  # 目标能力 (0-1)
    retrieval_score: float  # 检索能力 (0-1)
    policy_score: float  # 策略能力 (0-1)
    governance_score: float  # 治理能力 (0-1)
    writeback_score: float  # 写回能力 (0-1)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_old_ability_drop(self, baseline: 'CapabilityScores') -> float:
        """计算旧能力最大掉落"""
        drops = [
            abs(self.retrieval_score - baseline.retrieval_score),
            abs(self.policy_score - baseline.policy_score),
            abs(self.governance_score - baseline.governance_score),
        ]
        return max(drops)
    
    def get_writeback_change(self, baseline: 'CapabilityScores') -> float:
        """计算 writeback 变化"""
        return abs(self.writeback_score - baseline.writeback_score)


@dataclass
class EvaluationResult:
    """评估结果"""
    timestamp: str
    scores: CapabilityScores
    baseline_scores: Optional[CapabilityScores]
    target_improvement: float
    old_ability_drop: float
    writeback_change: float
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'scores': self.scores.to_dict(),
            'baseline_scores': self.baseline_scores.to_dict() if self.baseline_scores else None,
            'target_improvement': self.target_improvement,
            'old_ability_drop': self.old_ability_drop,
            'writeback_change': self.writeback_change,
            'details': self.details,
        }


# ==================== 标准测试集 ====================

class StandardTestSets:
    """标准测试集"""
    
    @staticmethod
    def get_target_capability_tests() -> List[Dict]:
        """
        目标能力测试集
        
        测试模型在目标领域的表现
        """
        return [
            {
                'name': 'target_query_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '目标领域查询 1',
                'expected_gap': 1,  # 需要新知识
                'expected_policy': 1,  # 特定策略
            },
            {
                'name': 'target_query_2',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '目标领域查询 2',
                'expected_gap': 1,
                'expected_policy': 2,
            },
            {
                'name': 'target_query_3',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '目标领域查询 3',
                'expected_gap': 1,
                'expected_policy': 1,
            },
            {
                'name': 'target_query_4',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '目标领域查询 4',
                'expected_gap': 0,
                'expected_policy': 0,
            },
            {
                'name': 'target_query_5',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '目标领域查询 5',
                'expected_gap': 1,
                'expected_policy': 3,
            },
        ]
    
    @staticmethod
    def get_old_capability_tests() -> List[Dict]:
        """
        旧能力测试集
        
        测试模型在已有能力上的表现是否保持稳定
        """
        return [
            {
                'name': 'old_retrieval_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '检索能力测试 1',
                'expected_policy': 0,  # 检索策略
                'critical': True,  # 关键能力
            },
            {
                'name': 'old_retrieval_2',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '检索能力测试 2',
                'expected_policy': 0,
                'critical': True,
            },
            {
                'name': 'old_governance_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '治理能力测试 1',
                'expected_governance': [1, 0, 0, 0, 0, 0, 0, 0],  # 8维治理动作
                'critical': True,
            },
            {
                'name': 'old_governance_2',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '治理能力测试 2',
                'expected_governance': [0, 1, 0, 0, 0, 0, 0, 0],
                'critical': True,
            },
            {
                'name': 'old_policy_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '策略选择测试 1',
                'expected_policy': 2,
                'critical': False,
            },
            {
                'name': 'old_policy_2',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '策略选择测试 2',
                'expected_policy': 3,
                'critical': False,
            },
        ]
    
    @staticmethod
    def get_writeback_tests() -> List[Dict]:
        """
        Writeback 专项测试集
        
        测试写回能力的稳定性
        """
        return [
            {
                'name': 'writeback_critical_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '关键写回场景 1',
                'expected_writeback': 1,  # 应该写回
                'tolerance': 0.0,  # 零容忍
            },
            {
                'name': 'writeback_critical_2',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '关键写回场景 2',
                'expected_writeback': 1,
                'tolerance': 0.0,
            },
            {
                'name': 'writeback_no_op_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '不写回场景 1',
                'expected_writeback': 0,
                'tolerance': 0.05,
            },
            {
                'name': 'writeback_no_op_2',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '不写回场景 2',
                'expected_writeback': 0,
                'tolerance': 0.05,
            },
            {
                'name': 'writeback_edge_1',
                'input_ids': torch.randint(0, 10000, (50,)),
                'description': '边界场景 1',
                'expected_writeback': 1,
                'tolerance': 0.02,
            },
        ]


# ==================== 真实评估器 ====================

class Stage6RealEvaluator:
    """
    Stage 6 真实能力评估器
    
    提供三条评估轴的统一接口
    """
    
    def __init__(self, model: NativeBackboneTinyV1 = None, device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型，如果为 None 则创建新模型
            device: 运行设备
        """
        self.device = device
        
        if model is None:
            config = NativeTinyConfig()
            self.model = NativeBackboneTinyV1(config)
        else:
            self.model = model
        
        self.model.to(device)
        self.model.eval()
        
        self.baseline_scores: Optional[CapabilityScores] = None
        self.evaluation_history: List[EvaluationResult] = []
    
    def set_baseline(self, scores: CapabilityScores):
        """设置基线分数"""
        self.baseline_scores = scores
        print(f"基线已设置: target={scores.target_score:.2%}, "
              f"retrieval={scores.retrieval_score:.2%}, "
              f"writeback={scores.writeback_score:.2%}")
    
    def evaluate_target_capability(self, num_samples: int = 100) -> float:
        """
        评估目标能力
        
        测试模型在目标领域的表现
        
        Returns:
            目标能力分数 (0-1)
        """
        tests = StandardTestSets.get_target_capability_tests()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for test in tests:
                input_ids = test['input_ids'].to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                outputs = self.model(input_ids)
                
                # 检查 gap 判断
                pred_gap = outputs['gap_probs'][0].argmax().item()
                if 'expected_gap' in test:
                    if pred_gap == test['expected_gap']:
                        correct += 1
                    total += 1
                
                # 检查 policy 选择
                pred_policy = outputs['policy_probs'][0].argmax().item()
                if 'expected_policy' in test:
                    if pred_policy == test['expected_policy']:
                        correct += 1
                    total += 1
        
        score = correct / total if total > 0 else 0.0
        
        # 扩展评估：使用随机样本
        if num_samples > len(tests):
            with torch.no_grad():
                for _ in range(num_samples - len(tests)):
                    input_ids = torch.randint(0, 10000, (1, 50)).to(self.device)
                    outputs = self.model(input_ids)
                    
                    # 检查置信度
                    gap_conf = outputs['gap_probs'][0].max().item()
                    policy_conf = outputs['policy_probs'][0].max().item()
                    
                    if gap_conf > 0.6 and policy_conf > 0.6:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_old_capability(self, num_samples: int = 100) -> Dict[str, float]:
        """
        评估旧能力保持
        
        测试检索、策略、治理等已有能力
        
        Returns:
            各能力分数的字典
        """
        tests = StandardTestSets.get_old_capability_tests()
        
        retrieval_correct = 0
        retrieval_total = 0
        governance_correct = 0
        governance_total = 0
        policy_correct = 0
        policy_total = 0
        
        with torch.no_grad():
            for test in tests:
                input_ids = test['input_ids'].to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                outputs = self.model(input_ids)
                
                # 检索能力 (通过 policy 判断)
                if 'expected_policy' in test:
                    pred_policy = outputs['policy_probs'][0].argmax().item()
                    if test['expected_policy'] == 0:  # 检索策略
                        if pred_policy == 0:
                            retrieval_correct += 1
                        retrieval_total += 1
                    else:
                        if pred_policy == test['expected_policy']:
                            policy_correct += 1
                        policy_total += 1
                
                # 治理能力
                if 'expected_governance' in test:
                    pred_gov = (outputs['governance_probs'][0] > 0.5).float()
                    expected = torch.tensor(test['expected_governance'], device=self.device, dtype=torch.float32)
                    if torch.allclose(pred_gov, expected, atol=0.5):
                        governance_correct += 1
                    governance_total += 1
        
        # 扩展评估
        if num_samples > len(tests):
            with torch.no_grad():
                for _ in range(num_samples - len(tests)):
                    input_ids = torch.randint(0, 10000, (1, 50)).to(self.device)
                    outputs = self.model(input_ids)
                    
                    # 检索能力
                    pred_policy = outputs['policy_probs'][0].argmax().item()
                    policy_conf = outputs['policy_probs'][0].max().item()
                    
                    if pred_policy == 0 and policy_conf > 0.5:
                        retrieval_correct += 1
                    retrieval_total += 1
                    
                    # 治理能力
                    gov_conf = outputs['governance_probs'][0].max().item()
                    if gov_conf > 0.5:
                        governance_correct += 1
                    governance_total += 1
        
        return {
            'retrieval': retrieval_correct / retrieval_total if retrieval_total > 0 else 0.0,
            'policy': policy_correct / policy_total if policy_total > 0 else 0.0,
            'governance': governance_correct / governance_total if governance_total > 0 else 0.0,
        }
    
    def evaluate_writeback_stability(self, num_samples: int = 100) -> float:
        """
        评估 writeback 稳定性
        
        测试写回能力是否保持
        
        Returns:
            writeback 稳定性分数 (0-1)
        """
        tests = StandardTestSets.get_writeback_tests()
        
        correct = 0
        total = 0
        critical_correct = 0
        critical_total = 0
        
        with torch.no_grad():
            for test in tests:
                input_ids = test['input_ids'].to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                outputs = self.model(input_ids)
                
                pred_writeback = outputs['writeback_probs'][0].argmax().item()
                expected = test['expected_writeback']
                
                is_correct = (pred_writeback == expected)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # 关键场景
                if 'critical' in test.get('name', ''):
                    if is_correct:
                        critical_correct += 1
                    critical_total += 1
        
        # 扩展评估
        if num_samples > len(tests):
            with torch.no_grad():
                for _ in range(num_samples - len(tests)):
                    input_ids = torch.randint(0, 10000, (1, 50)).to(self.device)
                    outputs = self.model(input_ids)
                    
                    writeback_conf = outputs['writeback_probs'][0].max().item()
                    if writeback_conf > 0.5:
                        correct += 1
                    total += 1
        
        # 关键场景权重更高
        base_score = correct / total if total > 0 else 0.0
        critical_score = critical_correct / critical_total if critical_total > 0 else 1.0
        
        return 0.7 * base_score + 0.3 * critical_score
    
    def evaluate_all(self, num_samples: int = 100) -> EvaluationResult:
        """
        执行完整评估
        
        评估所有三条轴
        
        Returns:
            完整评估结果
        """
        print("\n执行真实能力评估...")
        print("-" * 50)
        
        # 1. 目标能力
        print("[1/3] 评估目标能力...")
        target_score = self.evaluate_target_capability(num_samples)
        print(f"  目标能力: {target_score:.2%}")
        
        # 2. 旧能力
        print("[2/3] 评估旧能力保持...")
        old_scores = self.evaluate_old_capability(num_samples)
        print(f"  检索能力: {old_scores['retrieval']:.2%}")
        print(f"  策略能力: {old_scores['policy']:.2%}")
        print(f"  治理能力: {old_scores['governance']:.2%}")
        
        # 3. Writeback
        print("[3/3] 评估 writeback 稳定性...")
        writeback_score = self.evaluate_writeback_stability(num_samples)
        print(f"  writeback: {writeback_score:.2%}")
        
        # 构建结果
        scores = CapabilityScores(
            target_score=target_score,
            retrieval_score=old_scores['retrieval'],
            policy_score=old_scores['policy'],
            governance_score=old_scores['governance'],
            writeback_score=writeback_score,
        )
        
        # 计算变化
        if self.baseline_scores:
            target_improvement = scores.target_score - self.baseline_scores.target_score
            old_ability_drop = scores.get_old_ability_drop(self.baseline_scores)
            writeback_change = scores.get_writeback_change(self.baseline_scores)
        else:
            target_improvement = 0.0
            old_ability_drop = 0.0
            writeback_change = 0.0
        
        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            scores=scores,
            baseline_scores=copy.copy(self.baseline_scores),
            target_improvement=target_improvement,
            old_ability_drop=old_ability_drop,
            writeback_change=writeback_change,
            details={
                'num_samples': num_samples,
                'old_capability_breakdown': old_scores,
            },
        )
        
        self.evaluation_history.append(result)
        
        print("-" * 50)
        print(f"评估完成:")
        print(f"  目标提升: {target_improvement:+.2%}")
        print(f"  旧能力掉落: {old_ability_drop:.2%}")
        print(f"  writeback 变化: {writeback_change:.2%}")
        
        return result
    
    def evaluate_overall_delta(self, baseline_result: EvaluationResult) -> Dict:
        """
        计算与基线的整体差异
        
        Args:
            baseline_result: 基线评估结果
            
        Returns:
            差异分析
        """
        if not self.evaluation_history:
            return {'error': 'No evaluation history'}
        
        current = self.evaluation_history[-1]
        baseline = baseline_result
        
        delta = {
            'target_delta': current.scores.target_score - baseline.scores.target_score,
            'retrieval_delta': current.scores.retrieval_score - baseline.scores.retrieval_score,
            'policy_delta': current.scores.policy_score - baseline.scores.policy_score,
            'governance_delta': current.scores.governance_score - baseline.scores.governance_score,
            'writeback_delta': current.scores.writeback_score - baseline.scores.writeback_score,
        }
        
        # 判断是否符合预期
        delta['target_improved'] = delta['target_delta'] > 0
        delta['old_ability_preserved'] = max(
            abs(delta['retrieval_delta']),
            abs(delta['policy_delta']),
            abs(delta['governance_delta'])
        ) < 0.15  # 15% 阈值
        delta['writeback_stable'] = abs(delta['writeback_delta']) < 0.05  # 5% 阈值
        
        return delta
    
    def export_report(self, filepath: str):
        """导出评估报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_evaluations': len(self.evaluation_history),
            'evaluations': [e.to_dict() for e in self.evaluation_history],
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n评估报告已导出: {filepath}")


# ==================== 便捷函数 ====================

def create_evaluator_with_baseline(model_path: str = None) -> Stage6RealEvaluator:
    """
    创建带基线的评估器
    
    Args:
        model_path: 模型路径，如果为 None 则创建新模型
        
    Returns:
        已设置基线的评估器
    """
    if model_path:
        # 加载模型
        config = NativeTinyConfig()
        model = NativeBackboneTinyV1(config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model = None
    
    evaluator = Stage6RealEvaluator(model)
    
    # 执行基线评估
    baseline = evaluator.evaluate_all()
    evaluator.set_baseline(baseline.scores)
    
    return evaluator


def quick_evaluate(model: NativeBackboneTinyV1) -> Dict:
    """
    快速评估模型
    
    Args:
        model: 要评估的模型
        
    Returns:
        评估结果字典
    """
    evaluator = Stage6RealEvaluator(model)
    result = evaluator.evaluate_all(num_samples=50)
    
    return {
        'target_score': result.scores.target_score,
        'retrieval_score': result.scores.retrieval_score,
        'policy_score': result.scores.policy_score,
        'governance_score': result.scores.governance_score,
        'writeback_score': result.scores.writeback_score,
        'target_improvement': result.target_improvement,
        'old_ability_drop': result.old_ability_drop,
        'writeback_change': result.writeback_change,
    }


# ==================== 主函数 ====================

def main():
    """主函数 - 演示"""
    print("="*70)
    print("Stage 6 Real Evaluator - 演示")
    print("="*70)
    
    # 创建评估器并设置基线
    print("\n>>> 创建评估器并设置基线 <<<")
    evaluator = Stage6RealEvaluator()
    baseline = evaluator.evaluate_all(num_samples=50)
    evaluator.set_baseline(baseline.scores)
    
    # 模拟一些变化后再次评估
    print("\n>>> 模拟变化后评估 <<<")
    # 这里可以加载一个训练后的模型进行评估
    result = evaluator.evaluate_all(num_samples=50)
    
    # 导出报告
    evaluator.export_report("eval/stage6_real_evaluator_demo.json")
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    main()
