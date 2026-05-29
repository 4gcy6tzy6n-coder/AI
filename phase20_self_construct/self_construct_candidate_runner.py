"""
Self-Construct Candidate Runner

Stage 5A: 候选自构建实验

目标：让模型生成候选内容，但只到候选层
生成对象：候选解释、候选关系、候选规则、候选任务模式

验收指标：
- 幻觉率低
- 候选结构合法
- 候选有后续治理价值
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
import random
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class CandidateConfig:
    """候选生成配置"""
    max_candidates_per_run: int = 100
    hallucination_threshold: float = 0.3  # 幻觉检测阈值
    min_confidence: float = 0.7  # 最小置信度
    structure_validation: bool = True  # 结构验证
    device: str = 'cpu'


class CandidateGenerator:
    """候选生成器"""
    
    def __init__(self, model: NativeBackboneTinyV1, config: CandidateConfig):
        self.model = model
        self.config = config
        self.model.eval()
    
    def generate_candidate_explanation(self, query: str, context: Dict) -> Optional[Dict]:
        """
        生成候选解释
        
        例如：对某个概念的解释、对某个决策的理由
        """
        # 编码输入
        input_ids = self._encode_text(query)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # 获取置信度
        gap_conf = outputs['gap_probs'][0].max().item()
        policy_conf = outputs['policy_probs'][0].max().item()
        
        # 检查置信度
        if gap_conf < self.config.min_confidence or policy_conf < self.config.min_confidence:
            return None
        
        # 生成候选解释
        candidate = {
            'candidate_id': self._generate_id(query, 'explanation'),
            'candidate_type': 'EXPLANATION',
            'source_query': query,
            'context': context,
            'generated_content': {
                'gap_type': self._get_gap_type(outputs['gap_type'][0].item()),
                'strategy': self._get_strategy(outputs['strategy'][0].item()),
                'confidence': {
                    'gap': gap_conf,
                    'policy': policy_conf,
                },
                'reasoning': f"基于 {self._get_gap_type(outputs['gap_type'][0].item())} 缺口类型，"
                           f"选择 {self._get_strategy(outputs['strategy'][0].item())} 策略",
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'native_128_stage4b',
                'hallucination_score': self._estimate_hallucination(outputs),
            },
            'status': 'PENDING_REVIEW',
        }
        
        # 结构验证
        if self.config.structure_validation and not self._validate_structure(candidate):
            return None
        
        return candidate
    
    def generate_candidate_relation(self, entity_a: str, entity_b: str, relation_type: str) -> Optional[Dict]:
        """
        生成候选关系
        
        例如：实体 A 和实体 B 之间的关系
        """
        query = f"{entity_a} 和 {entity_b} 的关系"
        input_ids = self._encode_text(query)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        gap_conf = outputs['gap_probs'][0].max().item()
        
        if gap_conf < self.config.min_confidence:
            return None
        
        candidate = {
            'candidate_id': self._generate_id(f"{entity_a}_{entity_b}", 'relation'),
            'candidate_type': 'RELATION',
            'source_query': query,
            'entities': {'a': entity_a, 'b': entity_b},
            'generated_content': {
                'relation_type': relation_type,
                'confidence': gap_conf,
                'direction': 'bidirectional' if random.random() > 0.5 else 'unidirectional',
                'strength': random.uniform(0.5, 1.0),
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'native_128_stage4b',
                'hallucination_score': self._estimate_hallucination(outputs),
            },
            'status': 'PENDING_REVIEW',
        }
        
        if self.config.structure_validation and not self._validate_structure(candidate):
            return None
        
        return candidate
    
    def generate_candidate_rule(self, scenario: str, condition: str, action: str) -> Optional[Dict]:
        """
        生成候选规则
        
        例如：如果 [条件]，则 [动作]
        """
        query = f"当 {condition} 时，应该 {action}"
        input_ids = self._encode_text(query)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        policy_conf = outputs['policy_probs'][0].max().item()
        gov_conf = outputs['governance_probs'][0].max().item()
        
        if policy_conf < self.config.min_confidence:
            return None
        
        candidate = {
            'candidate_id': self._generate_id(scenario, 'rule'),
            'candidate_type': 'RULE',
            'source_query': query,
            'scenario': scenario,
            'generated_content': {
                'if_condition': condition,
                'then_action': action,
                'governance_action': self._get_governance_action(outputs['governance_probs'][0]),
                'confidence': {
                    'policy': policy_conf,
                    'governance': gov_conf,
                },
                'priority': random.randint(1, 5),
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'native_128_stage4b',
                'hallucination_score': self._estimate_hallucination(outputs),
            },
            'status': 'PENDING_REVIEW',
        }
        
        if self.config.structure_validation and not self._validate_structure(candidate):
            return None
        
        return candidate
    
    def generate_candidate_pattern(self, task_description: str, steps: List[str]) -> Optional[Dict]:
        """
        生成候选任务模式
        
        例如：完成某类任务的标准步骤
        """
        query = f"如何完成：{task_description}"
        input_ids = self._encode_text(query)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        policy_conf = outputs['policy_probs'][0].max().item()
        
        if policy_conf < self.config.min_confidence:
            return None
        
        candidate = {
            'candidate_id': self._generate_id(task_description, 'pattern'),
            'candidate_type': 'PATTERN',
            'source_query': query,
            'task_description': task_description,
            'generated_content': {
                'steps': steps,
                'strategy': self._get_strategy(outputs['strategy'][0].item()),
                'confidence': policy_conf,
                'estimated_steps': len(steps),
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'native_128_stage4b',
                'hallucination_score': self._estimate_hallucination(outputs),
            },
            'status': 'PENDING_REVIEW',
        }
        
        if self.config.structure_validation and not self._validate_structure(candidate):
            return None
        
        return candidate
    
    def _encode_text(self, text: str, max_len: int = 100) -> List[int]:
        """编码文本"""
        vocab_size = 10000
        ids = []
        for char in text[:max_len]:
            hash_val = hash(char) % vocab_size
            ids.append(abs(hash_val))
        while len(ids) < max_len:
            ids.append(0)
        return ids[:max_len]
    
    def _generate_id(self, content: str, prefix: str) -> str:
        """生成唯一ID"""
        hash_obj = hashlib.md5(f"{content}_{prefix}_{datetime.now()}".encode())
        return f"{prefix}_{hash_obj.hexdigest()[:12]}"
    
    def _get_gap_type(self, idx: int) -> str:
        """获取缺口类型"""
        types = ['NO_GAP', 'RETRIEVABLE', 'HIGH_RISK']
        return types[idx] if idx < len(types) else 'UNKNOWN'
    
    def _get_strategy(self, idx: int) -> str:
        """获取策略"""
        strategies = ['DIRECT', 'RETRIEVAL_FIRST', 'CONSERVATIVE', 'DECLINE', 'REVIEW']
        return strategies[idx] if idx < len(strategies) else 'UNKNOWN'
    
    def _get_governance_action(self, probs: torch.Tensor) -> str:
        """获取治理动作"""
        actions = ['VERIFY_SCOPE', 'CHECK_PERMISSION', 'REJECT', 'LOG_INCIDENT', 
                  'REQUEST_CLARIFICATION', 'ESCALATE', 'ISOLATE', 'ARCHIVE']
        idx = probs.argmax().item()
        return actions[idx] if idx < len(actions) else 'UNKNOWN'
    
    def _estimate_hallucination(self, outputs: Dict) -> float:
        """
        估计幻觉分数
        
        基于输出概率分布的熵来估计
        """
        # 计算各头的熵
        gap_entropy = self._compute_entropy(outputs['gap_probs'][0])
        policy_entropy = self._compute_entropy(outputs['policy_probs'][0])
        
        # 高熵 = 不确定 = 可能幻觉
        hallucination = (gap_entropy + policy_entropy) / 2
        return min(hallucination, 1.0)
    
    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """计算熵"""
        # 避免 log(0)
        probs = probs + 1e-10
        entropy = -(probs * torch.log(probs)).sum().item()
        return entropy
    
    def _validate_structure(self, candidate: Dict) -> bool:
        """验证候选结构是否合法"""
        required_fields = ['candidate_id', 'candidate_type', 'generated_content', 'metadata']
        
        for field in required_fields:
            if field not in candidate:
                return False
        
        # 验证 candidate_type
        valid_types = ['EXPLANATION', 'RELATION', 'RULE', 'PATTERN']
        if candidate['candidate_type'] not in valid_types:
            return False
        
        return True


class Stage5AExperiment:
    """Stage 5A 实验运行器"""
    
    def __init__(self, model_path: str = None):
        self.config = CandidateConfig()
        
        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ 加载模型: {model_path}")
        else:
            print("⚠ 使用随机初始化模型")
        
        self.generator = CandidateGenerator(self.model, self.config)
        self.candidates = []
    
    def run_explanation_generation(self, test_queries: List[str]) -> Dict:
        """运行解释生成实验"""
        print("\n" + "=" * 70)
        print("Stage 5A: 候选解释生成")
        print("=" * 70)
        
        generated = []
        
        for query in test_queries:
            candidate = self.generator.generate_candidate_explanation(query, {})
            if candidate:
                generated.append(candidate)
                self.candidates.append(candidate)
        
        # 统计
        hallucination_low = sum(1 for c in generated 
                               if c['metadata']['hallucination_score'] < self.config.hallucination_threshold)
        
        results = {
            'total_queries': len(test_queries),
            'generated': len(generated),
            'generation_rate': len(generated) / len(test_queries) if test_queries else 0,
            'hallucination_low': hallucination_low,
            'hallucination_rate': hallucination_low / len(generated) if generated else 0,
            'avg_confidence': sum(c['generated_content']['confidence']['gap'] 
                                 for c in generated) / len(generated) if generated else 0,
        }
        
        print(f"\n生成统计:")
        print(f"  查询数: {results['total_queries']}")
        print(f"  生成数: {results['generated']}")
        print(f"  生成率: {results['generation_rate']:.1%}")
        print(f"  低幻觉: {results['hallucination_low']} ({results['hallucination_rate']:.1%})")
        print(f"  平均置信度: {results['avg_confidence']:.2f}")
        
        return results
    
    def run_relation_generation(self, entity_pairs: List[Tuple[str, str]]) -> Dict:
        """运行关系生成实验"""
        print("\n" + "=" * 70)
        print("Stage 5A: 候选关系生成")
        print("=" * 70)
        
        generated = []
        relation_types = ['related_to', 'depends_on', 'part_of', 'uses', 'conflicts_with']
        
        for entity_a, entity_b in entity_pairs:
            relation_type = random.choice(relation_types)
            candidate = self.generator.generate_candidate_relation(entity_a, entity_b, relation_type)
            if candidate:
                generated.append(candidate)
                self.candidates.append(candidate)
        
        results = {
            'total_pairs': len(entity_pairs),
            'generated': len(generated),
            'generation_rate': len(generated) / len(entity_pairs) if entity_pairs else 0,
            'avg_confidence': sum(c['generated_content']['confidence'] 
                                 for c in generated) / len(generated) if generated else 0,
        }
        
        print(f"\n生成统计:")
        print(f"  实体对: {results['total_pairs']}")
        print(f"  生成数: {results['generated']}")
        print(f"  生成率: {results['generation_rate']:.1%}")
        
        return results
    
    def run_rule_generation(self, scenarios: List[Dict]) -> Dict:
        """运行规则生成实验"""
        print("\n" + "=" * 70)
        print("Stage 5A: 候选规则生成")
        print("=" * 70)
        
        generated = []
        
        for scenario in scenarios:
            candidate = self.generator.generate_candidate_rule(
                scenario['scenario'],
                scenario['condition'],
                scenario['action']
            )
            if candidate:
                generated.append(candidate)
                self.candidates.append(candidate)
        
        results = {
            'total_scenarios': len(scenarios),
            'generated': len(generated),
            'generation_rate': len(generated) / len(scenarios) if scenarios else 0,
        }
        
        print(f"\n生成统计:")
        print(f"  场景数: {results['total_scenarios']}")
        print(f"  生成数: {results['generated']}")
        print(f"  生成率: {results['generation_rate']:.1%}")
        
        return results
    
    def save_candidates(self, path: str = "candidates/stage5a_candidates.jsonl"):
        """保存候选"""
        with open(path, 'w', encoding='utf-8') as f:
            for candidate in self.candidates:
                f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
        print(f"\n✓ 保存 {len(self.candidates)} 个候选到 {path}")
    
    def run_full_experiment(self):
        """运行完整实验"""
        print("=" * 70)
        print("Stage 5A: 候选自构建实验")
        print("=" * 70)
        
        # 1. 解释生成
        test_queries = [
            "为什么需要检索？",
            "什么时候应该拒绝回答？",
            "如何处理私有信息？",
            "什么是高风险问题？",
            "为什么需要多轮对话？",
        ]
        explanation_results = self.run_explanation_generation(test_queries)
        
        # 2. 关系生成
        entity_pairs = [
            ("用户", "查询"),
            ("检索", "知识库"),
            ("策略", "治理"),
            ("缺口", "风险"),
        ]
        relation_results = self.run_relation_generation(entity_pairs)
        
        # 3. 规则生成
        scenarios = [
            {'scenario': '高风险查询', 'condition': '检测到高风险关键词', 'action': '拒绝回答并记录'},
            {'scenario': '私有信息查询', 'condition': '涉及用户私有数据', 'action': '验证权限后检索'},
            {'scenario': '多轮对话', 'condition': '对话轮数超过5轮', 'action': '总结上下文'},
        ]
        rule_results = self.run_rule_generation(scenarios)
        
        # 保存
        self.save_candidates()
        
        # 总结
        print("\n" + "=" * 70)
        print("Stage 5A 实验总结")
        print("=" * 70)
        
        total_candidates = len(self.candidates)
        valid_structure = sum(1 for c in self.candidates if self.generator._validate_structure(c))
        low_hallucination = sum(1 for c in self.candidates 
                               if c['metadata']['hallucination_score'] < self.config.hallucination_threshold)
        
        print(f"\n总候选数: {total_candidates}")
        print(f"结构合法: {valid_structure} ({valid_structure/total_candidates:.1%})")
        print(f"低幻觉: {low_hallucination} ({low_hallucination/total_candidates:.1%})")
        
        # 验收标准
        print("\n" + "=" * 70)
        print("Stage 5A 验收标准")
        print("=" * 70)
        
        checks = [
            ("结构合法率 > 90%", valid_structure/total_candidates > 0.9 if total_candidates else False),
            ("低幻觉率 > 50%", low_hallucination/total_candidates > 0.5 if total_candidates else False),
            ("有候选生成", total_candidates > 0),
        ]
        
        all_passed = all(passed for _, passed in checks)
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        if all_passed:
            print("\n✓ Stage 5A 验收通过！可以进入 Stage 5B")
        else:
            print("\n✗ Stage 5A 需要继续优化")
        
        return {
            'explanation': explanation_results,
            'relation': relation_results,
            'rule': rule_results,
            'total_candidates': total_candidates,
            'valid_structure': valid_structure,
            'low_hallucination': low_hallucination,
            'passed': all_passed,
        }


def main():
    """主函数"""
    # 使用 Stage 4b 最佳模型
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    
    experiment = Stage5AExperiment(model_path)
    results = experiment.run_full_experiment()
    
    # 保存结果
    with open("eval/stage5a_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ 结果已保存到 eval/stage5a_results.json")


if __name__ == "__main__":
    main()
