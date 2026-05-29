"""
Candidate Quality Benchmark

候选质量基准测试器

功能：
1. 评估候选生成质量
2. 对比不同候选类型的质量
3. 建立质量基线
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class QualityConfig:
    """质量评估配置"""
    min_structure_score: float = 0.8
    min_consistency_score: float = 0.7
    min_novelty_score: float = 0.3  # 允许一定重复，但要有新意


class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
    
    def evaluate_structure(self, candidate: Dict) -> float:
        """评估结构完整性"""
        score = 0.0
        
        # 检查必需字段
        required_fields = ['candidate_id', 'candidate_type', 'generated_content', 'metadata']
        for field in required_fields:
            if field in candidate:
                score += 0.2
        
        # 检查 generated_content 内容
        content = candidate.get('generated_content', {})
        if content:
            score += 0.1
            
            # 检查是否有实质内容
            if any(v for v in content.values() if v):
                score += 0.1
        
        return min(1.0, score)
    
    def evaluate_consistency(self, candidate: Dict) -> float:
        """评估内容一致性"""
        score = 1.0
        
        content = candidate.get('generated_content', {})
        candidate_type = candidate.get('candidate_type', '')
        
        # 检查类型与内容匹配
        if candidate_type == 'EXPLANATION':
            # 解释类应该有 reasoning
            if 'reasoning' not in content:
                score -= 0.3
        
        elif candidate_type == 'RELATION':
            # 关系类应该有 relation_type
            if 'relation_type' not in content:
                score -= 0.3
        
        elif candidate_type == 'RULE':
            # 规则类应该有 if/then
            if 'if_condition' not in content or 'then_action' not in content:
                score -= 0.3
        
        elif candidate_type == 'PATTERN':
            # 模式类应该有 steps
            if 'steps' not in content:
                score -= 0.3
        
        # 检查置信度合理性
        confidence = content.get('confidence', 0.5)
        if isinstance(confidence, dict):
            confidence = sum(confidence.values()) / len(confidence) if confidence else 0.5
        
        if confidence < 0.3 or confidence > 1.0:
            score -= 0.2
        
        return max(0.0, score)
    
    def evaluate_novelty(self, candidate: Dict, existing_candidates: List[Dict]) -> float:
        """评估新颖性"""
        if not existing_candidates:
            return 1.0  # 第一个候选总是新颖的
        
        # 简单的相似度检查
        candidate_type = candidate.get('candidate_type', '')
        content = str(candidate.get('generated_content', {}))
        
        # 检查是否有相似候选
        similar_count = 0
        for existing in existing_candidates:
            if existing.get('candidate_type') == candidate_type:
                existing_content = str(existing.get('generated_content', {}))
                # 简单字符串相似度
                if self._simple_similarity(content, existing_content) > 0.8:
                    similar_count += 1
        
        # 相似度越高，新颖性越低
        novelty = 1.0 - (similar_count / len(existing_candidates)) * 0.5
        return max(0.0, novelty)
    
    def _simple_similarity(self, str1: str, str2: str) -> float:
        """计算简单字符串相似度"""
        if not str1 or not str2:
            return 0.0
        
        # 使用 Jaccard 相似度
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_hallucination(self, candidate: Dict) -> float:
        """评估幻觉程度（越低越好）"""
        hallucination_score = candidate.get('metadata', {}).get('hallucination_score', 1.0)
        
        # 幻觉分数越低越好
        return 1.0 - hallucination_score
    
    def evaluate(self, candidate: Dict, existing_candidates: List[Dict] = None) -> Dict:
        """完整评估"""
        if existing_candidates is None:
            existing_candidates = []
        
        structure_score = self.evaluate_structure(candidate)
        consistency_score = self.evaluate_consistency(candidate)
        novelty_score = self.evaluate_novelty(candidate, existing_candidates)
        hallucination_resistance = self.evaluate_hallucination(candidate)
        
        # 综合分数
        overall_score = (
            structure_score * 0.3 +
            consistency_score * 0.3 +
            novelty_score * 0.2 +
            hallucination_resistance * 0.2
        )
        
        return {
            'structure': structure_score,
            'consistency': consistency_score,
            'novelty': novelty_score,
            'hallucination_resistance': hallucination_resistance,
            'overall': overall_score,
        }


class QualityBenchmark:
    """质量基准测试"""
    
    def __init__(self):
        self.config = QualityConfig()
        self.evaluator = QualityEvaluator(self.config)
        self.candidates = []
    
    def load_candidates(self, path: str = "candidates/stage5a_candidates.jsonl") -> List[Dict]:
        """加载候选"""
        candidates = []
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    candidates.append(json.loads(line))
        return candidates
    
    def run_benchmark(self, candidates: List[Dict]) -> Dict:
        """运行基准测试"""
        print("\n" + "=" * 70)
        print("候选质量基准测试")
        print("=" * 70)
        
        results = []
        type_scores = defaultdict(list)
        
        for i, candidate in enumerate(candidates):
            # 评估当前候选
            existing = candidates[:i]  # 之前的候选
            scores = self.evaluator.evaluate(candidate, existing)
            results.append(scores)
            
            # 按类型分组
            candidate_type = candidate.get('candidate_type', 'UNKNOWN')
            type_scores[candidate_type].append(scores)
        
        # 计算总体统计
        avg_scores = {
            'structure': sum(r['structure'] for r in results) / len(results) if results else 0,
            'consistency': sum(r['consistency'] for r in results) / len(results) if results else 0,
            'novelty': sum(r['novelty'] for r in results) / len(results) if results else 0,
            'hallucination_resistance': sum(r['hallucination_resistance'] for r in results) / len(results) if results else 0,
            'overall': sum(r['overall'] for r in results) / len(results) if results else 0,
        }
        
        # 计算各类型统计
        type_avg_scores = {}
        for candidate_type, scores_list in type_scores.items():
            type_avg_scores[candidate_type] = {
                'structure': sum(s['structure'] for s in scores_list) / len(scores_list),
                'consistency': sum(s['consistency'] for s in scores_list) / len(scores_list),
                'novelty': sum(s['novelty'] for s in scores_list) / len(scores_list),
                'hallucination_resistance': sum(s['hallucination_resistance'] for s in scores_list) / len(scores_list),
                'overall': sum(s['overall'] for s in scores_list) / len(scores_list),
                'count': len(scores_list),
            }
        
        print(f"\n总体质量统计:")
        print(f"  候选总数: {len(candidates)}")
        print(f"  结构完整性: {avg_scores['structure']:.2f}")
        print(f"  内容一致性: {avg_scores['consistency']:.2f}")
        print(f"  新颖性: {avg_scores['novelty']:.2f}")
        print(f"  抗幻觉: {avg_scores['hallucination_resistance']:.2f}")
        print(f"  综合分数: {avg_scores['overall']:.2f}")
        
        print(f"\n各类型质量统计:")
        for candidate_type, scores in type_avg_scores.items():
            print(f"\n  {candidate_type} (n={scores['count']}):")
            print(f"    综合分数: {scores['overall']:.2f}")
            print(f"    结构完整性: {scores['structure']:.2f}")
            print(f"    内容一致性: {scores['consistency']:.2f}")
        
        return {
            'total_candidates': len(candidates),
            'avg_scores': avg_scores,
            'type_scores': type_avg_scores,
            'detailed_results': results,
        }
    
    def check_quality_standards(self, results: Dict) -> Dict:
        """检查质量标准"""
        print("\n" + "=" * 70)
        print("质量标准检查")
        print("=" * 70)
        
        avg_scores = results['avg_scores']
        
        checks = [
            ('结构完整性 >= 0.8', avg_scores['structure'] >= self.config.min_structure_score),
            ('内容一致性 >= 0.7', avg_scores['consistency'] >= self.config.min_consistency_score),
            ('抗幻觉 >= 0.5', avg_scores['hallucination_resistance'] >= 0.5),
            ('综合分数 >= 0.6', avg_scores['overall'] >= 0.6),
        ]
        
        passed = sum(1 for _, p in checks if p)
        total = len(checks)
        
        for check_name, passed_check in checks:
            status = "✓" if passed_check else "✗"
            print(f"  {status} {check_name}")
        
        print(f"\n通过率: {passed}/{total} ({passed/total:.1%})")
        
        return {
            'checks': checks,
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total else 0,
        }
    
    def run_full_benchmark(self):
        """运行完整基准测试"""
        print("=" * 70)
        print("候选质量基准测试")
        print("=" * 70)
        
        # 加载候选
        candidates = self.load_candidates()
        
        if not candidates:
            print("⚠ 没有候选，请先运行 Stage 5A")
            return None
        
        # 运行基准测试
        results = self.run_benchmark(candidates)
        
        # 检查质量标准
        quality_check = self.check_quality_standards(results)
        
        # 保存结果
        output = {
            'benchmark': {
                'total_candidates': results['total_candidates'],
                'avg_scores': results['avg_scores'],
                'type_scores': results['type_scores'],
            },
            'quality_check': {
                'passed': quality_check['passed'],
                'total': quality_check['total'],
                'pass_rate': quality_check['pass_rate'],
            },
        }
        
        with open("eval/quality_benchmark_results.json", 'w') as f:
            json.dump(output, f, indent=2)
        
        # 验收
        print("\n" + "=" * 70)
        print("质量基准测试验收")
        print("=" * 70)
        
        if quality_check['pass_rate'] >= 0.75:
            print("✓ 质量基准测试通过")
        else:
            print("✗ 质量基准测试未通过，需要优化")
        
        return output


def main():
    """主函数"""
    benchmark = QualityBenchmark()
    results = benchmark.run_full_benchmark()
    
    if results:
        print("\n✓ 结果已保存到 eval/quality_benchmark_results.json")


if __name__ == "__main__":
    main()
