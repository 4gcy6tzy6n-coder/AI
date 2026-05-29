"""
Evaluate Product Behavior - 产品行为评估

Phase 14 Stage 14.6: 训练后闭环评估
使用 product_behavior_acceptance_v1.md 六大维度
"""

import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ProductBehaviorEvaluator:
    """产品行为评估器"""
    
    def __init__(self):
        self.results = {}
        self.metrics = defaultdict(list)
    
    def evaluate_system_perception(self, test_cases: List[Dict]) -> Dict:
        """
        评估维度1: 系统特色感知
        用户能否识别并描述系统的独特之处
        """
        print("\n" + "="*70)
        print("维度1: 系统特色感知评估")
        print("="*70)
        
        scores = []
        features_detected = defaultdict(int)
        
        for case in test_cases:
            response = case['response']
            
            # 检测特色特征
            features = {
                'retrieval_based': any(kw in response for kw in ['根据', '记忆', '检索', '来源']),
                'conservative': any(kw in response for kw in ['不确定', '置信度', '可能', '建议']),
                'governance_visible': any(kw in response for kw in ['审查', '复核', '谨慎']),
                'memory_aware': any(kw in response for kw in ['记得', '之前', '上文']),
                'citation': '[' in response and ']' in response
            }
            
            # 计算得分
            feature_count = sum(features.values())
            score = min(feature_count / 3, 1.0)  # 至少3个特征才算感知到
            scores.append(score)
            
            for feat, detected in features.items():
                if detected:
                    features_detected[feat] += 1
            
            print(f"\n查询: {case['query'][:50]}...")
            print(f"  检测到的特征: {[k for k, v in features.items() if v]}")
            print(f"  得分: {score:.2f}")
        
        avg_score = np.mean(scores)
        perception_rate = sum(1 for s in scores if s > 0.5) / len(scores)
        
        print(f"\n汇总:")
        print(f"  平均得分: {avg_score:.2f}")
        print(f"  感知率: {perception_rate:.1%}")
        print(f"  特征检测统计:")
        for feat, count in features_detected.items():
            print(f"    {feat}: {count}/{len(test_cases)} ({count/len(test_cases):.1%})")
        
        return {
            'average_score': avg_score,
            'perception_rate': perception_rate,
            'feature_detection': dict(features_detected),
            'target_met': perception_rate >= 0.70
        }
    
    def evaluate_persona_consistency(self, conversation_history: List[Dict]) -> Dict:
        """
        评估维度2: AI 人格一致性
        多轮对话中 AI 的行为风格保持一致
        """
        print("\n" + "="*70)
        print("维度2: AI 人格一致性评估")
        print("="*70)
        
        if len(conversation_history) < 5:
            print("对话历史不足，无法评估一致性")
            return {'consistency_score': 0.0, 'target_met': False}
        
        # 提取各轮特征
        strategies = [turn.get('strategy', 'UNKNOWN') for turn in conversation_history]
        confidences = [turn.get('confidence', 0.5) for turn in conversation_history]
        
        # 策略一致性
        strategy_consistency = 1.0 - (len(set(strategies)) / len(strategies))
        
        # 置信度稳定性
        confidence_std = np.std(confidences)
        confidence_stability = max(0, 1.0 - confidence_std)
        
        # 综合一致性
        consistency_score = 0.6 * strategy_consistency + 0.4 * confidence_stability
        
        print(f"对话轮数: {len(conversation_history)}")
        print(f"策略分布: {dict(zip(*np.unique(strategies, return_counts=True)))}")
        print(f"策略一致性: {strategy_consistency:.2f}")
        print(f"置信度稳定性: {confidence_stability:.2f}")
        print(f"综合一致性得分: {consistency_score:.2f}")
        
        return {
            'consistency_score': consistency_score,
            'strategy_consistency': strategy_consistency,
            'confidence_stability': confidence_stability,
            'target_met': consistency_score >= 0.90
        }
    
    def evaluate_governance_visibility(self, test_cases: List[Dict]) -> Dict:
        """
        评估维度3: 治理可见性
        用户能否感知到系统的治理逻辑
        """
        print("\n" + "="*70)
        print("维度3: 治理可见性评估")
        print("="*70)
        
        governance_markers = {
            'conservative_signals': ['不确定', '置信度', '可能', '也许', '建议'],
            'decline_signals': ['抱歉', '不了解', '无法', '不确定'],
            'review_signals': ['审查', '复核', '待审核', '标记'],
            'retrieval_signals': ['根据', '检索', '记忆', '来源']
        }
        
        visible_count = 0
        marker_counts = defaultdict(int)
        
        for case in test_cases:
            response = case['response']
            
            detected = False
            for marker_type, markers in governance_markers.items():
                if any(m in response for m in markers):
                    marker_counts[marker_type] += 1
                    detected = True
            
            if detected:
                visible_count += 1
            
            print(f"\n查询: {case['query'][:50]}...")
            print(f"  治理可见: {'是' if detected else '否'}")
        
        visibility_rate = visible_count / len(test_cases)
        
        print(f"\n汇总:")
        print(f"  治理可见率: {visibility_rate:.1%}")
        print(f"  各类标记检测:")
        for marker, count in marker_counts.items():
            print(f"    {marker}: {count}/{len(test_cases)}")
        
        return {
            'visibility_rate': visibility_rate,
            'marker_detection': dict(marker_counts),
            'target_met': visibility_rate >= 0.50
        }
    
    def evaluate_retrieval_value(self, test_cases: List[Dict]) -> Dict:
        """
        评估维度4: 检索价值感知
        用户是否认为检索增强了回答质量
        """
        print("\n" + "="*70)
        print("维度4: 检索价值感知评估")
        print("="*70)
        
        value_indicators = {
            'has_citation': 0,
            'has_source': 0,
            'improved_confidence': 0,
            'specific_details': 0
        }
        
        scores = []
        
        for case in test_cases:
            response = case['response']
            
            # 评估指标
            has_citation = '[' in response and ']' in response
            has_source = any(kw in response for kw in ['根据', '来源', '检索', '记忆'])
            has_details = len(response) > 100  # 详细回答
            
            # 计算得分 (模拟 1-5 分)
            score = 3.0  # 基础分
            if has_citation:
                score += 0.5
            if has_source:
                score += 0.5
            if has_details:
                score += 0.5
            if case.get('retrieval_triggered', False):
                score += 0.5
            
            scores.append(min(score, 5.0))
            
            if has_citation:
                value_indicators['has_citation'] += 1
            if has_source:
                value_indicators['has_source'] += 1
            if has_details:
                value_indicators['specific_details'] += 1
            
            print(f"\n查询: {case['query'][:50]}...")
            print(f"  有引用: {has_citation}, 有来源: {has_source}, 详细: {has_details}")
            print(f"  得分: {score:.1f}/5.0")
        
        avg_score = np.mean(scores)
        
        print(f"\n汇总:")
        print(f"  平均得分: {avg_score:.2f}/5.0")
        print(f"  指标统计:")
        for indicator, count in value_indicators.items():
            print(f"    {indicator}: {count}/{len(test_cases)}")
        
        return {
            'average_score': avg_score,
            'value_indicators': value_indicators,
            'target_met': avg_score >= 4.2
        }
    
    def evaluate_memory_effect(self, long_conversation: List[Dict]) -> Dict:
        """
        评估维度5: 记忆效果
        长对话中记忆是否改善体验
        """
        print("\n" + "="*70)
        print("维度5: 记忆效果评估")
        print("="*70)
        
        if len(long_conversation) < 10:
            print("对话轮数不足，无法评估记忆效果")
            return {'memory_score': 0.0, 'target_met': False}
        
        # 检测记忆引用
        memory_references = 0
        context_continuity = []
        
        for i, turn in enumerate(long_conversation[5:], start=5):  # 从第5轮开始检查
            response = turn.get('response', '')
            
            # 检测是否引用历史
            refers_history = any(kw in response for kw in ['之前', '上文', '刚才', '记得'])
            if refers_history:
                memory_references += 1
            
            # 检测话题连贯性
            if i > 0:
                prev_topic = long_conversation[i-1].get('topic', '')
                curr_topic = turn.get('topic', '')
                continuity = 1.0 if prev_topic == curr_topic else 0.5
                context_continuity.append(continuity)
        
        # 计算得分
        reference_rate = memory_references / max(len(long_conversation) - 5, 1)
        avg_continuity = np.mean(context_continuity) if context_continuity else 0.5
        
        memory_score = 0.4 * reference_rate + 0.6 * avg_continuity
        
        print(f"对话轮数: {len(long_conversation)}")
        print(f"记忆引用次数: {memory_references}")
        print(f"记忆引用率: {reference_rate:.1%}")
        print(f"话题连贯性: {avg_continuity:.2f}")
        print(f"记忆效果得分: {memory_score:.2f}")
        
        return {
            'memory_score': memory_score,
            'reference_rate': reference_rate,
            'continuity': avg_continuity,
            'target_met': memory_score >= 0.80  # 对应 4.0/5.0
        }
    
    def evaluate_differentiation(self, comparison_cases: List[Dict]) -> Dict:
        """
        评估维度6: 差异化
        与普通聊天模型相比的独特性
        """
        print("\n" + "="*70)
        print("维度6: 差异化评估")
        print("="*70)
        
        differentiation_markers = {
            'retrieval_based': 0,
            'governance_aware': 0,
            'memory_enabled': 0,
            'conservative_when_uncertain': 0,
            'cites_sources': 0
        }
        
        for case in comparison_cases:
            response = case['response']
            
            if any(kw in response for kw in ['根据', '检索', '记忆', '来源']):
                differentiation_markers['retrieval_based'] += 1
            if any(kw in response for kw in ['不确定', '置信度', '谨慎']):
                differentiation_markers['governance_aware'] += 1
            if any(kw in response for kw in ['记得', '之前', '上文']):
                differentiation_markers['memory_enabled'] += 1
            if '置信度' in response or '不确定' in response:
                differentiation_markers['conservative_when_uncertain'] += 1
            if '[' in response and ']' in response:
                differentiation_markers['cites_sources'] += 1
        
        # 计算差异化得分
        total_markers = sum(1 for v in differentiation_markers.values() if v > 0)
        differentiation_score = total_markers / len(differentiation_markers)
        
        # 映射到 1-5 分
        score_5 = 1.0 + 4.0 * differentiation_score
        
        print(f"差异化标记检测:")
        for marker, count in differentiation_markers.items():
            print(f"  {marker}: {count}/{len(comparison_cases)}")
        
        print(f"\n差异化得分: {differentiation_score:.2f}")
        print(f"5分制评分: {score_5:.2f}/5.0")
        
        return {
            'differentiation_score': differentiation_score,
            'score_5_scale': score_5,
            'markers': differentiation_markers,
            'target_met': score_5 >= 4.0
        }
    
    def generate_report(self) -> Dict:
        """生成完整评估报告"""
        print("\n" + "="*70)
        print("产品行为评估报告")
        print("="*70)
        
        dimensions = [
            ('系统特色感知', self.results.get('system_perception', {})),
            ('AI 人格一致性', self.results.get('persona_consistency', {})),
            ('治理可见性', self.results.get('governance_visibility', {})),
            ('检索价值感知', self.results.get('retrieval_value', {})),
            ('记忆效果', self.results.get('memory_effect', {})),
            ('差异化', self.results.get('differentiation', {}))
        ]
        
        print("\n评估结果汇总:")
        print("-" * 70)
        
        all_passed = True
        for name, result in dimensions:
            target_met = result.get('target_met', False)
            status = "✓ 通过" if target_met else "✗ 未通过"
            all_passed = all_passed and target_met
            
            print(f"{name:20s}: {status}")
        
        print("-" * 70)
        print(f"总体评估: {'通过' if all_passed else '未通过'}")
        
        return {
            'dimensions': {name: result for name, result in dimensions},
            'overall_passed': all_passed,
            'timestamp': '2026-04-18'
        }


def run_demo_evaluation():
    """运行演示评估"""
    evaluator = ProductBehaviorEvaluator()
    
    # 模拟测试数据
    test_cases = [
        {
            'query': 'Python是什么？',
            'response': 'Python是一种编程语言。根据我的记忆[来源:1]，它由Guido van Rossum创建。',
            'retrieval_triggered': True
        },
        {
            'query': '最新的法律规定是什么？',
            'response': '关于最新法律，我不太确定（置信度：60%）。建议查阅官方来源确认。',
            'retrieval_triggered': True
        },
        {
            'query': 'xyzabc是什么？',
            'response': '抱歉，我不了解xyzabc是什么。',
            'retrieval_triggered': False
        },
        {
            'query': '我们之前讨论过什么？',
            'response': '根据上文[来源:对话历史]，我们之前讨论了项目目标。',
            'retrieval_triggered': True
        }
    ]
    
    # 模拟长对话
    long_conversation = [
        {'turn': 1, 'strategy': 'DIRECT', 'confidence': 0.95, 'topic': 'project', 'response': '我们来讨论项目'},
        {'turn': 2, 'strategy': 'DIRECT', 'confidence': 0.90, 'topic': 'project', 'response': '项目目标是Q3完成'},
        {'turn': 3, 'strategy': 'RETRIEVAL_FIRST', 'confidence': 0.88, 'topic': 'project', 'response': '根据上文，项目预算100万'},
        {'turn': 4, 'strategy': 'DIRECT', 'confidence': 0.92, 'topic': 'project', 'response': '技术方案已确定'},
        {'turn': 5, 'strategy': 'DIRECT', 'confidence': 0.91, 'topic': 'project', 'response': '团队已组建'},
    ] + [{'turn': i, 'strategy': 'DIRECT', 'confidence': 0.90, 'topic': 'project', 'response': '继续讨论'} for i in range(6, 21)]
    
    # 执行评估
    evaluator.results['system_perception'] = evaluator.evaluate_system_perception(test_cases)
    evaluator.results['persona_consistency'] = evaluator.evaluate_persona_consistency(long_conversation)
    evaluator.results['governance_visibility'] = evaluator.evaluate_governance_visibility(test_cases)
    evaluator.results['retrieval_value'] = evaluator.evaluate_retrieval_value(test_cases)
    evaluator.results['memory_effect'] = evaluator.evaluate_memory_effect(long_conversation)
    evaluator.results['differentiation'] = evaluator.evaluate_differentiation(test_cases)
    
    # 生成报告
    report = evaluator.generate_report()
    
    # 保存报告
    output_path = 'product_behavior_eval_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告已保存: {output_path}")
    
    return report


if __name__ == "__main__":
    run_demo_evaluation()
