"""
Training Evaluator v1 - 训练评估器 v1

Phase 16: 训练前后对比评估
目标：验证训练后模型是否更像"你们的AI"
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 策略准确率
    strategy_accuracy: float
    retrieval_trigger_accuracy: float
    governance_action_accuracy: float
    memory_writeback_accuracy: float
    
    # 高风险指标
    high_risk_false_answer_rate: float
    conservative_decline_rate: float
    
    # 记忆指标
    memory_citation_accuracy: float
    memory_writeback_correctness: float
    
    # 双语指标
    english_relation_f1: float
    bilingual_consistency: float
    
    # 产品行为六维评分
    product_behavior_score: Dict[str, float]
    
    # 综合评分
    overall_score: float


@dataclass
class ComparisonResult:
    """对比结果"""
    before_metrics: EvaluationMetrics
    after_metrics: EvaluationMetrics
    improvements: Dict[str, float]
    regression_points: List[str]
    key_wins: List[str]


class TrainingEvaluator:
    """训练评估器"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_before_training(
        self,
        test_data_path: str,
        baseline_results: Optional[List[Dict]] = None
    ) -> EvaluationMetrics:
        """
        训练前评估（基线）
        
        评估维度：
        1. 五类响应策略准确率
        2. 检索触发正确率
        3. 高风险误答率
        4. 治理动作匹配率
        5. 记忆写回正确率
        6. 英文关系检测F1
        7. 双语一致性
        8. 产品行为六维评分
        """
        
        print("\n" + "="*70)
        print("训练前评估 (基线)")
        print("="*70)
        
        # 加载测试数据
        test_data = self._load_jsonl(test_data_path)
        
        # 模拟基线评估（实际应使用真实模型输出）
        # 基线：未训练模型的表现
        baseline = EvaluationMetrics(
            strategy_accuracy=0.45,  # 随机猜测水平
            retrieval_trigger_accuracy=0.50,
            governance_action_accuracy=0.40,
            memory_writeback_accuracy=0.55,
            high_risk_false_answer_rate=0.30,  # 高风险误答率较高
            conservative_decline_rate=0.25,
            memory_citation_accuracy=0.35,
            memory_writeback_correctness=0.40,
            english_relation_f1=0.45,
            bilingual_consistency=0.50,
            product_behavior_score={
                "helpfulness": 0.60,
                "accuracy": 0.50,
                "safety": 0.55,
                "memory_awareness": 0.40,
                "retrieval_habit": 0.35,
                "governance_compliance": 0.45,
            },
            overall_score=0.48
        )
        
        self._print_metrics(baseline, "训练前")
        return baseline
    
    def evaluate_after_training(
        self,
        test_data_path: str,
        model_paths: Dict[str, str],
        trained_results: Optional[List[Dict]] = None
    ) -> EvaluationMetrics:
        """
        训练后评估
        
        验证训练后模型是否更像"你们的AI"：
        - RETRIEVAL_FIRST 更稳定
        - CONSERVATIVE/DECLINE/REVIEW 边界更合理
        - 写回更稳
        - 用户明显感到"这是你们自己的AI"
        """
        
        print("\n" + "="*70)
        print("训练后评估")
        print("="*70)
        
        # 加载测试数据
        test_data = self._load_jsonl(test_data_path)
        
        # 模拟训练后评估（实际应使用训练后的模型）
        # 训练后：模型行为更稳定
        trained = EvaluationMetrics(
            strategy_accuracy=0.85,  # 显著提升
            retrieval_trigger_accuracy=0.88,
            governance_action_accuracy=0.82,
            memory_writeback_accuracy=0.80,
            high_risk_false_answer_rate=0.08,  # 显著降低
            conservative_decline_rate=0.75,  # 更保守但更准确
            memory_citation_accuracy=0.87,
            memory_writeback_correctness=0.85,
            english_relation_f1=0.83,
            bilingual_consistency=0.86,
            product_behavior_score={
                "helpfulness": 0.88,
                "accuracy": 0.85,
                "safety": 0.90,
                "memory_awareness": 0.87,
                "retrieval_habit": 0.88,
                "governance_compliance": 0.86,
            },
            overall_score=0.86
        )
        
        self._print_metrics(trained, "训练后")
        return trained
    
    def compare(
        self,
        before: EvaluationMetrics,
        after: EvaluationMetrics
    ) -> ComparisonResult:
        """对比训练前后"""
        
        print("\n" + "="*70)
        print("训练前后对比")
        print("="*70)
        
        # 计算改善
        improvements = {
            "strategy_accuracy": after.strategy_accuracy - before.strategy_accuracy,
            "retrieval_trigger_accuracy": after.retrieval_trigger_accuracy - before.retrieval_trigger_accuracy,
            "high_risk_false_answer_rate": before.high_risk_false_answer_rate - after.high_risk_false_answer_rate,  # 降低是改善
            "memory_citation_accuracy": after.memory_citation_accuracy - before.memory_citation_accuracy,
            "overall_score": after.overall_score - before.overall_score,
        }
        
        # 识别退化点
        regression_points = []
        if after.conservative_decline_rate > before.conservative_decline_rate + 0.3:
            regression_points.append("过度保守：拒答率上升过多")
        
        # 关键胜利
        key_wins = []
        if after.strategy_accuracy > 0.80:
            key_wins.append("策略选择准确率超过80%")
        if after.high_risk_false_answer_rate < 0.10:
            key_wins.append("高风险误答率降至10%以下")
        if after.retrieval_trigger_accuracy > 0.85:
            key_wins.append("检索触发准确率超过85%")
        if after.memory_citation_accuracy > 0.85:
            key_wins.append("记忆引用准确率超过85%")
        
        # 打印对比
        print("\n关键指标改善:")
        for metric, improvement in improvements.items():
            direction = "↑" if improvement > 0 else "↓"
            print(f"  {metric}: {improvement:+.3f} {direction}")
        
        print(f"\n退化点 ({len(regression_points)}):")
        for point in regression_points:
            print(f"  ⚠ {point}")
        
        print(f"\n关键胜利 ({len(key_wins)}):")
        for win in key_wins:
            print(f"  ✓ {win}")
        
        return ComparisonResult(
            before_metrics=before,
            after_metrics=after,
            improvements=improvements,
            regression_points=regression_points,
            key_wins=key_wins
        )
    
    def evaluate_conversation_quality(
        self,
        conversation_logs: List[Dict],
        evaluation_type: str = "private_knowledge"
    ) -> Dict[str, Any]:
        """
        对话质量评估
        
        评估类型：
        - private_knowledge: 私有知识问答
        - multi_turn_history: 多轮历史回顾
        - high_risk: 高风险问题
        """
        
        print(f"\n{'='*70}")
        print(f"对话质量评估: {evaluation_type}")
        print("="*70)
        
        if evaluation_type == "private_knowledge":
            return self._evaluate_private_knowledge(conversation_logs)
        elif evaluation_type == "multi_turn_history":
            return self._evaluate_multi_turn_history(conversation_logs)
        elif evaluation_type == "high_risk":
            return self._evaluate_high_risk(conversation_logs)
        else:
            return {}
    
    def _evaluate_private_knowledge(self, logs: List[Dict]) -> Dict[str, Any]:
        """评估私有知识问答"""
        
        metrics = {
            "retrieval_trigger_rate": 0.0,
            "correct_answer_rate": 0.0,
            "memory_citation_rate": 0.0,
            "confidence_score": 0.0,
        }
        
        total = len(logs)
        if total == 0:
            return metrics
        
        retrieval_count = sum(1 for log in logs if log.get("retrieval_triggered", False))
        correct_count = sum(1 for log in logs if log.get("answer_correct", False))
        citation_count = sum(1 for log in logs if log.get("memory_cited", False))
        
        metrics["retrieval_trigger_rate"] = retrieval_count / total
        metrics["correct_answer_rate"] = correct_count / total
        metrics["memory_citation_rate"] = citation_count / total
        metrics["confidence_score"] = (metrics["retrieval_trigger_rate"] + 
                                       metrics["correct_answer_rate"]) / 2
        
        print(f"  检索触发率: {metrics['retrieval_trigger_rate']:.1%}")
        print(f"  正确回答率: {metrics['correct_answer_rate']:.1%}")
        print(f"  记忆引用率: {metrics['memory_citation_rate']:.1%}")
        print(f"  综合置信度: {metrics['confidence_score']:.2f}")
        
        return metrics
    
    def _evaluate_multi_turn_history(self, logs: List[Dict]) -> Dict[str, Any]:
        """评估多轮历史回顾"""
        
        metrics = {
            "history_consistency": 0.0,
            "memory_accuracy": 0.0,
            "persona_stability": 0.0,
            "conversation_coherence": 0.0,
        }
        
        total = len(logs)
        if total == 0:
            return metrics
        
        # 模拟评估
        consistent_count = sum(1 for log in logs if log.get("consistent_with_history", False))
        accurate_count = sum(1 for log in logs if log.get("memory_accurate", False))
        
        metrics["history_consistency"] = consistent_count / total
        metrics["memory_accuracy"] = accurate_count / total
        metrics["persona_stability"] = 0.85  # 模拟
        metrics["conversation_coherence"] = 0.88  # 模拟
        
        print(f"  历史一致性: {metrics['history_consistency']:.1%}")
        print(f"  记忆准确度: {metrics['memory_accuracy']:.1%}")
        print(f"  人格稳定性: {metrics['persona_stability']:.1%}")
        print(f"  对话连贯性: {metrics['conversation_coherence']:.1%}")
        
        return metrics
    
    def _evaluate_high_risk(self, logs: List[Dict]) -> Dict[str, Any]:
        """评估高风险问题处理"""
        
        metrics = {
            "appropriate_decline_rate": 0.0,
            "false_answer_rate": 0.0,
            "review_trigger_rate": 0.0,
            "safety_score": 0.0,
        }
        
        total = len(logs)
        if total == 0:
            return metrics
        
        appropriate_decline = sum(1 for log in logs if log.get("appropriate_decline", False))
        false_answers = sum(1 for log in logs if log.get("false_answer", False))
        review_triggered = sum(1 for log in logs if log.get("review_triggered", False))
        
        metrics["appropriate_decline_rate"] = appropriate_decline / total
        metrics["false_answer_rate"] = false_answers / total
        metrics["review_trigger_rate"] = review_triggered / total
        metrics["safety_score"] = 1.0 - metrics["false_answer_rate"]
        
        print(f"  恰当拒答率: {metrics['appropriate_decline_rate']:.1%}")
        print(f"  错误回答率: {metrics['false_answer_rate']:.1%}")
        print(f"  审查触发率: {metrics['review_trigger_rate']:.1%}")
        print(f"  安全评分: {metrics['safety_score']:.2f}")
        
        return metrics
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载JSONL数据"""
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"警告: 文件不存在 {path}")
        return data
    
    def _print_metrics(self, metrics: EvaluationMetrics, label: str):
        """打印指标"""
        
        print(f"\n{label}指标:")
        print("-"*50)
        print(f"  策略准确率: {metrics.strategy_accuracy:.1%}")
        print(f"  检索触发准确率: {metrics.retrieval_trigger_accuracy:.1%}")
        print(f"  治理动作准确率: {metrics.governance_action_accuracy:.1%}")
        print(f"  记忆写回准确率: {metrics.memory_writeback_accuracy:.1%}")
        print(f"  高风险误答率: {metrics.high_risk_false_answer_rate:.1%}")
        print(f"  记忆引用准确率: {metrics.memory_citation_accuracy:.1%}")
        print(f"  英文关系F1: {metrics.english_relation_f1:.2f}")
        print(f"  双语一致性: {metrics.bilingual_consistency:.1%}")
        print(f"  综合评分: {metrics.overall_score:.2f}")
        
        print(f"\n  产品行为六维评分:")
        for dim, score in metrics.product_behavior_score.items():
            print(f"    {dim}: {score:.2f}")


# 便捷函数
def create_training_evaluator() -> TrainingEvaluator:
    """创建训练评估器"""
    return TrainingEvaluator()


# 测试
if __name__ == "__main__":
    print("="*70)
    print("Training Evaluator v1 - 测试模式")
    print("="*70)
    
    evaluator = create_training_evaluator()
    
    # 训练前后对比
    before = evaluator.evaluate_before_training("phase16/test_data.jsonl")
    after = evaluator.evaluate_after_training("phase16/test_data.jsonl", {})
    comparison = evaluator.compare(before, after)
    
    # 对话质量评估
    print("\n" + "="*70)
    print("对话质量评估")
    print("="*70)
    
    # 模拟对话日志
    private_knowledge_logs = [
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
        {"retrieval_triggered": False, "answer_correct": False, "memory_cited": False},
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
    ]
    
    high_risk_logs = [
        {"appropriate_decline": True, "false_answer": False, "review_triggered": True},
        {"appropriate_decline": True, "false_answer": False, "review_triggered": False},
        {"appropriate_decline": False, "false_answer": True, "review_triggered": False},
    ]
    
    evaluator.evaluate_conversation_quality(private_knowledge_logs, "private_knowledge")
    evaluator.evaluate_conversation_quality(high_risk_logs, "high_risk")
    
    print("\n" + "="*70)
    print("评估完成")
    print("="*70)
