"""
Bilingual Alignment Evaluation - 双语对齐评估

WP2 核心组件：
评估中英文治理结果的一致性

功能：
1. 语义对齐评估
2. 治理动作对齐评估
3. 质量分数对齐评估
4. Unit 类型映射一致性
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class AlignmentResult:
    """对齐评估结果"""
    semantic_alignment: float  # 语义对齐分数
    governance_alignment: float  # 治理动作对齐分数
    quality_score_diff: float  # 质量分数差异
    unit_type_consistency: float  # Unit 类型一致性
    overall_score: float  # 综合对齐分数
    details: Dict[str, Any]  # 详细结果


@dataclass
class BilingualTestCase:
    """双语测试用例"""
    zh_text: str
    en_text: str
    expected_unit_type: str
    expected_governance_action: str
    description: str


class BilingualAlignmentEvaluator:
    """
    双语对齐评估器
    
    评估维度：
    1. 语义对齐：中英文是否表达相同语义
    2. 治理动作对齐：是否触发相同的治理动作
    3. 质量分数对齐：质量分数差异是否在阈值内
    4. Unit 类型一致性：识别的 Unit 类型是否一致
    """
    
    def __init__(self, quality_score_threshold: float = 0.1):
        self.quality_score_threshold = quality_score_threshold
        self._init_test_cases()
    
    def _init_test_cases(self):
        """初始化双语测试用例"""
        self.test_cases = [
            # 概念对齐
            BilingualTestCase(
                zh_text="机器学习是人工智能的一个分支。",
                en_text="Machine learning is a branch of artificial intelligence.",
                expected_unit_type="concept",
                expected_governance_action="promote_to_longterm",
                description="概念定义对齐"
            ),
            # 关系对齐
            BilingualTestCase(
                zh_text="猫坐在垫子上。",
                en_text="The cat sat on the mat.",
                expected_unit_type="relation",
                expected_governance_action="promote_to_longterm",
                description="空间关系对齐"
            ),
            # 任务对齐
            BilingualTestCase(
                zh_text="如何训练神经网络？",
                en_text="How to train a neural network?",
                expected_unit_type="task_pattern",
                expected_governance_action="activate_task_pattern",
                description="任务模式对齐"
            ),
            # 规则对齐
            BilingualTestCase(
                zh_text="如果下雨，地面就会湿。",
                en_text="If it rains, the ground will be wet.",
                expected_unit_type="rule",
                expected_governance_action="promote_to_permanent",
                description="条件规则对齐"
            ),
            # 偏好表达
            BilingualTestCase(
                zh_text="他喜欢读书。",
                en_text="He likes reading.",
                expected_unit_type="concept",
                expected_governance_action="promote_to_longterm",
                description="偏好表达对齐"
            ),
            # 因果关系
            BilingualTestCase(
                zh_text="压力导致健康问题。",
                en_text="Stress causes health problems.",
                expected_unit_type="relation",
                expected_governance_action="promote_to_longterm",
                description="因果关系对齐"
            ),
            # 比较关系
            BilingualTestCase(
                zh_text="约翰比玛丽高。",
                en_text="John is taller than Mary.",
                expected_unit_type="relation",
                expected_governance_action="promote_to_longterm",
                description="比较关系对齐"
            ),
            # 拥有关系
            BilingualTestCase(
                zh_text="这家公司拥有很多专利。",
                en_text="This company owns many patents.",
                expected_unit_type="relation",
                expected_governance_action="promote_to_longterm",
                description="拥有关系对齐"
            ),
        ]
    
    def _mock_zh_governance(self, text: str) -> Dict[str, Any]:
        """模拟中文治理结果"""
        # 简化的模拟实现
        if "如何" in text or "怎么" in text:
            return {
                "unit_type": "task_pattern",
                "governance_action": "activate_task_pattern",
                "quality_score": 0.85,
                "units_detected": 1
            }
        elif "如果" in text:
            return {
                "unit_type": "rule",
                "governance_action": "promote_to_permanent",
                "quality_score": 0.88,
                "units_detected": 1
            }
        elif "是" in text and ("分支" in text or "一种" in text):
            return {
                "unit_type": "concept",
                "governance_action": "promote_to_longterm",
                "quality_score": 0.90,
                "units_detected": 1
            }
        else:
            return {
                "unit_type": "relation",
                "governance_action": "promote_to_longterm",
                "quality_score": 0.82,
                "units_detected": 1
            }
    
    def _mock_en_governance(self, text: str) -> Dict[str, Any]:
        """模拟英文治理结果"""
        # 简化的模拟实现
        text_lower = text.lower()
        
        if "how to" in text_lower or "how do" in text_lower:
            return {
                "unit_type": "task_pattern",
                "governance_action": "activate_task_pattern",
                "quality_score": 0.83,
                "units_detected": 1
            }
        elif "if" in text_lower and ("will" in text_lower or "then" in text_lower):
            return {
                "unit_type": "rule",
                "governance_action": "promote_to_permanent",
                "quality_score": 0.85,
                "units_detected": 1
            }
        elif "is a" in text_lower or "is an" in text_lower or "is the" in text_lower:
            return {
                "unit_type": "concept",
                "governance_action": "promote_to_longterm",
                "quality_score": 0.88,
                "units_detected": 1
            }
        else:
            return {
                "unit_type": "relation",
                "governance_action": "promote_to_longterm",
                "quality_score": 0.80,
                "units_detected": 1
            }
    
    def evaluate_alignment(self, test_case: BilingualTestCase) -> Dict[str, Any]:
        """
        评估单个测试用例的对齐情况
        
        Returns:
            Dict with alignment metrics
        """
        # 获取治理结果
        zh_result = self._mock_zh_governance(test_case.zh_text)
        en_result = self._mock_en_governance(test_case.en_text)
        
        # 1. 语义对齐 (基于 Unit 类型是否相同)
        semantic_match = zh_result["unit_type"] == en_result["unit_type"]
        semantic_alignment = 1.0 if semantic_match else 0.0
        
        # 2. 治理动作对齐
        governance_match = zh_result["governance_action"] == en_result["governance_action"]
        governance_alignment = 1.0 if governance_match else 0.0
        
        # 3. 质量分数差异
        quality_diff = abs(zh_result["quality_score"] - en_result["quality_score"])
        quality_aligned = quality_diff <= self.quality_score_threshold
        
        # 4. Unit 类型一致性 (与期望对比)
        zh_type_correct = zh_result["unit_type"] == test_case.expected_unit_type
        en_type_correct = en_result["unit_type"] == test_case.expected_unit_type
        unit_type_consistency = (1.0 if zh_type_correct else 0.0 + 
                                1.0 if en_type_correct else 0.0) / 2
        
        # 综合分数
        overall_score = (
            semantic_alignment * 0.3 +
            governance_alignment * 0.3 +
            (1.0 - min(quality_diff / self.quality_score_threshold, 1.0)) * 0.2 +
            unit_type_consistency * 0.2
        )
        
        return {
            "test_case": test_case.description,
            "zh_result": zh_result,
            "en_result": en_result,
            "semantic_alignment": semantic_alignment,
            "governance_alignment": governance_alignment,
            "quality_score_diff": quality_diff,
            "quality_aligned": quality_aligned,
            "unit_type_consistency": unit_type_consistency,
            "overall_score": overall_score,
            "passed": overall_score >= 0.7
        }
    
    def run_full_evaluation(self) -> AlignmentResult:
        """
        运行完整评估
        
        Returns:
            AlignmentResult: 综合评估结果
        """
        results = []
        
        for test_case in self.test_cases:
            result = self.evaluate_alignment(test_case)
            results.append(result)
        
        # 计算平均指标
        avg_semantic = sum(r["semantic_alignment"] for r in results) / len(results)
        avg_governance = sum(r["governance_alignment"] for r in results) / len(results)
        avg_quality_diff = sum(r["quality_score_diff"] for r in results) / len(results)
        avg_unit_type = sum(r["unit_type_consistency"] for r in results) / len(results)
        avg_overall = sum(r["overall_score"] for r in results) / len(results)
        
        pass_count = sum(1 for r in results if r["passed"])
        pass_rate = pass_count / len(results)
        
        return AlignmentResult(
            semantic_alignment=avg_semantic,
            governance_alignment=avg_governance,
            quality_score_diff=avg_quality_diff,
            unit_type_consistency=avg_unit_type,
            overall_score=avg_overall,
            details={
                "total_cases": len(results),
                "passed_cases": pass_count,
                "pass_rate": pass_rate,
                "individual_results": results
            }
        )
    
    def generate_report(self, result: AlignmentResult) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 70)
        report.append("Bilingual Alignment Evaluation Report")
        report.append("=" * 70)
        report.append("")
        
        # 总体指标
        report.append("Overall Metrics:")
        report.append("-" * 50)
        report.append(f"  Semantic Alignment:     {result.semantic_alignment:.2f} (target: > 0.80)")
        report.append(f"  Governance Alignment:   {result.governance_alignment:.2f} (target: > 0.80)")
        report.append(f"  Quality Score Diff:     {result.quality_score_diff:.3f} (target: < 0.10)")
        report.append(f"  Unit Type Consistency:  {result.unit_type_consistency:.2f} (target: > 0.90)")
        report.append(f"  Overall Score:          {result.overall_score:.2f} (target: > 0.70)")
        report.append("")
        
        # 通过率
        report.append(f"Pass Rate: {result.details['pass_rate']:.1%} "
                     f"({result.details['passed_cases']}/{result.details['total_cases']})")
        report.append("")
        
        # 详细结果
        report.append("Detailed Results:")
        report.append("-" * 50)
        
        for i, r in enumerate(result.details["individual_results"], 1):
            status = "✓" if r["passed"] else "✗"
            report.append(f"\n  {status} Test {i}: {r['test_case']}")
            report.append(f"    中文: {r['zh_result']['unit_type']} "
                         f"(质量: {r['zh_result']['quality_score']:.2f})")
            report.append(f"    英文: {r['en_result']['unit_type']} "
                         f"(质量: {r['en_result']['quality_score']:.2f})")
            report.append(f"    对齐分数: {r['overall_score']:.2f}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def demo_bilingual_alignment():
    """演示双语对齐评估"""
    print("\n" + "="*70)
    print("Bilingual Alignment Evaluation - 演示")
    print("="*70)
    
    evaluator = BilingualAlignmentEvaluator(quality_score_threshold=0.1)
    
    print("\n1. 运行对齐评估")
    print("-" * 50)
    
    result = evaluator.run_full_evaluation()
    
    print(f"\n  评估完成!")
    print(f"  测试用例数: {result.details['total_cases']}")
    print(f"  通过数: {result.details['passed_cases']}")
    print(f"  通过率: {result.details['pass_rate']:.1%}")
    
    print("\n2. 评估指标")
    print("-" * 50)
    print(f"  语义对齐:     {result.semantic_alignment:.2f}")
    print(f"  治理动作对齐: {result.governance_alignment:.2f}")
    print(f"  质量分数差异: {result.quality_score_diff:.3f}")
    print(f"  Unit类型一致: {result.unit_type_consistency:.2f}")
    print(f"  综合分数:     {result.overall_score:.2f}")
    
    print("\n3. 目标对比")
    print("-" * 50)
    print(f"  语义对齐目标:     > 0.80  {'✓' if result.semantic_alignment >= 0.80 else '✗'}")
    print(f"  治理对齐目标:     > 0.80  {'✓' if result.governance_alignment >= 0.80 else '✗'}")
    print(f"  质量差异目标:     < 0.10  {'✓' if result.quality_score_diff < 0.10 else '✗'}")
    print(f"  Unit一致目标:     > 0.90  {'✓' if result.unit_type_consistency >= 0.90 else '✗'}")
    print(f"  综合分数目标:     > 0.70  {'✓' if result.overall_score >= 0.70 else '✗'}")
    
    print("\n4. 详细报告")
    print("-" * 50)
    report = evaluator.generate_report(result)
    print(report)
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_bilingual_alignment()
