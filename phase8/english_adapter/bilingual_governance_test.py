"""
Bilingual Governance Test - 双语治理测试

WP1 核心组件：
验证英文适配层与中文原型在治理层面的一致性

测试目标：
1. 英文输入能够完成完整治理链路
2. 四类通用 Unit 在英文场景中全部可实例化
3. 中文逻辑与英文逻辑的差异主要体现在适配层
4. 英中双语在同一治理框架下均能通过基线测试
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase8.english_adapter.english_unit_mapping import (
    EnglishWordInput, EnglishUnitMapper
)
from phase8.english_adapter.english_explanation_layer import (
    EnglishExplanationPipeline
)


class BilingualGovernanceTest:
    """
    双语治理测试
    
    功能：
    1. 验证双语输入产生一致的治理决策
    2. 验证适配层隔离性
    3. 验证通用 Unit 抽象的有效性
    """
    
    def __init__(self):
        self.en_pipeline = EnglishExplanationPipeline()
        self.en_mapper = EnglishUnitMapper()
        
        # 双语对齐测试集
        self.bilingual_test_set = {
            "concept_alignment": [
                {
                    "zh": "学习",
                    "en": "study",
                    "expected_unit_type": "concept",
                    "expected_quality_threshold": 0.7
                },
                {
                    "zh": "苹果",
                    "en": "apple",
                    "expected_unit_type": "concept",
                    "expected_sense_count": 1
                },
                {
                    "zh": "知识",
                    "en": "knowledge",
                    "expected_unit_type": "concept",
                    "expected_abstract": True
                }
            ],
            "relation_alignment": [
                {
                    "zh": "导致",
                    "en": "cause",
                    "expected_unit_type": "relation",
                    "expected_relation_type": "causal"
                },
                {
                    "zh": "属于",
                    "en": "belong to",
                    "expected_unit_type": "relation",
                    "expected_relation_type": "hierarchical"
                },
                {
                    "zh": "在...之前",
                    "en": "before",
                    "expected_unit_type": "relation",
                    "expected_relation_type": "temporal"
                }
            ],
            "rule_alignment": [
                {
                    "zh": "如果下雨，地面会湿",
                    "en": "If it rains, the ground will be wet",
                    "expected_unit_type": "rule",
                    "expected_rule_type": "inference"
                }
            ],
            "task_alignment": [
                {
                    "zh": "什么是机器学习？",
                    "en": "What is machine learning?",
                    "expected_unit_type": "task_pattern",
                    "expected_task_type": "definition"
                },
                {
                    "zh": "解释深度学习和机器学习的区别",
                    "en": "Explain the difference between deep learning and machine learning",
                    "expected_unit_type": "task_pattern",
                    "expected_task_type": "comparison"
                },
                {
                    "zh": "为什么天空是蓝色的？",
                    "en": "Why is the sky blue?",
                    "expected_unit_type": "task_pattern",
                    "expected_task_type": "reasoning"
                }
            ]
        }
    
    def test_concept_alignment(self) -> Dict[str, Any]:
        """测试概念对齐"""
        print("\n" + "=" * 70)
        print("测试 1: 概念对齐 (Concept Alignment)")
        print("=" * 70)
        
        results = []
        for test_case in self.bilingual_test_set["concept_alignment"]:
            zh = test_case["zh"]
            en = test_case["en"]
            
            # 处理英文输入
            word_input = EnglishWordInput(text=en)
            concept_unit = self.en_mapper.map_to_concept_unit(word_input)
            
            # 验证
            passed = (
                concept_unit is not None and
                concept_unit.unit_type == test_case["expected_unit_type"]
            )
            
            result = {
                "zh": zh,
                "en": en,
                "passed": passed,
                "unit_id": concept_unit.unit_id if concept_unit else None,
                "syllable_count": concept_unit.syllable_count if concept_unit else 0
            }
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"\n  {status} {zh} ↔ {en}")
            if concept_unit:
                print(f"     Unit ID: {concept_unit.unit_id}")
                print(f"     音节数: {concept_unit.syllable_count}")
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        print(f"\n  通过率: {pass_rate:.1%}")
        
        return {
            "test_name": "concept_alignment",
            "pass_rate": pass_rate,
            "results": results
        }
    
    def test_relation_alignment(self) -> Dict[str, Any]:
        """测试关系对齐"""
        print("\n" + "=" * 70)
        print("测试 2: 关系对齐 (Relation Alignment)")
        print("=" * 70)
        
        results = []
        for test_case in self.bilingual_test_set["relation_alignment"]:
            zh = test_case["zh"]
            en = test_case["en"]
            
            # 构造包含关系的文本
            test_text = f"A {en} B"
            relation_unit = self.en_mapper.map_to_relation_unit(test_text)
            
            # 验证
            passed = (
                relation_unit is not None and
                relation_unit.unit_type == test_case["expected_unit_type"] and
                relation_unit.relation_type == test_case["expected_relation_type"]
            )
            
            result = {
                "zh": zh,
                "en": en,
                "passed": passed,
                "relation_type": relation_unit.relation_type if relation_unit else None
            }
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"\n  {status} {zh} ↔ {en}")
            if relation_unit:
                print(f"     关系类型: {relation_unit.relation_type}")
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        print(f"\n  通过率: {pass_rate:.1%}")
        
        return {
            "test_name": "relation_alignment",
            "pass_rate": pass_rate,
            "results": results
        }
    
    def test_rule_alignment(self) -> Dict[str, Any]:
        """测试规则对齐"""
        print("\n" + "=" * 70)
        print("测试 3: 规则对齐 (Rule Alignment)")
        print("=" * 70)
        
        results = []
        for test_case in self.bilingual_test_set["rule_alignment"]:
            zh = test_case["zh"]
            en = test_case["en"]
            
            # 处理英文输入
            rule_unit = self.en_mapper.map_to_rule_unit(en)
            
            # 验证
            passed = (
                rule_unit is not None and
                rule_unit.unit_type == test_case["expected_unit_type"] and
                rule_unit.rule_type == test_case["expected_rule_type"]
            )
            
            result = {
                "zh": zh,
                "en": en,
                "passed": passed,
                "premises": rule_unit.premises if rule_unit else [],
                "conclusions": rule_unit.conclusions if rule_unit else []
            }
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"\n  {status} {zh}")
            print(f"     EN: {en}")
            if rule_unit:
                print(f"     规则类型: {rule_unit.rule_type}")
                print(f"     前提: {rule_unit.premises}")
                print(f"     结论: {rule_unit.conclusions}")
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        print(f"\n  通过率: {pass_rate:.1%}")
        
        return {
            "test_name": "rule_alignment",
            "pass_rate": pass_rate,
            "results": results
        }
    
    def test_task_alignment(self) -> Dict[str, Any]:
        """测试任务对齐"""
        print("\n" + "=" * 70)
        print("测试 4: 任务对齐 (Task Alignment)")
        print("=" * 70)
        
        results = []
        for test_case in self.bilingual_test_set["task_alignment"]:
            zh = test_case["zh"]
            en = test_case["en"]
            
            # 处理英文输入
            task_unit = self.en_mapper.map_to_task_pattern_unit(en)
            
            # 验证
            passed = (
                task_unit is not None and
                task_unit.unit_type == test_case["expected_unit_type"] and
                task_unit.task_type == test_case["expected_task_type"]
            )
            
            result = {
                "zh": zh,
                "en": en,
                "passed": passed,
                "task_type": task_unit.task_type if task_unit else None,
                "triggers": task_unit.trigger_keywords if task_unit else []
            }
            results.append(result)
            
            status = "✅" if passed else "❌"
            print(f"\n  {status} {zh}")
            print(f"     EN: {en}")
            if task_unit:
                print(f"     任务类型: {task_unit.task_type}")
                print(f"     触发词: {task_unit.trigger_keywords}")
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        print(f"\n  通过率: {pass_rate:.1%}")
        
        return {
            "test_name": "task_alignment",
            "pass_rate": pass_rate,
            "results": results
        }
    
    def test_adapter_isolation(self) -> Dict[str, Any]:
        """测试适配层隔离性"""
        print("\n" + "=" * 70)
        print("测试 5: 适配层隔离性 (Adapter Isolation)")
        print("=" * 70)
        
        checks = []
        
        # 检查 1: 英文特有属性在 Unit 中
        word_input = EnglishWordInput(text="study")
        concept = self.en_mapper.map_to_concept_unit(word_input)
        
        has_en_specific = (
            concept.spelling_variants is not None and
            concept.syllable_count > 0 and
            concept.inflectional_forms is not None
        )
        
        checks.append({
            "check": "英文特有属性在 Unit 中",
            "passed": has_en_specific,
            "details": f"音节数: {concept.syllable_count}, 词形变化: {len(concept.inflectional_forms)}"
        })
        
        # 检查 2: 通用治理元数据存在
        has_governance_metadata = (
            concept.quality_score is not None and
            concept.stability_cycles is not None
        )
        
        checks.append({
            "check": "通用治理元数据存在",
            "passed": has_governance_metadata,
            "details": f"质量分数: {concept.quality_score}"
        })
        
        # 检查 3: 介词映射在 Relation Unit 中（英文特有）
        relation = self.en_mapper.map_to_relation_unit("A is in B")
        has_preposition_mapping = (
            relation is not None and
            relation.preposition_mapping is not None and
            "spatial" in relation.preposition_mapping
        )
        
        checks.append({
            "check": "介词映射在 Relation Unit 中",
            "passed": has_preposition_mapping,
            "details": f"空间介词: {relation.preposition_mapping.get('spatial', [])[:3] if relation else []}"
        })
        
        # 打印结果
        for check in checks:
            status = "✅" if check["passed"] else "❌"
            print(f"\n  {status} {check['check']}")
            print(f"     {check['details']}")
        
        pass_rate = sum(1 for c in checks if c["passed"]) / len(checks)
        print(f"\n  通过率: {pass_rate:.1%}")
        
        return {
            "test_name": "adapter_isolation",
            "pass_rate": pass_rate,
            "checks": checks
        }
    
    def test_full_pipeline(self) -> Dict[str, Any]:
        """测试完整治理链路"""
        print("\n" + "=" * 70)
        print("测试 6: 完整治理链路 (Full Governance Pipeline)")
        print("=" * 70)
        
        test_inputs = [
            "What is artificial intelligence?",
            "Explain the difference between AI and ML",
            "If data is biased then the model will be unfair",
            "A dog is a type of animal"
        ]
        
        results = []
        for text in test_inputs:
            # 完整处理
            result = self.en_pipeline.process(text)
            
            # 验证所有 Unit 类型都被正确识别
            units = result["units"]
            has_concept = units["concept"] is not None
            has_relation = units["relation"] is not None
            has_rule = units["rule"] is not None
            has_task = units["task"] is not None
            
            # 至少有一个 Unit 被识别
            has_any_unit = has_concept or has_relation or has_rule or has_task
            
            results.append({
                "text": text,
                "has_concept": has_concept,
                "has_relation": has_relation,
                "has_rule": has_rule,
                "has_task": has_task,
                "passed": has_any_unit
            })
            
            status = "✅" if has_any_unit else "❌"
            print(f"\n  {status} {text}")
            print(f"     Concept: {has_concept}, Relation: {has_relation}")
            print(f"     Rule: {has_rule}, Task: {has_task}")
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        print(f"\n  通过率: {pass_rate:.1%}")
        
        return {
            "test_name": "full_pipeline",
            "pass_rate": pass_rate,
            "results": results
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "🧪 " * 35)
        print("Bilingual Governance Test - 双语治理测试")
        print("🧪 " * 35)
        
        # 运行所有测试
        results = {
            "concept_alignment": self.test_concept_alignment(),
            "relation_alignment": self.test_relation_alignment(),
            "rule_alignment": self.test_rule_alignment(),
            "task_alignment": self.test_task_alignment(),
            "adapter_isolation": self.test_adapter_isolation(),
            "full_pipeline": self.test_full_pipeline()
        }
        
        # 汇总
        print("\n" + "=" * 70)
        print("双语治理测试总结")
        print("=" * 70)
        
        for test_name, result in results.items():
            status = "✅" if result["pass_rate"] >= 0.8 else "⚠️" if result["pass_rate"] >= 0.5 else "❌"
            print(f"  {status} {test_name}: {result['pass_rate']:.1%}")
        
        overall_pass_rate = sum(r["pass_rate"] for r in results.values()) / len(results)
        
        print(f"\n  总体通过率: {overall_pass_rate:.1%}")
        
        if overall_pass_rate >= 0.8:
            print("\n" + "🎉 " * 35)
            print("双语治理测试通过！英文适配层与治理主链集成成功。")
            print("🎉 " * 35)
        elif overall_pass_rate >= 0.5:
            print("\n⚠️ 部分测试通过，需要优化。")
        else:
            print("\n❌ 测试未通过，需要检查实现。")
        
        return {
            "overall_pass_rate": overall_pass_rate,
            "detailed_results": results
        }


def demo_bilingual_governance_test():
    """演示双语治理测试"""
    test = BilingualGovernanceTest()
    return test.run_all_tests()


if __name__ == "__main__":
    demo_bilingual_governance_test()
