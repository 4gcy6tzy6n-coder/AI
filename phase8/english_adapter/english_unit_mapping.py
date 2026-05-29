"""
English Unit Mapping - 英文 Unit 映射实现

WP1 核心组件：
将英文输入映射到四类通用 Unit：
- Concept Unit
- Relation Unit
- Rule Unit
- Task Pattern Unit

原则：
- 所有英文特有逻辑限制在本文件
- 通过标准接口与 Core Governance 交互
- 不污染核心治理层
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EnglishUnitType(Enum):
    """英文 Unit 类型"""
    CONCEPT = "concept"
    RELATION = "relation"
    RULE = "rule"
    TASK_PATTERN = "task_pattern"


@dataclass
class EnglishWordInput:
    """英文单词输入结构"""
    # 基础信息
    text: str
    context: Optional[str] = None
    
    # 预处理结果
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    
    # 语言检测
    detected_language: str = "en"
    language_confidence: float = 1.0
    
    # 适配层标记
    adapter_version: str = "v1.0"
    preprocessing_timestamp: str = ""


@dataclass
class EnglishConceptUnit:
    """英文 Concept Unit 实现"""
    unit_id: str
    unit_type: str = "concept"
    language: str = "en"
    
    # 感知核心（英文特有）
    surface_form: str = ""
    spelling_variants: List[str] = field(default_factory=list)
    phonetic_representation: str = ""
    pronunciation_variants: List[str] = field(default_factory=list)
    syllable_count: int = 0
    stress_pattern: str = ""
    
    # 词形变化（英文特有）
    inflectional_forms: Dict[str, str] = field(default_factory=dict)
    
    # 语义核心
    word_senses: List[Dict] = field(default_factory=list)
    domain_labels: List[str] = field(default_factory=list)
    register: List[str] = field(default_factory=list)
    
    # 治理元数据
    quality_score: Dict[str, float] = field(default_factory=lambda: {
        "Q": 0.0, "T": 0.0, "S": 0.0, "C": 0.0, "L": 0.0
    })
    stability_cycles: int = 0
    source: str = ""


@dataclass
class EnglishRelationUnit:
    """英文 Relation Unit 实现"""
    unit_id: str
    unit_type: str = "relation"
    language: str = "en"
    
    # 关系标记词
    relation_markers: List[str] = field(default_factory=list)
    
    # 句法模式
    syntactic_patterns: List[str] = field(default_factory=list)
    
    # 介词映射（英文特有）
    preposition_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        "spatial": ["in", "on", "at", "under", "over", "behind", "in front of"],
        "temporal": ["before", "after", "during", "since", "until"],
        "causal": ["because", "due to", "as a result", "therefore"],
        "possessive": ["of", "'s"],
        "associative": ["with", "about", "regarding"]
    })
    
    # 语义核心
    relation_type: str = ""  # hierarchical/causal/temporal/spatial/functional
    arity: int = 2
    domain_constraints: List[str] = field(default_factory=list)
    range_constraints: List[str] = field(default_factory=list)
    
    # 治理元数据
    quality_score: Dict[str, float] = field(default_factory=lambda: {
        "Q": 0.0, "T": 0.0, "S": 0.0, "C": 0.0, "L": 0.0
    })


@dataclass
class EnglishRuleUnit:
    """英文 Rule Unit 实现"""
    unit_id: str
    unit_type: str = "rule"
    language: str = "en"
    
    # 规则触发词（英文特有）
    trigger_words: List[str] = field(default_factory=list)
    condition_markers: List[str] = field(default_factory=lambda: [
        "if", "when", "unless", "provided that", "assuming",
        "given that", "in case"
    ])
    conclusion_markers: List[str] = field(default_factory=lambda: [
        "then", "therefore", "thus", "consequently", "as a result"
    ])
    
    # 语义核心
    rule_type: str = ""  # inference/transformation/constraint/preference
    premises: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # 治理元数据
    quality_score: Dict[str, float] = field(default_factory=lambda: {
        "Q": 0.0, "T": 0.0, "S": 0.0, "C": 0.0, "L": 0.0
    })


@dataclass
class EnglishTaskPatternUnit:
    """英文 Task Pattern Unit 实现"""
    unit_id: str
    unit_type: str = "task_pattern"
    language: str = "en"
    
    # 任务触发关键词（英文特有）
    trigger_keywords: List[str] = field(default_factory=list)
    question_patterns: List[str] = field(default_factory=list)
    instruction_patterns: List[str] = field(default_factory=list)
    
    # 语义核心
    task_type: str = ""  # classification/generation/reasoning/retrieval/verification
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    required_units: List[str] = field(default_factory=list)
    
    # 治理元数据
    quality_score: Dict[str, float] = field(default_factory=lambda: {
        "Q": 0.0, "T": 0.0, "S": 0.0, "C": 0.0, "L": 0.0
    })


class EnglishUnitMapper:
    """
    英文 Unit 映射器
    
    功能：
    1. 将英文输入映射到通用 Unit 抽象
    2. 管理英文特有属性
    3. 提供与 Core Governance 的标准接口
    """
    
    def __init__(self):
        self.mappings: List[Dict] = []
        
        # 英文词形变化规则（简化版）
        self.inflection_rules = {
            "noun": {
                "plural": {
                    "regular_s": lambda w: w + "s",
                    "regular_es": lambda w: w + "es" if w.endswith(("s", "x", "z", "ch", "sh")) else None,
                    "y_to_ies": lambda w: w[:-1] + "ies" if w.endswith("y") and w[-2] not in "aeiou" else None,
                },
                "irregular": {
                    "child": "children",
                    "mouse": "mice",
                    "person": "people",
                    "foot": "feet",
                }
            },
            "verb": {
                "third_person": lambda w: w + "es" if w.endswith(("s", "x", "z", "ch", "sh", "o")) else w + "s",
                "past_tense": {
                    "regular_ed": lambda w: w + "ed",
                    "irregular": {
                        "go": "went",
                        "eat": "ate",
                        "see": "saw",
                        "take": "took",
                    }
                },
                "progressive": lambda w: w + "ing",
            }
        }
        
        # 关系标记词词典
        self.relation_markers = {
            "hierarchical": ["is a", "is an", "are", "belongs to", "part of", "type of"],
            "causal": ["causes", "leads to", "results in", "because", "due to", "therefore"],
            "temporal": ["before", "after", "during", "while", "when", "until", "since"],
            "spatial": ["in", "on", "at", "under", "over", "between", "next to"],
            "functional": ["used for", "enables", "allows", "helps", "serves as"],
        }
        
        # 任务触发词词典
        self.task_triggers = {
            "definition": ["what is", "define", "explain", "describe", "meaning of"],
            "comparison": ["difference between", "compare", "vs", "versus", "similarities"],
            "reasoning": ["why", "how come", "what if", "explain why", "reason for"],
            "verification": ["is it true", "verify", "check", "confirm", "validate"],
            "classification": ["what type", "which category", "classify", "categorize"],
        }
    
    def map_to_concept_unit(self, word_input: EnglishWordInput) -> EnglishConceptUnit:
        """
        将英文单词输入映射为 Concept Unit
        
        Args:
            word_input: 英文单词输入
            
        Returns:
            EnglishConceptUnit
        """
        word = word_input.text.lower()
        
        # 生成词形变化形式
        inflections = self._generate_inflections(word)
        
        # 检测音节数和重音模式（简化）
        syllable_count = self._count_syllables(word)
        
        concept_unit = EnglishConceptUnit(
            unit_id=f"en_concept_{word}_{hash(word) % 10000}",
            surface_form=word_input.text,
            spelling_variants=[word, word.capitalize(), word.upper()],
            phonetic_representation=self._generate_phonetic(word),
            syllable_count=syllable_count,
            stress_pattern=self._estimate_stress_pattern(word, syllable_count),
            inflectional_forms=inflections,
            word_senses=self._generate_word_senses(word),
            domain_labels=["general"],
            register=["neutral"]
        )
        
        return concept_unit
    
    def map_to_relation_unit(self, text: str) -> Optional[EnglishRelationUnit]:
        """
        从文本中识别并映射 Relation Unit
        
        Args:
            text: 输入文本
            
        Returns:
            EnglishRelationUnit 或 None
        """
        text_lower = text.lower()
        
        # 检测关系类型
        detected_relation = None
        for rel_type, markers in self.relation_markers.items():
            for marker in markers:
                if marker in text_lower:
                    detected_relation = rel_type
                    break
            if detected_relation:
                break
        
        if not detected_relation:
            return None
        
        relation_unit = EnglishRelationUnit(
            unit_id=f"en_relation_{detected_relation}_{hash(text) % 10000}",
            relation_markers=[detected_relation],
            syntactic_patterns=[text],
            relation_type=detected_relation,
            domain_constraints=["entity"],
            range_constraints=["entity"]
        )
        
        return relation_unit
    
    def map_to_rule_unit(self, text: str) -> Optional[EnglishRuleUnit]:
        """
        从文本中识别并映射 Rule Unit
        
        Args:
            text: 输入文本
            
        Returns:
            EnglishRuleUnit 或 None
        """
        text_lower = text.lower()
        
        # 检测条件从句标记
        has_condition = any(marker in text_lower for marker in 
                          ["if", "when", "unless", "provided that"])
        
        # 检测结论从句标记
        has_conclusion = any(marker in text_lower for marker in 
                           ["then", "therefore", "thus", "consequently"])
        
        if not (has_condition and has_conclusion):
            return None
        
        # 解析前提和结论（简化）
        premises, conclusions = self._parse_rule_structure(text)
        
        rule_unit = EnglishRuleUnit(
            unit_id=f"en_rule_{hash(text) % 10000}",
            trigger_words=["if", "then"],
            rule_type="inference",
            premises=premises,
            conclusions=conclusions,
            confidence=0.7
        )
        
        return rule_unit
    
    def map_to_task_pattern_unit(self, text: str) -> Optional[EnglishTaskPatternUnit]:
        """
        从文本中识别并映射 Task Pattern Unit
        
        Args:
            text: 输入文本
            
        Returns:
            EnglishTaskPatternUnit 或 None
        """
        text_lower = text.lower()
        
        # 检测任务类型
        detected_task = None
        detected_triggers = []
        
        for task_type, triggers in self.task_triggers.items():
            for trigger in triggers:
                if trigger in text_lower:
                    detected_task = task_type
                    detected_triggers.append(trigger)
        
        if not detected_task:
            return None
        
        task_unit = EnglishTaskPatternUnit(
            unit_id=f"en_task_{detected_task}_{hash(text) % 10000}",
            trigger_keywords=detected_triggers,
            question_patterns=[text] if "?" in text else [],
            instruction_patterns=[text] if "?" not in text else [],
            task_type=detected_task,
            input_schema={"type": "text", "language": "en"},
            output_schema={"type": "answer", "format": "text"},
            required_units=[]
        )
        
        return task_unit
    
    def _generate_inflections(self, word: str) -> Dict[str, str]:
        """生成词形变化形式（简化版）"""
        inflections = {}
        
        # 名词复数
        if word not in self.inflection_rules["noun"]["irregular"]:
            if word.endswith(("s", "x", "z", "ch", "sh")):
                inflections["plural"] = word + "es"
            elif word.endswith("y") and word[-2] not in "aeiou":
                inflections["plural"] = word[:-1] + "ies"
            else:
                inflections["plural"] = word + "s"
        else:
            inflections["plural"] = self.inflection_rules["noun"]["irregular"][word]
        
        # 动词形式
        if word not in self.inflection_rules["verb"]["past_tense"]["irregular"]:
            inflections["past_tense"] = word + "ed"
            inflections["progressive"] = word + "ing"
        else:
            inflections["past_tense"] = self.inflection_rules["verb"]["past_tense"]["irregular"][word]
            inflections["progressive"] = word + "ing"
        
        return inflections
    
    def _count_syllables(self, word: str) -> int:
        """计算音节数（简化算法）"""
        vowels = "aeiouy"
        word = word.lower()
        count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # 处理尾音e
        if word.endswith("e") and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _estimate_stress_pattern(self, word: str, syllable_count: int) -> str:
        """估计重音模式（简化）"""
        if syllable_count == 1:
            return "P"
        elif syllable_count == 2:
            # 大多数双音节名词重音在第一音节
            return "P-p"
        else:
            # 简化：假设重音在倒数第三音节
            return "p-p-P" if syllable_count == 3 else "p-P-p"
    
    def _generate_phonetic(self, word: str) -> str:
        """生成音标表示（占位符）"""
        # 实际实现需要集成音标词典或音标生成算法
        return f"/{word}/"
    
    def _generate_word_senses(self, word: str) -> List[Dict]:
        """生成词义（简化版，实际应查询词典）"""
        # 返回基本词义结构
        return [{
            "sense_id": f"{word}_1",
            "definition": f"Definition of {word}",
            "pos": "unknown",
            "frequency_rank": 1000
        }]
    
    def _parse_rule_structure(self, text: str) -> (List[str], List[str]):
        """解析规则结构，提取前提和结论（简化版）"""
        text_lower = text.lower()
        
        # 简单分割
        if "then" in text_lower:
            parts = text_lower.split("then")
            premises = [parts[0].replace("if", "").strip()]
            conclusions = [parts[1].strip()]
        else:
            premises = [text]
            conclusions = ["conclusion"]
        
        return premises, conclusions


def demo_english_unit_mapping():
    """演示英文 Unit 映射"""
    print("\n" + "=" * 70)
    print("English Unit Mapping Demo - 英文 Unit 映射演示")
    print("=" * 70)
    
    mapper = EnglishUnitMapper()
    
    # 测试 Concept Unit
    print("\n1. Concept Unit 映射")
    print("-" * 40)
    
    test_words = ["study", "apple", "run", "beautiful"]
    for word in test_words:
        word_input = EnglishWordInput(text=word)
        concept = mapper.map_to_concept_unit(word_input)
        print(f"\n  单词: {word}")
        print(f"    Unit ID: {concept.unit_id}")
        print(f"    音节数: {concept.syllable_count}")
        print(f"    重音模式: {concept.stress_pattern}")
        print(f"    词形变化: {concept.inflectional_forms}")
    
    # 测试 Relation Unit
    print("\n2. Relation Unit 映射")
    print("-" * 40)
    
    test_relations = [
        "A dog is a type of animal",
        "Smoking causes cancer",
        "The meeting is before lunch",
        "The book is on the table"
    ]
    
    for text in test_relations:
        relation = mapper.map_to_relation_unit(text)
        if relation:
            print(f"\n  文本: {text}")
            print(f"    Unit ID: {relation.unit_id}")
            print(f"    关系类型: {relation.relation_type}")
            print(f"    标记词: {relation.relation_markers}")
    
    # 测试 Rule Unit
    print("\n3. Rule Unit 映射")
    print("-" * 40)
    
    test_rules = [
        "If it rains, then the ground will be wet",
        "When the temperature drops below 0, water freezes"
    ]
    
    for text in test_rules:
        rule = mapper.map_to_rule_unit(text)
        if rule:
            print(f"\n  文本: {text}")
            print(f"    Unit ID: {rule.unit_id}")
            print(f"    规则类型: {rule.rule_type}")
            print(f"    前提: {rule.premises}")
            print(f"    结论: {rule.conclusions}")
    
    # 测试 Task Pattern Unit
    print("\n4. Task Pattern Unit 映射")
    print("-" * 40)
    
    test_tasks = [
        "What is machine learning?",
        "Explain the difference between AI and ML",
        "Why is the sky blue?",
        "Verify if this statement is true"
    ]
    
    for text in test_tasks:
        task = mapper.map_to_task_pattern_unit(text)
        if task:
            print(f"\n  文本: {text}")
            print(f"    Unit ID: {task.unit_id}")
            print(f"    任务类型: {task.task_type}")
            print(f"    触发词: {task.trigger_keywords}")
    
    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    demo_english_unit_mapping()
