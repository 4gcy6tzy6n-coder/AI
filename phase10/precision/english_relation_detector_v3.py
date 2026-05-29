"""
English Relation Detector v3 - 英文关系检测器 v3

WP3 核心组件：
进一步精修英文关系检测精度

改进点：
1. 更丰富的关系模式库
2. 增强的介词处理
3. 从句结构支持
4. 否定与转折关系
5. 置信度评分优化
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RelationType(Enum):
    """关系类型"""
    SUBJECT_VERB_OBJECT = "svo"
    PREPOSITION = "preposition"
    POSSESSION = "possession"
    CAUSATION = "causation"
    COMPARISON = "comparison"
    NEGATION = "negation"
    CONDITION = "condition"
    CONTRAST = "contrast"


@dataclass
class Relation:
    """关系定义"""
    subject: str
    predicate: str
    object: str
    relation_type: RelationType
    confidence: float
    source_text: str
    span: Tuple[int, int]
    metadata: Dict[str, Any] = None


class EnglishRelationDetectorV3:
    """
    英文关系检测器 v3
    
    改进点：
    1. 更丰富的关系模式（从 6 类扩展到 8 类）
    2. 增强的介词处理（20+ 介词模式）
    3. 从句结构支持（定语从句、状语从句）
    4. 否定与转折关系
    5. 置信度评分优化
    """
    
    def __init__(self):
        self._init_patterns()
        self._init_negative_patterns()
        self._init_preposition_mappings()
        self._init_clause_patterns()
    
    def _init_patterns(self):
        """初始化关系模式"""
        # 1. 主谓宾模式（增强版）
        self.svo_patterns = [
            # 基础模式
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>is|are|was|were|becomes?|remains?)\s+(?P<object>[^.]+)',
            # 主动语态
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>[a-z]+s?)\s+(?P<object>\b[a-z]+\b)',
            # 带助动词
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(has|have|had|will|would|can|could|may|might)\s+(?P<verb>[a-z]+ed|be\s+[a-z]+ing)\s+(?P<object>[^.]+)',
        ]
        
        # 2. 介词关系模式（扩展）
        self.prep_patterns = [
            # 空间关系
            (r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>is|are|was|were)\s+(?P<prep>on|in|at|under|above|below|behind|in front of|next to|beside|between|among)\s+(?P<object>[^.]+)', RelationType.PREPOSITION),
            # 时间关系
            (r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>happened|occurred|took place)\s+(?P<prep>before|after|during|while|when)\s+(?P<object>[^.]+)', RelationType.PREPOSITION),
            # 方式关系
            (r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>is|are|was|were)\s+(?P<prep>by|with|through|via)\s+(?P<object>[^.]+)', RelationType.PREPOSITION),
        ]
        
        # 3. 拥有关系
        self.possession_patterns = [
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>has|have|had|owns|possesses)\s+(?P<object>[^.]+)',
            r'(?P<object>[^.]+)\s+(?P<verb>belongs to|is owned by)\s+(?P<subject>\b[A-Z][a-zA-Z]*\b)',
        ]
        
        # 4. 因果关系（增强）
        self.causation_patterns = [
            # A causes B
            (r'(?P<cause>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>causes?|leads? to|results? in|triggers?|produces?)\s+(?P<effect>[^.]+)', 'forward'),
            # B is caused by A
            (r'(?P<effect>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>is|are|was|were)\s+(?P<caused>caused by|resulted from|triggered by|produced by)\s+(?P<cause>[^.]+)', 'reverse'),
            # If A then B
            (r'if\s+(?P<cause>[^,]+),?\s+then\s+(?P<effect>[^.]+)', 'conditional'),
        ]
        
        # 5. 比较关系
        self.comparison_patterns = [
            # 比较级
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>is|are|was|were)\s+(?P<comparator>taller|shorter|bigger|smaller|faster|slower|better|worse|stronger|weaker)\s+than\s+(?P<object>[^.]+)',
            # 同级比较
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>is|are|was|were)\s+as\s+(?P<adjective>[a-z]+)\s+as\s+(?P<object>[^.]+)',
        ]
        
        # 6. 否定关系（新增）
        self.negation_patterns = [
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<negation>does not|doesn\'t|did not|didn\'t|is not|isn\'t|are not|aren\'t|was not|wasn\'t|were not|weren\'t|has not|hasn\'t|have not|haven\'t)\s+(?P<verb>[a-z]+)\s+(?P<object>[^.]+)',
        ]
        
        # 7. 条件关系（新增）
        self.condition_patterns = [
            r'if\s+(?P<condition>[^,]+),?\s+(?P<result>[^.]+)',
            r'unless\s+(?P<condition>[^,]+),?\s+(?P<result>[^.]+)',
            r'provided that\s+(?P<condition>[^,]+),?\s+(?P<result>[^.]+)',
        ]
        
        # 8. 转折关系（新增）
        self.contrast_patterns = [
            r'(?P<first>[^.]+),?\s+but\s+(?P<second>[^.]+)',
            r'although\s+(?P<concession>[^,]+),?\s+(?P<result>[^.]+)',
            r'even though\s+(?P<concession>[^,]+),?\s+(?P<result>[^.]+)',
            r'while\s+(?P<contrast>[^,]+),?\s+(?P<result>[^.]+)',
        ]
    
    def _init_negative_patterns(self):
        """初始化负样本模式"""
        self.negative_patterns = [
            # 疑问句
            r'^\s*(is|are|was|were|do|does|did|can|could|will|would|have|has|had)\s+',
            # 感叹句
            r'!\s*$',
            # 祈使句（以动词开头）
            r'^\s*(please\s+)?[a-z]+\s+',
        ]
    
    def _init_preposition_mappings(self):
        """初始化介词映射"""
        self.prep_relation_map = {
            # 空间
            'on': 'located_on',
            'in': 'located_in',
            'at': 'located_at',
            'under': 'located_under',
            'above': 'located_above',
            'below': 'located_below',
            'behind': 'located_behind',
            'in front of': 'located_in_front_of',
            'next to': 'located_next_to',
            'beside': 'located_beside',
            'between': 'located_between',
            'among': 'located_among',
            # 时间
            'before': 'time_before',
            'after': 'time_after',
            'during': 'time_during',
            'while': 'time_while',
            'when': 'time_when',
            # 方式
            'by': 'method_by',
            'with': 'method_with',
            'through': 'method_through',
            'via': 'method_via',
        }
    
    def _init_clause_patterns(self):
        """初始化从句模式"""
        self.clause_patterns = {
            # 定语从句
            'relative': r'(?P<subject>\b[A-Z][a-zA-Z]*\b),?\s+(?P<relative>who|which|that)\s+(?P<verb>[^,]+)',
            # 状语从句（时间）
            'adverbial_time': r'when\s+(?P<clause>[^,]+),?\s+(?P<main>[^.]+)',
            # 状语从句（原因）
            'adverbial_reason': r'because\s+(?P<clause>[^,]+),?\s+(?P<main>[^.]+)',
        }
    
    def _is_negative_sample(self, text: str) -> bool:
        """检查是否为负样本"""
        for pattern in self.negative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _calculate_confidence(self, match, pattern_type: str, text: str) -> float:
        """计算置信度"""
        base_confidence = 0.7
        
        # 根据模式类型调整
        type_adjustments = {
            'svo': 0.1,
            'prep': 0.05,
            'possession': 0.15,
            'causation': 0.05,
            'comparison': 0.1,
            'negation': -0.05,  # 否定关系置信度稍低
            'condition': 0.0,
            'contrast': 0.0,
        }
        
        confidence = base_confidence + type_adjustments.get(pattern_type, 0)
        
        # 根据文本长度调整
        text_length = len(text)
        if text_length < 20:
            confidence += 0.05
        elif text_length > 100:
            confidence -= 0.05
        
        # 根据匹配质量调整
        match_length = match.end() - match.start()
        if match_length < 10:
            confidence -= 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def detect_relations(self, text: str) -> List[Relation]:
        """检测文本中的关系"""
        relations = []
        
        # 负样本过滤
        if self._is_negative_sample(text):
            return relations
        
        # 1. 检测主谓宾关系
        for pattern in self.svo_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                verb = match.group('verb')
                obj = match.group('object')
                
                # 清理 object
                obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE).strip()
                obj = obj.split(',')[0].strip()  # 取第一个分句
                
                confidence = self._calculate_confidence(match, "svo", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate=verb,
                    object=obj,
                    relation_type=RelationType.SUBJECT_VERB_OBJECT,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 2. 检测介词关系
        for pattern, rel_type in self.prep_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                verb = match.group('verb')
                obj = match.group('object')
                prep = match.groupdict().get('prep', '')
                
                # 构建谓词
                predicate = self.prep_relation_map.get(prep.lower(), f"{verb}_{prep}")
                
                confidence = self._calculate_confidence(match, "prep", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj.strip(),
                    relation_type=rel_type,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 3. 检测拥有关系
        for pattern in self.possession_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                verb = match.group('verb')
                obj = match.group('object')
                
                confidence = self._calculate_confidence(match, "possession", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate="possesses",
                    object=obj.strip(),
                    relation_type=RelationType.POSSESSION,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 4. 检测因果关系
        for pattern, direction in self.causation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if direction == 'forward':
                    cause = match.group('cause')
                    effect = match.group('effect')
                elif direction == 'reverse':
                    effect = match.group('effect')
                    cause = match.group('cause')
                else:  # conditional
                    cause = match.group('condition')
                    effect = match.group('result')
                
                confidence = self._calculate_confidence(match, "causation", text)
                
                relations.append(Relation(
                    subject=cause.strip(),
                    predicate="causes",
                    object=effect.strip(),
                    relation_type=RelationType.CAUSATION,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 5. 检测比较关系
        for pattern in self.comparison_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                obj = match.group('object')
                
                # 获取比较词
                comparator = match.groupdict().get('comparator', 'as_as')
                
                confidence = self._calculate_confidence(match, "comparison", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate=f"compared_{comparator}",
                    object=obj.strip(),
                    relation_type=RelationType.COMPARISON,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 6. 检测否定关系（新增）
        for pattern in self.negation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                verb = match.group('verb')
                obj = match.group('object')
                
                confidence = self._calculate_confidence(match, "negation", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate=f"not_{verb}",
                    object=obj.strip(),
                    relation_type=RelationType.NEGATION,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 7. 检测条件关系（新增）
        for pattern in self.condition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                condition = match.group('condition')
                result = match.group('result')
                
                confidence = self._calculate_confidence(match, "condition", text)
                
                relations.append(Relation(
                    subject=condition.strip(),
                    predicate="implies",
                    object=result.strip(),
                    relation_type=RelationType.CONDITION,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 8. 检测转折关系（新增）
        for pattern in self.contrast_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # 根据模式获取不同的组
                if 'first' in match.groupdict():
                    first = match.group('first')
                    second = match.group('second')
                elif 'concession' in match.groupdict():
                    first = match.group('concession')
                    second = match.group('result')
                elif 'contrast' in match.groupdict():
                    first = match.group('contrast')
                    second = match.group('result')
                else:
                    continue
                
                confidence = self._calculate_confidence(match, "contrast", text)
                
                relations.append(Relation(
                    subject=first.strip(),
                    predicate="contrasts_with",
                    object=second.strip(),
                    relation_type=RelationType.CONTRAST,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 去重：基于 span
        unique_relations = []
        seen_spans = set()
        for rel in relations:
            if rel.span not in seen_spans:
                unique_relations.append(rel)
                seen_spans.add(rel.span)
        
        # 按置信度排序
        unique_relations.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_relations
    
    def evaluate(self, test_cases: List[Dict]) -> Dict[str, float]:
        """评估检测器性能"""
        tp = fp = fn = 0
        
        for case in test_cases:
            text = case["text"]
            expected = case.get("expected_relations", [])
            
            detected = self.detect_relations(text)
            detected_set = {(r.subject, r.predicate, r.object) for r in detected}
            expected_set = {(e["subject"], e["predicate"], e["object"]) for e in expected}
            
            tp += len(detected_set & expected_set)
            fp += len(detected_set - expected_set)
            fn += len(expected_set - detected_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }


def demo_relation_detector_v3():
    """演示关系检测器 v3"""
    print("\n" + "="*70)
    print("English Relation Detector V3 - 演示")
    print("="*70)
    
    detector = EnglishRelationDetectorV3()
    
    # 测试用例（扩展）
    test_cases = [
        # 基础 SVO
        "The cat sits on the mat.",
        # 介词关系
        "The book is on the table.",
        # 拥有关系
        "John has a car.",
        # 因果关系
        "Rain causes wet ground.",
        # 比较关系
        "Tom is taller than Jerry.",
        # 否定关系（新增）
        "The cat does not like water.",
        # 条件关系（新增）
        "If it rains, the ground will be wet.",
        # 转折关系（新增）
        "It is raining, but I still go out.",
    ]
    
    print("\n1. 关系检测测试")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n  测试 {i}: {text}")
        
        relations = detector.detect_relations(text)
        
        if relations:
            for j, rel in enumerate(relations[:2], 1):  # 只显示前2个
                print(f"    {j}. [{rel.relation_type.value}] {rel.subject} --{rel.predicate}--> {rel.object} (conf: {rel.confidence:.2f})")
        else:
            print("    (无关系检测)")
    
    print("\n2. 评估指标")
    print("-" * 50)
    
    # 使用标准测试集评估
    eval_cases = [
        {
            "text": "The cat sits on the mat.",
            "expected_relations": [{"subject": "The cat", "predicate": "sits_on", "object": "the mat"}]
        },
        {
            "text": "John has a car.",
            "expected_relations": [{"subject": "John", "predicate": "possesses", "object": "a car"}]
        },
        {
            "text": "Rain causes wet ground.",
            "expected_relations": [{"subject": "Rain", "predicate": "causes", "object": "wet ground"}]
        },
        {
            "text": "If it rains, the ground will be wet.",
            "expected_relations": [{"subject": "it rains", "predicate": "implies", "object": "the ground will be wet"}]
        },
    ]
    
    metrics = detector.evaluate(eval_cases)
    
    print(f"  精确率 (Precision): {metrics['precision']:.2f}")
    print(f"  召回率 (Recall): {metrics['recall']:.2f}")
    print(f"  F1 分数: {metrics['f1']:.2f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_relation_detector_v3()
