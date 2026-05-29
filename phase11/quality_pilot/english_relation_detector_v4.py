"""
English Relation Detector v4 - 英文关系检测器 v4

Phase 11 WP4 核心组件：
进一步提升英文关系检测精度至 F1 0.72+

改进点：
1. 更精确的模式匹配
2. 增强的边界检测
3. 误召回抑制
4. 检索质量联动
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


class EnglishRelationDetectorV4:
    """
    英文关系检测器 v4
    
    目标：F1 >= 0.72
    
    改进：
    1. 更精确的模式匹配
    2. 边界检测优化
    3. 误召回抑制
    4. 置信度校准
    """
    
    def __init__(self):
        self._init_patterns()
        self._init_negative_patterns()
        self.min_confidence = 0.6
    
    def _init_patterns(self):
        """初始化关系模式（优化版）"""
        # 1. 主谓宾模式（精确版）
        self.svo_patterns = [
            # 标准模式：The cat sits on the mat
            (r'\b([A-Z][a-zA-Z]*(?:\s+[a-z]+){0,2})\s+(sits?|stands?|lies?|runs?|walks?|jumps?)\s+(on|in|under|at|by)\s+(the\s+)?([a-z]+(?:\s+[a-z]+){0,2})\b',
             lambda m: (m.group(1), f"{m.group(2)}_{m.group(3)}", m.group(5))),
            
            # BE 动词模式：The book is on the table
            (r'\b([A-Z][a-zA-Z]*(?:\s+[a-z]+){0,2})\s+(is|are|was|were)\s+(on|in|under|above|below|at|beside|near)\s+(the\s+)?([a-z]+(?:\s+[a-z]+){0,2})\b',
             lambda m: (m.group(1), f"located_{m.group(3)}", m.group(5))),
            
            # 拥有模式：John has a car
            (r'\b([A-Z][a-zA-Z]*)\s+(has|have|had|owns|possesses)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+){0,2})\b',
             lambda m: (m.group(1), "possesses", m.group(3))),
        ]
        
        # 2. 因果关系（精确版）
        self.causation_patterns = [
            # A causes B
            (r'\b([A-Z][a-zA-Z]*(?:\s+[a-z]+){0,2})\s+(causes?|leads?\s+to|results?\s+in|triggers?)\s+([a-z]+(?:\s+[a-z]+){0,3})\b',
             lambda m: (m.group(1), "causes", m.group(3))),
        ]
        
        # 3. 比较关系（精确版）
        self.comparison_patterns = [
            # A is taller than B
            (r'\b([A-Z][a-zA-Z]*)\s+(is|are|was|were)\s+(taller|shorter|bigger|smaller|faster|slower|better|worse)\s+than\s+([A-Z][a-zA-Z]*)\b',
             lambda m: (m.group(1), f"is_{m.group(3)}_than", m.group(4))),
        ]
        
        # 4. 否定关系
        self.negation_patterns = [
            (r'\b([A-Z][a-zA-Z]*)\s+(does\s+not|doesn\'t|is\s+not|isn\'t)\s+([a-z]+)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
             lambda m: (m.group(1), f"not_{m.group(3)}", m.group(4))),
        ]
        
        # 5. 条件关系
        self.condition_patterns = [
            (r'\bif\s+([a-z]+(?:\s+[a-z]+){0,3}),?\s+then\s+([a-z]+(?:\s+[a-z]+){0,3})\b',
             lambda m: (m.group(1), "implies", m.group(2))),
        ]
    
    def _init_negative_patterns(self):
        """初始化负样本模式"""
        self.negative_patterns = [
            r'^\s*(is|are|was|were|do|does|did|can|could|will|would)\s+',  # 疑问句
            r'!\s*$',  # 感叹句
            r'^\s*(please\s+)?[a-z]+\s+[a-z]+',  # 祈使句
            r'\?$',  # 问句
        ]
    
    def _is_negative_sample(self, text: str) -> bool:
        """检查是否为负样本"""
        for pattern in self.negative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _clean_entity(self, entity: str) -> str:
        """清理实体"""
        # 去除首尾空格
        entity = entity.strip()
        # 去除冠词
        entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        # 去除标点
        entity = re.sub(r'[.,;!?]$', '', entity)
        return entity.strip()
    
    def _calculate_confidence(self, pattern_type: str, match_quality: float) -> float:
        """计算置信度"""
        base_confidence = {
            'svo': 0.85,
            'causation': 0.75,
            'comparison': 0.80,
            'negation': 0.70,
            'condition': 0.75
        }.get(pattern_type, 0.70)
        
        # 根据匹配质量调整
        confidence = base_confidence * match_quality
        
        return min(1.0, max(0.0, confidence))
    
    def detect_relations(self, text: str) -> List[Relation]:
        """检测文本中的关系"""
        relations = []
        
        # 负样本过滤
        if self._is_negative_sample(text):
            return relations
        
        # 1. 检测 SVO 关系
        for pattern, extractor in self.svo_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject, predicate, obj = extractor(match)
                    subject = self._clean_entity(subject)
                    obj = self._clean_entity(obj)
                    
                    if subject and obj and len(subject) > 1 and len(obj) > 1:
                        confidence = self._calculate_confidence('svo', 1.0)
                        
                        if confidence >= self.min_confidence:
                            relations.append(Relation(
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                                relation_type=RelationType.SUBJECT_VERB_OBJECT,
                                confidence=confidence,
                                source_text=text,
                                span=(match.start(), match.end())
                            ))
                except Exception:
                    continue
        
        # 2. 检测因果关系
        for pattern, extractor in self.causation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject, predicate, obj = extractor(match)
                    subject = self._clean_entity(subject)
                    obj = self._clean_entity(obj)
                    
                    if subject and obj:
                        confidence = self._calculate_confidence('causation', 0.9)
                        
                        if confidence >= self.min_confidence:
                            relations.append(Relation(
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                                relation_type=RelationType.CAUSATION,
                                confidence=confidence,
                                source_text=text,
                                span=(match.start(), match.end())
                            ))
                except Exception:
                    continue
        
        # 3. 检测比较关系
        for pattern, extractor in self.comparison_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject, predicate, obj = extractor(match)
                    subject = self._clean_entity(subject)
                    obj = self._clean_entity(obj)
                    
                    if subject and obj:
                        confidence = self._calculate_confidence('comparison', 0.95)
                        
                        if confidence >= self.min_confidence:
                            relations.append(Relation(
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                                relation_type=RelationType.COMPARISON,
                                confidence=confidence,
                                source_text=text,
                                span=(match.start(), match.end())
                            ))
                except Exception:
                    continue
        
        # 4. 检测否定关系
        for pattern, extractor in self.negation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject, predicate, obj = extractor(match)
                    subject = self._clean_entity(subject)
                    obj = self._clean_entity(obj)
                    
                    if subject and obj:
                        confidence = self._calculate_confidence('negation', 0.85)
                        
                        if confidence >= self.min_confidence:
                            relations.append(Relation(
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                                relation_type=RelationType.NEGATION,
                                confidence=confidence,
                                source_text=text,
                                span=(match.start(), match.end())
                            ))
                except Exception:
                    continue
        
        # 5. 检测条件关系
        for pattern, extractor in self.condition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject, predicate, obj = extractor(match)
                    subject = self._clean_entity(subject)
                    obj = self._clean_entity(obj)
                    
                    if subject and obj:
                        confidence = self._calculate_confidence('condition', 0.9)
                        
                        if confidence >= self.min_confidence:
                            relations.append(Relation(
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                                relation_type=RelationType.CONDITION,
                                confidence=confidence,
                                source_text=text,
                                span=(match.start(), match.end())
                            ))
                except Exception:
                    continue
        
        # 去重
        unique_relations = []
        seen = set()
        for rel in relations:
            key = (rel.subject.lower(), rel.predicate.lower(), rel.object.lower())
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
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
            
            # 简化匹配：检查 subject 和 object 是否匹配
            detected_set = set()
            for r in detected:
                detected_set.add((r.subject.lower(), r.object.lower()))
            
            expected_set = set()
            for e in expected:
                expected_set.add((e["subject"].lower(), e["object"].lower()))
            
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


def demo_relation_detector_v4():
    """演示关系检测器 v4"""
    print("\n" + "="*70)
    print("English Relation Detector V4 - 演示")
    print("="*70)
    
    detector = EnglishRelationDetectorV4()
    
    # 测试用例
    test_cases = [
        "The cat sits on the mat.",
        "The book is on the table.",
        "John has a car.",
        "Rain causes wet ground.",
        "Tom is taller than Jerry.",
        "The cat does not like water.",
        "If it rains, then the ground will be wet.",
    ]
    
    print("\n1. 关系检测测试")
    print("-" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n  测试 {i}: {text}")
        
        relations = detector.detect_relations(text)
        
        if relations:
            for j, rel in enumerate(relations[:2], 1):
                print(f"    {j}. [{rel.relation_type.value}] {rel.subject} --{rel.predicate}--> {rel.object} (conf: {rel.confidence:.2f})")
        else:
            print("    (无关系检测)")
    
    # 评估
    print("\n2. 评估指标")
    print("-" * 50)
    
    eval_cases = [
        {
            "text": "The cat sits on the mat.",
            "expected_relations": [{"subject": "The cat", "object": "mat"}]
        },
        {
            "text": "John has a car.",
            "expected_relations": [{"subject": "John", "object": "car"}]
        },
        {
            "text": "Rain causes wet ground.",
            "expected_relations": [{"subject": "Rain", "object": "wet ground"}]
        },
        {
            "text": "Tom is taller than Jerry.",
            "expected_relations": [{"subject": "Tom", "object": "Jerry"}]
        },
    ]
    
    metrics = detector.evaluate(eval_cases)
    
    print(f"  精确率 (Precision): {metrics['precision']:.2f}")
    print(f"  召回率 (Recall): {metrics['recall']:.2f}")
    print(f"  F1 分数: {metrics['f1']:.2f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    # 目标检查
    print("\n3. 目标检查")
    print("-" * 50)
    if metrics['f1'] >= 0.72:
        print(f"  ✓ F1 达到目标 (>= 0.72): {metrics['f1']:.2f}")
    else:
        print(f"  ⚠ F1 未达目标 (>= 0.72): {metrics['f1']:.2f}")
        print(f"    建议: 继续优化模式匹配")
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_relation_detector_v4()
