"""
English Relation Detector v2 - 英文关系检测器 v2

WP2 核心组件：
解决 Phase 8 关系检测精度低的问题 (33.3%)

优化策略：
1. 扩展关系模式库
2. 增强介词结构识别
3. 支持从句结构
4. 引入置信度评分
5. 负样本过滤
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
    SUBJECT_VERB_OBJECT = "svo"           # 主谓宾
    VERB_PREPOSITION = "vp"               # 动词+介词
    POSSESSION = "possession"             # 拥有关系
    SPATIAL = "spatial"                   # 空间关系
    TEMPORAL = "temporal"                 # 时间关系
    CAUSAL = "causal"                     # 因果关系
    COMPARISON = "comparison"             # 比较关系
    ATTRIBUTION = "attribution"           # 归因关系
    PART_WHOLE = "part_whole"             # 部分整体
    ASSOCIATION = "association"           # 关联关系


@dataclass
class Relation:
    """关系定义"""
    subject: str
    predicate: str
    object: str
    relation_type: RelationType
    confidence: float
    source_text: str
    span: Tuple[int, int]  # 在原文中的位置


@dataclass
class DetectionResult:
    """检测结果"""
    relations: List[Relation]
    precision: float
    recall: float
    f1_score: float


class EnglishRelationDetectorV2:
    """
    英文关系检测器 v2
    
    改进点：
    1. 更丰富的关系模式
    2. 介词结构专项处理
    3. 从句结构支持
    4. 置信度评分
    5. 负样本过滤
    """
    
    def __init__(self):
        self._init_patterns()
        self._init_negative_patterns()
        self._init_preposition_mappings()
    
    def _init_patterns(self):
        """初始化关系模式库"""
        
        # 主谓宾模式 (SVO)
        self.svo_patterns = [
            # 基础模式: Subject Verb Object
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>\w+ed|\w+ing|\w+s?)\s+(?P<object>\b[a-z]*[A-Z][a-zA-Z]*\b|\bthe\s+\w+|\ba\s+\w+)',
            # 带助动词: Subject has/have/had Verb Object
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?:has|have|had)\s+(?P<verb>\w+ed|\w+ing)\s+(?P<object>\bthe\s+\w+)',
            # 被动语态: Object was/were Verb by Subject
            r'(?P<object>\b[A-Z][a-zA-Z]*\b)\s+(?:was|were)\s+(?P<verb>\w+ed)\s+by\s+(?P<subject>\b[a-z]+\b)',
        ]
        
        # 介词关系模式
        self.prep_patterns = [
            # 空间介词
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>is|are|was|were)\s+(?P<prep>on|in|at|under|above|below|beside|near|between)\s+(?P<object>\b\w+\b)', RelationType.SPATIAL),
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>sat|stood|placed|located)\s+(?P<prep>on|in|at|under)\s+(?P<object>\b\w+\b)', RelationType.SPATIAL),
            # 时间介词
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>happened|occurred|took place)\s+(?P<prep>on|in|at|during|before|after)\s+(?P<object>\b\w+\b)', RelationType.TEMPORAL),
            # 拥有介词
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>has|have|had|owns|possesses)\s+(?P<object>\b\w+\b)', RelationType.POSSESSION),
            # 部分整体
            (r'(?P<object>\b\w+\b)\s+(?P<verb>has|have|had|contains|includes)\s+(?P<subject>\b\w+\b)', RelationType.PART_WHOLE),
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>is|are|was|were)\s+part\s+of\s+(?P<object>\b\w+\b)', RelationType.PART_WHOLE),
        ]
        
        # 因果关系模式
        self.causal_patterns = [
            (r'(?P<cause>.*?)\s+(?:causes?|leads? to|results? in|makes?)\s+(?P<effect>\b\w+.*?)(?:\.|,|;|$)', RelationType.CAUSAL),
            (r'(?P<effect>\b\w+.*?)\s+(?:is|are|was|were)\s+(?:caused by|due to|because of|result of)\s+(?P<cause>\b\w+.*?)(?:\.|,|;|$)', RelationType.CAUSAL),
            (r'(?:if|when)\s+(?P<cause>\b\w+.*?)\s*,?\s+(?:then)?\s*(?P<effect>\b\w+.*?)(?:\.|,|;|$)', RelationType.CAUSAL),
        ]
        
        # 比较关系模式
        self.comparison_patterns = [
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>is|are|was|were)\s+(?P<comparator>more|less|better|worse|taller|shorter)\s+than\s+(?P<object>\b\w+\b)', RelationType.COMPARISON),
            (r'(?P<subject>\b\w+\b)\s+(?P<verb>is|are|was|were)\s+as\s+(?P<adjective>\w+)\s+as\s+(?P<object>\b\w+\b)', RelationType.COMPARISON),
        ]
        
        # 双宾语模式 (Subject Verb Object1 Object2)
        self.ditransitive_patterns = [
            r'(?P<subject>\b[A-Z][a-zA-Z]*\b)\s+(?P<verb>gave|gives?|given|sent|sends?|offered|offers?|told|tells?)\s+(?P<obj1>\b\w+\b)\s+(?P<obj2>\b\w+\b)',
        ]
    
    def _init_negative_patterns(self):
        """初始化负样本过滤模式"""
        self.negative_patterns = [
            # 虚词误检
            r'\b(that|this|it|there|here)\s+(is|are|was|were)\b',
            # 时间表达误检
            r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b',
            # 日期表达误检
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b',
            # 疑问句
            r'^(What|Who|Where|When|Why|How|Is|Are|Was|Were|Do|Does|Did|Can|Could|Would|Will)\b',
        ]
    
    def _init_preposition_mappings(self):
        """初始化介词到关系类型的映射"""
        self.prep_to_relation = {
            # 空间介词
            'on': RelationType.SPATIAL,
            'in': RelationType.SPATIAL,
            'at': RelationType.SPATIAL,
            'under': RelationType.SPATIAL,
            'above': RelationType.SPATIAL,
            'below': RelationType.SPATIAL,
            'beside': RelationType.SPATIAL,
            'near': RelationType.SPATIAL,
            'between': RelationType.SPATIAL,
            'inside': RelationType.SPATIAL,
            'outside': RelationType.SPATIAL,
            
            # 时间介词
            'before': RelationType.TEMPORAL,
            'after': RelationType.TEMPORAL,
            'during': RelationType.TEMPORAL,
            'while': RelationType.TEMPORAL,
            
            # 因果介词
            'because': RelationType.CAUSAL,
            'since': RelationType.CAUSAL,
            'due': RelationType.CAUSAL,
        }
    
    def _is_negative_sample(self, text: str) -> bool:
        """检查是否为负样本"""
        for pattern in self.negative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _calculate_confidence(
        self,
        match: re.Match,
        pattern_type: str,
        text: str
    ) -> float:
        """计算置信度"""
        base_confidence = 0.7
        
        # 根据模式类型调整
        if pattern_type == "svo":
            base_confidence += 0.1
        elif pattern_type == "prep":
            base_confidence += 0.05
        
        # 根据匹配长度调整
        match_len = match.end() - match.start()
        if match_len > 20:
            base_confidence += 0.05
        
        # 根据上下文调整
        if text[match.start():match.end()].count(' ') > 3:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def detect_relations(self, text: str) -> List[Relation]:
        """
        检测文本中的关系
        
        Returns:
            List[Relation]: 检测到的关系列表
        """
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
                obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                
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
                
                # 获取介词
                prep = match.groupdict().get('prep', '')
                predicate = f"{verb}_{prep}" if prep else verb
                
                confidence = self._calculate_confidence(match, "prep", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    relation_type=rel_type,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 3. 检测因果关系
        for pattern, rel_type in self.causal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                cause = match.group('cause')
                effect = match.group('effect')
                
                confidence = self._calculate_confidence(match, "causal", text)
                
                relations.append(Relation(
                    subject=cause,
                    predicate="causes",
                    object=effect,
                    relation_type=rel_type,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 4. 检测比较关系
        for pattern, rel_type in self.comparison_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                obj = match.group('object')
                
                confidence = self._calculate_confidence(match, "comparison", text)
                
                relations.append(Relation(
                    subject=subject,
                    predicate="compared_to",
                    object=obj,
                    relation_type=rel_type,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
        
        # 5. 检测双宾语关系
        for pattern in self.ditransitive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject = match.group('subject')
                verb = match.group('verb')
                obj1 = match.group('obj1')
                obj2 = match.group('obj2')
                
                confidence = self._calculate_confidence(match, "ditransitive", text)
                
                # 添加两个关系
                relations.append(Relation(
                    subject=subject,
                    predicate=verb,
                    object=obj1,
                    relation_type=RelationType.SUBJECT_VERB_OBJECT,
                    confidence=confidence,
                    source_text=text,
                    span=(match.start(), match.end())
                ))
                
                relations.append(Relation(
                    subject=subject,
                    predicate=verb,
                    object=obj2,
                    relation_type=RelationType.SUBJECT_VERB_OBJECT,
                    confidence=confidence * 0.9,  # 第二个宾语置信度稍低
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
    
    def evaluate_detection(
        self,
        text: str,
        expected_relations: List[Tuple[str, str, str]]
    ) -> DetectionResult:
        """
        评估检测性能
        
        Args:
            text: 输入文本
            expected_relations: 期望的关系列表 [(subject, predicate, object), ...]
        
        Returns:
            DetectionResult: 评估结果
        """
        detected = self.detect_relations(text)
        
        # 转换为集合便于比较
        detected_set = set((r.subject.lower(), r.predicate.lower(), r.object.lower()) 
                          for r in detected)
        expected_set = set((s.lower(), p.lower(), o.lower()) 
                          for s, p, o in expected_relations)
        
        # 计算指标
        true_positives = len(detected_set & expected_set)
        false_positives = len(detected_set - expected_set)
        false_negatives = len(expected_set - detected_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return DetectionResult(
            relations=detected,
            precision=precision,
            recall=recall,
            f1_score=f1
        )


def demo_relation_detector_v2():
    """演示关系检测器 v2"""
    print("\n" + "="*70)
    print("English Relation Detector v2 - 演示")
    print("="*70)
    
    detector = EnglishRelationDetectorV2()
    
    # 测试用例
    test_cases = [
        {
            "text": "The cat sat on the mat.",
            "expected": [("cat", "sat_on", "mat")],
            "description": "基础介词关系"
        },
        {
            "text": "She gave him a book.",
            "expected": [("She", "gave", "him"), ("She", "gave", "book")],
            "description": "双宾语关系"
        },
        {
            "text": "The company acquired the startup.",
            "expected": [("company", "acquired", "startup")],
            "description": "主谓宾关系"
        },
        {
            "text": "The meeting is on Monday.",
            "expected": [],  # 时间介词不应检测为空间关系
            "description": "时间介词（负样本）"
        },
        {
            "text": "Stress causes health problems.",
            "expected": [("Stress", "causes", "health problems")],
            "description": "因果关系"
        },
        {
            "text": "John is taller than Mary.",
            "expected": [("John", "compared_to", "Mary")],
            "description": "比较关系"
        },
    ]
    
    print("\n1. 关系检测测试")
    print("-" * 50)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n  测试 {i}: {test['description']}")
        print(f"  文本: \"{test['text']}\"")
        
        result = detector.evaluate_detection(test['text'], test['expected'])
        
        print(f"  检测结果:")
        for rel in result.relations:
            print(f"    - ({rel.subject}, {rel.predicate}, {rel.object}) "
                  f"[置信度: {rel.confidence:.2f}]")
        
        print(f"  评估指标:")
        print(f"    精确率: {result.precision:.2f}")
        print(f"    召回率: {result.recall:.2f}")
        print(f"    F1分数: {result.f1_score:.2f}")
        
        total_precision += result.precision
        total_recall += result.recall
        total_f1 += result.f1_score
    
    # 平均指标
    n = len(test_cases)
    print(f"\n2. 总体评估")
    print("-" * 50)
    print(f"  平均精确率: {total_precision/n:.2f}")
    print(f"  平均召回率: {total_recall/n:.2f}")
    print(f"  平均F1分数: {total_f1/n:.2f}")
    
    print("\n3. 与 Phase 8 对比")
    print("-" * 50)
    print(f"  Phase 8 关系检测 F1: ~0.30")
    print(f"  Phase 9 目标 F1: > 0.72")
    print(f"  当前 F1: {total_f1/n:.2f}")
    
    if total_f1/n >= 0.72:
        print(f"  状态: ✅ 达到目标")
    else:
        print(f"  状态: 🔄 需继续优化")
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_relation_detector_v2()
