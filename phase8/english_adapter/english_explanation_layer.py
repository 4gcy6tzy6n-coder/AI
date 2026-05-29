"""
English Explanation Layer - 英文解释层实现

WP1 核心组件：
提供三层解释结构：
- Layer 1: Word Explanation (词形还原、词义消歧、词性标注)
- Layer 2: Phrase Explanation (短语识别、语义组合)
- Layer 3: Sentence-Level Composition (句法解析、依存关系)

原则：
- 所有英文特有逻辑限制在本文件
- 通过标准接口与 Core Governance 交互
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase8.english_adapter.english_unit_mapping import (
    EnglishWordInput, EnglishConceptUnit, EnglishRelationUnit,
    EnglishUnitMapper
)


@dataclass
class WordExplanation:
    """单词解释结果"""
    original: str
    lemma: str
    pos: str
    sense: Optional[str] = None
    confidence: float = 0.0


@dataclass
class PhraseExplanation:
    """短语解释结果"""
    phrase_text: str
    phrase_type: str  # NP/VP/PP等
    head_word: str
    modifiers: List[str] = field(default_factory=list)
    semantic_composition: str = ""


@dataclass
class SyntacticRelation:
    """句法关系"""
    relation_type: str
    head: str
    dependent: str
    label: str


class WordExplanationLayer:
    """
    单词解释层 (Layer 1)
    
    功能：
    1. 词形还原 (Lemmatization)
    2. 词义消歧 (Word Sense Disambiguation)
    3. 词性标注 (POS Tagging)
    """
    
    def __init__(self):
        # 常见词形变化规则
        self.lemma_rules = {
            # 名词复数还原
            "plural_to_singular": {
                "ies": lambda w: w[:-3] + "y",
                "es": lambda w: w[:-2] if w.endswith(("ches", "shes", "xes", "zes", "ses")) else w[:-1],
                "s": lambda w: w[:-1],
            },
            # 动词时态还原
            "verb_conjugation": {
                "ied": lambda w: w[:-3] + "y",
                "ied": lambda w: w[:-3] + "y",
                "ed": lambda w: w[:-2] if len(w) > 3 else w,
                "ing": lambda w: w[:-3] if len(w) > 4 else w,
            }
        }
        
        # 不规则词形表
        self.irregular_forms = {
            # 名词
            "children": "child",
            "people": "person",
            "mice": "mouse",
            "feet": "foot",
            "teeth": "tooth",
            # 动词
            "went": "go",
            "ate": "eat",
            "saw": "see",
            "took": "take",
            "came": "come",
            "did": "do",
            "had": "have",
            "was": "be",
            "were": "be",
            # 形容词
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
        }
        
        # 简单词性标注规则
        self.pos_rules = {
            "noun_suffixes": ["tion", "sion", "ness", "ity", "er", "or", "ist", "ism"],
            "verb_suffixes": ["ate", "ify", "ize", "ise"],
            "adj_suffixes": ["ful", "ous", "ive", "able", "ible", "al", "ic"],
            "adv_suffixes": ["ly"],
        }
    
    def lemmatize(self, word: str, pos: Optional[str] = None) -> str:
        """
        词形还原
        
        Args:
            word: 输入单词
            pos: 词性提示（可选）
            
        Returns:
            还原后的基本形式
        """
        word_lower = word.lower()
        
        # 检查不规则词形
        if word_lower in self.irregular_forms:
            return self.irregular_forms[word_lower]
        
        # 应用规则还原
        # 处理复数
        if word_lower.endswith("ies"):
            return word_lower[:-3] + "y"
        elif word_lower.endswith(("ches", "shes", "xes", "zes", "ses")):
            return word_lower[:-2]
        elif word_lower.endswith("s") and not word_lower.endswith(("ss", "us")):
            return word_lower[:-1]
        
        # 处理动词时态
        if word_lower.endswith("ied"):
            return word_lower[:-3] + "y"
        elif word_lower.endswith("ed") and len(word_lower) > 3:
            return word_lower[:-2]
        elif word_lower.endswith("ing") and len(word_lower) > 4:
            # 处理双写辅音
            if len(word_lower) > 5 and word_lower[-5] == word_lower[-4]:
                return word_lower[:-4]
            return word_lower[:-3]
        
        return word_lower
    
    def tag_pos(self, words: List[str]) -> List[str]:
        """
        词性标注（简化规则版）
        
        Args:
            words: 单词列表
            
        Returns:
            词性标签列表
        """
        pos_tags = []
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            pos = self._determine_pos(word_lower, i, words)
            pos_tags.append(pos)
        
        return pos_tags
    
    def _determine_pos(self, word: str, index: int, context: List[str]) -> str:
        """确定单个词的词性"""
        # 检查常见功能词
        if word in ["the", "a", "an"]:
            return "DT"  # 限定词
        elif word in ["is", "are", "was", "were", "be", "been", "being"]:
            return "VB"  # 动词
        elif word in ["in", "on", "at", "to", "for", "with", "by"]:
            return "IN"  # 介词
        elif word in ["and", "or", "but"]:
            return "CC"  # 连词
        elif word in ["what", "who", "where", "when", "why", "how"]:
            return "WP"  # 疑问词
        
        # 检查后缀
        for suffix in self.pos_rules["adv_suffixes"]:
            if word.endswith(suffix):
                return "RB"  # 副词
        
        for suffix in self.pos_rules["adj_suffixes"]:
            if word.endswith(suffix):
                return "JJ"  # 形容词
        
        for suffix in self.pos_rules["verb_suffixes"]:
            if word.endswith(suffix):
                return "VB"  # 动词
        
        for suffix in self.pos_rules["noun_suffixes"]:
            if word.endswith(suffix):
                return "NN"  # 名词
        
        # 默认假设
        if word.endswith("s") and len(word) > 2:
            return "NNS"  # 复数名词
        elif word.endswith("ed") or word.endswith("ing"):
            return "VBG" if word.endswith("ing") else "VBD"  # 动词形式
        
        return "NN"  # 默认名词
    
    def disambiguate_sense(self, word: str, context: str) -> Tuple[str, float]:
        """
        词义消歧（简化版）
        
        Args:
            word: 目标单词
            context: 上下文
            
        Returns:
            (词义, 置信度)
        """
        word_lower = word.lower()
        context_lower = context.lower()
        
        # 常见多义词的简单消歧规则
        disambiguation_rules = {
            "bank": {
                "river": "financial_institution",
                "money": "financial_institution",
                "account": "financial_institution",
                "water": "river_side",
                "stream": "river_side",
            },
            "run": {
                "fast": "move_quickly",
                "business": "operate",
                "program": "execute",
            },
            "book": {
                "read": "publication",
                "reserve": "reserve",
                "ticket": "reserve",
            }
        }
        
        if word_lower in disambiguation_rules:
            for cue, sense in disambiguation_rules[word_lower].items():
                if cue in context_lower:
                    return sense, 0.8
        
        return "default_sense", 0.5
    
    def explain_word(self, word: str, context: str = "") -> WordExplanation:
        """
        完整单词解释
        
        Args:
            word: 单词
            context: 上下文
            
        Returns:
            WordExplanation
        """
        lemma = self.lemmatize(word)
        pos = self.tag_pos([word])[0]
        sense, confidence = self.disambiguate_sense(word, context)
        
        return WordExplanation(
            original=word,
            lemma=lemma,
            pos=pos,
            sense=sense,
            confidence=confidence
        )


class PhraseExplanationLayer:
    """
    短语解释层 (Layer 2)
    
    功能：
    1. 短语识别 (Phrase Identification)
    2. 短语语义组合 (Semantic Composition)
    """
    
    def __init__(self):
        self.word_layer = WordExplanationLayer()
        
        # 短语识别模式
        self.phrase_patterns = {
            "NP": {
                "pattern": r"(DT\s+)?(JJ\s+)*(NN|NNS)",
                "description": "名词短语: (限定词) + (形容词)* + 名词"
            },
            "VP": {
                "pattern": r"(VB|VBD|VBG|VBN|VBP|VBZ)\s+(NP)?",
                "description": "动词短语: 动词 + (名词短语)?"
            },
            "PP": {
                "pattern": r"IN\s+NP",
                "description": "介词短语: 介词 + 名词短语"
            }
        }
    
    def identify_phrases(self, words: List[str], pos_tags: List[str]) -> List[PhraseExplanation]:
        """
        识别短语
        
        Args:
            words: 单词列表
            pos_tags: 词性标签列表
            
        Returns:
            短语解释列表
        """
        phrases = []
        i = 0
        
        while i < len(words):
            # 尝试识别名词短语
            np_phrase = self._identify_np(words, pos_tags, i)
            if np_phrase:
                phrases.append(np_phrase)
                i += len(np_phrase.modifiers) + 1
                continue
            
            # 尝试识别动词短语
            vp_phrase = self._identify_vp(words, pos_tags, i)
            if vp_phrase:
                phrases.append(vp_phrase)
                i += 1
                continue
            
            i += 1
        
        return phrases
    
    def _identify_np(self, words: List[str], pos_tags: List[str], start: int) -> Optional[PhraseExplanation]:
        """识别名词短语"""
        if start >= len(words):
            return None
        
        # 简单规则：限定词 + 形容词* + 名词
        i = start
        modifiers = []
        head = None
        
        # 检查限定词
        if pos_tags[i] == "DT":
            modifiers.append(words[i])
            i += 1
        
        # 收集形容词
        while i < len(words) and pos_tags[i] == "JJ":
            modifiers.append(words[i])
            i += 1
        
        # 检查名词
        if i < len(words) and pos_tags[i] in ["NN", "NNS"]:
            head = words[i]
            phrase_text = " ".join(words[start:i+1])
            
            return PhraseExplanation(
                phrase_text=phrase_text,
                phrase_type="NP",
                head_word=head,
                modifiers=modifiers,
                semantic_composition=f"HEAD:{head}, MODS:{modifiers}"
            )
        
        return None
    
    def _identify_vp(self, words: List[str], pos_tags: List[str], start: int) -> Optional[PhraseExplanation]:
        """识别动词短语"""
        if start >= len(words):
            return None
        
        # 检查动词
        if pos_tags[start] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            head = words[start]
            
            return PhraseExplanation(
                phrase_text=head,
                phrase_type="VP",
                head_word=head,
                modifiers=[],
                semantic_composition=f"VERB:{head}"
            )
        
        return None
    
    def compose_phrase_meaning(self, phrase: PhraseExplanation, word_explanations: List[WordExplanation]) -> str:
        """
        组合短语语义
        
        Args:
            phrase: 短语
            word_explanations: 单词解释列表
            
        Returns:
            组合后的语义描述
        """
        head_meaning = phrase.head_word
        modifier_meanings = [we.lemma for we in word_explanations if we.original in phrase.modifiers]
        
        if phrase.phrase_type == "NP":
            return f"{head_meaning}({', '.join(modifier_meanings)})"
        elif phrase.phrase_type == "VP":
            return f"ACTION:{head_meaning}"
        else:
            return f"{phrase.phrase_type}:{head_meaning}"


class SentenceCompositionLayer:
    """
    句子组合层 (Layer 3)
    
    功能：
    1. 句法解析
    2. 依存关系提取
    3. Task Pattern 映射
    """
    
    def __init__(self):
        self.word_layer = WordExplanationLayer()
        self.phrase_layer = PhraseExplanationLayer()
    
    def parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        解析句子
        
        Args:
            sentence: 输入句子
            
        Returns:
            解析结果
        """
        # 分词
        words = sentence.split()
        
        # 词性标注
        pos_tags = self.word_layer.tag_pos(words)
        
        # 单词解释
        word_explanations = [
            self.word_layer.explain_word(word, sentence)
            for word in words
        ]
        
        # 短语识别
        phrases = self.phrase_layer.identify_phrases(words, pos_tags)
        
        # 提取依存关系
        relations = self._extract_dependencies(words, pos_tags, phrases)
        
        return {
            "sentence": sentence,
            "words": words,
            "pos_tags": pos_tags,
            "word_explanations": word_explanations,
            "phrases": phrases,
            "dependencies": relations
        }
    
    def _extract_dependencies(self, words: List[str], pos_tags: List[str], phrases: List[PhraseExplanation]) -> List[SyntacticRelation]:
        """提取句法依存关系"""
        relations = []
        
        # 简单规则：找主语-谓语-宾语关系
        np_heads = [p.head_word for p in phrases if p.phrase_type == "NP"]
        vp_heads = [p.head_word for p in phrases if p.phrase_type == "VP"]
        
        # 假设第一个NP是主语，VP是谓语，第二个NP是宾语
        if len(np_heads) >= 1 and len(vp_heads) >= 1:
            relations.append(SyntacticRelation(
                relation_type="nsubj",
                head=vp_heads[0],
                dependent=np_heads[0],
                label="主语"
            ))
        
        if len(np_heads) >= 2 and len(vp_heads) >= 1:
            relations.append(SyntacticRelation(
                relation_type="dobj",
                head=vp_heads[0],
                dependent=np_heads[1],
                label="宾语"
            ))
        
        return relations
    
    def map_to_task_pattern(self, sentence: str) -> Optional[str]:
        """
        映射到 Task Pattern
        
        Args:
            sentence: 输入句子
            
        Returns:
            Task Pattern 类型
        """
        sentence_lower = sentence.lower()
        
        task_patterns = {
            "definition": ["what is", "define", "explain", "describe"],
            "comparison": ["difference between", "compare", "vs"],
            "reasoning": ["why", "how come", "what if"],
            "verification": ["is it true", "verify", "check"],
        }
        
        for task_type, triggers in task_patterns.items():
            for trigger in triggers:
                if trigger in sentence_lower:
                    return task_type
        
        return None


class EnglishExplanationPipeline:
    """
    英文解释流水线
    
    整合三层解释层，提供统一接口
    """
    
    def __init__(self):
        self.word_layer = WordExplanationLayer()
        self.phrase_layer = PhraseExplanationLayer()
        self.sentence_layer = SentenceCompositionLayer()
        self.unit_mapper = EnglishUnitMapper()
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        处理英文输入
        
        Args:
            text: 输入文本
            
        Returns:
            完整处理结果
        """
        # 句子级解析
        parse_result = self.sentence_layer.parse_sentence(text)
        
        # 映射到 Unit
        word_input = EnglishWordInput(text=text.split()[0] if text else "")
        concept_unit = self.unit_mapper.map_to_concept_unit(word_input)
        relation_unit = self.unit_mapper.map_to_relation_unit(text)
        rule_unit = self.unit_mapper.map_to_rule_unit(text)
        task_unit = self.unit_mapper.map_to_task_pattern_unit(text)
        
        return {
            "parse_result": parse_result,
            "units": {
                "concept": concept_unit,
                "relation": relation_unit,
                "rule": rule_unit,
                "task": task_unit
            }
        }


def demo_english_explanation_layer():
    """演示英文解释层"""
    print("\n" + "=" * 70)
    print("English Explanation Layer Demo - 英文解释层演示")
    print("=" * 70)
    
    # Word Layer
    print("\n1. Word Explanation Layer")
    print("-" * 40)
    
    word_layer = WordExplanationLayer()
    test_words = [
        ("running", "I am running fast"),
        ("better", "This is better"),
        ("children", "The children are playing"),
        ("bank", "The bank of the river"),
    ]
    
    for word, context in test_words:
        explanation = word_layer.explain_word(word, context)
        print(f"\n  单词: {word}")
        print(f"    原形: {explanation.lemma}")
        print(f"    词性: {explanation.pos}")
        print(f"    词义: {explanation.sense}")
        print(f"    置信度: {explanation.confidence:.2f}")
    
    # Phrase Layer
    print("\n2. Phrase Explanation Layer")
    print("-" * 40)
    
    phrase_layer = PhraseExplanationLayer()
    test_sentences = [
        "the quick brown fox",
        "a beautiful garden",
        "the old man"
    ]
    
    for sentence in test_sentences:
        words = sentence.split()
        pos_tags = word_layer.tag_pos(words)
        phrases = phrase_layer.identify_phrases(words, pos_tags)
        
        print(f"\n  句子: {sentence}")
        print(f"    词性: {pos_tags}")
        for phrase in phrases:
            print(f"    短语: {phrase.phrase_text} ({phrase.phrase_type})")
            print(f"      中心词: {phrase.head_word}")
            print(f"      修饰语: {phrase.modifiers}")
    
    # Sentence Layer
    print("\n3. Sentence Composition Layer")
    print("-" * 40)
    
    sentence_layer = SentenceCompositionLayer()
    test_sentences = [
        "The cat sits on the mat",
        "What is machine learning?",
        "If it rains then the ground will be wet"
    ]
    
    for sentence in test_sentences:
        result = sentence_layer.parse_sentence(sentence)
        task = sentence_layer.map_to_task_pattern(sentence)
        
        print(f"\n  句子: {sentence}")
        print(f"    分词: {result['words']}")
        print(f"    词性: {result['pos_tags']}")
        print(f"    短语数: {len(result['phrases'])}")
        print(f"    依存关系: {len(result['dependencies'])}")
        if task:
            print(f"    任务类型: {task}")
    
    # Full Pipeline
    print("\n4. Full Explanation Pipeline")
    print("-" * 40)
    
    pipeline = EnglishExplanationPipeline()
    test_input = "What is artificial intelligence?"
    
    result = pipeline.process(test_input)
    
    print(f"\n  输入: {test_input}")
    print(f"    Concept Unit: {result['units']['concept'].surface_form if result['units']['concept'] else 'None'}")
    print(f"    Relation Unit: {result['units']['relation'].relation_type if result['units']['relation'] else 'None'}")
    print(f"    Rule Unit: {result['units']['rule'].rule_type if result['units']['rule'] else 'None'}")
    print(f"    Task Unit: {result['units']['task'].task_type if result['units']['task'] else 'None'}")
    
    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


if __name__ == "__main__":
    demo_english_explanation_layer()
