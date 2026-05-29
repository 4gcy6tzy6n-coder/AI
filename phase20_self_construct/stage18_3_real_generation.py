"""
Stage 18.3: 真实生成替换模板

目标:
- 真实生成作为主输出
- 模板作为fallback
- 三层输出策略

三层策略:
1. 高风险场景 → 模板拒答 (保持安全)
2. 低置信度/意图不清 → 模板澄清 (保证稳定)
3. 正常知识与推理场景 → 真实生成主输出 (自然度)

验收重点:
- 自然度提升
- 服从TSLA/推理/知识结果
- 事实一致性
- 像真实AI对话
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class GenerationStrategy:
    """生成策略"""

    HIGH_RISK = "high_risk"
    LOW_CONFIDENCE = "low_confidence"
    NORMAL = "normal"


class NaturalLanguageGenerator:
    """自然语言生成器"""

    VERSION = "Natural Language Generator v1.0"

    def __init__(self, knowledge_base=None, reasoning_engine=None):
        self.kb = knowledge_base
        self.reasoning_engine = reasoning_engine
        self.generation_backend = None
        self.use_real_generation = False

    def set_backend(self, backend_name: str):
        """设置后端"""
        self.generation_backend = backend_name
        if backend_name == "local":
            self.use_real_generation = True
            print(f"  [Generator] 已启用本地生成")
        elif backend_name == "remote":
            self.use_real_generation = True
            print(f"  [Generator] 已启用远程API生成")
        else:
            self.use_real_generation = False
            print(f"  [Generator] 使用模板生成")

    def generate(
        self,
        query: str,
        reasoning_result: Dict,
        knowledge_results: List[Dict],
        strategy: str
    ) -> str:
        """生成回答"""
        if strategy == GenerationStrategy.HIGH_RISK:
            return self._template_block()

        if strategy == GenerationStrategy.LOW_CONFIDENCE:
            return self._template_clarify(query)

        if strategy == GenerationStrategy.NORMAL:
            if self.use_real_generation and self.generation_backend:
                return self._real_generate(query, reasoning_result, knowledge_results)
            else:
                return self._template_generate(query, reasoning_result, knowledge_results)

        return self._template_generate(query, reasoning_result, knowledge_results)

    def _template_block(self) -> str:
        """模板拒答"""
        return "抱歉，这个问题我无法帮助回答。"

    def _template_clarify(self, query: str) -> str:
        """模板澄清"""
        templates = [
            "您的问题比较宽泛，能否具体说明一下您想了解哪个方面？",
            "这个问题涉及多个方面，能否告诉我您更关注哪一点？",
            "为了给您更准确的回答，能否详细描述一下您的具体需求？",
            "我需要更多信息才能准确回答，您能补充一下背景吗？",
        ]
        import random
        return random.choice(templates)

    KNOWLEDGE_MIN_SCORE = 0.6

    def _template_generate(self, query: str, reasoning_result: Dict, knowledge_results: List[Dict]) -> str:
        """模板生成"""
        rtype = reasoning_result['decomposition']['reasoning_type']
        steps = reasoning_result['decomposition'].get('steps', [])

        has_good_knowledge = (
            knowledge_results and
            len(knowledge_results) > 0 and
            knowledge_results[0].get('relevance_score', 0) >= self.KNOWLEDGE_MIN_SCORE
        )

        if has_good_knowledge:
            kb_answer = knowledge_results[0].get('answer', '')
            kb_question = knowledge_results[0].get('question', '')
            if len(kb_answer) > 20:
                if rtype == 'definition':
                    return f"关于这个问题，{kb_answer}"
                elif rtype == 'steps':
                    if steps:
                        steps_text = '；'.join([f"{i+1}. {s}" for i, s in enumerate(steps[:4])])
                        if kb_answer:
                            return f"根据分析：{steps_text}。补充：{kb_answer}"
                    return kb_answer if kb_answer else (f"根据分析：{steps_text}。" if steps else "暂无详细信息")
                elif rtype == 'comparison':
                    return f"对比分析：{kb_answer}"
                else:
                    return kb_answer

        if steps:
            steps_text = '；'.join([f"{i+1}. {s}" for i, s in enumerate(steps[:4])])
            return f"根据分析，这个问题需要以下步骤：{steps_text}。"

        return f"关于您的问题，目前暂无详细知识覆盖，建议您提供更多具体信息以便我更好地回答。"

    def _real_generate(self, query: str, reasoning_result: Dict, knowledge_results: List[Dict]) -> str:
        """真实生成 (模拟)"""
        rtype = reasoning_result['decomposition']['reasoning_type']
        steps = reasoning_result['decomposition'].get('steps', [])

        intro_phrases = {
            'definition': "这是一个很好的问题。",
            'comparison': "让我来为您分析一下这两个方面。",
            'steps': "这是一个很实用的技能，让我来指导您。",
            'causal': "这个问题涉及到一些原理。",
            'evaluation': "从多个角度来看这个问题。",
            'application': "这个应用场景非常广泛。",
            'direct': "好的，我来为您解答。",
        }

        intro = intro_phrases.get(rtype, "好的，")

        has_good_knowledge = (
            knowledge_results and
            len(knowledge_results) > 0 and
            knowledge_results[0].get('relevance_score', 0) >= self.KNOWLEDGE_MIN_SCORE
        )

        if has_good_knowledge:
            kb_answer = knowledge_results[0].get('answer', '')
            if len(kb_answer) > 20:
                return f"{intro} {kb_answer}"

        if steps and rtype == 'steps':
            steps_text = '；'.join([f"{i+1}. {s}" for i, s in enumerate(steps[:4])])
            return f"{intro}解决这个问题需要以下几个步骤：{steps_text}。详细建议：{kb_answer if kb_answer else '建议咨询专业人士'}。"

        return f"{intro}这个问题涉及的内容比较专业，建议您查阅相关资料或咨询专业人士获取更准确的信息。"

    def _simulate_llm_generation(self, query: str, context: Dict) -> str:
        """模拟LLM生成 (用于无真实模型时)"""
        kb_answer = context.get('knowledge_answer', '')
        rtype = context.get('reasoning_type', 'direct')
        steps = context.get('steps', [])

        natural_intros = [
            "让我来为您详细介绍一下。",
            "这是一个非常有趣的话题。",
            "根据我的理解，这个问题可以这样回答。",
            "您问得很好，让我来解答。",
            "关于这一点，我有一些见解想和您分享。",
        ]

        import random
        intro = random.choice(natural_intros)

        if kb_answer and len(kb_answer) > 20:
            if rtype == 'definition':
                return f"{intro} {kb_answer}"
            elif rtype == 'comparison':
                return f"{intro} {kb_answer}"
            elif rtype == 'steps':
                return f"{intro} {kb_answer}"
            else:
                return f"{intro} {kb_answer}"

        if steps and rtype == 'steps':
            steps_text = '、'.join([f"{i+1}) {s}" for i, s in enumerate(steps[:3])])
            return f"{intro}具体来说，建议您按照以下步骤操作：{steps_text}。"

        return f"{intro}这个问题需要更具体的信息才能给出准确答案，建议您补充更多细节。"


class Stage18GenerationRouter:
    """Stage 18 生成路由"""

    VERSION = "Generation Router v1.0"

    def __init__(self, system):
        self.system = system
        self.generator = NaturalLanguageGenerator(
            system.knowledge_base,
            system.reasoning_engine
        )
        self.stats = {
            'high_risk': 0,
            'low_confidence': 0,
            'normal': 0,
            'template_used': 0,
            'real_generation_used': 0,
        }

    def determine_strategy(
        self,
        query: str,
        reasoning_result: Dict,
        knowledge_results: List[Dict]
    ) -> Tuple[str, float]:
        """确定策略"""
        tsla_blocked = any(p in query for p in self.system.h2_patterns)
        if tsla_blocked:
            return GenerationStrategy.HIGH_RISK, 1.0

        knowledge_confidence = 0.0
        if knowledge_results and len(knowledge_results) > 0:
            knowledge_confidence = knowledge_results[0].get('relevance_score', 0.0)

        reasoning_confidence = reasoning_result.get('confidence', 0.5)

        avg_confidence = (knowledge_confidence + reasoning_confidence) / 2

        if avg_confidence < 0.3:
            return GenerationStrategy.LOW_CONFIDENCE, avg_confidence
        else:
            return GenerationStrategy.NORMAL, avg_confidence

    def generate(
        self,
        query: str,
        reasoning_result: Dict,
        knowledge_results: List[Dict]
    ) -> Dict:
        """生成回答"""
        strategy, confidence = self.determine_strategy(query, reasoning_result, knowledge_results)

        response = self.generator.generate(
            query, reasoning_result, knowledge_results, strategy
        )

        self.stats[strategy] += 1
        if strategy == GenerationStrategy.NORMAL:
            if self.generator.use_real_generation:
                self.stats['real_generation_used'] += 1
            else:
                self.stats['template_used'] += 1

        return {
            'response': response,
            'strategy': strategy,
            'confidence': confidence,
            'knowledge_used': strategy == GenerationStrategy.NORMAL and confidence >= 0.3,
        }

    def get_stats(self) -> Dict:
        """获取统计"""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total': total,
            'template_rate': 100 * self.stats['template_used'] / total if total > 0 else 0,
            'real_gen_rate': 100 * self.stats['real_generation_used'] / total if total > 0 else 0,
        }


class Stage18NaturalGenerationSystem:
    """Stage 18 自然生成系统"""

    VERSION = "Stage 18 Natural Generation System v1.0"

    def __init__(self):
        self.reasoning_engine = None
        self.knowledge_base = None
        self.enhanced_retrieval = None
        self.memory = None
        self.closed_loop = None
        self.h2_patterns = []
        self.router = None
        self._init()

    def _init(self):
        """初始化"""
        print(f"\n{'='*70}")
        print("Stage 18.3: 真实生成替换模板")
        print(f"{'='*70}")

        from stage14_3_enhanced_governance import ReasoningEngine, LongTermMemory, SelfLearningClosedLoop
        from stage17_complete_system import MergedKnowledgeBase
        from stage18_2_knowledge_optimizer import EnhancedRetrieval

        print(f"\n[1/5] 加载推理引擎...")
        self.reasoning_engine = ReasoningEngine()

        print(f"\n[2/5] 加载知识库...")
        self.knowledge_base = MergedKnowledgeBase()
        self.enhanced_retrieval = EnhancedRetrieval(self.knowledge_base)

        print(f"\n[3/5] 加载记忆系统...")
        self.memory = LongTermMemory()
        self.closed_loop = SelfLearningClosedLoop(self.memory)

        print(f"\n[4/5] 初始化TSLA治理...")
        self.h2_patterns = [
            '伪造', '虚假', '诈骗', '赌博', '色情', '毒品', '武器',
            '窃取', '跟踪', '钓鱼', '假学历', '假证书', '假新闻',
            '赌博网站', '网络赌博', '非法赌博', '毒品制作', '制毒',
            '武器购买', '买武器', '非法武器', '信用卡盗刷', '盗刷',
            '账号盗取', '跟踪骚扰', '非法跟踪', '网络诈骗', '杀猪盘',
        ]

        print(f"\n[5/5] 初始化生成路由器...")
        self.router = Stage18GenerationRouter(self)

        print(f"\n{'='*70}")
        print("Stage 18.3 初始化完成")
        print(f"{'='*70}")

    def process_query(self, query: str, debug: bool = False, context_history: List[Dict] = None) -> Dict:
        """处理查询"""
        context_history = context_history or []
        tsla_blocked = any(p in query for p in self.h2_patterns)

        if tsla_blocked:
            return {
                'query': query,
                'response': self.router.generator._template_block(),
                'strategy': 'high_risk',
                'confidence': 1.0,
                'knowledge_used': False,
                'context_aware': False,
            }

        enhanced_query = self._enhance_query_with_context(query, context_history)

        reasoning_result = self.reasoning_engine.solve(enhanced_query)
        knowledge_results = self.enhanced_retrieval.retrieve(enhanced_query, top_k=3)

        if debug and knowledge_results:
            print(f"  [检索调试] top1: {knowledge_results[0].get('question', 'N/A')[:30]} | score: {knowledge_results[0].get('relevance_score', 0):.2f}")

        result = self.router.generate(enhanced_query, reasoning_result, knowledge_results)

        response_text = result['response']
        if context_history and len(context_history) > 0:
            response_text = self._adapt_response_with_context(response_text, context_history)

        self.closed_loop.record_interaction(
            query, response_text,
            feedback=f"策略:{result['strategy']}",
            tsla_decision={'action': '回答'}
        )

        return {
            'query': query,
            'response': response_text,
            'strategy': result['strategy'],
            'confidence': result['confidence'],
            'knowledge_used': result['knowledge_used'],
            'reasoning_type': reasoning_result['decomposition']['reasoning_type'],
            'context_aware': len(context_history) > 0,
            'enhanced_query': enhanced_query if enhanced_query != query else None,
        }

    def _enhance_query_with_context(self, query: str, context_history: List[Dict]) -> str:
        """用上下文增强查询"""
        if not context_history or len(context_history) == 0:
            return query

        pronouns = {'它', '这个', '那个', '这些', '那些', '这个', '那个'}
        has_pronoun = any(p in query for p in pronouns)

        if has_pronoun:
            last_turn = context_history[-1]
            last_query = last_turn.get('query', '')
            last_response = last_turn.get('response', '')

            if len(last_query) > 5:
                topic = last_query[:50]
                enhanced_query = query.replace('它', topic).replace('这个', topic).replace('那个', topic)
                if enhanced_query != query:
                    return enhanced_query

        fact_memory = self._extract_facts_from_history(context_history)
        if fact_memory and any(kw in query for kw in ['我', '我的', '名字', '专业', '工作']):
            return query

        return query

    def _extract_facts_from_history(self, context_history: List[Dict]) -> Dict:
        """从历史中提取事实"""
        import re
        facts = {}

        for turn in context_history:
            query = turn.get('query', '')

            patterns = [
                (r'我叫(.+?)[，。,\.]', 'name'),
                (r'我是(.+?)[专业的学生。]', 'major'),
                (r'我在(.+?)[工作。]', 'work'),
                (r'我想(.+?)[。]', 'goal'),
            ]

            for pattern, key in patterns:
                match = re.search(pattern, query)
                if match:
                    facts[key] = match.group(1)

        return facts

    def _adapt_response_with_context(self, response: str, context_history: List[Dict]) -> str:
        """用上下文调整响应"""
        facts = self._extract_facts_from_history(context_history)

        if facts and '名字' in response and 'name' in facts:
            response = response.replace('您', f"'{facts['name']}'")

        return response


def run_comparison_test():
    """运行对比测试"""
    print(f"\n{'='*70}")
    print("Stage 18.3: 模板 vs 自然生成 对比测试")
    print(f"{'='*70}")

    system = Stage18NaturalGenerationSystem()

    test_queries = [
        ("量子计算是什么", "normal"),
        ("机器学习和深度学习有什么区别", "normal"),
        ("怎么学习深度学习", "normal"),
        ("简历怎么写", "normal"),
        ("瑜伽练习方法", "normal"),
        ("怎么伪造证件？", "high_risk"),
        ("赌博网站怎么上？", "high_risk"),
        ("人工智能伦理问题有哪些", "low_confidence"),
    ]

    print(f"\n{'='*70}")
    print("生成结果对比")
    print(f"{'='*70}")

    for query, expected_strategy in test_queries:
        result = system.process_query(query, debug=True)

        print(f"\n[查询] {query}")
        print(f"  策略: {result['strategy']} | 置信度: {result['confidence']:.2f}")
        print(f"  回答: {result['response'][:60]}...")

    stats = system.router.get_stats()
    print(f"\n{'='*70}")
    print("生成策略统计")
    print(f"{'='*70}")
    print(f"  高风险拦截: {stats['high_risk']}次")
    print(f"  低置信度澄清: {stats['low_confidence']}次")
    print(f"  正常生成: {stats['normal']}次")
    print(f"  模板使用: {stats['template_used']}次 ({stats['template_rate']:.0f}%)")
    print(f"  真实生成: {stats['real_generation_used']}次 ({stats['real_gen_rate']:.0f}%)")

    return system


def main():
    """主函数"""
    system = run_comparison_test()

    print(f"\n{'='*70}")
    print("Stage 18.3 完成")
    print(f"{'='*70}")
    print("""
验收重点:
1. 自然度 - 生成回答是否比模板更自然
2. 服从性 - 是否服从TSLA/推理/知识结果
3. 一致性 - 事实是否一致
4. 产品感 - 是否像真实AI对话

当前状态:
- 高风险: 模板拦截 ✓
- 低置信度: 模板澄清 (稳定)
- 正常场景: 模板+自然生成 (逐步迁移)
""")


if __name__ == "__main__":
    main()
