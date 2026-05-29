"""
Stage 18.2: 知识命中率优化

目标: 从78.6%提升到90%+

优化策略:
1. 检索增强 (Retrieval Enhancement)
   - 添加查询改写/扩展
   - 同义词匹配
   - 口语化转专业术语
   - 模糊匹配

2. 评分优化 (Scoring Optimization)
   - 多特征融合评分
   - 知识类别权重
   - 标签命中加分

3. 知识库结构化
   - 问题模板
   - 同义表达
   - 别名索引
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import re
from typing import Dict, List, Tuple


class EnhancedRetrieval:
    """增强检索系统"""

    VERSION = "Enhanced Retrieval v1.0"

    SYNONYMS = {
        '怎么': ['如何', '怎样', '怎么样', '啥方法', '什么方法'],
        '什么': ['啥', '哪个', '哪些', '哪些是'],
        '为什么': ['为何', '原因为何', '原因是'],
        '是否': ['能不能', '可不可以', '能不能够'],
        '股票': ['股市', 'A股', '炒股'],
        '基金': ['公募基金', '私募基金', '证券投资基金'],
        '买房': ['购房', '买房子', '置产'],
        '简历': ['履历', 'CV'],
        '面试': ['面谈', '口试'],
        '健康': ['身体健康', '保健', '养生'],
        '投资': ['理财', '资产管理'],
        'Python': ['python', 'Python编程'],
        '深度学习': ['deep learning', 'DeepLearning'],
        '机器学习': ['machine learning', 'ML'],
        '量子计算': ['量子计算机', '量子信息'],
        'AI': ['人工智能', 'Artificial Intelligence'],
        '区块链': ['blockchain', '分布式账本'],
    }

    QUESTION_PATTERNS = [
        (r'.*是什么.*', 'definition'),
        (r'.*有什么区别.*', 'comparison'),
        (r'.*怎么办.*', 'steps'),
        (r'.*为什么.*', 'causal'),
        (r'.*有哪些.*', 'list'),
        (r'.*如何.*', 'steps'),
    ]

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.query_cache = {}

    def expand_query(self, query: str) -> List[str]:
        """扩展查询"""
        expansions = [query]

        generic_terms = ['怎么', '什么', '如何', '怎样', '是不是', '能不能', '可不可以']
        for key in generic_terms:
            if key in query:
                query_without_generic = query.replace(key, '').strip()
                if query_without_generic and query_without_generic not in expansions:
                    expansions.append(query_without_generic)

        words = query.split()
        if len(words) >= 2:
            for i in range(len(words)):
                for j in range(i+1, len(words)+1):
                    phrase = ' '.join(words[i:j])
                    if phrase not in expansions and len(phrase) > 4:
                        expansions.append(phrase)

        return expansions[:8]

    def get_question_type(self, query: str) -> str:
        """获取问题类型"""
        for pattern, qtype in self.QUESTION_PATTERNS:
            if re.search(pattern, query):
                return qtype
        return 'direct'

    def score_match(self, entry: Dict, query: str, expanded_queries: List[str]) -> float:
        """计算匹配分数"""
        score = 0.0

        GENERIC_TAGS = ['方法', '技术', '原理', '基础', '基本', '介绍', '概述', '学习']
        STOPWORDS = {'怎么', '什么', '如何', '为什么', '是否', '吗', '呢', '吧', '啊', '学习'}

        entry_question = entry.get('question', '').lower()
        entry_tags = [t.lower() for t in entry.get('tags', [])]
        entry_answer = entry.get('answer', '').lower()
        entry_category = entry.get('category', '').lower()

        original_query_lower = query.lower()
        filtered_query_words = [w for w in original_query_lower.split() if w not in STOPWORDS and len(w) > 1]
        filtered_query_set = set(filtered_query_words)

        for exp_q in expanded_queries:
            exp_q_lower = exp_q.lower()

            if exp_q_lower in entry_question:
                score += 0.5
            if exp_q_lower in entry_answer:
                score += 0.2

            exp_words = set(re.findall(r'[\w]+', exp_q_lower))
            entry_words = set(re.findall(r'[\w]+', entry_question))

            filtered_exp_words = exp_words - STOPWORDS
            overlap = filtered_exp_words & entry_words
            if overlap:
                overlap_ratio = len(overlap) / max(len(filtered_exp_words), 1)
                score += 0.3 * overlap_ratio

        for tag in entry_tags:
            if tag in GENERIC_TAGS:
                continue
            if tag in original_query_lower:
                score += 0.4
            for exp_q in expanded_queries:
                if tag in exp_q.lower():
                    score += 0.3
                    break

        for word in filtered_query_set:
            if len(word) > 2:
                for tag in entry_tags:
                    if tag in GENERIC_TAGS:
                        continue
                    if word in tag or tag in word:
                        score += 0.15

        category_weights = {
            'programming': 0.1, 'frontier': 0.1, 'finance': 0.1,
            'career': 0.1, 'health': 0.1, 'education': 0.1,
        }
        if entry_category in category_weights:
            score += category_weights[entry_category]

        return min(score, 1.0)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索"""
        if query in self.query_cache:
            cached = self.query_cache[query]
            return cached[:top_k]

        expanded = self.expand_query(query)

        candidates = []
        for kid, entry in self.kb.entries.items():
            score = self.score_match(entry, query, expanded)
            if score > 0:
                result = entry.copy()
                result['relevance_score'] = score
                result['match_type'] = self._get_match_type(entry, query, expanded)
                candidates.append(result)

        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        self.query_cache[query] = candidates
        return candidates[:top_k]

    def _get_match_type(self, entry: Dict, query: str, expanded: List[str]) -> str:
        """获取匹配类型"""
        query_lower = query.lower()
        entry_question = entry.get('question', '').lower()

        if query_lower in entry_question:
            return 'exact_question'

        for exp in expanded:
            if exp.lower() in entry_question:
                return 'expanded_match'

        entry_tags = [t.lower() for t in entry.get('tags', [])]
        for tag in entry_tags:
            if tag in query_lower:
                return 'tag_match'

        return 'semantic_match'


class Stage18KnowledgeOptimizer:
    """Stage 18 知识优化器"""

    VERSION = "Stage 18 Knowledge Optimizer v1.0"

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.enhanced_retrieval = EnhancedRetrieval(knowledge_base)

    def analyze_failures(self, queries: List[str]) -> Dict:
        """分析失败案例"""
        failures = []
        for query in queries:
            results = self.enhanced_retrieval.retrieve(query, top_k=3)
            hit = len(results) > 0 and results[0].get('relevance_score', 0) >= 0.3

            if not hit:
                failures.append({
                    'query': query,
                    'top_score': results[0].get('relevance_score', 0) if results else 0,
                    'top_match': results[0].get('question', 'N/A')[:30] if results else 'N/A',
                    'reason': self._analyze_failure_reason(query, results),
                })

        return {
            'total': len(queries),
            'failures': failures,
            'failure_rate': len(failures) / len(queries) if queries else 0,
        }

    def _analyze_failure_reason(self, query: str, results: List[Dict]) -> str:
        """分析失败原因"""
        if not results:
            return 'no_candidates'

        top_score = results[0].get('relevance_score', 0)
        if top_score < 0.1:
            return 'low_overlap'
        elif top_score < 0.3:
            return 'threshold_cut'
        else:
            return 'topic_mismatch'

    def add_aliases_to_knowledge(self):
        """为知识库添加别名"""
        alias_additions = {
            'stock_market_basics': {'aliases': ['炒股入门', '股市投资', '如何买股票']},
            'etf_introduction': {'aliases': ['指数基金', 'ETF基金', '交易所交易基金']},
            'resume_writing': {'aliases': ['简历制作', '求职简历', '怎么写简历']},
            'exercise_guidelines': {'aliases': ['如何锻炼', '健身方法', '运动建议']},
            'nutrition_basics': {'aliases': ['健康饮食', '怎么吃健康', '营养均衡']},
        }

        added = 0
        for kid, info in alias_additions.items():
            if kid in self.kb.entries:
                if 'aliases' not in self.kb.entries[kid]:
                    self.kb.entries[kid]['aliases'] = []
                for alias in info['aliases']:
                    if alias not in self.kb.entries[kid]['aliases']:
                        self.kb.entries[kid]['aliases'].append(alias)
                        added += 1

        return added


def run_optimization_test():
    """运行优化测试"""
    print(f"\n{'='*70}")
    print("Stage 18.2: 知识命中率优化")
    print(f"{'='*70}")

    from stage17_complete_system import MergedKnowledgeBase
    kb = MergedKnowledgeBase()

    optimizer = Stage18KnowledgeOptimizer(kb)
    enhancer = optimizer.enhanced_retrieval

    test_queries = [
        "量子计算是什么",
        "mRNA疫苗原理",
        "DeFi是什么",
        "怎么投资股票",
        "ETF和基金区别",
        "如何买房",
        "简历怎么写",
        "怎么保持健康",
        "瑜伽练习方法",
        "高考志愿填报",
        "面试准备技巧",
        "人工智能伦理",
    ]

    print(f"\n[增强检索测试]")
    print(f"-" * 50)

    hits = 0
    for q in test_queries:
        results = enhancer.retrieve(q, top_k=3)
        hit = len(results) > 0 and results[0].get('relevance_score', 0) >= 0.3

        if hit:
            hits += 1
            print(f"  [✓ {results[0]['relevance_score']:.2f}] {q}")
            print(f"      → {results[0]['question']} ({results[0].get('match_type', 'N/A')})")
        else:
            print(f"  [✗] {q}")
            if results:
                print(f"      最佳: {results[0]['question']} ({results[0]['relevance_score']:.2f})")

    hit_rate = 100 * hits / len(test_queries) if test_queries else 0
    print(f"\n[结果] {hits}/{len(test_queries)} 命中 ({hit_rate:.1f}%)")

    print(f"\n[查询扩展示例]")
    expanded = enhancer.expand_query("量子计算是什么")
    print(f"  原始: 量子计算是什么")
    print(f"  扩展: {', '.join(expanded[:5])}")

    return optimizer, enhancer


def main():
    """主函数"""
    optimizer, enhancer = run_optimization_test()

    print(f"\n{'='*70}")
    print("Stage 18.2 优化完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
