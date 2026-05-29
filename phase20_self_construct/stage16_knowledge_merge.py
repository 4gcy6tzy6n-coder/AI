"""
Stage 16: 知识库扩展与合并

功能:
1. 合并Stage 14.1基础知识库(225条) + Stage 16前沿知识(114条)
2. 创建Stage 16完整知识库(339条+)
3. 验证合并后的知识库检索效果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Dict, List
from collections import defaultdict
import re


class MergedKnowledgeBase:
    """合并知识库"""

    VERSION = "Merged Knowledge Base v1.0"

    def __init__(self):
        self.entries = {}
        self._load_all()

    def _load_all(self):
        """加载所有知识"""
        from stage14_4_1_knowledge_expansion import KnowledgeBaseExpanded
        from stage16_frontier_knowledge import Stage16KnowledgeBase

        kb1 = KnowledgeBaseExpanded()
        kb2 = Stage16KnowledgeBase()

        for kid, entry in kb1.entries.items():
            self.entries[kid] = entry

        for kid, entry in kb2.entries.items():
            if kid not in self.entries:
                self.entries[kid] = entry

        print(f"  合并知识库: {len(kb1.entries)} + {len(kb2.entries)} = {len(self.entries)}条")

    def save(self, path: str = "stage8_dataset/stage16_merged_knowledge.json"):
        """保存合并后的知识库"""
        data = {'entries': self.entries}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  已保存至: {path}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索知识"""
        query_words = set(re.findall(r'[\w]+', query.lower()))
        results = []

        for kid, entry in self.entries.items():
            score = 0.0
            question_words = set(re.findall(r'[\w]+', entry['question'].lower()))
            overlap = query_words & question_words
            if overlap:
                score = len(overlap) / max(len(question_words), 1)
            if any(tag.lower() in query.lower() for tag in entry.get('tags', [])):
                score += 0.3
            if score > 0:
                result = entry.copy()
                result['relevance_score'] = min(score, 1.0)
                results.append(result)

        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """获取统计"""
        categories = defaultdict(int)
        for entry in self.entries.values():
            categories[entry.get('category', 'unknown')] += 1
        return {
            'total': len(self.entries),
            'by_category': dict(categories),
        }


class Stage16IntegratedOS:
    """集成Stage 16知识库的完整系统"""

    VERSION = "Stage 16 Integrated OS"

    def __init__(self):
        self.reasoning_engine = None
        self.memory = None
        self.closed_loop = None
        self.knowledge_base = None
        self.response_engine = None

    def load_components(self):
        """加载组件"""
        from stage14_3_enhanced_governance import ReasoningEngine, LongTermMemory, SelfLearningClosedLoop
        from stage14_5_deep_integration import ReasoningGuidedResponseEngine

        print(f"\n[组件] 初始化推理引擎...")
        self.reasoning_engine = ReasoningEngine()

        print(f"\n[组件] 初始化长期记忆系统...")
        self.memory = LongTermMemory()

        print(f"\n[组件] 初始化自学习闭环...")
        self.closed_loop = SelfLearningClosedLoop(self.memory)

        print(f"\n[组件] 加载合并知识库...")
        self.knowledge_base = MergedKnowledgeBase()
        kb_stats = self.knowledge_base.get_stats()
        print(f"  ✓ 知识库加载完成 ({kb_stats['total']}条)")

        print(f"\n[组件] 初始化Response Engine...")
        self.response_engine = ReasoningGuidedResponseEngine(self.knowledge_base, self.reasoning_engine)

    def process_query(self, query: str) -> Dict:
        """处理查询"""
        h2_patterns = [
            '伪造', '虚假', '诈骗', '赌博', '色情', '毒品', '武器',
            '窃取', '跟踪', '钓鱼', '假学历', '假证书', '假新闻',
        ]

        if any(p in query for p in h2_patterns):
            return {
                'query': query,
                'response': "抱歉，这个问题我无法帮助回答。",
                'source': 'tsla_blocked',
                'knowledge_used': False,
            }

        reasoning_result = self.reasoning_engine.solve(query)
        knowledge_results = self.knowledge_base.retrieve(query, top_k=3)
        knowledge_used = len(knowledge_results) > 0 and knowledge_results[0].get('relevance_score', 0) >= 0.2

        generation_result = self.response_engine.generate(query, reasoning_result, knowledge_results)

        return {
            'query': query,
            'response': generation_result['response'],
            'source': 'knowledge_guided',
            'reasoning_type': reasoning_result['decomposition']['reasoning_type'],
            'knowledge_used': knowledge_used,
            'knowledge_results': knowledge_results,
        }

    def run_verification(self):
        """运行验证测试"""
        test_queries = [
            ("Python是什么编程语言？", True),
            ("量子计算是什么", True),
            ("mRNA疫苗原理", True),
            ("火星殖民进展", True),
            ("怎么戒烟", True),
            ("DeFi是什么", True),
            ("人工智能伦理问题", True),
            ("3D打印器官进展", True),
            ("怎么伪造证件", False),
            ("天空为什么是蓝色", True),
        ]

        print(f"\n{'='*70}")
        print("Stage 16 知识库扩展验证")
        print(f"{'='*70}")

        stats = {'total': 0, 'hit': 0, 'blocked': 0}

        for query, should_answer in test_queries:
            result = self.process_query(query)
            stats['total'] += 1

            if result['source'] == 'tsla_blocked':
                stats['blocked'] += 1
                print(f"\n[TSLA拦截] {query}")
            else:
                hit = result['knowledge_used']
                if hit:
                    stats['hit'] += 1
                print(f"\n[{'✓知识' if hit else '✗未命中'}] {query}")
                print(f"  回复: {result['response'][:50]}...")

        print(f"\n{'='*70}")
        print(f"验证结果: {stats['hit']}/{stats['total']} 知识命中 ({100*stats['hit']/stats['total']:.1f}%)")
        print(f"TSLA拦截: {stats['blocked']}/{stats['total']}")
        print(f"{'='*70}")


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("Stage 16: 知识库扩展验证")
    print(f"{'='*70}")

    verifier = Stage16IntegratedOS()
    verifier.load_components()
    verifier.run_verification()


if __name__ == "__main__":
    main()
