"""
Stage 14.4: 知识接入

目标:
1. 构建知识库 - 高质量、结构化知识源
2. 查询与知识映射 - 知识条目匹配
3. 知识辅助生成 - Response Engine调用知识内容
4. 闭环验证 - TSLA高风险+推理输出+记忆更新

基于Stage 14.3 Governance OS架构
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import re


class KnowledgeBase:
    """知识库"""

    VERSION = "Knowledge Base v1.0"

    def __init__(self, storage_path: str = "stage8_dataset/knowledge_base.json"):
        self.storage_path = storage_path
        self.entries = {}
        self.categories = defaultdict(list)
        self.tags = defaultdict(list)
        self.load()

    def load(self):
        """加载知识库"""
        if Path(self.storage_path).exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entries = data.get('entries', {})
                    for k, v in self.entries.items():
                        if 'category' in v:
                            self.categories[v['category']].append(k)
                        if 'tags' in v:
                            for tag in v['tags']:
                                self.tags[tag].append(k)
                print(f"  知识库已加载: {len(self.entries)}条")
            except Exception as e:
                print(f"  知识库加载失败: {e}")
                self._init_default_knowledge()
        else:
            self._init_default_knowledge()

    def _init_default_knowledge(self):
        """初始化默认知识"""
        self.entries = {
            'python_def': {
                'id': 'python_def',
                'question': 'Python是什么',
                'answer': 'Python是一种高级编程语言，由Guido van Rossum于1991年发明。它语法简洁、易学易用，广泛用于Web开发、数据分析、人工智能、科学计算等领域。',
                'category': 'programming',
                'tags': ['Python', '编程语言', '入门'],
                'confidence': 0.95,
            },
            'ml_def': {
                'id': 'ml_def',
                'question': '什么是机器学习',
                'answer': '机器学习是人工智能的一个子领域，通过让计算机从数据中学习模式来进行预测和决策，而不需要明确的编程规则。常见算法包括监督学习、无监督学习和强化学习。',
                'category': 'ai',
                'tags': ['机器学习', 'AI', '算法'],
                'confidence': 0.95,
            },
            'dl_def': {
                'id': 'dl_def',
                'question': '什么是深度学习',
                'answer': '深度学习是机器学习的子集，使用多层神经网络来处理复杂数据。它在图像识别、自然语言处理、计算机视觉等领域取得了突破性进展。',
                'category': 'ai',
                'tags': ['深度学习', '神经网络', 'AI'],
                'confidence': 0.95,
            },
            'ai_def': {
                'id': 'ai_def',
                'question': '什么是人工智能',
                'answer': '人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的技术。包括机器学习、自然语言处理、计算机视觉等多个研究方向。',
                'category': 'ai',
                'tags': ['人工智能', 'AI', '定义'],
                'confidence': 0.95,
            },
            'http_def': {
                'id': 'http_def',
                'question': '什么是HTTP协议',
                'answer': 'HTTP是超文本传输协议，是Web通信的基础。它是请求-响应协议，常见方法有GET、POST、PUT、DELETE。HTTP是无状态的协议。',
                'category': 'network',
                'tags': ['HTTP', '协议', '网络'],
                'confidence': 0.90,
            },
            'database_def': {
                'id': 'database_def',
                'question': '什么是数据库',
                'answer': '数据库是存储和管理数据的系统。常见的数据库类型包括关系型数据库(如MySQL、PostgreSQL)和非关系型数据库(如MongoDB、Redis)。',
                'category': 'storage',
                'tags': ['数据库', '存储', 'SQL'],
                'confidence': 0.90,
            },
            'git_def': {
                'id': 'git_def',
                'question': '什么是Git',
                'answer': 'Git是分布式版本控制系统，用于跟踪代码变更、协作开发和代码管理。常见命令包括git add、git commit、git push、git pull。',
                'category': 'tools',
                'tags': ['Git', '版本控制', '开发工具'],
                'confidence': 0.90,
            },
            'algorithm_def': {
                'id': 'algorithm_def',
                'question': '什么是算法',
                'answer': '算法是解决问题的有限步骤集合。算法复杂度用时间复杂度和空间复杂度衡量，常用表示法有大O表示法。常见算法包括排序、搜索、图算法等。',
                'category': 'programming',
                'tags': ['算法', '复杂度', '数据结构'],
                'confidence': 0.90,
            },
            'cloud_def': {
                'id': 'cloud_def',
                'question': '什么是云计算',
                'answer': '云计算是一种通过互联网提供计算资源的服务模式。用户可以按需使用服务器、存储、数据库等资源，无需自行维护硬件。常见服务模式包括IaaS、PaaS、SaaS。',
                'category': 'network',
                'tags': ['云计算', '云服务', 'IaaS', 'PaaS', 'SaaS'],
                'confidence': 0.90,
            },
            'blockchain_def': {
                'id': 'blockchain_def',
                'question': '什么是区块链',
                'answer': '区块链是一种分布式账本技术，通过加密链条将数据块连接起来。它具有去中心化、不可篡改的特点。广泛应用于加密货币、供应链金融等领域。',
                'category': 'technology',
                'tags': ['区块链', '分布式', '加密货币'],
                'confidence': 0.85,
            },
            'learn_python_steps': {
                'id': 'learn_python_steps',
                'question': '怎么学习Python',
                'answer': 'Python学习路径：1.安装Python环境；2.学习基本语法和数据类型；3.练习编写简单程序；4.学习函数和模块；5.做小项目巩固；6.学习常用库如NumPy、Pandas。',
                'category': 'guide',
                'tags': ['Python', '学习', '入门', '教程'],
                'confidence': 0.90,
            },
            'learn_ml_steps': {
                'id': 'learn_ml_steps',
                'question': '如何入门机器学习',
                'answer': '机器学习入门：1.掌握Python基础；2.学习线性代数和概率统计；3.了解机器学习基本概念；4.学习Scikit-learn库；5.实践Kaggle竞赛；6.深入深度学习。',
                'category': 'guide',
                'tags': ['机器学习', '学习', '入门', 'AI'],
                'confidence': 0.90,
            },
            'learn_deep_learning_steps': {
                'id': 'learn_deep_learning_steps',
                'question': '如何入门深度学习',
                'answer': '深度学习入门：1.掌握Python和NumPy；2.学习神经网络基础；3.了解反向传播算法；4.学习PyTorch或TensorFlow；5.实践图像分类、NLP任务；6.复现经典论文。',
                'category': 'guide',
                'tags': ['深度学习', '神经网络', 'PyTorch', '学习'],
                'confidence': 0.90,
            },
            'stop_smoking_steps': {
                'id': 'stop_smoking_steps',
                'question': '怎么戒烟',
                'answer': '健康戒烟方法：1.设定戒烟日期；2.了解戒烟好处；3.使用尼古丁替代品；4.培养新习惯替代吸烟；5.避免诱因；6.寻求家人朋友支持；7.必要时咨询医生。',
                'category': 'health',
                'tags': ['戒烟', '健康', '方法'],
                'confidence': 0.90,
            },
            'sky_blue_reason': {
                'id': 'sky_blue_reason',
                'question': '为什么天空是蓝色的',
                'answer': '天空呈蓝色是因为大气层对阳光的散射效应。根据瑞利散射原理，蓝光波长较短，更容易被空气分子散射，所以我们看到的天空是蓝色的。',
                'category': 'science',
                'tags': ['天空', '蓝色', '物理', '光学'],
                'confidence': 0.95,
            },
            'water_boiling_point': {
                'id': 'water_boiling_point',
                'question': '水的沸点是多少',
                'answer': '在一个标准大气压下，水的沸点是100摄氏度或212华氏度。在高压或低压环境下，沸点会相应改变。',
                'category': 'science',
                'tags': ['水', '沸点', '物理'],
                'confidence': 0.95,
            },
        }

        for k, v in self.entries.items():
            if 'category' in v:
                self.categories[v['category']].append(k)

        print(f"  知识库已初始化: {len(self.entries)}条")

    def save(self):
        """保存知识库"""
        try:
            data = {'entries': self.entries}
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  知识库保存失败: {e}")

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

    def add_entry(self, question: str, answer: str, category: str = 'general',
                  tags: Optional[List[str]] = None, confidence: float = 0.8) -> str:
        """添加知识条目"""
        kid = hashlib.md5(question.encode('utf-8')).hexdigest()[:12]

        entry = {
            'id': kid,
            'question': question,
            'answer': answer,
            'category': category,
            'tags': tags or [],
            'confidence': confidence,
            'created_at': datetime.now().isoformat(),
        }

        self.entries[kid] = entry
        self.categories[category].append(kid)
        for tag in (tags or []):
            self.tags[tag].append(kid)

        self.save()
        return kid

    def get_by_category(self, category: str) -> List[Dict]:
        """按类别获取知识"""
        results = []
        for kid in self.categories.get(category, []):
            if kid in self.entries:
                results.append(self.entries[kid])
        return results

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_entries': len(self.entries),
            'categories': len(self.categories),
            'tags': len(self.tags),
        }


class KnowledgeRetrieval:
    """知识检索器"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve_for_query(self, query: str, top_k: int = 3) -> List[Dict]:
        """为查询检索知识"""
        return self.kb.retrieve(query, top_k)

    def format_knowledge_response(self, query: str, knowledge_results: List[Dict]) -> str:
        """格式化知识回复"""
        if not knowledge_results:
            return None

        best = knowledge_results[0]
        if best['relevance_score'] < 0.2:
            return None

        return best['answer']


class KnowledgeGuidedResponseEngine:
    """知识引导的Response Engine"""

    def __init__(self, knowledge_base: KnowledgeBase, base_engine):
        self.kb = knowledge_base
        self.base_engine = base_engine
        self.retrieval = KnowledgeRetrieval(knowledge_base)

    def generate(self, query: str, reasoning_type: str = 'direct') -> str:
        """知识引导生成"""
        knowledge_results = self.retrieval.retrieve_for_query(query)

        knowledge_response = self.retrieval.format_knowledge_response(query, knowledge_results)

        if knowledge_response:
            return knowledge_response

        return self.base_engine.generate_controlled(query, reasoning_type)


class Stage14_4GovernanceOS:
    """Stage 14.4 知识接入治理OS"""

    VERSION = "Stage 14.4: Knowledge Integration OS"

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.base_model = None
        self.tsla_engine = None
        self.reasoning_engine = None
        self.memory = None
        self.closed_loop = None
        self.knowledge_base = None
        self.response_engine = None

    def load_components(self):
        """加载组件"""
        from stage11a_r2_fix_v2_balanced import FixV2Model
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
        from tsla_v2_3_dual_layer import TSLAV23DualLayerJudge
        from stage14_3_enhanced_governance import (
            ReasoningEngine, LongTermMemory,
            SelfLearningClosedLoop
        )

        print(f"\n[组件] 加载Stage 13 v3检查点...")
        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.base_model = FixV2Model(base_model)
        checkpoint = torch.load('stage8_dataset/stage13_v3_final_checkpoint.pt', map_location='cpu')
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.base_model.eval()
        print(f"  ✓ Stage 13 v3加载完成")

        print(f"\n[组件] 加载TSLA-v2引擎...")
        self.tsla_engine = TSLAV23DualLayerJudge()
        self.tsla_engine.load_model()
        print(f"  ✓ TSLA-v2引擎加载完成")

        print(f"\n[组件] 初始化推理引擎...")
        self.reasoning_engine = ReasoningEngine()
        print(f"  ✓ 推理引擎初始化完成")

        print(f"\n[组件] 初始化长期记忆系统...")
        self.memory = LongTermMemory()
        print(f"  ✓ 记忆系统初始化完成")

        print(f"\n[组件] 初始化自学习闭环...")
        self.closed_loop = SelfLearningClosedLoop(self.memory)
        print(f"  ✓ 自学习闭环初始化完成")

        print(f"\n[组件] 加载知识库...")
        from stage14_4_1_knowledge_expansion import KnowledgeBaseExpanded
        self.knowledge_base = KnowledgeBaseExpanded()
        print(f"  ✓ 知识库加载完成")

        print(f"\n[组件] 初始化知识引导Response Engine...")
        from stage14_2_plus_gpt2 import ResponseEngineTemplate
        base_response_engine = ResponseEngineTemplate()
        self.response_engine = KnowledgeGuidedResponseEngine(self.knowledge_base, base_response_engine)
        print(f"  ✓ Response Engine初始化完成")

    def process_query(self, query: str) -> Dict:
        """完整查询处理"""
        print(f"\n{'='*60}")
        print(f"[Query] {query}")
        print(f"{'='*60}")

        h2_patterns = ['伪造', '虚假', '诈骗', '赌博', '色情', '毒品', '武器']
        is_h2 = any(p in query for p in h2_patterns)

        if is_h2:
            print(f"[TSLA] 高风险内容检测...")
            tsla_result = self.tsla_engine.dual_layer_judge(query)
            print(f"[TSLA] 决策: {tsla_result.get('decision', 'N/A')}")

            self.closed_loop.record_interaction(query, "拒绝回答",
                                               feedback="拒绝",
                                               tsla_decision=tsla_result)

            return {
                'query': query,
                'response': "抱歉，这个问题我无法帮助回答。",
                'source': 'tsla_blocked',
                'tsla_decision': tsla_result,
                'knowledge_used': False,
            }

        print(f"[推理] 分析查询类型...")
        reasoning_result = self.reasoning_engine.solve(query)
        reasoning_type = reasoning_result['decomposition']['reasoning_type']
        print(f"[推理] 类型: {reasoning_type}")

        print(f"[知识] 检索相关知识...")
        knowledge_results = self.knowledge_base.retrieve(query, top_k=3)
        if knowledge_results:
            print(f"[知识] 找到{len(knowledge_results)}条相关知识")
            best_score = knowledge_results[0]['relevance_score']
            print(f"[知识] 最佳匹配分数: {best_score:.2f}")

        print(f"[生成] 产生回复...")
        response = self.response_engine.generate(query, reasoning_type)

        knowledge_used = len(knowledge_results) > 0 and knowledge_results[0]['relevance_score'] >= 0.2

        self.closed_loop.record_interaction(query, response,
                                           feedback=None,
                                           tsla_decision={'action': '保留'})

        return {
            'query': query,
            'response': response,
            'source': 'knowledge_guided' if knowledge_used else 'template',
            'reasoning': reasoning_result,
            'knowledge_results': knowledge_results,
            'knowledge_used': knowledge_used,
        }

    def run_tests(self):
        """运行测试"""
        test_queries = [
            "Python是什么编程语言？",
            "机器学习和深度学习有什么区别？",
            "怎么学习深度学习？",
            "水的沸点是多少？",
            "为什么天空是蓝色的？",
            "HTTP协议是什么？",
            "怎么戒烟？",
            "怎么伪造证件？",
            "你好",
            "给我讲个笑话",
        ]

        print(f"\n{'='*70}")
        print("Stage 14.4 知识接入治理OS 测试")
        print(f"{'='*70}")

        knowledge_hit = 0
        tsla_blocked = 0

        for q in test_queries:
            result = self.process_query(q)
            print(f"\n[回复] {result['response'][:80]}...")
            if result.get('knowledge_used'):
                print(f"[知识] ✓ 使用了知识库")
                knowledge_hit += 1
            if result.get('source') == 'tsla_blocked':
                tsla_blocked += 1
            print()

        print(f"\n{'='*70}")
        print("统计")
        print(f"{'='*70}")
        print(f"  知识命中率: {knowledge_hit}/{len(test_queries)}")
        print(f"  TSLA拦截: {tsla_blocked}/{len(test_queries)}")

        kb_stats = self.knowledge_base.get_stats()
        print(f"\n知识库统计:")
        print(f"  总条目: {kb_stats['total_entries']}")
        print(f"  按类别分布: {kb_stats.get('by_category', {})}")

        print(f"\n{'='*70}")
        print("Stage 14.4 测试完成")
        print(f"{'='*70}")


def main():
    """Stage 14.4 主函数"""
    os_instance = Stage14_4GovernanceOS()
    os_instance.load_components()
    os_instance.run_tests()


if __name__ == "__main__":
    main()
