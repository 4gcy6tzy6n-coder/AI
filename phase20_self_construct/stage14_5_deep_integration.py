"""
Stage 14.5: 推理引擎与Response Engine深度整合

目标:
1. 推理驱动生成 - 多步推理输出作为Response Engine条件输入
2. 知识+推理融合 - 查询→推理分解→知识检索→Response Engine
3. 闭环强化 - TSLA拦截+记忆更新+自学习优化

整合策略:
- 接口对齐: 推理引擎输出结构 → Response Engine 输入条件
- 控制实验: 50-100查询进行融合测试
- 逐步升级: 模板Response Engine → 轻量LLM
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


class ReasoningGuidedResponseEngine:
    """推理引导的Response Engine - 深度整合版"""

    VERSION = "Reasoning Guided Response Engine v1.0"

    RESPONSE_STRATEGIES = {
        'definition': {
            'template': "根据分析，{core_concept}是指{features}。{example}",
            'structure': ['概念明确', '特征说明', '实例补充'],
        },
        'comparison': {
            'template': "对比分析如下：{item1}的特点是{features1}，而{item2}的特点是{features2}。主要区别在于{differences}。",
            'structure': ['分别说明', '对比差异', '总结结论'],
        },
        'steps': {
            'template': "解决这个问题需要{num_steps}个步骤：{steps_text}。每一步都至关重要，请按顺序执行。",
            'structure': ['目标确认', '步骤分解', '执行验证'],
        },
        'causal': {
            'template': "分析原因如下：{direct_causes}。此外还有{indirect_causes}。结论：{conclusion}。",
            'structure': ['直接原因', '间接原因', '综合结论'],
        },
        'evaluation': {
            'template': "综合评估：优点包括{pros}，缺点包括{cons}。整体评价：{verdict}。",
            'structure': ['优点分析', '缺点分析', '综合评价'],
        },
        'application': {
            'template': "主要应用场景包括：{scenarios}。具体使用方式：{usage_methods}。效果：{effects}。",
            'structure': ['场景识别', '使用方法', '效果评估'],
        },
        'direct': {
            'template': "{answer}",
            'structure': ['直接回答'],
        },
    }

    def __init__(self, knowledge_base, reasoning_engine):
        self.kb = knowledge_base
        self.reasoning = reasoning_engine
        self.generation_history = []

    def generate(self, query: str, reasoning_result: Dict,
                 knowledge_results: List[Dict] = None) -> Dict:
        """推理引导的生成"""
        reasoning_type = reasoning_result['decomposition']['reasoning_type']
        sub_steps = reasoning_result['decomposition']['sub_steps']

        strategy = self.RESPONSE_STRATEGIES.get(reasoning_type, self.RESPONSE_STRATEGIES['direct'])

        knowledge_context = ""
        if knowledge_results and len(knowledge_results) > 0:
            best_knowledge = knowledge_results[0]
            if best_knowledge.get('relevance_score', 0) >= 0.2:
                knowledge_context = self._build_knowledge_context(knowledge_results)

        response_text = self._synthesize_response(
            query, reasoning_type, sub_steps, strategy, knowledge_context
        )

        generation_info = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'reasoning_type': reasoning_type,
            'sub_steps_count': len(sub_steps),
            'knowledge_used': bool(knowledge_context),
            'knowledge_count': len(knowledge_results) if knowledge_results else 0,
            'response_preview': response_text[:50],
        }

        self.generation_history.append(generation_info)

        return {
            'response': response_text,
            'reasoning_type': reasoning_type,
            'strategy': strategy,
            'sub_steps': sub_steps,
            'knowledge_context': knowledge_context,
            'generation_info': generation_info,
        }

    def _build_knowledge_context(self, knowledge_results: List[Dict]) -> str:
        """构建知识上下文"""
        if not knowledge_results:
            return ""

        contexts = []
        for i, kb_entry in enumerate(knowledge_results[:2], 1):
            contexts.append(f"{kb_entry['question']}：{kb_entry['answer'][:100]}")

        return " | ".join(contexts)

    def _synthesize_response(self, query: str, reasoning_type: str,
                            sub_steps: List[str], strategy: Dict,
                            knowledge_context: str) -> str:
        """综合生成回复"""
        if knowledge_context and reasoning_type == 'direct':
            return f"基于我的分析：{knowledge_context.split('：')[1] if '：' in knowledge_context else knowledge_context}"

        if reasoning_type == 'definition':
            core = query.replace('是什么', '').replace('什么是', '')
            features = "这是一个重要概念，具有特定的应用价值"
            example = "在实际应用中有广泛用途"
            return strategy['template'].format(
                core_concept=core,
                features=features,
                example=example
            )

        elif reasoning_type == 'comparison':
            parts = query.replace('和', ' ').replace('与', ' ').split()
            item1 = parts[0] if len(parts) > 0 else "A"
            item2 = parts[1] if len(parts) > 1 else "B"
            return strategy['template'].format(
                item1=item1, item2=item2,
                features1="各有特点",
                features2="各有优势",
                differences="应用场景和实现方式不同"
            )

        elif reasoning_type == 'steps':
            steps_text = "；".join([f"{i+1}.{s.split('：')[-1] if '：' in s else s}" for i, s in enumerate(sub_steps[:4])])
            return strategy['template'].format(
                num_steps=len(sub_steps),
                steps_text=steps_text
            )

        elif reasoning_type == 'causal':
            causes = sub_steps[0] if sub_steps else "多种因素共同作用"
            indirect = sub_steps[1] if len(sub_steps) > 1 else "深层原因"
            conclusion = sub_steps[-1] if sub_steps else "需要综合分析"
            return strategy['template'].format(
                direct_causes=causes,
                indirect_causes=indirect,
                conclusion=conclusion
            )

        elif reasoning_type == 'evaluation':
            pros = "功能强大，应用广泛"
            cons = "需要一定学习成本"
            verdict = "总体来说值得推荐"
            return strategy['template'].format(
                pros=pros, cons=cons, verdict=verdict
            )

        elif reasoning_type == 'application':
            scenarios = "教育、工作、研究等领域"
            usage_methods = "根据具体场景选择合适的方式"
            effects = "效果显著"
            return strategy['template'].format(
                scenarios=scenarios,
                usage_methods=usage_methods,
                effects=effects
            )

        else:
            if knowledge_context:
                kb_answer = knowledge_context.split('：')[1] if '：' in knowledge_context else knowledge_context
                return f"综合分析：{kb_answer}"
            return f"根据我的分析，这需要从多个角度来考虑。"


class Stage14_5IntegratedOS:
    """Stage 14.5 深度整合治理OS"""

    VERSION = "Stage 14.5: Deep Integration OS"

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
        from stage14_3_enhanced_governance import ReasoningEngine, LongTermMemory, SelfLearningClosedLoop
        from stage14_4_1_knowledge_expansion import KnowledgeBaseExpanded
        from stage14_2_plus_gpt2 import ResponseEngineTemplate

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
        self.knowledge_base = KnowledgeBaseExpanded()
        print(f"  ✓ 知识库加载完成 (225条)")

        print(f"\n[组件] 初始化推理引导Response Engine...")
        self.response_engine = ReasoningGuidedResponseEngine(self.knowledge_base, self.reasoning_engine)
        print(f"  ✓ Response Engine初始化完成")

    def process_query(self, query: str) -> Dict:
        """深度整合的查询处理"""
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
                'reasoning_type': 'blocked',
                'knowledge_used': False,
            }

        print(f"[推理] 分析查询类型...")
        reasoning_result = self.reasoning_engine.solve(query)
        reasoning_type = reasoning_result['decomposition']['reasoning_type']
        sub_steps = reasoning_result['decomposition']['sub_steps']
        print(f"[推理] 类型: {reasoning_type}")
        print(f"[推理] 步骤数: {len(sub_steps)}")
        for i, step in enumerate(sub_steps[:3], 1):
            print(f"  步骤{i}: {step[:50]}...")

        print(f"[知识] 检索相关知识...")
        knowledge_results = self.knowledge_base.retrieve(query, top_k=3)
        knowledge_used = len(knowledge_results) > 0 and knowledge_results[0].get('relevance_score', 0) >= 0.2
        if knowledge_results:
            print(f"[知识] 找到{len(knowledge_results)}条, 最佳匹配: {knowledge_results[0]['relevance_score']:.2f}")

        print(f"[生成] 推理引导生成...")
        generation_result = self.response_engine.generate(query, reasoning_result, knowledge_results)

        response = generation_result['response']
        print(f"[生成] 策略: {generation_result['strategy']}")

        self.closed_loop.record_interaction(query, response,
                                           feedback=None,
                                           tsla_decision={'action': '保留'})

        return {
            'query': query,
            'response': response,
            'source': 'reasoning_guided',
            'reasoning': reasoning_result,
            'reasoning_type': reasoning_type,
            'knowledge_results': knowledge_results,
            'knowledge_used': knowledge_used,
            'generation_info': generation_result.get('generation_info'),
        }

    def run_tests(self):
        """运行深度整合测试"""
        test_queries = [
            "Python是什么编程语言？",
            "机器学习和深度学习有什么区别？",
            "怎么学习深度学习？",
            "水的沸点是多少？",
            "为什么天空是蓝色的？",
            "HTTP协议是什么？",
            "怎么戒烟？",
            "如何缓解工作压力？",
            "项目管理有哪些方法？",
            "Git怎么使用？",
            "怎么伪造证件？",
            "你好",
            "给我讲个笑话",
            "人工智能的发展历史是什么？",
            "云计算有哪些应用场景？",
        ]

        print(f"\n{'='*70}")
        print("Stage 14.5 深度整合测试")
        print(f"{'='*70}")

        stats = {
            'total': len(test_queries),
            'knowledge_hit': 0,
            'tsla_blocked': 0,
            'reasoning_types': defaultdict(int),
        }

        for q in test_queries:
            result = self.process_query(q)
            print(f"\n[回复] {result['response'][:80]}...")
            print(f"[类型] {result['reasoning_type']} | [知识] {'✓' if result.get('knowledge_used') else '✗'}")
            if result.get('source') == 'tsla_blocked':
                stats['tsla_blocked'] += 1
            if result.get('knowledge_used'):
                stats['knowledge_hit'] += 1
            stats['reasoning_types'][result['reasoning_type']] += 1
            print()

        print(f"\n{'='*70}")
        print("统计报告")
        print(f"{'='*70}")
        print(f"  总查询数: {stats['total']}")
        print(f"  知识命中: {stats['knowledge_hit']}/{stats['total']} ({100*stats['knowledge_hit']/stats['total']:.1f}%)")
        print(f"  TSLA拦截: {stats['tsla_blocked']}/{stats['total']} ({100*stats['tsla_blocked']/stats['total']:.1f}%)")
        print(f"\n推理类型分布:")
        for rtype, count in sorted(stats['reasoning_types'].items()):
            print(f"  {rtype}: {count}")

        print(f"\n{'='*70}")
        print("生成历史")
        print(f"{'='*70}")
        for i, gen in enumerate(self.response_engine.generation_history[-5:], 1):
            print(f"  {i}. [{gen['reasoning_type']}] {gen['response_preview']}...")

        print(f"\n{'='*70}")
        print("Stage 14.5 测试完成")
        print(f"{'='*70}")

        return stats

    def run_extended_test(self, num_queries: int = 50):
        """运行扩展测试 (50-100查询)"""
        extended_queries = [
            "什么是人工智能？",
            "机器学习怎么入门？",
            "深度学习需要什么基础？",
            "Python和JavaScript的区别？",
            "React和Vue哪个好？",
            "Docker容器技术原理？",
            "Kubernetes是什么？",
            "Git工作流有哪些？",
            "SQL和NoSQL的区别？",
            "MongoDB适用场景？",
            "Redis数据类型？",
            "微服务架构的优缺点？",
            "REST API设计原则？",
            "GraphQL和REST对比？",
            "云计算的三种服务模式？",
            "AWS有哪些核心服务？",
            "如何设计数据库表结构？",
            "索引的原理和注意事项？",
            "事务的ACID特性？",
            "什么是CAP定理？",
            "怎么提高代码质量？",
            "代码重构的原则？",
            "设计模式有哪些？",
            "什么是TDD开发？",
            "敏捷开发的核心是什么？",
            "Scrum有哪些角色？",
            "如何做时间管理？",
            "怎么提高团队协作效率？",
            "职业生涯如何规划？",
            "如何写好简历？",
            "面试技巧有哪些？",
            "远程工作注意事项？",
            "怎么保护眼睛健康？",
            "运动有哪些好处？",
            "如何提高睡眠质量？",
            "怎样缓解焦虑？",
            "健康饮食原则？",
            "水的重要性？",
            "心脏健康保护方法？",
            "大脑如何保持活力？",
            "区块链的工作原理？",
            "比特币和以太坊区别？",
            "NFT是什么？",
            "元宇宙概念？",
            "5G技术的特点？",
            "IPv6和IPv4的区别？",
            "HTTPS安全机制？",
            "DNS解析过程？",
            "CDN的作用？",
            "负载均衡算法？",
            "SSL/TLS握手过程？",
            "大数据的5V特征？",
            "数据湖和数据仓库的区别？",
            "ETL流程是什么？",
            "什么是数据挖掘？",
            "机器学习常用算法？",
            "监督学习和无监督学习区别？",
            "神经网络为什么有效？",
            "Transformer架构原理？",
            "BERT和GPT区别？",
            "什么是RAG？",
            "Prompt工程技巧？",
            "Fine-tuning微调方法？",
            "模型过拟合怎么办？",
            "交叉验证的作用？",
            "准确率和精确率的区别？",
        ]

        queries_to_test = extended_queries[:num_queries]
        return self._run_batch_test(queries_to_test)

    def _run_batch_test(self, queries: List[str]) -> Dict:
        """批量测试"""
        print(f"\n{'='*70}")
        print(f"批量测试: {len(queries)} 查询")
        print(f"{'='*70}")

        stats = {
            'total': len(queries),
            'knowledge_hit': 0,
            'tsla_blocked': 0,
            'reasoning_types': defaultdict(int),
            'queries': [],
        }

        for q in queries:
            result = self.process_query(q)
            if result.get('source') == 'tsla_blocked':
                stats['tsla_blocked'] += 1
            if result.get('knowledge_used'):
                stats['knowledge_hit'] += 1
            stats['reasoning_types'][result['reasoning_type']] += 1

            stats['queries'].append({
                'query': q,
                'reasoning_type': result['reasoning_type'],
                'knowledge_used': result.get('knowledge_used', False),
                'blocked': result.get('source') == 'tsla_blocked',
            })

        print(f"\n{'='*70}")
        print("批量测试报告")
        print(f"{'='*70}")
        print(f"  总查询数: {stats['total']}")
        print(f"  知识命中率: {stats['knowledge_hit']}/{stats['total']} ({100*stats['knowledge_hit']/stats['total']:.1f}%)")
        print(f"  TSLA拦截率: {stats['tsla_blocked']}/{stats['total']} ({100*stats['tsla_blocked']/stats['total']:.1f}%)")
        print(f"\n推理类型分布:")
        for rtype, count in sorted(stats['reasoning_types'].items()):
            bar = '█' * count
            print(f"  {rtype:12s}: {count:3d} {bar}")

        return stats


def main():
    """主函数"""
    os_instance = Stage14_5IntegratedOS()
    os_instance.load_components()

    mode = input("\n选择测试模式:\n  1. 标准测试 (15查询)\n  2. 扩展测试 (50查询)\n  3. 完整测试 (100查询)\n请输入选项 (1/2/3): ").strip()

    if mode == '2':
        os_instance.run_extended_test(50)
    elif mode == '3':
        os_instance.run_extended_test(100)
    else:
        os_instance.run_tests()


if __name__ == "__main__":
    main()
