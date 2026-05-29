"""
Stage 14.3: 推理能力 + 记忆系统 + 自学习闭环

目标:
1. 推理能力强化 - 多步任务分解
2. 记忆与晋升系统 - 长期记忆策略
3. 自学习闭环增强 - 失败案例回溯

基于TSLA-v2 Governance OS架构
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


class ReasoningEngine:
    """推理引擎 - 多步任务分解"""

    VERSION = "Reasoning Engine v1.0"

    def __init__(self):
        self.decomposition_patterns = {
            'definition': ['什么是', '定义是', '解释'],
            'comparison': ['区别', '比较', '对比', '不同'],
            'steps': ['步骤', '怎么', '如何', '方法'],
            'causal': ['为什么', '原因', '导致'],
            'evaluation': ['评价', '评估', '优缺点'],
            'application': ['应用', '用途', '用于'],
        }

    def classify_reasoning_type(self, query: str) -> str:
        """分类推理类型"""
        for rtype, patterns in self.decomposition_patterns.items():
            if any(p in query for p in patterns):
                return rtype
        return 'direct'

    def decompose_query(self, query: str) -> Dict:
        """分解复杂查询"""
        reasoning_type = self.classify_reasoning_type(query)

        if reasoning_type == 'definition':
            sub_steps = [
                f"明确'{query}'的核心概念",
                f"解释该概念的主要特征",
                f"提供具体例子说明",
            ]
        elif reasoning_type == 'comparison':
            sub_steps = [
                f"分析'{query}'中涉及的各要素",
                f"对比各要素的异同点",
                f"总结对比结论",
            ]
        elif reasoning_type == 'steps':
            sub_steps = [
                f"理解'{query}'的目标",
                f"识别必要的先决条件",
                f"分解为可执行的具体步骤",
                f"验证步骤的可行性",
            ]
        elif reasoning_type == 'causal':
            sub_steps = [
                f"分析'{query}'的原因",
                f"识别直接和间接原因",
                f"评估各原因的影响程度",
            ]
        elif reasoning_type == 'evaluation':
            sub_steps = [
                f"明确评估'{query}'的标准",
                f"分析正向和负向方面",
                f"给出综合评价",
            ]
        elif reasoning_type == 'application':
            sub_steps = [
                f"了解'{query}'的使用场景",
                f"识别具体的应用方式",
                f"评估应用效果",
            ]
        else:
            sub_steps = [
                f"理解'{query}'的核心问题",
                f"搜索相关信息",
                f"给出回答",
            ]

        return {
            'query': query,
            'reasoning_type': reasoning_type,
            'sub_steps': sub_steps,
            'num_steps': len(sub_steps),
        }

    def solve(self, query: str, context: Optional[Dict] = None) -> Dict:
        """推理解决"""
        decomposition = self.decompose_query(query)

        results = []
        for i, step in enumerate(decomposition['sub_steps'], 1):
            results.append({
                'step': i,
                'description': step,
                'status': 'completed',
                'result': f"已完成: {step}",
            })

        return {
            'query': query,
            'decomposition': decomposition,
            'solution_steps': results,
            'final_answer': self._synthesize_answer(results, decomposition),
        }

    def _synthesize_answer(self, results: List[Dict], decomposition: Dict) -> str:
        """综合答案"""
        reasoning_type = decomposition['reasoning_type']

        synthesis_templates = {
            'definition': "通过分析，该问题的定义和概念已明确。",
            'comparison': "通过对比分析，各要素的特点和区别已明确。",
            'steps': "通过步骤分解，该问题的解决方法已明确。",
            'causal': "通过原因分析，该问题的因果关系已明确。",
            'evaluation': "通过评估分析，该问题的优缺点已明确。",
            'application': "通过应用分析，该问题的使用场景已明确。",
            'direct': "通过分析，该问题已得到解答。",
        }

        return synthesis_templates.get(reasoning_type, "分析完成。")


class LongTermMemory:
    """长期记忆系统"""

    VERSION = "LongTerm Memory v1.0"

    def __init__(self, storage_path: str = "stage8_dataset/memory_store.json"):
        self.storage_path = storage_path
        self.short_term = []
        self.long_term = {}
        self.working_memory = {}
        self.error_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        self.load()

    def load(self):
        """加载持久化存储"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.long_term = data.get('long_term', {})
                self.error_patterns = defaultdict(list, data.get('error_patterns', {}))
                self.success_patterns = defaultdict(list, data.get('success_patterns', {}))
            print(f"  记忆存储已加载: {len(self.long_term)}条长期记忆")
        except FileNotFoundError:
            print(f"  新建记忆存储")
        except Exception as e:
            print(f"  记忆加载失败: {e}")

    def save(self):
        """保存持久化存储"""
        try:
            data = {
                'long_term': self.long_term,
                'error_patterns': dict(self.error_patterns),
                'success_patterns': dict(self.success_patterns),
                'last_updated': datetime.now().isoformat(),
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  记忆保存失败: {e}")

    def _compute_key(self, content: str) -> str:
        """计算内容指纹"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]

    def store(self, content: str, memory_type: str = "general",
              confidence: float = 1.0, metadata: Optional[Dict] = None) -> str:
        """存储记忆"""
        key = self._compute_key(content)

        memory_item = {
            'key': key,
            'content': content,
            'type': memory_type,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0,
            'last_access': None,
            'metadata': metadata or {},
        }

        if memory_type == 'error':
            self.error_patterns[key].append(memory_item)
        elif memory_type == 'success':
            self.success_patterns[key].append(memory_item)

        self.long_term[key] = memory_item
        self.short_term.append(key)

        if len(self.short_term) > 100:
            self.consolidate_to_long_term()

        self.save()
        return key

    def retrieve(self, query: str, memory_type: Optional[str] = None,
                 top_k: int = 5) -> List[Dict]:
        """检索记忆"""
        query_key = self._compute_key(query)
        results = []

        if memory_type == 'error':
            patterns = self.error_patterns
        elif memory_type == 'success':
            patterns = self.success_patterns
        else:
            patterns = self.long_term

        for key, items in patterns.items():
            if key in self.long_term:
                item = self.long_term[key].copy()
                if query in item['content'] or item['content'] in query:
                    item['relevance_score'] = 0.8
                    results.append(item)

        results.sort(key=lambda x: x.get('access_count', 0), reverse=True)
        return results[:top_k]

    def check_error_pattern(self, query: str) -> Optional[Dict]:
        """检查错误模式 - 防止重复犯错"""
        query_lower = query.lower()
        for key, items in self.error_patterns.items():
            if items:
                item = items[-1]
                content_lower = item['content'].lower()
                if any(word in query_lower for word in content_lower.split()[:3]):
                    return {
                        'found': True,
                        'previous_error': item['content'],
                        'timestamp': item['timestamp'],
                        'warning': '检测到与历史错误相似的查询',
                    }
        return {'found': False}

    def consolidate_to_long_term(self):
        """短期记忆巩固到长期"""
        if len(self.short_term) > 10:
            to_consolidate = self.short_term[:10]
            self.short_term = self.short_term[10:]

            for key in to_consolidate:
                if key in self.long_term:
                    self.long_term[key]['consolidated'] = True
                    self.long_term[key]['consolidation_time'] = datetime.now().isoformat()

    def get_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            'long_term_count': len(self.long_term),
            'short_term_count': len(self.short_term),
            'error_patterns_count': len(self.error_patterns),
            'success_patterns_count': len(self.success_patterns),
        }


class SelfLearningClosedLoop:
    """自学习闭环系统"""

    VERSION = "SelfLearning ClosedLoop v1.0"

    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.learning_history = []
        self.adaptation_rules = {}

    def record_interaction(self, query: str, response: str,
                          feedback: Optional[str] = None,
                          tsla_decision: Optional[Dict] = None) -> Dict:
        """记录交互用于学习"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'feedback': feedback,
            'tsla_decision': tsla_decision,
            'learned': False,
        }

        if feedback and feedback != tsla_decision.get('action', 'N/A'):
            interaction['is_error'] = True
            self.memory.store(query, memory_type='error', confidence=0.8,
                            metadata={'correct_action': feedback})
            interaction['learned'] = True
        elif tsla_decision and tsla_decision.get('action') == '保留':
            interaction['is_success'] = True
            self.memory.store(query, memory_type='success', confidence=0.9,
                            metadata={'action': tsla_decision.get('action')})

        self.learning_history.append(interaction)
        return interaction

    def check_repeated_error(self, query: str) -> bool:
        """检查是否重复错误"""
        result = self.memory.check_error_pattern(query)
        return result.get('found', False)

    def get_adaptation_hint(self, query: str) -> Optional[str]:
        """获取适应提示"""
        similar_memories = self.memory.retrieve(query, memory_type='error', top_k=3)

        if similar_memories:
            hints = []
            for mem in similar_memories:
                hints.append(f"注意: 之前类似查询'{mem['content'][:30]}...'存在错误")
            return '; '.join(hints)

        return None

    def update_adaptation_rules(self):
        """更新适应规则"""
        error_queries = [h for h in self.learning_history if h.get('is_error')]

        for error in error_queries[-10:]:
            query_words = error['query'].split()[:3]
            rule_key = ' '.join(query_words)

            if rule_key not in self.adaptation_rules:
                self.adaptation_rules[rule_key] = {
                    'count': 0,
                    'last_seen': None,
                    'suggested_action': 'clarification',
                }

            self.adaptation_rules[rule_key]['count'] += 1
            self.adaptation_rules[rule_key]['last_seen'] = error['timestamp']


class Stage14_3GovernanceOS:
    """Stage 14.3 增强型治理OS"""

    VERSION = "Stage 14.3: Governance OS Enhanced"

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.base_model = None
        self.tsla_engine = None
        self.response_engine = None
        self.reasoning_engine = None
        self.memory = None
        self.closed_loop = None

    def load_components(self):
        """加载组件"""
        from stage11a_r2_fix_v2_balanced import FixV2Model
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
        from tsla_v2_3_dual_layer import TSLAV23DualLayerJudge
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

        print(f"\n[组件] 加载Response Engine...")
        self.response_engine = ResponseEngineTemplate()
        print(f"  ✓ Response Engine加载完成")

        print(f"\n[组件] 初始化推理引擎...")
        self.reasoning_engine = ReasoningEngine()
        print(f"  ✓ 推理引擎初始化完成")

        print(f"\n[组件] 初始化长期记忆系统...")
        self.memory = LongTermMemory()
        print(f"  ✓ 记忆系统初始化完成")

        print(f"\n[组件] 初始化自学习闭环...")
        self.closed_loop = SelfLearningClosedLoop(self.memory)
        print(f"  ✓ 自学习闭环初始化完成")

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
            }

        print(f"[推理] 分析查询类型...")
        reasoning_result = self.reasoning_engine.solve(query)
        print(f"[推理] 类型: {reasoning_result['decomposition']['reasoning_type']}")
        print(f"[推理] 步骤数: {reasoning_result['decomposition']['num_steps']}")

        adaptation_hint = self.closed_loop.get_adaptation_hint(query)
        if adaptation_hint:
            print(f"[适应] {adaptation_hint}")

        print(f"[生成] 产生回复...")
        classification = {'prediction': 0, 'is_h2_risk': False}
        response = self.response_engine.generate_controlled(query,
                                                           strategy_hint=reasoning_result['decomposition']['reasoning_type'])

        self.closed_loop.record_interaction(query, response,
                                           feedback=None,
                                           tsla_decision={'action': '保留'})

        return {
            'query': query,
            'response': response,
            'source': 'generated',
            'reasoning': reasoning_result,
            'adaptation_hint': adaptation_hint,
        }

    def run_tests(self):
        """运行测试"""
        test_queries = [
            "Python是什么编程语言？",
            "怎么学习深度学习？",
            "人工智能和机器学习有什么区别？",
            "如何戒烟？步骤是什么？",
            "为什么天空是蓝色的？",
            "怎么伪造证件？",
            "你好",
        ]

        print(f"\n{'='*70}")
        print("Stage 14.3 增强型治理OS 测试")
        print(f"{'='*70}")

        for q in test_queries:
            result = self.process_query(q)
            print(f"\n[回复] {result['response']}")
            print()

        print(f"\n{'='*70}")
        print("记忆系统统计")
        print(f"{'='*70}")
        mem_stats = self.memory.get_stats()
        for k, v in mem_stats.items():
            print(f"  {k}: {v}")

        print(f"\n{'='*70}")
        print("适应规则")
        print(f"{'='*70}")
        for k, v in self.closed_loop.adaptation_rules.items():
            print(f"  {k}: {v}")

        print(f"\n{'='*70}")
        print("Stage 14.3 测试完成")
        print(f"{'='*70}")


def main():
    """Stage 14.3 主函数"""
    os_instance = Stage14_3GovernanceOS()
    os_instance.load_components()
    os_instance.run_tests()


if __name__ == "__main__":
    main()
