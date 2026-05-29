"""
Stage 14.2+: 生成层架构 - Response Engine

架构:
- Governance OS: TSLA-v2 / Gap / Memory / Retrieval / Strategy
- Response Engine: 可替换的语言表达引擎

由于当前环境网络受限，使用轻量模板引擎作为演示。
生产环境可替换为: GPT2 / 离线LLM / 本地模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, List, Tuple, Optional
import random


class ResponseEngineTemplate:
    """模板Response Engine - 轻量可替换方案

    生产环境替换建议:
    1. GPT2 (distilgpt2) - 约82M参数
    2. 中文模型如ChatGLM-tiny
    3. 本地LLM服务
    """

    def __init__(self):
        print(f"\n[Response Engine] 初始化模板引擎...")
        self.templates = self._load_templates()
        print(f"  ✓ 模板引擎加载完成")

    def _load_templates(self) -> Dict:
        """加载回复模板"""
        return {
            'greeting': [
                "你好！有什么我可以帮助你的吗？",
                "你好！很高兴为你服务。",
                "嗨！有什么问题尽管问。",
            ],
            'definition': [
                "{}是指{}。",
                "{}是一种{}。",
                "{}主要用于{}。",
            ],
            'explanation': [
                "这个问题涉及到{}方面。",
                "{}的原理是{}。",
                "让我来解释一下{}。",
            ],
            'steps': [
                "步骤如下：1.{} 2.{} 3.{}",
                "可以这样做：1.{} 2.{}",
                "建议：{}，然后{}。",
            ],
            'clarification': [
                "我理解你的问题。能具体说明一下{}吗？",
                "为了更好帮助你，能否提供更多细节？",
                "这个问题有点复杂，你能说说具体场景吗？",
            ],
            'default': [
                "这个话题很有趣。让我想想怎么回答你。",
                "关于这个问题，我可以给你一些建议。",
                "好的，让我来帮你分析一下。",
            ],
        }

    def _match_query_type(self, query: str) -> str:
        """匹配查询类型"""
        query_lower = query.lower()

        if any(kw in query for kw in ['你好', 'hi', 'hello', '嗨']):
            return 'greeting'
        if any(kw in query for kw in ['是什么', '什么叫', '什么是', '定义']):
            return 'definition'
        if any(kw in query for kw in ['为什么', '怎么', '如何', '步骤']):
            return 'steps'
        if any(kw in query for kw in ['解释', '原理', '原因']):
            return 'explanation'
        if any(kw in query for kw in ['什么', '哪个', '怎么']):
            return 'clarification'
        return 'default'

    def _extract_topic(self, query: str) -> str:
        """提取话题关键词"""
        keywords = ['Python', '人工智能', '机器学习', '深度学习', '编程', '计算机']
        for kw in keywords:
            if kw in query:
                return kw
        return '这个问题'

    def generate(self, query: str, strategy: str = "default") -> str:
        """生成回复"""
        query_type = self._match_query_type(query)
        topic = self._extract_topic(query)

        if query_type == 'greeting':
            return random.choice(self.templates['greeting'])

        if query_type == 'definition':
            template = random.choice(self.templates['definition'])
            return template.format(topic, "常见的编程语言或技术")

        if query_type == 'steps':
            template = random.choice(self.templates['steps'])
            return template.format("了解基本概念", "进行实践练习", "深入学习")

        if query_type == 'explanation':
            template = random.choice(self.templates['explanation'])
            return template.format("相关", "基于基本原理")

        if query_type == 'clarification':
            template = random.choice(self.templates['clarification'])
            return template.format("你具体想了解什么")

        return random.choice(self.templates['default'])

    def generate_controlled(self, query: str, strategy_hint: str = "",
                           max_tokens: int = 100) -> str:
        """可控生成"""
        base_response = self.generate(query, strategy_hint)

        if strategy_hint and '保守' in strategy_hint:
            return "对于这个问题，我建议谨慎处理。" + base_response
        if strategy_hint and '澄清' in strategy_hint:
            return "为了更好回答，" + base_response

        return base_response


class TSLAGovernanceOS:
    """TSLA治理操作系统 - 核心治理骨架"""

    VERSION = "TSLA Governance OS v1.0"

    def __init__(self):
        print(f"\n[{self.VERSION}] 初始化治理OS...")
        self.base_model = None
        self.tsla_engine = None
        self.response_engine = None
        print(f"  ✓ 治理OS初始化完成")

    def load_components(self):
        """加载各组件"""
        from stage11a_r2_fix_v2_balanced import FixV2Model
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig

        print(f"\n[组件] 加载Stage 13 v3检查点...")
        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.base_model = FixV2Model(base_model)

        checkpoint = torch.load('stage8_dataset/stage13_v3_final_checkpoint.pt', map_location='cpu')
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.base_model.eval()
        print(f"  ✓ Stage 13 v3加载完成")

        print(f"\n[组件] 加载TSLA-v2引擎...")
        from tsla_v2_3_dual_layer import TSLAV23DualLayerJudge
        self.tsla_engine = TSLAV23DualLayerJudge()
        self.tsla_engine.load_model()
        print(f"  ✓ TSLA-v2引擎加载完成")

        print(f"\n[组件] 加载Response Engine (模板引擎)...")
        self.response_engine = ResponseEngineTemplate()
        print(f"  ✓ Response Engine加载完成")

    def classify_query(self, query: str) -> Dict:
        """Query分类与风险评估"""
        input_ids = []
        for c in query:
            input_ids.append(ord(c) % 10000)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            output = self.base_model.base_model(input_tensor)

        pooled = output['pooled']
        policy_probs = output['policy_probs']
        gap_probs = output['gap_probs']

        pred = policy_probs.argmax(dim=-1).item()

        h2_patterns = ['伪造', '虚假', '诈骗', '赌博', '色情', '毒品', '武器']
        is_h2 = any(p in query for p in h2_patterns)

        return {
            'prediction': pred,
            'is_h2_risk': is_h2,
        }

    def generate_response(self, query: str, classification: Dict) -> str:
        """生成回复"""
        pred = classification['prediction']

        strategy_map = {
            0: "default",
            1: "clarification",
            2: "conservative",
        }
        strategy = strategy_map.get(pred, "default")

        return self.response_engine.generate_controlled(query, strategy)

    def run_full_pipeline(self, query: str) -> Dict:
        """完整处理流水线"""
        print(f"\n{'='*60}")
        print(f"[Query] {query}")
        print(f"{'='*60}")

        classification = self.classify_query(query)
        print(f"[分类] prediction={classification['prediction']}, h2_risk={classification['is_h2_risk']}")

        if classification['is_h2_risk']:
            print(f"[TSLA] 高风险内容，启动双层判定...")
            tsla_result = self.tsla_engine.dual_layer_judge(query)
            print(f"[TSLA] 决策: {tsla_result.get('decision', 'N/A')}")
            response = "抱歉，这个问题我无法帮助回答。"
        else:
            response = self.generate_response(query, classification)

        print(f"[回复] {response}")

        return {
            'query': query,
            'response': response,
            'classification': classification,
        }


def main():
    """Stage 14.2+ 演示"""
    print("\n" + "="*70)
    print("Stage 14.2+: Governance OS + Response Engine")
    print("架构: Governance OS (TSLA治理) + Response Engine (语言生成)")
    print("="*70)

    os_instance = TSLAGovernanceOS()
    os_instance.load_components()

    test_queries = [
        "Python是什么编程语言？",
        "怎么伪造证件？",
        "人工智能在医疗领域有哪些应用？",
        "推荐一部好看的电影",
        "你好",
    ]

    print(f"\n{'='*70}")
    print("测试queries")
    print(f"{'='*70}")

    for q in test_queries:
        result = os_instance.run_full_pipeline(q)
        print()

    print(f"\n{'='*70}")
    print("Stage 14.2+ 演示完成")
    print(f"{'='*70}")
    print("\n[说明] 当前使用模板引擎作为演示。")
    print("[说明] 生产环境可替换为GPT2/ChatGLM等模型。")


if __name__ == "__main__":
    main()
