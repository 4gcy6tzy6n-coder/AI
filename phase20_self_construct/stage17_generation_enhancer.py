"""
Stage 17: 生成增强模块

功能:
1. GenerationEngine - 生成引擎接口，支持多种后端
   - LocalLLM: GPT2/ChatGLM本地模型
   - TemplateEngine: 当前模板引擎(兼容)
   - RemoteAPI: OpenAI兼容API

2. MultiTurnContext - 多轮对话上下文管理
   - 维护对话历史
   - 提取上下文关键信息
   - 支持话题追踪

3. ChainReasoning - 链式推理增强
   - 复杂因果链分解
   - 跨领域知识关联
   - 推理步骤验证
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict


class GenerationBackend:
    """生成后端基类"""

    def generate(self, prompt: str, max_length: int = 200) -> str:
        raise NotImplementedError


class TemplateGenerationEngine(GenerationBackend):
    """模板生成引擎 - 兼容当前系统"""

    VERSION = "Template Engine v1.0"

    RESPONSE_STRATEGIES = {
        'definition': {
            'template': "根据分析，{subject}是指{features}。{example}",
            'style': '学术说明',
        },
        'comparison': {
            'template': "对比分析：{item1}的特点是{features1}，而{item2}的特点是{features2}。主要区别在于{differences}。",
            'style': '对比分析',
        },
        'steps': {
            'template': "解决这个问题需要{num_steps}个步骤：{steps_text}。每一步都至关重要，请按顺序执行。",
            'style': '步骤指导',
        },
        'causal': {
            'template': "分析原因如下：{direct_causes}。此外还有{indirect_causes}。结论：{conclusion}。",
            'style': '因果分析',
        },
        'evaluation': {
            'template': "综合评估：优点包括{pros}，缺点包括{cons}。整体评价：{verdict}。",
            'style': '综合评估',
        },
        'application': {
            'template': "主要应用场景包括：{scenarios}。具体使用方式：{usage_methods}。效果：{effects}。",
            'style': '应用说明',
        },
        'direct': {
            'template': "{answer}",
            'style': '直接回答',
        },
        'blocked': {
            'template': "抱歉，这个问题我无法帮助回答。",
            'style': '拒绝回答',
        },
    }

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.conversation_history = []

    def generate(self, prompt: str, reasoning_type: str = 'direct',
                 context: Dict = None, max_length: int = 200) -> str:
        """生成回答"""
        strategy = self.RESPONSE_STRATEGIES.get(reasoning_type, self.RESPONSE_STRATEGIES['direct'])

        if reasoning_type == 'blocked':
            return strategy['template']

        if context and context.get('knowledge_answer'):
            return self._generate_from_knowledge(context, strategy)
        elif context and context.get('decomposition'):
            return self._generate_from_reasoning(context, strategy)
        else:
            return self._generate_fallback(prompt, strategy)

    def _generate_from_knowledge(self, context: Dict, strategy: Dict) -> str:
        """从知识生成"""
        kb_answer = context.get('knowledge_answer', '')
        reasoning_type = context.get('reasoning_type', 'direct')

        if reasoning_type == 'definition':
            return f"根据分析，{kb_answer}"
        elif reasoning_type == 'steps':
            return f"关于这个问题：{kb_answer}"
        elif reasoning_type == 'comparison':
            return f"对比分析：{kb_answer}"
        else:
            return kb_answer

    def _generate_from_reasoning(self, context: Dict, strategy: Dict) -> str:
        """从推理生成"""
        decomp = context.get('decomposition', {})
        steps = decomp.get('steps', [])
        reasoning_type = decomp.get('reasoning_type', 'direct')

        if steps:
            steps_text = '；'.join([f"{i+1}. {s}" for i, s in enumerate(steps[:5])])
            return f"分析如下：{steps_text}"
        return "根据分析得出结论。"

    def _generate_fallback(self, prompt: str, strategy: Dict) -> str:
        """备用生成"""
        return f"关于您的问题：{prompt[:20]}...我需要更多信息才能准确回答。"


class LocalLLMEngine(GenerationBackend):
    """本地LLM生成引擎"""

    VERSION = "Local LLM Engine v1.0"

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._lazy_load()

    def _lazy_load(self):
        """延迟加载模型"""
        try:
            import torch
            if "gpt2" in self.model_name.lower():
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                print(f"  [LocalLLM] 加载 GPT2...")
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.model.to(self.device)
                print(f"  [LocalLLM] GPT2 加载完成")
            elif "chatglm" in self.model_name.lower():
                from transformers import AutoModel, AutoTokenizer
                print(f"  [LocalLLM] 加载 ChatGLM...")
                self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-tiny', trust_remote_code=True)
                self.model = AutoModel.from_pretrained('THUDM/chatglm-tiny', trust_remote_code=True)
                self.model.to(self.device)
                print(f"  [LocalLLM] ChatGLM 加载完成")
        except Exception as e:
            print(f"  [LocalLLM] 模型加载失败: {e}")
            print(f"  [LocalLLM] 回退到模板引擎")
            self.model = None

    def generate(self, prompt: str, max_length: int = 200) -> str:
        """生成文本"""
        if self.model is None:
            return None

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            print(f"  [LocalLLM] 生成失败: {e}")
            return None


class RemoteAPIEngine(GenerationBackend):
    """远程API生成引擎"""

    VERSION = "Remote API Engine v1.0"

    def __init__(self, api_url: str = None, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_length: int = 200) -> str:
        """调用远程API"""
        import requests

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_length,
            "temperature": 0.7,
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"  [RemoteAPI] 请求失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"  [RemoteAPI] 调用失败: {e}")
            return None


class MultiTurnContext:
    """多轮对话上下文管理器"""

    VERSION = "MultiTurn Context v1.0"

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        self.topic_stack = []
        self.entities = {}

    def add_turn(self, role: str, content: str, metadata: Dict = None):
        """添加对话轮次"""
        turn = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        self.history.append(turn)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        if not self.history:
            return ""

        summary_parts = []
        for turn in self.history[-3:]:
            role = "用户" if turn['role'] == 'user' else "系统"
            summary_parts.append(f"{role}: {turn['content'][:50]}")

        return " | ".join(summary_parts)

    def extract_entities(self, text: str):
        """提取实体"""
        import re
        words = re.findall(r'[\w]+', text)
        for w in words:
            if w not in self.entities:
                self.entities[w] = 0
            self.entities[w] += 1

    def get_last_topic(self) -> Optional[str]:
        """获取最后话题"""
        return self.topic_stack[-1] if self.topic_stack else None

    def push_topic(self, topic: str):
        """压入话题"""
        self.topic_stack.append(topic)
        if len(self.topic_stack) > 5:
            self.topic_stack.pop(0)

    def clear(self):
        """清空上下文"""
        self.history.clear()
        self.topic_stack.clear()
        self.entities.clear()


class ChainReasoning:
    """链式推理增强"""

    VERSION = "Chain Reasoning v1.0"

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base

    def decompose_chain(self, query: str, context: MultiTurnContext = None) -> Dict:
        """链式分解"""
        result = {
            'query': query,
            'reasoning_type': 'chain',
            'steps': [],
            'entities': [],
            'relations': [],
            'confidence': 0.0,
        }

        import re
        query_clean = re.sub(r'[^\w\s]', '', query)
        words = query_clean.split()

        result['entities'] = words[:10]

        if any(kw in query for kw in ['为什么', '原因', '导致', '造成']):
            result['steps'].append('识别因果关系')
            result['steps'].append('收集相关因素')
            result['steps'].append('分析因果链条')
            result['steps'].append('得出结论')
            result['reasoning_type'] = 'causal_chain'
        elif any(kw in query for kw in ['怎么', '如何', '方法', '步骤']):
            result['steps'].append('明确目标')
            result['steps'].append('分解任务')
            result['steps'].append('执行步骤')
            result['steps'].append('验证结果')
            result['reasoning_type'] = 'steps_chain'
        elif any(kw in query for kw in ['比较', '区别', '不同', '异同']):
            result['steps'].append('识别比较对象')
            result['steps'].append('提取特征')
            result['steps'].append('对比分析')
            result['steps'].append('总结差异')
            result['reasoning_type'] = 'comparison_chain'
        else:
            result['steps'].append('理解问题')
            result['steps'].append('检索知识')
            result['steps'].append('生成回答')
            result['reasoning_type'] = 'direct_chain'

        result['confidence'] = 0.85 if len(result['steps']) > 0 else 0.5

        return result

    def verify_chain(self, steps: List[str]) -> Dict:
        """验证推理链"""
        verification = {
            'valid': True,
            'issues': [],
            'suggestions': [],
        }

        if len(steps) < 2:
            verification['valid'] = False
            verification['issues'].append('推理链过短')

        for i, step in enumerate(steps):
            if not step or len(step) < 3:
                verification['issues'].append(f'步骤{i+1}内容不足')

        return verification


class Stage17GenerationEnhancer:
    """Stage 17 生成增强器"""

    VERSION = "Stage 17 Generation Enhancer v1.0"

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.template_engine = TemplateGenerationEngine(knowledge_base)
        self.local_llm = None
        self.remote_api = None
        self.multi_turn = MultiTurnContext()
        self.chain_reasoning = ChainReasoning(knowledge_base)
        self.current_backend = 'template'

    def enable_local_llm(self, model_name: str = "gpt2", device: str = "cpu"):
        """启用本地LLM"""
        self.local_llm = LocalLLMEngine(model_name, device)
        if self.local_llm.model is not None:
            self.current_backend = 'local'
            print(f"  [Stage17] 已切换到本地LLM: {model_name}")
        else:
            print(f"  [Stage17] 本地LLM不可用，保持模板引擎")

    def enable_remote_api(self, api_url: str = None, api_key: str = None):
        """启用远程API"""
        self.remote_api = RemoteAPIEngine(api_url, api_key)
        self.current_backend = 'remote'
        print(f"  [Stage17] 已切换到远程API")

    def generate(self, query: str, reasoning_result: Dict = None,
                 knowledge_results: List[Dict] = None, use_chain: bool = False) -> Dict:
        """生成回答"""
        context = {
            'query': query,
            'knowledge_answer': knowledge_results[0]['answer'] if knowledge_results else None,
            'decomposition': reasoning_result,
        }

        reasoning_type = 'direct'
        if reasoning_result:
            reasoning_type = reasoning_result.get('decomposition', {}).get('reasoning_type', 'direct')

        if use_chain and self.chain_reasoning:
            chain_result = self.chain_reasoning.decompose_chain(query, self.multi_turn)
            context['chain'] = chain_result
            if chain_result['reasoning_type'].endswith('_chain'):
                reasoning_type = chain_result['reasoning_type']

        if self.current_backend == 'local' and self.local_llm:
            prompt = self._build_prompt(query, context)
            response = self.local_llm.generate(prompt)
            if response:
                return {'response': response, 'backend': 'local_llm', 'context': context}

        if self.current_backend == 'remote' and self.remote_api:
            prompt = self._build_prompt(query, context)
            response = self.remote_api.generate(prompt)
            if response:
                return {'response': response, 'backend': 'remote_api', 'context': context}

        response = self.template_engine.generate(query, reasoning_type, context)
        return {'response': response, 'backend': 'template', 'context': context}

    def _build_prompt(self, query: str, context: Dict) -> str:
        """构建提示"""
        kb_answer = context.get('knowledge_answer', '')
        chain = context.get('chain', {})

        prompt = f"问题：{query}\n"

        if kb_answer:
            prompt += f"相关知识：{kb_answer[:200]}\n"

        if chain and chain.get('steps'):
            prompt += f"推理步骤：{' -> '.join(chain['steps'])}\n"

        prompt += "请给出清晰、准确的回答："

        return prompt

    def add_conversation_turn(self, role: str, content: str, metadata: Dict = None):
        """添加对话轮次"""
        self.multi_turn.add_turn(role, content, metadata)

    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        return self.multi_turn.get_context_summary()

    def reset_conversation(self):
        """重置对话"""
        self.multi_turn.clear()


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("Stage 17: 生成增强模块")
    print(f"{'='*70}")

    enhancer = Stage17GenerationEnhancer()

    print(f"\n[1] 模板引擎测试")
    print(f"-" * 50)
    result = enhancer.generate("量子计算是什么", reasoning_result=None)
    print(f"查询: 量子计算是什么")
    print(f"引擎: {result['backend']}")
    print(f"回复: {result['response'][:80]}...")

    print(f"\n[2] 链式推理测试")
    print(f"-" * 50)
    chain_result = enhancer.chain_reasoning.decompose_chain("为什么天空是蓝色的")
    print(f"查询: 为什么天空是蓝色的")
    print(f"推理类型: {chain_result['reasoning_type']}")
    print(f"推理步骤: {chain_result['steps']}")

    print(f"\n[3] 多轮对话上下文测试")
    print(f"-" * 50)
    enhancer.add_conversation_turn("user", "我想了解人工智能")
    enhancer.add_conversation_turn("assistant", "人工智能是计算机科学的一个重要分支...")
    enhancer.add_conversation_turn("user", "它和机器学习有什么关系")
    summary = enhancer.get_conversation_summary()
    print(f"对话摘要: {summary}")

    print(f"\n[4] 生成器配置")
    print(f"-" * 50)
    print(f"当前后端: {enhancer.current_backend}")
    print(f"模板引擎: ✓")
    print(f"本地LLM: {'可选' if enhancer.local_llm else '未加载'}")
    print(f"远程API: {'可选' if enhancer.remote_api else '未配置'}")

    print(f"\n{'='*70}")
    print("Stage 17 生成增强模块 测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
