"""
LLM Client v1 - LLM客户端封装 v1

Phase 15 核心组件：
统一封装 OpenAI/Claude/本地模型 API
实现与 Conversation Orchestrator 的无缝集成
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class ResponseStrategy(Enum):
    """响应策略枚举"""
    DIRECT = "DIRECT"
    RETRIEVAL_FIRST = "RETRIEVAL_FIRST"
    CONSERVATIVE = "CONSERVATIVE"
    DECLINE = "DECLINE"
    REVIEW = "REVIEW"


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    source: str
    confidence: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class GovernanceContext:
    """治理上下文"""
    strategy: ResponseStrategy
    confidence: float
    risk_level: str
    requires_review: bool
    reasoning: str


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    layer: str
    timestamp: str


@dataclass
class LLMInput:
    """LLM输入结构"""
    system_prompt: str
    conversation_history: List[Dict[str, str]]
    current_query: str
    retrieval_context: Optional[List[RetrievalResult]] = None
    governance_context: Optional[GovernanceContext] = None
    memory_context: Optional[List[MemoryEntry]] = None
    
    def to_messages(self) -> List[Dict[str, str]]:
        """转换为LLM消息格式"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加历史（最近5轮）
        for turn in self.conversation_history[-5:]:
            messages.append(turn)
        
        # 构建当前输入
        user_input = self._build_user_input()
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _build_user_input(self) -> str:
        """构建用户输入（包含上下文）"""
        parts = [f"用户问题：{self.current_query}"]
        
        # 添加检索上下文
        if self.retrieval_context:
            parts.append("\n检索到的相关信息：")
            for i, ctx in enumerate(self.retrieval_context[:3], 1):
                parts.append(f"[{i}] {ctx.content[:200]}...")
        
        # 添加记忆上下文
        if self.memory_context:
            parts.append("\n相关记忆：")
            for mem in self.memory_context[:2]:
                parts.append(f"- {mem.content[:100]}...")
        
        return "\n".join(parts)


@dataclass
class LLMOutput:
    """LLM输出结构"""
    strategy: ResponseStrategy
    confidence: float
    response: str
    reasoning: str
    citations: List[str]
    raw_output: str
    latency_ms: float
    token_usage: Dict[str, int]
    
    @classmethod
    def from_json(cls, json_str: str, **metadata) -> 'LLMOutput':
        """从JSON解析"""
        try:
            data = json.loads(json_str)
            return cls(
                strategy=ResponseStrategy(data.get('strategy', 'CONSERVATIVE')),
                confidence=float(data.get('confidence', 0.5)),
                response=data.get('response', '处理响应时出现问题'),
                reasoning=data.get('reasoning', ''),
                citations=data.get('citations', []),
                raw_output=json_str,
                **metadata
            )
        except Exception as e:
            # 解析失败，降级处理
            return cls(
                strategy=ResponseStrategy.CONSERVATIVE,
                confidence=0.5,
                response=f"抱歉，处理响应时出现问题。原始输出：{json_str[:100]}...",
                reasoning=f"解析错误: {str(e)}",
                citations=[],
                raw_output=json_str,
                **metadata
            )


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    async def generate(
        self,
        input_data: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        """生成响应"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
    
    def _build_system_prompt(
        self,
        governance_context: Optional[GovernanceContext] = None
    ) -> str:
        """构建系统Prompt"""
        base_prompt = """你是一个具有治理意识的AI助手。你的回答必须遵循以下协议：

## 响应策略
根据置信度和风险选择策略：
- DIRECT: 置信度>0.9，直接回答
- RETRIEVAL_FIRST: 需要验证，先检索再回答  
- CONSERVATIVE: 置信度0.6-0.9，说明不确定性
- DECLINE: 置信度<0.6，诚实拒绝
- REVIEW: 高风险内容，标记审查

## 回答要求
1. 如果使用了检索结果，必须标注来源[来源:N]
2. 如果不确定，必须说明置信度
3. 如果高风险，必须保守或拒绝
4. 保持人格一致性

## 输出格式
你必须以JSON格式输出，不要包含其他内容：
{
    "strategy": "DIRECT|RETRIEVAL_FIRST|CONSERVATIVE|DECLINE|REVIEW",
    "confidence": 0.0-1.0,
    "response": "最终回答",
    "reasoning": "推理过程",
    "citations": ["来源1", "来源2"]
}"""
        
        if governance_context:
            governance_section = f"""

## 当前治理建议
- 建议策略：{governance_context.strategy.value}
- 置信度评估：{governance_context.confidence:.2f}
- 风险等级：{governance_context.risk_level}
- 需要审查：{'是' if governance_context.requires_review else '否'}
"""
            base_prompt += governance_section
        
        return base_prompt


class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端（用于测试）"""
    
    def __init__(self):
        self.responses = {
            "python": {
                "strategy": "DIRECT",
                "confidence": 0.95,
                "response": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁、易读的语法著称。",
                "reasoning": "这是广为人知的事实，置信度高",
                "citations": []
            },
            "之前": {
                "strategy": "RETRIEVAL_FIRST",
                "confidence": 0.88,
                "response": "根据我的记忆[来源:1]，我们之前讨论了项目目标。",
                "reasoning": "用户提到'之前'，需要检索记忆",
                "citations": ["记忆记录"]
            },
            "不确定": {
                "strategy": "CONSERVATIVE",
                "confidence": 0.65,
                "response": "关于这个问题，我不太确定（置信度：65%）。根据现有信息，可能是...",
                "reasoning": "信息不足，需要保守回答",
                "citations": []
            },
            "xyz": {
                "strategy": "DECLINE",
                "confidence": 0.2,
                "response": "抱歉，我不了解'xyzabc'是什么。",
                "reasoning": "对主题完全不了解",
                "citations": []
            }
        }
    
    async def generate(
        self,
        input_data: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        """模拟生成"""
        start_time = time.time()
        
        query = input_data.current_query.lower()
        
        # 简单关键词匹配
        if "python" in query:
            data = self.responses["python"]
        elif "之前" in query or "previous" in query:
            data = self.responses["之前"]
        elif "不确定" in query or "latest" in query:
            data = self.responses["不确定"]
        elif "xyz" in query:
            data = self.responses["xyz"]
        else:
            data = {
                "strategy": "DIRECT",
                "confidence": 0.8,
                "response": f"关于'{input_data.current_query}'，这是一个有趣的问题。",
                "reasoning": "一般性问题",
                "citations": []
            }
        
        latency = (time.time() - start_time) * 1000
        
        return LLMOutput(
            strategy=ResponseStrategy(data["strategy"]),
            confidence=data["confidence"],
            response=data["response"],
            reasoning=data["reasoning"],
            citations=data["citations"],
            raw_output=json.dumps(data, ensure_ascii=False),
            latency_ms=latency,
            token_usage={"prompt": 100, "completion": 50}
        )
    
    async def health_check(self) -> bool:
        return True


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    async def generate(
        self,
        input_data: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        """调用OpenAI API生成响应"""
        start_time = time.time()
        
        # 构建系统prompt
        system_prompt = self._build_system_prompt(input_data.governance_context)
        input_data.system_prompt = system_prompt
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=input_data.to_messages(),
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            latency = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            return LLMOutput.from_json(
                content,
                latency_ms=latency,
                token_usage={
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            # API调用失败，降级处理
            latency = (time.time() - start_time) * 1000
            return LLMOutput(
                strategy=ResponseStrategy.CONSERVATIVE,
                confidence=0.5,
                response=f"服务暂时不可用，请稍后重试。错误：{str(e)}",
                reasoning="API调用失败",
                citations=[],
                raw_output=str(e),
                latency_ms=latency,
                token_usage={"prompt": 0, "completion": 0}
            )
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 简单测试调用
            test_input = LLMInput(
                system_prompt="You are a helpful assistant.",
                conversation_history=[],
                current_query="Hi"
            )
            await self.generate(test_input, max_tokens=10)
            return True
        except:
            return False


class ClaudeClient(BaseLLMClient):
    """Claude客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    async def generate(
        self,
        input_data: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        """调用Claude API生成响应"""
        start_time = time.time()
        
        system_prompt = self._build_system_prompt(input_data.governance_context)
        
        try:
            # Claude使用不同的消息格式
            messages = []
            for msg in input_data.to_messages():
                if msg["role"] != "system":
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages
            )
            
            latency = (time.time() - start_time) * 1000
            content = response.content[0].text
            
            # 尝试解析JSON，Claude可能不严格遵循格式
            return LLMOutput.from_json(
                content,
                latency_ms=latency,
                token_usage={
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens
                }
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return LLMOutput(
                strategy=ResponseStrategy.CONSERVATIVE,
                confidence=0.5,
                response=f"服务暂时不可用，请稍后重试。错误：{str(e)}",
                reasoning="API调用失败",
                citations=[],
                raw_output=str(e),
                latency_ms=latency,
                token_usage={"prompt": 0, "completion": 0}
            )
    
    async def health_check(self) -> bool:
        try:
            test_input = LLMInput(
                system_prompt="You are a helpful assistant.",
                conversation_history=[],
                current_query="Hi"
            )
            await self.generate(test_input, max_tokens=10)
            return True
        except:
            return False


class DeepSeekClient(BaseLLMClient):
    """DeepSeek客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com"
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    async def generate(
        self,
        input_data: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        """调用DeepSeek API生成响应"""
        start_time = time.time()
        
        # 构建系统prompt
        system_prompt = self._build_system_prompt(input_data.governance_context)
        input_data.system_prompt = system_prompt
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=input_data.to_messages(),
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            latency = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            return LLMOutput.from_json(
                content,
                latency_ms=latency,
                token_usage={
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            # API调用失败，降级处理
            latency = (time.time() - start_time) * 1000
            return LLMOutput(
                strategy=ResponseStrategy.CONSERVATIVE,
                confidence=0.5,
                response=f"服务暂时不可用，请稍后重试。错误：{str(e)}",
                reasoning="API调用失败",
                citations=[],
                raw_output=str(e),
                latency_ms=latency,
                token_usage={"prompt": 0, "completion": 0}
            )
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            test_input = LLMInput(
                system_prompt="You are a helpful assistant.",
                conversation_history=[],
                current_query="Hi"
            )
            await self.generate(test_input, max_tokens=10)
            return True
        except:
            return False


class LLMClientFactory:
    """LLM客户端工厂"""
    
    @staticmethod
    def create_client(
        provider: str,
        **config
    ) -> BaseLLMClient:
        """创建客户端"""
        
        if provider == "mock":
            return MockLLMClient()
        elif provider == "openai":
            return OpenAIClient(
                api_key=config.get('api_key'),
                model=config.get('model', 'gpt-4')
            )
        elif provider == "claude":
            return ClaudeClient(
                api_key=config.get('api_key'),
                model=config.get('model', 'claude-3-opus-20240229')
            )
        elif provider == "deepseek":
            return DeepSeekClient(
                api_key=config.get('api_key'),
                model=config.get('model', 'deepseek-chat')
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")


# 便捷函数
def create_mock_client() -> BaseLLMClient:
    """创建模拟客户端"""
    return LLMClientFactory.create_client("mock")


def create_openai_client(model: str = "gpt-4") -> BaseLLMClient:
    """创建OpenAI客户端"""
    return LLMClientFactory.create_client("openai", model=model)


def create_claude_client(model: str = "claude-3-opus-20240229") -> BaseLLMClient:
    """创建Claude客户端"""
    return LLMClientFactory.create_client("claude", model=model)


def create_deepseek_client(model: str = "deepseek-chat") -> BaseLLMClient:
    """创建DeepSeek客户端"""
    return LLMClientFactory.create_client("deepseek", model=model)


async def test_client():
    """测试客户端"""
    print("="*70)
    print("LLM Client Test")
    print("="*70)
    
    # 测试模拟客户端
    print("\n1. 测试模拟客户端")
    mock_client = create_mock_client()
    
    test_input = LLMInput(
        system_prompt="You are a helpful assistant.",
        conversation_history=[],
        current_query="Python是什么？"
    )
    
    output = await mock_client.generate(test_input)
    print(f"策略: {output.strategy.value}")
    print(f"置信度: {output.confidence:.2f}")
    print(f"响应: {output.response[:100]}...")
    print(f"延迟: {output.latency_ms:.2f}ms")
    
    # 测试OpenAI（如果有API key）
    if os.getenv("OPENAI_API_KEY"):
        print("\n2. 测试OpenAI客户端")
        try:
            openai_client = create_openai_client(model="gpt-3.5-turbo")
            
            if await openai_client.health_check():
                print("✓ 健康检查通过")
                
                output = await openai_client.generate(test_input)
                print(f"策略: {output.strategy.value}")
                print(f"置信度: {output.confidence:.2f}")
                print(f"响应: {output.response[:100]}...")
            else:
                print("✗ 健康检查失败")
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    else:
        print("\n2. 跳过OpenAI测试（未设置API key）")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())
