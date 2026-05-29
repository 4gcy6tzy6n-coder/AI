"""
LLM Client with Streaming v1 - 支持流式输出的LLM客户端 v1

关键改进：
1. 流式生成接口
2. TTFT (Time To First Token) 监控
3. 渐进式响应输出
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from openai import AsyncOpenAI


@dataclass
class StreamingChunk:
    """流式输出块"""
    content: str
    is_first: bool = False
    is_last: bool = False
    timestamp_ms: float = 0


@dataclass
class StreamingResult:
    """流式生成结果"""
    full_response: str
    ttft_ms: float  # Time To First Token
    total_latency_ms: float
    token_usage: Dict[str, int]
    chunk_count: int


@dataclass
class LLMInput:
    """LLM输入结构"""
    system_prompt: str
    conversation_history: List[Dict]
    current_query: str
    retrieval_context: Optional[Dict] = None
    governance_context: Optional[Dict] = None
    memory_context: Optional[List[Dict]] = None


@dataclass
class LLMOutput:
    """LLM输出结构"""
    strategy: str
    confidence: float
    response: str
    reasoning: str
    citations: List[str]
    raw_output: str
    latency_ms: float
    token_usage: Dict[str, int]


class BaseStreamingLLMClient(ABC):
    """流式LLM客户端基类"""
    
    @abstractmethod
    async def generate_stream(self, input_data: LLMInput) -> AsyncGenerator[StreamingChunk, None]:
        """流式生成"""
        pass
    
    @abstractmethod
    async def generate(self, input_data: LLMInput) -> LLMOutput:
        """非流式生成"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class DeepSeekStreamingClient(BaseStreamingLLMClient):
    """DeepSeek流式客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com"
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _build_messages(self, input_data: LLMInput) -> List[Dict]:
        """构建消息列表"""
        messages = [
            {"role": "system", "content": input_data.system_prompt}
        ]
        
        # 添加历史对话
        for turn in input_data.conversation_history[-5:]:  # 只取最近5轮
            messages.append({"role": "user", "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("ai", "")})
        
        # 添加当前查询
        messages.append({"role": "user", "content": input_data.current_query})
        
        return messages
    
    async def generate_stream(self, input_data: LLMInput) -> AsyncGenerator[StreamingChunk, None]:
        """流式生成，监控TTFT"""
        
        start_time = time.time()
        first_token_time = None
        chunk_count = 0
        
        messages = self._build_messages(input_data)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=1000
            )
            
            async for chunk in stream:
                chunk_count += 1
                
                # 记录首token时间
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = (first_token_time - start_time) * 1000
                else:
                    ttft = None
                
                # 提取内容
                content = chunk.choices[0].delta.content or ""
                
                if content:
                    yield StreamingChunk(
                        content=content,
                        is_first=(chunk_count == 1),
                        is_last=False,
                        timestamp_ms=(time.time() - start_time) * 1000
                    )
            
            # 发送结束标记
            yield StreamingChunk(
                content="",
                is_first=False,
                is_last=True,
                timestamp_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            print(f"[Streaming Error] {e}")
            yield StreamingChunk(
                content=f"[生成错误: {str(e)}]",
                is_first=True,
                is_last=True,
                timestamp_ms=(time.time() - start_time) * 1000
            )
    
    async def generate(self, input_data: LLMInput) -> LLMOutput:
        """非流式生成（兼容旧接口）"""
        
        start_time = time.time()
        full_content = []
        
        async for chunk in self.generate_stream(input_data):
            full_content.append(chunk.content)
        
        total_latency = (time.time() - start_time) * 1000
        raw_output = "".join(full_content)
        
        # 解析输出
        return self._parse_output(raw_output, total_latency)
    
    def _parse_output(self, raw_output: str, latency_ms: float) -> LLMOutput:
        """解析LLM输出"""
        try:
            # 尝试解析JSON
            if raw_output.strip().startswith("{"):
                data = json.loads(raw_output)
                return LLMOutput(
                    strategy=data.get("strategy", "CONSERVATIVE"),
                    confidence=data.get("confidence", 0.6),
                    response=data.get("response", raw_output),
                    reasoning=data.get("reasoning", ""),
                    citations=data.get("citations", []),
                    raw_output=raw_output,
                    latency_ms=latency_ms,
                    token_usage={"total": len(raw_output) // 4}  # 粗略估算
                )
        except:
            pass
        
        # 非JSON格式，直接返回
        return LLMOutput(
            strategy="CONSERVATIVE",
            confidence=0.6,
            response=raw_output,
            reasoning="",
            citations=[],
            raw_output=raw_output,
            latency_ms=latency_ms,
            token_usage={"total": len(raw_output) // 4}
        )
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            return response is not None
        except:
            return False


class MockStreamingClient(BaseStreamingLLMClient):
    """模拟流式客户端（用于测试）"""
    
    async def generate_stream(self, input_data: LLMInput) -> AsyncGenerator[StreamingChunk, None]:
        """模拟流式生成"""
        
        start_time = time.time()
        
        # 模拟延迟
        await asyncio.sleep(0.5)  # 500ms TTFT
        
        # 模拟分块输出
        chunks = [
            "根据",
            "检索",
            "到的",
            "信息",
            "，",
            "项目",
            "目标",
            "是",
            "在",
            "2024",
            "年",
            "Q3",
            "完成",
            "核心",
            "功能",
            "开发",
            "。",
        ]
        
        for i, chunk in enumerate(chunks):
            yield StreamingChunk(
                content=chunk,
                is_first=(i == 0),
                is_last=False,
                timestamp_ms=(time.time() - start_time) * 1000
            )
            await asyncio.sleep(0.05)  # 50ms 每块
        
        yield StreamingChunk(
            content="",
            is_first=False,
            is_last=True,
            timestamp_ms=(time.time() - start_time) * 1000
        )
    
    async def generate(self, input_data: LLMInput) -> LLMOutput:
        """非流式生成"""
        
        start_time = time.time()
        full_content = []
        
        async for chunk in self.generate_stream(input_data):
            full_content.append(chunk.content)
        
        total_latency = (time.time() - start_time) * 1000
        
        return LLMOutput(
            strategy="RETRIEVAL_FIRST",
            confidence=0.85,
            response="".join(full_content),
            reasoning="基于检索结果",
            citations=["来源1"],
            raw_output="",
            latency_ms=total_latency,
            token_usage={"total": 50}
        )
    
    async def health_check(self) -> bool:
        return True


# 便捷函数
def create_deepseek_streaming_client() -> DeepSeekStreamingClient:
    """创建DeepSeek流式客户端"""
    return DeepSeekStreamingClient()


def create_mock_streaming_client() -> MockStreamingClient:
    """创建模拟流式客户端"""
    return MockStreamingClient()


# 测试
async def test_streaming():
    """测试流式输出"""
    print("="*70)
    print("流式输出测试")
    print("="*70)
    
    client = create_mock_streaming_client()
    
    input_data = LLMInput(
        system_prompt="你是一个有帮助的AI助手",
        conversation_history=[],
        current_query="项目的目标是什么？"
    )
    
    print("\n开始流式生成...")
    start_time = time.time()
    first_token_time = None
    
    async for chunk in client.generate_stream(input_data):
        if chunk.is_first:
            first_token_time = time.time()
            ttft = (first_token_time - start_time) * 1000
            print(f"\n[首Token] TTFT: {ttft:.0f}ms")
            print("响应: ", end="", flush=True)
        
        print(chunk.content, end="", flush=True)
        
        if chunk.is_last:
            total = (time.time() - start_time) * 1000
            print(f"\n\n[完成] 总耗时: {total:.0f}ms")


if __name__ == "__main__":
    asyncio.run(test_streaming())
