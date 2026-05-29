"""
Orchestrator with Real LLM v2 - 集成真实LLM的对话主控层 v2

改进：
1. 使用 PromptBuilderV2，修复上下文注入问题
2. 添加调试输出，打印完整 Prompt
3. 改进的上下文格式化
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase13.orchestrator.conversation_orchestrator_v1 import (
    ConversationOrchestrator as BaseOrchestrator,
    ConversationContext,
    OrchestratorResponse,
    ResponseStrategy
)
from phase11.core_services.retrieval_service_v1 import RetrievalService
from phase11.core_services.governance_service_v1 import GovernanceService
from phase11.core_services.memory_service_v1 import MemoryService
from phase15.llm_client_v1 import (
    BaseLLMClient,
    LLMInput,
    LLMOutput,
    ResponseStrategy as LLMResponseStrategy,
    create_mock_client
)
from phase15.prompt_builder_v2 import PromptBuilderV2, create_prompt_builder_v2


class LLMInputAdapter:
    """将 PromptBuilderV2 输出适配为 LLMInput"""
    
    @staticmethod
    def adapt(prompt_data: Dict[str, Any]) -> LLMInput:
        """适配为 LLMInput"""
        return LLMInput(
            system_prompt=prompt_data["system_prompt"],
            conversation_history=prompt_data["conversation_history"],
            current_query=prompt_data["user_prompt"],
            retrieval_context=None,  # 已包含在 user_prompt 中
            governance_context=None,  # 已包含在 user_prompt 中
            memory_context=None  # 已包含在 user_prompt 中
        )


class ConversationOrchestratorWithLLMV2(BaseOrchestrator):
    """
    集成真实LLM的对话主控层 v2
    
    改进：
    - 使用 PromptBuilderV2 正确注入上下文
    - 添加调试模式，可打印完整 Prompt
    """
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        governance_service: GovernanceService,
        memory_service: MemoryService,
        llm_client: BaseLLMClient,
        prompt_builder: Optional[PromptBuilderV2] = None,
        debug_mode: bool = False
    ):
        super().__init__(retrieval_service, governance_service, memory_service)
        self.llm = llm_client
        self.prompt_builder = prompt_builder or create_prompt_builder_v2()
        self.debug_mode = debug_mode
    
    def _generate_response(
        self,
        user_input: str,
        assembled_context: dict,
        governance_result: dict,
        retrieval_result: Optional[Any]
    ) -> OrchestratorResponse:
        """生成响应（真实LLM版本）"""
        import time
        start_time = time.time()
        
        # 获取会话上下文
        session_id = assembled_context.get('session_id', 'default')
        context = self.sessions.get(session_id)
        
        # 1. 构建 Prompt（使用 V2）
        prompt_data = self.prompt_builder.build(
            query=user_input,
            history=context.history if context else [],
            retrieval_results=retrieval_result,
            governance_result=governance_result,
            memory_context=context.retrieved_memories if context else []
        )
        
        # 调试模式：打印完整 Prompt
        if self.debug_mode:
            print("\n" + "="*70)
            print("DEBUG: 完整 Prompt 发送给 LLM")
            print("="*70)
            print(prompt_data["full_prompt"])
            print("="*70 + "\n")
        
        # 2. 适配为 LLMInput
        llm_input = LLMInputAdapter.adapt(prompt_data)
        
        # 3. 调用真实LLM生成响应
        try:
            llm_output = asyncio.get_event_loop().run_until_complete(
                self.llm.generate(llm_input)
            )
        except Exception as e:
            print(f"[警告] LLM调用失败: {e}")
            llm_output = LLMOutput(
                strategy=LLMResponseStrategy.CONSERVATIVE,
                confidence=0.5,
                response="服务暂时不可用，请稍后重试。",
                reasoning=f"LLM调用失败: {e}",
                citations=[],
                raw_output=str(e),
                latency_ms=0,
                token_usage={"prompt": 0, "completion": 0}
            )
        
        # 4. 验证和校准
        validated_output = self._validate_and_calibrate(llm_output, governance_result)
        
        # 5. 构建最终响应
        generation_latency = (time.time() - start_time) * 1000
        
        response = OrchestratorResponse(
            response_text=validated_output.response,
            strategy=validated_output.strategy,
            confidence=validated_output.confidence,
            sources=validated_output.citations,
            reasoning=validated_output.reasoning,
            metadata={
                "retrieval_count": len(retrieval_result.results) if retrieval_result else 0,
                "llm_latency_ms": llm_output.latency_ms,
                "total_latency_ms": generation_latency,
                "token_usage": llm_output.token_usage
            },
            timestamp=datetime.now()
        )
        
        return response
    
    def _validate_and_calibrate(
        self,
        llm_output: LLMOutput,
        governance_result: dict
    ) -> LLMOutput:
        """验证和校准LLM输出"""
        
        gov_strategy = governance_result.get('strategy', ResponseStrategy.DIRECT)
        gov_confidence = governance_result.get('confidence', 0.7)
        risk_level = governance_result.get('risk_level', 'low')
        
        # 转换为 ResponseStrategy
        if isinstance(gov_strategy, str):
            gov_strategy = ResponseStrategy(gov_strategy)
        
        # 检查1：高风险但LLM过于自信
        if risk_level == 'high' and llm_output.confidence > 0.8:
            print(f"[校准] 高风险但LLM过于自信 ({llm_output.confidence:.2f})，强制降级为CONSERVATIVE")
            return LLMOutput(
                strategy=LLMResponseStrategy.CONSERVATIVE,
                confidence=0.6,
                response=f"{llm_output.response}\n\n[注：此回答已根据风险评估调整为保守表述]",
                reasoning=f"{llm_output.reasoning} [策略校准：高风险强制保守]",
                citations=llm_output.citations,
                raw_output=llm_output.raw_output,
                latency_ms=llm_output.latency_ms,
                token_usage=llm_output.token_usage
            )
        
        # 检查2：置信度不足但LLM选择DIRECT
        if gov_confidence < 0.6 and llm_output.strategy.value == "DIRECT":
            print(f"[校准] 置信度不足 ({gov_confidence:.2f}) 但选择DIRECT，调整为CONSERVATIVE")
            return LLMOutput(
                strategy=LLMResponseStrategy.CONSERVATIVE,
                confidence=gov_confidence,
                response=f"根据现有信息，{llm_output.response}（置信度：{gov_confidence:.0%}）",
                reasoning=f"{llm_output.reasoning} [置信度校准]",
                citations=llm_output.citations,
                raw_output=llm_output.raw_output,
                latency_ms=llm_output.latency_ms,
                token_usage=llm_output.token_usage
            )
        
        # 检查3：策略偏离
        if llm_output.strategy.value != gov_strategy.value:
            print(f"[校准] LLM策略 ({llm_output.strategy.value}) 与治理建议 ({gov_strategy.value}) 不一致")
            
            # 如果LLM选择更保守的策略，接受
            if self._is_more_conservative(llm_output.strategy, gov_strategy):
                print("  -> LLM选择更保守，接受")
                return llm_output
            else:
                print(f"  -> 使用治理建议策略: {gov_strategy.value}")
                # 映射到 LLMResponseStrategy
                strategy_map = {
                    "DIRECT": LLMResponseStrategy.DIRECT,
                    "RETRIEVAL_FIRST": LLMResponseStrategy.RETRIEVAL_FIRST,
                    "CONSERVATIVE": LLMResponseStrategy.CONSERVATIVE,
                    "DECLINE": LLMResponseStrategy.DECLINE,
                    "REVIEW": LLMResponseStrategy.REVIEW
                }
                return LLMOutput(
                    strategy=strategy_map.get(gov_strategy.value, LLMResponseStrategy.CONSERVATIVE),
                    confidence=llm_output.confidence,
                    response=llm_output.response,
                    reasoning=f"{llm_output.reasoning} [策略调整：遵循治理建议]",
                    citations=llm_output.citations,
                    raw_output=llm_output.raw_output,
                    latency_ms=llm_output.latency_ms,
                    token_usage=llm_output.token_usage
                )
        
        return llm_output
    
    def _is_more_conservative(
        self,
        strategy1: LLMResponseStrategy,
        strategy2: ResponseStrategy
    ) -> bool:
        """判断strategy1是否比strategy2更保守"""
        
        # 转换为可比较的值
        def get_conservativeness(s):
            val = s.value if hasattr(s, 'value') else str(s)
            mapping = {
                "DECLINE": 4,
                "REVIEW": 3,
                "CONSERVATIVE": 2,
                "RETRIEVAL_FIRST": 1,
                "DIRECT": 0
            }
            return mapping.get(val, 0)
        
        return get_conservativeness(strategy1) > get_conservativeness(strategy2)
    
    async def process_turn_async(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> OrchestratorResponse:
        """异步处理单轮对话"""
        
        import time
        start_time = time.time()
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 1. 意图分析
        intent = self._analyze_intent(user_input, context)
        print(f"  [意图分析] 类型: {intent['type']}, 置信度: {intent['confidence']:.2f}")
        
        # 2. 检索决策
        retrieval_result = await self._decide_retrieval(user_input, intent, context)
        if retrieval_result:
            print(f"  [检索] 找到 {len(retrieval_result.results)} 条相关记忆")
            context.retrieved_memories.extend([
                {"content": r.content, "layer": r.layer, "id": r.id}
                for r in retrieval_result.results[:3]
            ])
        
        # 3. 上下文组装
        assembled_context = self._assemble_context(user_input, retrieval_result, context)
        assembled_context['session_id'] = session_id
        
        # 4. 治理审查
        governance_result = await self._governance_review(assembled_context)
        print(f"  [治理] 策略: {governance_result['strategy'].value}, 置信度: {governance_result['confidence']:.2f}")
        context.governance_trail.append(governance_result)
        
        # 5. 构建 Prompt（V2）
        prompt_data = self.prompt_builder.build(
            query=user_input,
            history=context.history,
            retrieval_results=retrieval_result,
            governance_result=governance_result,
            memory_context=context.retrieved_memories
        )
        
        # 调试输出
        if self.debug_mode:
            print("\n" + "="*70)
            print("DEBUG: 完整 Prompt 发送给 LLM")
            print("="*70)
            print(prompt_data["full_prompt"][:2000])  # 只打印前2000字符
            print("... [截断] ..." if len(prompt_data["full_prompt"]) > 2000 else "")
            print("="*70 + "\n")
        
        # 6. 适配并调用LLM
        llm_input = LLMInputAdapter.adapt(prompt_data)
        
        llm_start = time.time()
        llm_output = await self.llm.generate(llm_input)
        llm_latency = (time.time() - llm_start) * 1000
        
        print(f"  [LLM] 生成完成，延迟: {llm_latency:.0f}ms")
        
        # 7. 验证和校准
        validated_output = self._validate_and_calibrate(llm_output, governance_result)
        
        # 8. 记忆更新
        await self._update_memory(user_input, validated_output, context)
        
        # 9. 构建响应
        total_latency = (time.time() - start_time) * 1000
        
        response = OrchestratorResponse(
            response_text=validated_output.response,
            strategy=validated_output.strategy,
            confidence=validated_output.confidence,
            sources=validated_output.citations,
            reasoning=validated_output.reasoning,
            metadata={
                "retrieval_count": len(retrieval_result.results) if retrieval_result else 0,
                "llm_latency_ms": llm_latency,
                "total_latency_ms": total_latency,
                "token_usage": llm_output.token_usage
            },
            timestamp=datetime.now()
        )
        
        # 更新历史
        context.history.append({
            "turn": context.turn_number,
            "user": user_input,
            "ai": response.response_text,
            "strategy": response.strategy.value,
            "confidence": response.confidence
        })
        
        print(f"  [完成] 总延迟: {total_latency:.0f}ms\n")
        
        return response


async def demo_with_llm_v2():
    """演示集成LLM的Orchestrator v2"""
    print("="*70)
    print("Conversation Orchestrator with Real LLM v2 - 演示")
    print("="*70)
    
    # 初始化服务
    retrieval = RetrievalService()
    governance = GovernanceService()
    memory = MemoryService()
    
    await retrieval.start()
    await governance.start()
    await memory.start()
    
    # 创建LLM客户端
    llm_client = create_mock_client()
    
    # 创建主控器（启用调试模式）
    orchestrator = ConversationOrchestratorWithLLMV2(
        retrieval_service=retrieval,
        governance_service=governance,
        memory_service=memory,
        llm_client=llm_client,
        debug_mode=True  # 启用调试
    )
    
    # 准备测试数据
    from phase11.core_services.retrieval_service_v1 import MemoryLayer
    test_memories = [
        ("long_term", "python_basics", "Python 是一种高级编程语言，由 Guido van Rossum 创建"),
        ("long_term", "user_name", "用户的名字是 Alice"),
    ]
    
    for layer, key, content in test_memories:
        layer_enum = MemoryLayer.LONG_TERM
        retrieval.insert_memory(layer_enum, key, content, {"category": "knowledge"})
    
    print("\n准备测试数据完成\n")
    
    # 模拟对话
    session_id = "demo_session_001"
    
    test_inputs = [
        "你好",
        "Python 是什么？",
        "我叫什么名字？",
    ]
    
    for user_input in test_inputs:
        response = await orchestrator.process_turn_async(user_input, session_id)
        print(f"AI: {response.response_text}")
        print(f"   [策略: {response.strategy.value}, 置信度: {response.confidence:.2f}]")
        if response.sources:
            print(f"   [来源: {', '.join(response.sources)}]")
        print()
    
    # 清理
    await retrieval.stop()
    await governance.stop()
    await memory.stop()
    
    print("\n演示完成")


if __name__ == "__main__":
    asyncio.run(demo_with_llm_v2())
