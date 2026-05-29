"""
Orchestrator with Real LLM v3 - 集成真实LLM的对话主控层 v3

改进：
1. 集成 QueryRewriter 优化检索查询
2. 添加检索结果相关性过滤
3. 改进的调试输出
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase13.orchestrator.conversation_orchestrator_v1 import (
    ConversationOrchestrator as BaseOrchestrator,
    ConversationContext,
    OrchestratorResponse,
    ResponseStrategy
)
from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, RetrievalRequest, RetrievalType
)
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
from phase15.query_rewriter_v1 import QueryRewriter, create_query_rewriter


class LLMInputAdapter:
    """将 PromptBuilderV2 输出适配为 LLMInput"""
    
    @staticmethod
    def adapt(prompt_data: Dict[str, Any]) -> LLMInput:
        """适配为 LLMInput"""
        return LLMInput(
            system_prompt=prompt_data["system_prompt"],
            conversation_history=prompt_data["conversation_history"],
            current_query=prompt_data["user_prompt"],
            retrieval_context=None,
            governance_context=None,
            memory_context=None
        )


class ConversationOrchestratorWithLLMV3(BaseOrchestrator):
    """
    集成真实LLM的对话主控层 v3
    
    关键改进：
    - 使用 QueryRewriter 优化检索查询
    - 添加检索结果相关性过滤
    - 更详细的调试信息
    """
    
    # 相似度阈值
    MIN_RETRIEVAL_CONFIDENCE = 0.2
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        governance_service: GovernanceService,
        memory_service: MemoryService,
        llm_client: BaseLLMClient,
        prompt_builder: Optional[PromptBuilderV2] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        debug_mode: bool = False
    ):
        super().__init__(retrieval_service, governance_service, memory_service)
        self.llm = llm_client
        self.prompt_builder = prompt_builder or create_prompt_builder_v2()
        self.query_rewriter = query_rewriter or create_query_rewriter()
        self.debug_mode = debug_mode
    
    async def _decide_retrieval_v3(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> Optional[Any]:
        """
        改进的检索决策（v3）
        
        改进：
        1. 使用 QueryRewriter 优化查询
        2. 过滤低相关性结果
        """
        # 1. 改写查询
        rewrite_result = self.query_rewriter.rewrite(
            user_input=user_input,
            intent=intent,
            conversation_history=context.history
        )
        
        if self.debug_mode:
            print(f"  [QueryRewrite] 原始: {rewrite_result.original_query}")
            print(f"  [QueryRewrite] 改写: {rewrite_result.rewritten_query}")
            print(f"  [QueryRewrite] 原因: {rewrite_result.rewrite_reason}")
        
        # 2. 使用改写后的查询进行检索
        try:
            request = RetrievalRequest(
                query=rewrite_result.rewritten_query,
                query_id=f"{context.session_id}_{context.turn_number}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=5,
                timeout_ms=1000
            )
            
            response = await self.retrieval.retrieve(request)
            
            if response.status == "success" and response.total_found > 0:
                # 3. 过滤低相关性结果
                filtered_results = self._filter_by_relevance(response.results)
                
                if self.debug_mode:
                    print(f"  [Retrieval] 原始召回: {len(response.results)} 条")
                    print(f"  [Retrieval] 过滤后: {len(filtered_results)} 条")
                
                if filtered_results:
                    # 返回过滤后的结果（保持原有响应对象结构）
                    response.results = filtered_results
                    response.total_found = len(filtered_results)
                    return response
            
        except Exception as e:
            print(f"  [检索警告] 检索失败: {e}")
        
        return None
    
    def _filter_by_relevance(self, results: List[Any]) -> List[Any]:
        """按相关性过滤结果"""
        filtered = []
        for r in results:
            confidence = r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5
            if confidence >= self.MIN_RETRIEVAL_CONFIDENCE:
                filtered.append(r)
        
        # 按置信度排序
        filtered.sort(
            key=lambda x: x.metadata.get('confidence', 0.5) if hasattr(x, 'metadata') else 0.5,
            reverse=True
        )
        
        return filtered[:3]  # 最多返回3条
    
    async def process_turn_async(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> OrchestratorResponse:
        """异步处理单轮对话（v3）"""
        
        import time
        start_time = time.time()
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 1. 意图分析
        intent = self._analyze_intent(user_input, context)
        print(f"  [意图分析] 类型: {intent['type']}, 置信度: {intent['confidence']:.2f}")
        
        # 2. 检索决策（v3 - 使用改写查询）
        retrieval_start = time.time()
        retrieval_result = await self._decide_retrieval_v3(user_input, intent, context)
        retrieval_latency = (time.time() - retrieval_start) * 1000
        
        if retrieval_result:
            print(f"  [检索] 找到 {len(retrieval_result.results)} 条相关记忆 (耗时: {retrieval_latency:.0f}ms)")
            for i, r in enumerate(retrieval_result.results[:2]):
                conf = r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5
                print(f"    [{i+1}] {r.content[:50]}... (置信度: {conf:.2f})")
            
            context.retrieved_memories.extend([
                {"content": r.content, "layer": r.layer, "id": r.id}
                for r in retrieval_result.results[:3]
            ])
        else:
            print(f"  [检索] 未找到相关记忆 (耗时: {retrieval_latency:.0f}ms)")
        
        # 3. 上下文组装
        assembled_context = self._assemble_context(user_input, retrieval_result, context)
        assembled_context['session_id'] = session_id
        
        # 4. 治理审查
        governance_result = await self._governance_review(assembled_context)
        print(f"  [治理] 策略: {governance_result['strategy'].value}, 置信度: {governance_result['confidence']:.2f}")
        context.governance_trail.append(governance_result)
        
        # 5. 构建 Prompt
        prompt_data = self.prompt_builder.build(
            query=user_input,
            history=context.history,
            retrieval_results=retrieval_result,
            governance_result=governance_result,
            memory_context=context.retrieved_memories
        )
        
        # 6. 调用LLM
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
                "retrieval_latency_ms": retrieval_latency,
                "llm_latency_ms": llm_latency,
                "total_latency_ms": total_latency,
                "token_usage": llm_output.token_usage,
                "query_rewritten": True
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
    
    def _validate_and_calibrate(
        self,
        llm_output: LLMOutput,
        governance_result: dict
    ) -> LLMOutput:
        """验证和校准LLM输出"""
        
        gov_strategy = governance_result.get('strategy', ResponseStrategy.DIRECT)
        gov_confidence = governance_result.get('confidence', 0.7)
        risk_level = governance_result.get('risk_level', 'low')
        
        if isinstance(gov_strategy, str):
            gov_strategy = ResponseStrategy(gov_strategy)
        
        # 策略映射
        strategy_map = {
            "DIRECT": LLMResponseStrategy.DIRECT,
            "RETRIEVAL_FIRST": LLMResponseStrategy.RETRIEVAL_FIRST,
            "CONSERVATIVE": LLMResponseStrategy.CONSERVATIVE,
            "DECLINE": LLMResponseStrategy.DECLINE,
            "REVIEW": LLMResponseStrategy.REVIEW
        }
        
        # 高风险强制降级
        if risk_level == 'high' and llm_output.confidence > 0.8:
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
        
        # 策略偏离处理
        if llm_output.strategy.value != gov_strategy.value:
            # 如果LLM选择更保守，接受
            if self._is_more_conservative(llm_output.strategy, gov_strategy):
                return llm_output
            else:
                # 使用治理建议
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


async def demo_with_llm_v3():
    """演示集成LLM的Orchestrator v3"""
    print("="*70)
    print("Conversation Orchestrator with Real LLM v3 - 演示")
    print("="*70)
    
    # 初始化服务
    from phase11.core_services.retrieval_service_v1 import MemoryLayer
    
    retrieval = RetrievalService()
    governance = GovernanceService()
    memory = MemoryService()
    
    await retrieval.start()
    await governance.start()
    await memory.start()
    
    # 创建LLM客户端
    llm_client = create_mock_client()
    
    # 创建主控器（启用调试模式）
    orchestrator = ConversationOrchestratorWithLLMV3(
        retrieval_service=retrieval,
        governance_service=governance,
        memory_service=memory,
        llm_client=llm_client,
        debug_mode=True
    )
    
    # 准备测试数据
    test_memories = [
        ("long_term", "user_name", "用户的名字是Alice，是一名软件工程师"),
        ("long_term", "project_goal", "项目目标是在Q3完成核心功能开发，包括AI对话系统"),
        ("long_term", "tech_stack", "技术栈使用Python、React和PostgreSQL"),
        ("long_term", "python_info", "Python是一种高级编程语言，由Guido van Rossum于1991年创建"),
    ]
    
    for layer, key, content in test_memories:
        retrieval.insert_memory(MemoryLayer.LONG_TERM, key, content, {"category": "test"})
    
    print("\n准备测试数据完成\n")
    
    # 模拟对话
    session_id = "demo_session_v3_001"
    
    test_inputs = [
        "你好",
        "Python 是什么？",
        "我叫什么名字？",
        "项目的目标是什么？",
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
    asyncio.run(demo_with_llm_v3())
