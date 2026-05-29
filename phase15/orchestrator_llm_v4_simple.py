"""
Orchestrator with Real LLM v4 Simple - 简化版 v4

关键改进：
1. 检索触发增强 - 提高检索触发率
2. 保持 v3 的检索方式（工作正常）
"""

import asyncio
import sys
import time
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
    RetrievalService, RetrievalRequest, RetrievalType, RetrievalResult, MemoryLayer
)
from phase11.core_services.governance_service_v1 import GovernanceService
from phase11.core_services.memory_service_v1 import MemoryService
from phase15.llm_client_v1 import (
    BaseLLMClient,
    LLMInput,
    LLMOutput,
    ResponseStrategy as LLMResponseStrategy
)
from phase15.prompt_builder_v2 import create_prompt_builder_v2
from phase15.query_rewriter_v1 import create_query_rewriter
from phase15.retrieval_trigger_enhancer_v1 import create_retrieval_trigger_enhancer


class LLMInputAdapter:
    """适配器"""
    
    @staticmethod
    def adapt(prompt_data: Dict[str, Any]) -> LLMInput:
        return LLMInput(
            system_prompt=prompt_data["system_prompt"],
            conversation_history=prompt_data["conversation_history"],
            current_query=prompt_data["user_prompt"],
            retrieval_context=None,
            governance_context=None,
            memory_context=None
        )


class ConversationOrchestratorWithLLMV4Simple(BaseOrchestrator):
    """集成真实LLM的对话主控层 v4 简化版"""
    
    MIN_RETRIEVAL_CONFIDENCE = 0.2
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        governance_service: GovernanceService,
        memory_service: MemoryService,
        llm_client: BaseLLMClient,
        debug_mode: bool = False
    ):
        super().__init__(retrieval_service, governance_service, memory_service)
        self.llm = llm_client
        self.prompt_builder = create_prompt_builder_v2()
        self.query_rewriter = create_query_rewriter()
        self.retrieval_trigger = create_retrieval_trigger_enhancer()
        self.debug_mode = debug_mode
    
    async def _decide_retrieval_v4(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> Optional[Any]:
        """改进的检索决策（v4 简化版）"""
        
        # 1. 判断是否优先检索
        trigger_decision = self.retrieval_trigger.should_prioritize_retrieval(user_input, intent)
        
        if trigger_decision.should_retrieve:
            print(f"  [Trigger] 检索建议 (score: {trigger_decision.priority_score:.2f})")
            if self.debug_mode:
                print(f"    原因: {', '.join(trigger_decision.reasons[:3])}")
        
        # 2. 改写查询
        rewrite_result = self.query_rewriter.rewrite(
            user_input=user_input,
            intent=intent,
            conversation_history=context.history
        )
        
        if self.debug_mode:
            print(f"  [QueryRewrite] {rewrite_result.original_query} -> {rewrite_result.rewritten_query}")
        
        # 3. 决定是否检索（使用 v3 的逻辑 + 触发增强）
        should_search = (
            intent.get('needs_retrieval') or 
            trigger_decision.should_retrieve or
            trigger_decision.force_retrieval
        )
        
        if not should_search:
            return None
        
        # 4. 执行检索（使用 v3 的方式，工作正常）
        try:
            retrieval_start = time.time()
            
            request = RetrievalRequest(
                query=rewrite_result.rewritten_query,
                query_id=f"v4_{context.session_id}_{context.turn_number}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=5,
                timeout_ms=1000
            )
            
            response = await self.retrieval.retrieve(request)
            retrieval_latency = (time.time() - retrieval_start) * 1000
            
            if response.status == "success" and response.total_found > 0:
                # 过滤低相关性结果
                filtered_results = self._filter_by_relevance(response.results)
                
                print(f"  [Retrieval] 原始召回:{response.total_found} -> 过滤后:{len(filtered_results)} (耗时:{retrieval_latency:.0f}ms)")
                
                if filtered_results:
                    for i, r in enumerate(filtered_results[:2]):
                        conf = r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5
                        print(f"    [{i+1}] {r.content[:40]}... (conf:{conf:.2f})")
                    
                    # 更新响应对象
                    response.results = filtered_results
                    response.total_found = len(filtered_results)
                    return response
            else:
                print(f"  [Retrieval] 无结果 (status:{response.status})")
            
        except Exception as e:
            print(f"  [检索警告] {e}")
        
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
        
        return filtered[:3]
    
    async def process_turn_async(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> OrchestratorResponse:
        """处理单轮对话（v4 简化版）"""
        
        start_time = time.time()
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 1. 意图分析
        intent = self._analyze_intent(user_input, context)
        print(f"  [意图] {intent['type']}, 置信度:{intent['confidence']:.2f}")
        
        # 2. 检索（v4 简化版）
        retrieval_result = await self._decide_retrieval_v4(user_input, intent, context)
        
        if retrieval_result:
            print(f"  [检索成功] {len(retrieval_result.results)} 条")
            context.retrieved_memories.extend([
                {"content": r.content, "layer": r.layer, "id": r.id}
                for r in retrieval_result.results[:3]
            ])
        else:
            print(f"  [检索] 无结果")
        
        # 3. 上下文组装
        assembled_context = self._assemble_context(user_input, retrieval_result, context)
        assembled_context['session_id'] = session_id
        
        # 4. 治理审查
        governance_result = await self._governance_review(assembled_context)
        print(f"  [治理] {governance_result['strategy'].value}, 置信度:{governance_result['confidence']:.2f}")
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
        
        print(f"  [LLM] 延迟:{llm_latency:.0f}ms")
        
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
        
        print(f"  [完成] 总延迟:{total_latency:.0f}ms\n")
        
        return response
    
    def _validate_and_calibrate(self, llm_output: LLMOutput, governance_result: dict) -> LLMOutput:
        """验证和校准"""
        return llm_output


async def demo_v4_simple():
    """演示 v4 简化版"""
    print("="*70)
    print("Orchestrator v4 Simple - 检索触发增强演示")
    print("="*70)
    
    retrieval = RetrievalService()
    governance = GovernanceService()
    memory = MemoryService()
    
    await retrieval.start()
    await governance.start()
    await memory.start()
    
    from phase15.llm_client_v1 import create_mock_client
    llm_client = create_mock_client()
    
    orchestrator = ConversationOrchestratorWithLLMV4Simple(
        retrieval_service=retrieval,
        governance_service=governance,
        memory_service=memory,
        llm_client=llm_client,
        debug_mode=True
    )
    
    # 准备数据
    test_memories = [
        (MemoryLayer.LONG_TERM, "user_name", "用户的名字是Alice"),
        (MemoryLayer.LONG_TERM, "project_goal", "项目目标是在Q3完成核心功能"),
        (MemoryLayer.LONG_TERM, "tech_stack", "技术栈使用Python、React和PostgreSQL"),
    ]
    
    for layer, key, content in test_memories:
        retrieval.insert_memory(layer, key, content, {"confidence": 0.9})
    
    print("\n数据准备完成\n")
    
    # 测试
    session_id = "demo_v4s_001"
    test_inputs = ["我叫什么名字？", "项目的目标是什么？", "技术栈是什么？"]
    
    for query in test_inputs:
        response = await orchestrator.process_turn_async(query, session_id)
        print(f"AI: {response.response_text[:80]}...")
        print(f"   [策略:{response.strategy.value}, 检索:{response.metadata.get('retrieval_count', 0)}条]")
        print()
    
    await retrieval.stop()
    await governance.stop()
    await memory.stop()
    print("演示完成")


if __name__ == "__main__":
    asyncio.run(demo_v4_simple())
