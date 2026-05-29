"""
Orchestrator with Real LLM v5 - 集成真实LLM的对话主控层 v5

关键改进：
1. QueryRewriter v2 - 4类查询分别处理
2. 同义词扩展 - 提升召回率
3. 检索触发增强 - 保持 v4 的优势
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
    OrchestratorResponse
)
from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, RetrievalRequest, RetrievalType
)
from phase11.core_services.governance_service_v1 import GovernanceService
from phase11.core_services.memory_service_v1 import MemoryService
from phase15.llm_client_v1 import BaseLLMClient, LLMInput
from phase15.prompt_builder_v2 import create_prompt_builder_v2
from phase15.query_rewriter_v2 import create_query_rewriter_v2
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


class ConversationOrchestratorWithLLMV5(BaseOrchestrator):
    """集成真实LLM的对话主控层 v5"""
    
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
        self.query_rewriter = create_query_rewriter_v2()
        self.retrieval_trigger = create_retrieval_trigger_enhancer()
        self.debug_mode = debug_mode
    
    async def _decide_retrieval_v5(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> Optional[Any]:
        """改进的检索决策（v5）"""
        
        # 1. 判断是否优先检索
        trigger_decision = self.retrieval_trigger.should_prioritize_retrieval(user_input, intent)
        
        if trigger_decision.should_retrieve:
            print(f"  [Trigger] 检索建议 (score: {trigger_decision.priority_score:.2f})")
            if self.debug_mode:
                print(f"    原因: {', '.join(trigger_decision.reasons[:3])}")
        
        # 2. 改写查询（使用 v2）
        rewrite_result = self.query_rewriter.rewrite(
            user_input=user_input,
            intent=intent,
            conversation_history=context.history
        )
        
        if self.debug_mode:
            print(f"  [QueryRewrite] {rewrite_result.original_query}")
            print(f"    类型: {rewrite_result.query_type}")
            print(f"    改写: {rewrite_result.rewritten_query}")
        
        # 3. 决定是否检索
        should_search = (
            intent.get('needs_retrieval') or 
            trigger_decision.should_retrieve or
            trigger_decision.force_retrieval or
            rewrite_result.query_type != "general"  # 非通用查询优先检索
        )
        
        if not should_search:
            return None
        
        # 4. 执行检索
        try:
            retrieval_start = time.time()
            
            # 使用改写后的查询
            search_query = rewrite_result.rewritten_query if rewrite_result.rewritten_query else user_input
            
            request = RetrievalRequest(
                query=search_query,
                query_id=f"v5_{context.session_id}_{context.turn_number}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=5,
                timeout_ms=1000
            )
            
            response = await self.retrieval.retrieve(request)
            retrieval_latency = (time.time() - retrieval_start) * 1000
            
            if response.status == "success" and response.total_found > 0:
                # 过滤低相关性结果
                filtered_results = self._filter_by_relevance(response.results)
                
                print(f"  [Retrieval] 原始:{response.total_found} -> 过滤:{len(filtered_results)} (耗时:{retrieval_latency:.0f}ms)")
                
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
        """处理单轮对话（v5）"""
        
        start_time = time.time()
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 1. 意图分析
        intent = self._analyze_intent(user_input, context)
        print(f"  [意图] {intent['type']}, 置信度:{intent['confidence']:.2f}")
        
        # 2. 检索（v5）
        retrieval_result = await self._decide_retrieval_v5(user_input, intent, context)
        
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
        validated_output = llm_output
        
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


# 便捷函数
def create_orchestrator_v5(
    retrieval_service: RetrievalService,
    governance_service: GovernanceService,
    memory_service: MemoryService,
    llm_client: BaseLLMClient,
    debug_mode: bool = False
) -> ConversationOrchestratorWithLLMV5:
    """创建 Orchestrator v5"""
    return ConversationOrchestratorWithLLMV5(
        retrieval_service=retrieval_service,
        governance_service=governance_service,
        memory_service=memory_service,
        llm_client=llm_client,
        debug_mode=debug_mode
    )
