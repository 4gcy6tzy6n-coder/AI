"""
Orchestrator with Real LLM v6 Streaming - 流式输出版本

关键改进：
1. 集成 StreamingClient
2. 实现流式响应处理
3. TTFT监控和优化
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
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
from phase15.llm_client_streaming_v1 import (
    BaseStreamingLLMClient, LLMInput, StreamingChunk
)
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


class ConversationOrchestratorWithLLMV6Streaming(BaseOrchestrator):
    """集成真实LLM的对话主控层 v6（流式输出）"""
    
    MIN_RETRIEVAL_CONFIDENCE = 0.2
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        governance_service: GovernanceService,
        memory_service: MemoryService,
        llm_client: BaseStreamingLLMClient,
        debug_mode: bool = False
    ):
        super().__init__(retrieval_service, governance_service, memory_service)
        self.llm = llm_client
        self.prompt_builder = create_prompt_builder_v2()
        self.query_rewriter = create_query_rewriter_v2()
        self.retrieval_trigger = create_retrieval_trigger_enhancer()
        self.debug_mode = debug_mode
    
    async def _decide_retrieval_v6(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> Optional[Any]:
        """检索决策（v6）"""
        
        # 1. 判断是否优先检索
        trigger_decision = self.retrieval_trigger.should_prioritize_retrieval(user_input, intent)
        
        if trigger_decision.should_retrieve and self.debug_mode:
            print(f"  [Trigger] 检索建议 (score: {trigger_decision.priority_score:.2f})")
        
        # 2. 改写查询
        rewrite_result = self.query_rewriter.rewrite(
            user_input=user_input,
            intent=intent,
            conversation_history=context.history
        )
        
        if self.debug_mode:
            print(f"  [QueryRewrite] 类型:{rewrite_result.query_type}")
        
        # 3. 决定是否检索
        should_search = (
            intent.get('needs_retrieval') or 
            trigger_decision.should_retrieve or
            trigger_decision.force_retrieval or
            rewrite_result.query_type != "general"
        )
        
        if not should_search:
            return None
        
        # 4. 执行检索
        try:
            search_query = rewrite_result.rewritten_query if rewrite_result.rewritten_query else user_input
            
            request = RetrievalRequest(
                query=search_query,
                query_id=f"v6_{context.session_id}_{context.turn_number}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=5,
                timeout_ms=1000
            )
            
            response = await self.retrieval.retrieve(request)
            
            if response.status == "success" and response.total_found > 0:
                filtered_results = self._filter_by_relevance(response.results)
                
                if self.debug_mode:
                    print(f"  [Retrieval] 原始:{response.total_found} -> 过滤:{len(filtered_results)}")
                
                if filtered_results:
                    response.results = filtered_results
                    response.total_found = len(filtered_results)
                    return response
            
        except Exception as e:
            if self.debug_mode:
                print(f"  [检索警告] {e}")
        
        return None
    
    def _filter_by_relevance(self, results: List[Any]) -> List[Any]:
        """按相关性过滤结果"""
        filtered = []
        for r in results:
            confidence = r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5
            if confidence >= self.MIN_RETRIEVAL_CONFIDENCE:
                filtered.append(r)
        
        filtered.sort(
            key=lambda x: x.metadata.get('confidence', 0.5) if hasattr(x, 'metadata') else 0.5,
            reverse=True
        )
        
        return filtered[:3]
    
    async def process_turn_streaming(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        处理单轮对话（流式输出）
        
        Yields:
            {
                "type": "status" | "content" | "complete",
                "data": ...
            }
        """
        
        total_start = time.time()
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        if self.debug_mode:
            print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 发送开始状态
        yield {
            "type": "status",
            "data": {"status": "started", "turn": context.turn_number}
        }
        
        try:
            # 1. 意图分析
            intent = self._analyze_intent(user_input, context)
            if self.debug_mode:
                print(f"  [意图] {intent['type']}, 置信度:{intent['confidence']:.2f}")
            
            # 2. 检索
            retrieval_result = await self._decide_retrieval_v6(user_input, intent, context)
            
            if retrieval_result:
                if self.debug_mode:
                    print(f"  [检索成功] {len(retrieval_result.results)} 条")
                context.retrieved_memories.extend([
                    {"content": r.content, "layer": r.layer, "id": r.id}
                    for r in retrieval_result.results[:3]
                ])
            else:
                if self.debug_mode:
                    print(f"  [检索] 无结果")
            
            # 3. 上下文组装
            assembled_context = self._assemble_context(user_input, retrieval_result, context)
            assembled_context['session_id'] = session_id
            
            # 4. 治理审查
            governance_result = await self._governance_review(assembled_context)
            if self.debug_mode:
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
            
            # 发送准备完成状态（即将开始流式生成）
            pre_processing_time = (time.time() - total_start) * 1000
            yield {
                "type": "status",
                "data": {
                    "status": "generating",
                    "pre_processing_ms": pre_processing_time,
                    "retrieval_count": len(retrieval_result.results) if retrieval_result else 0,
                    "strategy": governance_result['strategy'].value
                }
            }
            
            # 6. 流式生成
            llm_input = LLMInputAdapter.adapt(prompt_data)
            
            full_response = []
            first_token_time = None
            chunk_count = 0
            
            async for chunk in self.llm.generate_stream(llm_input):
                chunk_count += 1
                
                # 记录首Token时间
                if first_token_time is None and chunk.content:
                    first_token_time = time.time()
                    ttft = (first_token_time - total_start) * 1000
                    
                    if self.debug_mode:
                        print(f"  [首Token] TTFT: {ttft:.0f}ms")
                
                # 发送内容块
                if chunk.content:
                    full_response.append(chunk.content)
                    yield {
                        "type": "content",
                        "data": {
                            "content": chunk.content,
                            "is_first": chunk.is_first,
                            "chunk_number": chunk_count
                        }
                    }
                
                # 流式结束
                if chunk.is_last:
                    break
            
            # 7. 组装完整响应
            complete_response = "".join(full_response)
            total_latency = (time.time() - total_start) * 1000
            
            # 8. 记忆更新
            # 创建模拟LLMOutput用于记忆更新
            from phase15.llm_client_streaming_v1 import LLMOutput
            llm_output = LLMOutput(
                strategy=governance_result['strategy'].value,
                confidence=governance_result['confidence'],
                response=complete_response,
                reasoning="",
                citations=[],
                raw_output=complete_response,
                latency_ms=total_latency,
                token_usage={"total": len(complete_response) // 4}
            )
            await self._update_memory(user_input, llm_output, context)
            
            # 9. 更新历史
            context.history.append({
                "turn": context.turn_number,
                "user": user_input,
                "ai": complete_response,
                "strategy": governance_result['strategy'].value,
                "confidence": governance_result['confidence']
            })
            
            # 发送完成状态
            yield {
                "type": "complete",
                "data": {
                    "response": complete_response,
                    "strategy": governance_result['strategy'].value,
                    "confidence": governance_result['confidence'],
                    "ttft_ms": (first_token_time - total_start) * 1000 if first_token_time else 0,
                    "total_latency_ms": total_latency,
                    "chunk_count": chunk_count,
                    "retrieval_count": len(retrieval_result.results) if retrieval_result else 0
                }
            }
            
            if self.debug_mode:
                print(f"  [完成] TTFT:{(first_token_time - total_start) * 1000:.0f}ms, 总延迟:{total_latency:.0f}ms\n")
            
        except Exception as e:
            # 发送错误状态
            yield {
                "type": "error",
                "data": {"error": str(e)}
            }
            if self.debug_mode:
                print(f"  [错误] {e}\n")


# 便捷函数
def create_orchestrator_v6_streaming(
    retrieval_service: RetrievalService,
    governance_service: GovernanceService,
    memory_service: MemoryService,
    llm_client: BaseStreamingLLMClient,
    debug_mode: bool = False
) -> ConversationOrchestratorWithLLMV6Streaming:
    """创建流式输出的 Orchestrator v6"""
    return ConversationOrchestratorWithLLMV6Streaming(
        retrieval_service=retrieval_service,
        governance_service=governance_service,
        memory_service=memory_service,
        llm_client=llm_client,
        debug_mode=debug_mode
    )
