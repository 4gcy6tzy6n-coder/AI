"""
Orchestrator with Real LLM v5 Profiled - 带延迟分析的 v5

关键改进：
1. 集成延迟分析器
2. 详细分解各环节耗时
3. 识别性能瓶颈
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
from phase15.latency_profiler_v1 import LatencyProfiler, LatencyBreakdown


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


class ConversationOrchestratorWithLLMV5Profiled(BaseOrchestrator):
    """集成真实LLM的对话主控层 v5（带延迟分析）"""
    
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
        self.profiler = LatencyProfiler()
    
    async def _decide_retrieval_profiled(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext,
        breakdown: LatencyBreakdown
    ) -> Optional[Any]:
        """带性能分析的检索决策"""
        
        # 1. 检索触发判断
        self.profiler.start("retrieval_trigger")
        trigger_decision = self.retrieval_trigger.should_prioritize_retrieval(user_input, intent)
        breakdown.retrieval_trigger_ms = self.profiler.end("retrieval_trigger")
        
        if trigger_decision.should_retrieve and self.debug_mode:
            print(f"  [Trigger] 检索建议 (score: {trigger_decision.priority_score:.2f})")
        
        # 2. 改写查询
        self.profiler.start("query_rewrite")
        rewrite_result = self.query_rewriter.rewrite(
            user_input=user_input,
            intent=intent,
            conversation_history=context.history
        )
        breakdown.query_rewrite_ms = self.profiler.end("query_rewrite")
        
        if self.debug_mode:
            print(f"  [QueryRewrite] 类型:{rewrite_result.query_type}, 改写:{rewrite_result.rewritten_query[:30]}...")
        
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
            self.profiler.start("retrieval_search")
            
            search_query = rewrite_result.rewritten_query if rewrite_result.rewritten_query else user_input
            
            request = RetrievalRequest(
                query=search_query,
                query_id=f"v5p_{context.session_id}_{context.turn_number}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=5,
                timeout_ms=1000
            )
            
            response = await self.retrieval.retrieve(request)
            breakdown.retrieval_search_ms = self.profiler.end("retrieval_search")
            
            if response.status == "success" and response.total_found > 0:
                filtered_results = self._filter_by_relevance(response.results)
                
                if self.debug_mode:
                    print(f"  [Retrieval] 原始:{response.total_found} -> 过滤:{len(filtered_results)}")
                
                if filtered_results:
                    response.results = filtered_results
                    response.total_found = len(filtered_results)
                    breakdown.retrieval_count = len(filtered_results)
                    return response
            
        except Exception as e:
            breakdown.retrieval_search_ms = self.profiler.end("retrieval_search")
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
    
    async def process_turn_profiled(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> tuple:
        """处理单轮对话（带性能分析）
        
        Returns:
            (OrchestratorResponse, LatencyBreakdown)
        """
        
        total_start = time.time()
        breakdown = LatencyBreakdown(total_ms=0)
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 1. 意图分析
        self.profiler.start("intent_analysis")
        intent = self._analyze_intent(user_input, context)
        breakdown.intent_analysis_ms = self.profiler.end("intent_analysis")
        print(f"  [意图] {intent['type']}, 置信度:{intent['confidence']:.2f}")
        
        # 2. 检索（带性能分析）
        retrieval_result = await self._decide_retrieval_profiled(user_input, intent, context, breakdown)
        
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
        self.profiler.start("governance_review")
        governance_result = await self._governance_review(assembled_context)
        breakdown.governance_review_ms = self.profiler.end("governance_review")
        print(f"  [治理] {governance_result['strategy'].value}, 置信度:{governance_result['confidence']:.2f}")
        context.governance_trail.append(governance_result)
        
        # 5. 构建 Prompt
        self.profiler.start("prompt_build")
        prompt_data = self.prompt_builder.build(
            query=user_input,
            history=context.history,
            retrieval_results=retrieval_result,
            governance_result=governance_result,
            memory_context=context.retrieved_memories
        )
        breakdown.prompt_build_ms = self.profiler.end("prompt_build")
        
        # 6. 调用LLM
        llm_input = LLMInputAdapter.adapt(prompt_data)
        self.profiler.start("llm_generation")
        llm_output = await self.llm.generate(llm_input)
        breakdown.llm_generation_ms = self.profiler.end("llm_generation")
        # 从 token_usage 获取总token数
        token_usage = getattr(llm_output, 'token_usage', {})
        breakdown.llm_tokens = token_usage.get('total', 0)
        print(f"  [LLM] 延迟:{breakdown.llm_generation_ms:.0f}ms, tokens:{breakdown.llm_tokens}")
        
        # 7. 验证和校准
        validated_output = llm_output
        
        # 8. 记忆更新
        await self._update_memory(user_input, validated_output, context)
        
        # 9. 构建响应
        self.profiler.start("response_build")
        total_latency = (time.time() - total_start) * 1000
        breakdown.total_ms = total_latency
        breakdown.response_build_ms = self.profiler.end("response_build")
        
        response = OrchestratorResponse(
            response_text=validated_output.response,
            strategy=validated_output.strategy,
            confidence=validated_output.confidence,
            sources=validated_output.citations,
            reasoning=validated_output.reasoning,
            metadata={
                "retrieval_count": len(retrieval_result.results) if retrieval_result else 0,
                "llm_latency_ms": breakdown.llm_generation_ms,
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
        
        # 记录延迟分解
        self.profiler.record(breakdown)
        
        # 打印延迟分解
        if self.debug_mode:
            print(f"\n  [延迟分解] 总计:{breakdown.total_ms:.0f}ms")
            print(f"    意图:{breakdown.intent_analysis_ms:.0f}ms | "
                  f"改写:{breakdown.query_rewrite_ms:.0f}ms | "
                  f"触发:{breakdown.retrieval_trigger_ms:.0f}ms | "
                  f"检索:{breakdown.retrieval_search_ms:.0f}ms")
            print(f"    治理:{breakdown.governance_review_ms:.0f}ms | "
                  f"Prompt:{breakdown.prompt_build_ms:.0f}ms | "
                  f"LLM:{breakdown.llm_generation_ms:.0f}ms")
        
        print(f"  [完成] 总延迟:{total_latency:.0f}ms\n")
        
        return response, breakdown
    
    def print_latency_report(self):
        """打印延迟分析报告"""
        self.profiler.print_report()


# 便捷函数
def create_orchestrator_v5_profiled(
    retrieval_service: RetrievalService,
    governance_service: GovernanceService,
    memory_service: MemoryService,
    llm_client: BaseLLMClient,
    debug_mode: bool = False
) -> ConversationOrchestratorWithLLMV5Profiled:
    """创建带延迟分析的 Orchestrator v5"""
    return ConversationOrchestratorWithLLMV5Profiled(
        retrieval_service=retrieval_service,
        governance_service=governance_service,
        memory_service=memory_service,
        llm_client=llm_client,
        debug_mode=debug_mode
    )
