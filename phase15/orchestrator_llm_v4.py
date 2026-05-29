"""
Orchestrator with Real LLM v4 - 集成真实LLM的对话主控层 v4

关键改进：
1. 检索触发增强 - 提高检索触发率
2. 混合检索 - 关键词 + 语义召回
3. Rerank 优化 - 结果重排序
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
    RetrievalService, RetrievalRequest, RetrievalType, RetrievalResult
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
from phase15.prompt_builder_v2 import create_prompt_builder_v2
from phase15.query_rewriter_v1 import create_query_rewriter
from phase15.retrieval_trigger_enhancer_v1 import create_retrieval_trigger_enhancer


class HybridRetriever:
    """混合检索器 - 关键词 + 语义召回"""
    
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval = retrieval_service
        self.keyword_weight = 0.4
        self.semantic_weight = 0.6
    
    async def retrieve_hybrid(
        self,
        query: str,
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """混合检索"""
        # 1. 关键词召回
        keyword_results = await self._keyword_retrieve(query, max_results * 2)
        
        # 2. 语义召回（查询扩展）
        semantic_results = await self._semantic_retrieve(query, max_results * 2)
        
        # 3. 合并去重并加权
        all_results = {}
        
        # 添加关键词结果
        for r in keyword_results:
            r.adjusted_score = (r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5) * self.keyword_weight
            all_results[r.id] = r
        
        # 添加语义结果
        for r in semantic_results:
            if r.id in all_results:
                all_results[r.id].adjusted_score += (r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5) * self.semantic_weight
            else:
                r.adjusted_score = (r.metadata.get('confidence', 0.5) if hasattr(r, 'metadata') else 0.5) * self.semantic_weight
                all_results[r.id] = r
        
        # 4. 排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: getattr(x, 'adjusted_score', 0),
            reverse=True
        )
        
        return sorted_results[:max_results]
    
    async def _keyword_retrieve(self, query: str, max_results: int) -> List[RetrievalResult]:
        """关键词检索"""
        try:
            request = RetrievalRequest(
                query=query,
                query_id=f"kw_{time.time()}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=max_results
            )
            response = await self.retrieval.retrieve(request)
            return response.results if response else []
        except Exception as e:
            print(f"  [Keyword检索警告] {e}")
            return []
    
    async def _semantic_retrieve(self, query: str, max_results: int) -> List[RetrievalResult]:
        """语义检索 - 查询扩展"""
        # 扩展查询
        expanded_queries = self._expand_query(query)
        
        all_results = []
        for eq in expanded_queries[:3]:  # 最多3个扩展查询
            try:
                request = RetrievalRequest(
                    query=eq,
                    query_id=f"sem_{time.time()}",
                    retrieval_type=RetrievalType.HYBRID,
                    max_results=max_results
                )
                response = await self.retrieval.retrieve(request)
                if response:
                    all_results.extend(response.results)
            except Exception as e:
                print(f"  [Semantic检索警告] {e}")
        
        # 去重
        seen = set()
        unique_results = []
        for r in all_results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)
        
        return unique_results[:max_results]
    
    def _expand_query(self, query: str) -> List[str]:
        """扩展查询"""
        expansions = [query]
        
        # 同义词映射
        synonyms = {
            "项目": ["project", "计划"],
            "目标": ["goal", "目的"],
            "技术栈": ["技术方案", "架构"],
            "名字": ["姓名", "名称"],
        }
        
        for word, alts in synonyms.items():
            if word in query:
                for alt in alts[:1]:  # 只取第一个同义词
                    expanded = query.replace(word, alt)
                    if expanded != query:
                        expansions.append(expanded)
        
        return expansions


class SimpleReranker:
    """简易重排序器"""
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        conversation_history: List[Dict] = None
    ) -> List[RetrievalResult]:
        """重排序"""
        for result in results:
            score = 0.0
            
            # 1. 基础分数 (40%)
            base_score = result.metadata.get('confidence', 0.5) if hasattr(result, 'metadata') else 0.5
            score += base_score * 0.4
            
            # 2. 关键词匹配 (30%)
            keyword_score = self._keyword_match(result.content, query)
            score += keyword_score * 0.3
            
            # 3. 主题相关性 (20%)
            if conversation_history:
                topic_score = self._topic_relevance(result, conversation_history)
                score += topic_score * 0.2
            
            # 4. 时效性 (10%)
            freshness = self._freshness(result)
            score += freshness * 0.1
            
            result.rerank_score = score
        
        # 排序
        results.sort(key=lambda x: getattr(x, 'rerank_score', 0), reverse=True)
        return results
    
    def _keyword_match(self, content: str, query: str) -> float:
        """关键词匹配度"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        matches = len(content_words & query_words)
        return min(matches / len(query_words), 1.0)
    
    def _topic_relevance(self, result: RetrievalResult, history: List[Dict]) -> float:
        """主题相关性"""
        if not history:
            return 0.5
        
        recent_text = " ".join([
            turn.get('user', '') + " " + turn.get('ai', '')
            for turn in history[-2:]
        ]).lower()
        
        result_text = result.content.lower()
        keywords = ['python', '项目', '技术', '目标', '名字', 'alice']
        matches = sum(1 for kw in keywords if kw in result_text and kw in recent_text)
        return min(matches / 3, 1.0)
    
    def _freshness(self, result: RetrievalResult) -> float:
        """时效性"""
        layer_scores = {
            'ephemeral': 1.0,
            'short_term': 0.8,
            'long_term': 0.6,
            'shallow_permanent': 0.4,
            'deep_permanent': 0.3,
        }
        return layer_scores.get(result.layer, 0.5)


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


class ConversationOrchestratorWithLLMV4(BaseOrchestrator):
    """集成真实LLM的对话主控层 v4"""
    
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
        self.hybrid_retriever = HybridRetriever(retrieval_service)
        self.reranker = SimpleReranker()
        self.debug_mode = debug_mode
    
    async def _decide_retrieval_v4(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> Optional[Any]:
        """改进的检索决策（v4）"""
        
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
        
        # 3. 决定是否检索
        should_search = (
            intent.get('needs_retrieval') or 
            trigger_decision.should_retrieve or
            trigger_decision.force_retrieval
        )
        
        if not should_search:
            return None
        
        # 4. 混合检索
        try:
            retrieval_start = time.time()
            results = await self.hybrid_retriever.retrieve_hybrid(
                rewrite_result.rewritten_query,
                max_results=8
            )
            retrieval_latency = (time.time() - retrieval_start) * 1000
            
            if not results:
                return None
            
            # 5. Rerank
            reranked = self.reranker.rerank(results, rewrite_result.rewritten_query, context.history)
            
            # 6. 过滤
            filtered = [r for r in reranked if getattr(r, 'rerank_score', 0) > 0.3]
            
            print(f"  [Hybrid] 召回:{len(results)} -> Rerank:{len(reranked)} -> 过滤:{len(filtered)} (耗时:{retrieval_latency:.0f}ms)")
            
            if filtered:
                # 创建响应对象
                from phase11.core_services.retrieval_service_v1 import RetrievalResponse
                return RetrievalResponse(
                    query_id=f"v4_{time.time()}",
                    results=filtered[:3],
                    total_found=len(filtered),
                    latency_ms=retrieval_latency,
                    cache_hit=False,
                    from_layer="hybrid",
                    status="success"
                )
            
        except Exception as e:
            print(f"  [检索警告] {e}")
        
        return None
    
    async def process_turn_async(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> OrchestratorResponse:
        """处理单轮对话（v4）"""
        
        start_time = time.time()
        
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 1. 意图分析
        intent = self._analyze_intent(user_input, context)
        print(f"  [意图] {intent['type']}, 置信度:{intent['confidence']:.2f}")
        
        # 2. 检索（v4）
        retrieval_result = await self._decide_retrieval_v4(user_input, intent, context)
        
        if retrieval_result:
            print(f"  [检索成功] {len(retrieval_result.results)} 条")
            for i, r in enumerate(retrieval_result.results[:2]):
                score = getattr(r, 'rerank_score', 0)
                print(f"    [{i+1}] {r.content[:40]}... (score:{score:.2f})")
            
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
        
        # 7. 验证
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
        # 简化版验证
        return llm_output


async def demo_v4():
    """演示 v4"""
    print("="*70)
    print("Orchestrator v4 - 检索优化演示")
    print("="*70)
    
    from phase11.core_services.retrieval_service_v1 import MemoryLayer
    
    retrieval = RetrievalService()
    governance = GovernanceService()
    memory = MemoryService()
    
    await retrieval.start()
    await governance.start()
    await memory.start()
    
    llm_client = create_mock_client()
    
    orchestrator = ConversationOrchestratorWithLLMV4(
        retrieval_service=retrieval,
        governance_service=governance,
        memory_service=memory,
        llm_client=llm_client,
        debug_mode=True
    )
    
    # 准备数据
    test_memories = [
        ("long_term", "user_name", "用户的名字是Alice"),
        ("long_term", "project_goal", "项目目标是在Q3完成核心功能"),
        ("long_term", "tech_stack", "技术栈使用Python、React和PostgreSQL"),
    ]
    
    for layer, key, content in test_memories:
        retrieval.insert_memory(MemoryLayer.LONG_TERM, key, content, {"confidence": 0.9})
    
    print("\n数据准备完成\n")
    
    # 测试
    session_id = "demo_v4_001"
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
    asyncio.run(demo_v4())
