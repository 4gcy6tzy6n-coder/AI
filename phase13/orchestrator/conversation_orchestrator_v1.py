"""
Conversation Orchestrator V1 - 对话主控层 V1

Phase 13 核心组件：
统一协调 API Gateway、Retrieval、Governance、Memory、Response Generation
形成每轮对话都执行的完整主链

主链流程：
用户输入 → Intent Analysis → Retrieval Decision → Context Assembly → 
Governance Review → Response Generation → Memory Update → 输出给用户
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, RetrievalRequest, RetrievalType, RetrievalResponse
)
from phase11.core_services.governance_service_v1 import (
    GovernanceService, UnitInfo, GovernanceDecision
)
from phase11.core_services.memory_service_v1 import (
    MemoryService, WritebackRequest, MemoryOperation
)


class ResponseStrategy(Enum):
    """响应策略"""
    DIRECT = "direct"              # 直接回答
    RETRIEVAL_FIRST = "retrieval"  # 先检索再回答
    CONSERVATIVE = "conservative"  # 保守回答
    DECLINE = "decline"            # 拒绝回答
    REVIEW = "review"              # 需要审查


class ConfidenceLevel(Enum):
    """置信度级别"""
    HIGH = "high"      # > 0.9
    MEDIUM = "medium"  # 0.6 - 0.9
    LOW = "low"        # < 0.6


@dataclass
class ConversationContext:
    """对话上下文"""
    session_id: str
    user_id: str
    turn_number: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_memories: List[Dict] = field(default_factory=list)
    governance_trail: List[Dict] = field(default_factory=list)


@dataclass
class OrchestratorResponse:
    """主控层响应"""
    response_text: str
    strategy: ResponseStrategy
    confidence: float
    sources: List[str]
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime


class ConversationOrchestrator:
    """
    对话主控层
    
    职责：
    1. 统一协调各服务
    2. 执行完整主链
    3. 管理对话上下文
    4. 实现 AI 回答协议
    """
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        governance_service: GovernanceService,
        memory_service: MemoryService
    ):
        self.retrieval = retrieval_service
        self.governance = governance_service
        self.memory = memory_service
        
        # 对话上下文管理
        self.sessions: Dict[str, ConversationContext] = {}
        
        # 协议配置
        self.config = {
            "retrieval_threshold": 0.7,
            "conservative_threshold": 0.6,
            "decline_threshold": 0.4,
            "max_history_turns": 10,
            "enable_citations": True
        }
    
    async def process_turn(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> OrchestratorResponse:
        """
        处理单轮对话
        
        完整主链：
        1. 意图分析
        2. 检索决策
        3. 上下文组装
        4. 治理审查
        5. 响应生成
        6. 记忆更新
        """
        start_time = time.time()
        
        # 1. 获取或创建会话上下文
        context = self._get_or_create_context(session_id, user_id)
        context.turn_number += 1
        
        print(f"\n[Turn {context.turn_number}] 用户: {user_input}")
        
        # 2. 意图分析
        intent = self._analyze_intent(user_input, context)
        print(f"  [意图分析] 类型: {intent['type']}, 置信度: {intent['confidence']:.2f}")
        
        # 3. 检索决策
        retrieval_result = await self._decide_retrieval(user_input, intent, context)
        if retrieval_result:
            print(f"  [检索] 找到 {len(retrieval_result.results)} 条相关记忆")
            context.retrieved_memories.extend([
                {"content": r.content, "layer": r.layer, "id": r.id}
                for r in retrieval_result.results[:3]
            ])
        
        # 4. 上下文组装
        assembled_context = self._assemble_context(user_input, retrieval_result, context)
        
        # 5. 治理审查
        governance_result = await self._governance_review(assembled_context)
        print(f"  [治理] 策略: {governance_result['strategy']}, 置信度: {governance_result['confidence']:.2f}")
        context.governance_trail.append(governance_result)
        
        # 6. 响应生成
        response = self._generate_response(
            user_input,
            assembled_context,
            governance_result,
            retrieval_result
        )
        
        # 7. 记忆更新
        await self._update_memory(user_input, response, context)
        
        # 8. 更新历史
        context.history.append({
            "turn": context.turn_number,
            "user": user_input,
            "ai": response.response_text,
            "strategy": response.strategy.value,
            "confidence": response.confidence
        })
        
        # 保留最近 N 轮
        if len(context.history) > self.config["max_history_turns"]:
            context.history = context.history[-self.config["max_history_turns"]:]
        
        latency = time.time() - start_time
        print(f"  [完成] 延迟: {latency*1000:.0f}ms\n")
        
        return response
    
    def _get_or_create_context(
        self,
        session_id: str,
        user_id: str
    ) -> ConversationContext:
        """获取或创建会话上下文"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                turn_number=0
            )
        return self.sessions[session_id]
    
    def _analyze_intent(
        self,
        user_input: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        意图分析
        
        分析：
        - 查询类型（事实/观点/操作）
        - 是否需要检索
        - 置信度评估
        """
        # 简单启发式分析（实际应使用更复杂的模型）
        input_lower = user_input.lower()
        
        # 事实查询特征
        fact_keywords = ["是什么", "什么是", "how to", "what is", "为什么"]
        is_fact_query = any(kw in input_lower for kw in fact_keywords)
        
        # 需要检索的特征
        retrieval_keywords = ["记得", "之前", "上次", "according to", "based on"]
        needs_retrieval = any(kw in input_lower for kw in retrieval_keywords)
        
        # 简单置信度估计
        if len(user_input) < 5:
            confidence = 0.5
        elif is_fact_query:
            confidence = 0.8
        else:
            confidence = 0.7
        
        return {
            "type": "fact" if is_fact_query else "general",
            "needs_retrieval": needs_retrieval or is_fact_query,
            "confidence": confidence,
            "language": "zh" if any('\u4e00' <= c <= '\u9fff' for c in user_input) else "en"
        }
    
    async def _decide_retrieval(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> Optional[RetrievalResponse]:
        """
        检索决策
        
        决策逻辑：
        - 明确需要检索 → 执行检索
        - 高置信度事实查询 → 执行检索
        - 其他 → 跳过检索
        """
        if not intent["needs_retrieval"] and intent["confidence"] < 0.8:
            return None
        
        try:
            request = RetrievalRequest(
                query=user_input,
                query_id=f"{context.session_id}_{context.turn_number}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=5,
                timeout_ms=1000
            )
            
            response = await self.retrieval.retrieve(request)
            
            if response.status == "success" and response.total_found > 0:
                return response
            
        except Exception as e:
            print(f"  [检索警告] 检索失败: {e}")
        
        return None
    
    def _assemble_context(
        self,
        user_input: str,
        retrieval_result: Optional[RetrievalResponse],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """组装上下文"""
        assembled = {
            "user_input": user_input,
            "history": context.history[-3:],  # 最近 3 轮
            "retrieved": [],
            "session_context": {
                "turn_number": context.turn_number,
                "user_id": context.user_id
            }
        }
        
        if retrieval_result:
            assembled["retrieved"] = [
                {
                    "content": r.content,
                    "layer": r.layer,
                    "confidence": r.metadata.get("confidence", 0.5)
                }
                for r in retrieval_result.results[:3]
            ]
        
        return assembled
    
    async def _governance_review(
        self,
        assembled_context: Dict
    ) -> Dict[str, Any]:
        """
        治理审查
        
        审查内容：
        - 内容风险
        - 置信度评估
        - 响应策略建议
        """
        user_input = assembled_context["user_input"]
        retrieved = assembled_context["retrieved"]
        
        # 基础置信度
        base_confidence = 0.7 if retrieved else 0.6
        
        # 根据检索结果调整
        if retrieved:
            avg_retrieval_conf = sum(r["confidence"] for r in retrieved) / len(retrieved)
            base_confidence = 0.5 + avg_retrieval_conf * 0.4
        
        # 决定响应策略
        if base_confidence > 0.9:
            strategy = ResponseStrategy.DIRECT
        elif base_confidence > 0.7:
            strategy = ResponseStrategy.RETRIEVAL_FIRST if retrieved else ResponseStrategy.DIRECT
        elif base_confidence > 0.5:
            strategy = ResponseStrategy.CONSERVATIVE
        else:
            strategy = ResponseStrategy.DECLINE
        
        return {
            "strategy": strategy,
            "confidence": base_confidence,
            "risk_level": "low" if base_confidence > 0.8 else "medium" if base_confidence > 0.6 else "high",
            "reasoning": f"基于{len(retrieved)}条检索结果，置信度{base_confidence:.2f}"
        }
    
    def _generate_response(
        self,
        user_input: str,
        assembled_context: Dict,
        governance_result: Dict,
        retrieval_result: Optional[RetrievalResponse]
    ) -> OrchestratorResponse:
        """
        响应生成
        
        根据策略生成不同风格的响应
        """
        strategy = governance_result["strategy"]
        confidence = governance_result["confidence"]
        retrieved = assembled_context["retrieved"]
        
        # 构建引用
        sources = []
        if retrieved and self.config["enable_citations"]:
            sources = [f"[记忆:{i+1}]" for i in range(len(retrieved))]
        
        # 根据策略生成响应
        if strategy == ResponseStrategy.DIRECT:
            response_text = self._generate_direct_response(user_input, retrieved)
            
        elif strategy == ResponseStrategy.RETRIEVAL_FIRST:
            response_text = self._generate_retrieval_response(user_input, retrieved)
            
        elif strategy == ResponseStrategy.CONSERVATIVE:
            response_text = self._generate_conservative_response(user_input, retrieved, confidence)
            
        elif strategy == ResponseStrategy.DECLINE:
            response_text = "抱歉，我对这个问题不太确定，无法给出可靠回答。"
            
        else:
            response_text = "此内容已标记待复核。"
        
        return OrchestratorResponse(
            response_text=response_text,
            strategy=strategy,
            confidence=confidence,
            sources=sources,
            reasoning=governance_result["reasoning"],
            metadata={
                "retrieval_count": len(retrieved),
                "risk_level": governance_result["risk_level"]
            },
            timestamp=datetime.now()
        )
    
    def _generate_direct_response(
        self,
        user_input: str,
        retrieved: List[Dict]
    ) -> str:
        """生成直接回答"""
        if retrieved:
            context = " ".join([r["content"] for r in retrieved[:2]])
            return f"根据我的了解，{context}"
        return f"关于'{user_input}'，这是一个很好的问题。"
    
    def _generate_retrieval_response(
        self,
        user_input: str,
        retrieved: List[Dict]
    ) -> str:
        """生成基于检索的回答"""
        if retrieved:
            sources_text = "\n".join([f"  [{i+1}] {r['content'][:50]}..." 
                                     for i, r in enumerate(retrieved[:2])])
            return f"根据检索到的信息：\n{sources_text}\n\n基于以上，我认为..."
        return "我尝试检索了相关信息，但没有找到足够的内容。"
    
    def _generate_conservative_response(
        self,
        user_input: str,
        retrieved: List[Dict],
        confidence: float
    ) -> str:
        """生成保守回答"""
        uncertainty = "我不太确定，但根据现有信息"
        if retrieved:
            context = retrieved[0]["content"][:50]
            return f"{uncertainty}，{context}...（置信度：{confidence:.0%}）"
        return f"{uncertainty}，这个问题可能需要更多信息才能准确回答。"
    
    async def _update_memory(
        self,
        user_input: str,
        response: OrchestratorResponse,
        context: ConversationContext
    ):
        """
        记忆更新
        
        有价值的信息写入记忆
        """
        # 简单策略：高价值对话写入记忆
        if response.confidence > 0.7 and len(user_input) > 10:
            try:
                operation = MemoryOperation(
                    operation_type="create",
                    object_id=f"conv_{context.session_id}_{context.turn_number}",
                    target_layer="long_term",
                    content={
                        "user_input": user_input,
                        "response": response.response_text,
                        "confidence": response.confidence
                    },
                    reason=f"对话轮次 {context.turn_number}，置信度 {response.confidence:.2f}"
                )
                
                request = WritebackRequest(
                    query_id=f"{context.session_id}_{context.turn_number}",
                    trace_id=context.session_id,
                    transaction_id=f"txn_{context.session_id}_{context.turn_number}",
                    operations=[operation]
                )
                
                await self.memory.writeback(request)
                
            except Exception as e:
                print(f"  [记忆警告] 写回失败: {e}")


async def demo_orchestrator():
    """演示对话主控层"""
    print("="*70)
    print("Conversation Orchestrator V1 - 演示")
    print("="*70)
    
    # 初始化服务
    retrieval = RetrievalService()
    governance = GovernanceService()
    memory = MemoryService()
    
    await retrieval.start()
    await governance.start()
    await memory.start()
    
    # 创建主控器
    orchestrator = ConversationOrchestrator(retrieval, governance, memory)
    
    # 准备测试数据
    test_memories = [
        ("long_term", "python_basics", "Python 是一种高级编程语言，由 Guido van Rossum 创建"),
        ("long_term", "ml_intro", "机器学习是人工智能的一个分支，让计算机从数据中学习"),
        ("deep_permanent", "project_x", "项目 X 的目标是在 Q3 完成核心功能开发"),
    ]
    
    from phase11.core_services.retrieval_service_v1 import MemoryLayer
    for layer, key, content in test_memories:
        layer_enum = MemoryLayer.LONG_TERM if layer == "long_term" else MemoryLayer.DEEP_PERMANENT
        retrieval.insert_memory(layer_enum, key, content, {"category": "knowledge"})
    
    print("\n准备测试数据完成\n")
    
    # 模拟对话
    session_id = "demo_session_001"
    
    test_inputs = [
        "你好",
        "Python 是什么？",
        "机器学习是什么",
        "记得我们之前讨论过什么吗",
        "我不太确定这个问题的答案",
        "xyzabc 是什么"  # 应该触发 decline
    ]
    
    for user_input in test_inputs:
        response = await orchestrator.process_turn(user_input, session_id)
        print(f"AI: {response.response_text}")
        print(f"   [策略: {response.strategy.value}, 置信度: {response.confidence:.2f}]")
        if response.sources:
            print(f"   [来源: {', '.join(response.sources)}]")
        print()
    
    # 显示对话历史
    context = orchestrator.sessions[session_id]
    print("="*70)
    print("对话历史摘要")
    print("="*70)
    for turn in context.history:
        print(f"Turn {turn['turn']}: [{turn['strategy']}] 置信度 {turn['confidence']:.2f}")
    
    # 清理
    await retrieval.stop()
    await governance.stop()
    await memory.stop()
    
    print("\n演示完成")


if __name__ == "__main__":
    asyncio.run(demo_orchestrator())
