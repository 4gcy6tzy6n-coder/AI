"""
Model Router v1 - 模型路由 v1

目标：
1. 简单查询走更快模型
2. 私有知识问题走主模型+检索
3. 高风险问题走完整治理链
4. 量化路由效果
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class RouteType(Enum):
    """路由类型"""
    FAST = "fast"              # 快速路由 - 缓存/轻量模型
    STANDARD = "standard"      # 标准路由 - 主模型+检索
    GOVERNANCE = "governance"  # 治理路由 - 完整治理链
    DIRECT = "direct"          # 直接路由 - 通用知识


class QueryComplexity(Enum):
    """查询复杂度"""
    LOW = "low"        # 简单问候/缓存命中
    MEDIUM = "medium"  # 标准私有知识查询
    HIGH = "high"      # 复杂/高风险问题


@dataclass
class RouteDecision:
    """路由决策"""
    route_type: RouteType
    complexity: QueryComplexity
    reasoning: str
    use_cache: bool
    use_retrieval: bool
    model_tier: str  # "fast", "standard", "premium"
    estimated_tokens: int
    confidence: float


@dataclass
class RouteMetrics:
    """路由指标"""
    route_type: str
    query_type: str
    ttft_ms: float
    total_latency_ms: float
    token_usage: int
    quality_score: float  # 回答质量评分
    cache_hit: bool
    retrieval_count: int


class ModelRouter:
    """模型路由器"""
    
    # 路由规则配置
    ROUTE_CONFIG = {
        RouteType.FAST: {
            "model_tier": "fast",
            "max_tokens": 500,
            "temperature": 0.5,
            "description": "快速响应 - 缓存优先，轻量模型",
        },
        RouteType.STANDARD: {
            "model_tier": "standard",
            "max_tokens": 1000,
            "temperature": 0.7,
            "description": "标准响应 - 主模型+检索",
        },
        RouteType.GOVERNANCE: {
            "model_tier": "premium",
            "max_tokens": 1500,
            "temperature": 0.3,
            "description": "治理增强 - 完整治理链",
        },
        RouteType.DIRECT: {
            "model_tier": "fast",
            "max_tokens": 800,
            "temperature": 0.7,
            "description": "直接响应 - 通用知识",
        },
    }
    
    def __init__(self, cache_client=None):
        self.cache = cache_client
        self.route_stats = {
            "total_routes": 0,
            "route_distribution": {rt.value: 0 for rt in RouteType},
            "complexity_distribution": {c.value: 0 for c in QueryComplexity},
        }
        self.route_history: List[RouteMetrics] = []
    
    def route(
        self,
        query: str,
        query_type: str,
        intent: Dict[str, Any],
        conversation_history: List[Dict],
        cache_entry: Optional[Any] = None
    ) -> RouteDecision:
        """
        路由决策
        
        路由规则：
        1. 缓存命中 -> FAST路由
        2. 简单问候/通用知识 -> DIRECT路由
        3. 身份/项目/技术栈类 -> STANDARD路由
        4. 历史回顾/复杂问题 -> GOVERNANCE路由
        """
        
        self.route_stats["total_routes"] += 1
        
        # 规则1: 缓存命中 -> FAST路由
        if cache_entry is not None:
            self.route_stats["route_distribution"][RouteType.FAST.value] += 1
            self.route_stats["complexity_distribution"][QueryComplexity.LOW.value] += 1
            
            return RouteDecision(
                route_type=RouteType.FAST,
                complexity=QueryComplexity.LOW,
                reasoning="缓存命中，直接返回",
                use_cache=True,
                use_retrieval=False,
                model_tier="fast",
                estimated_tokens=50,
                confidence=0.95
            )
        
        # 规则2: 简单问候 -> DIRECT路由
        if self._is_simple_greeting(query):
            self.route_stats["route_distribution"][RouteType.DIRECT.value] += 1
            self.route_stats["complexity_distribution"][QueryComplexity.LOW.value] += 1
            
            return RouteDecision(
                route_type=RouteType.DIRECT,
                complexity=QueryComplexity.LOW,
                reasoning="简单问候，直接响应",
                use_cache=False,
                use_retrieval=False,
                model_tier="fast",
                estimated_tokens=100,
                confidence=0.9
            )
        
        # 规则3: 根据查询类型路由
        if query_type in ["identity", "project_state", "tech_stack"]:
            # 身份/项目/技术栈类 -> STANDARD路由
            self.route_stats["route_distribution"][RouteType.STANDARD.value] += 1
            self.route_stats["complexity_distribution"][QueryComplexity.MEDIUM.value] += 1
            
            return RouteDecision(
                route_type=RouteType.STANDARD,
                complexity=QueryComplexity.MEDIUM,
                reasoning=f"{query_type}类查询，需要检索+主模型",
                use_cache=False,
                use_retrieval=True,
                model_tier="standard",
                estimated_tokens=300,
                confidence=0.85
            )
        
        elif query_type == "history_review":
            # 历史回顾类 -> GOVERNANCE路由（需要更谨慎处理）
            self.route_stats["route_distribution"][RouteType.GOVERNANCE.value] += 1
            self.route_stats["complexity_distribution"][QueryComplexity.HIGH.value] += 1
            
            return RouteDecision(
                route_type=RouteType.GOVERNANCE,
                complexity=QueryComplexity.HIGH,
                reasoning="历史回顾类，需要完整治理链",
                use_cache=False,
                use_retrieval=True,
                model_tier="premium",
                estimated_tokens=400,
                confidence=0.8
            )
        
        # 规则4: 根据意图置信度路由
        intent_confidence = intent.get("confidence", 0.5)
        intent_type = intent.get("type", "general")
        
        if intent_confidence > 0.8 and intent_type in ["fact", "query"]:
            # 高置信度事实查询 -> STANDARD路由
            self.route_stats["route_distribution"][RouteType.STANDARD.value] += 1
            self.route_stats["complexity_distribution"][QueryComplexity.MEDIUM.value] += 1
            
            return RouteDecision(
                route_type=RouteType.STANDARD,
                complexity=QueryComplexity.MEDIUM,
                reasoning="高置信度事实查询，标准处理",
                use_cache=False,
                use_retrieval=True,
                model_tier="standard",
                estimated_tokens=300,
                confidence=intent_confidence
            )
        
        elif intent_confidence < 0.5:
            # 低置信度 -> GOVERNANCE路由
            self.route_stats["route_distribution"][RouteType.GOVERNANCE.value] += 1
            self.route_stats["complexity_distribution"][QueryComplexity.HIGH.value] += 1
            
            return RouteDecision(
                route_type=RouteType.GOVERNANCE,
                complexity=QueryComplexity.HIGH,
                reasoning="低置信度查询，需要增强治理",
                use_cache=False,
                use_retrieval=True,
                model_tier="premium",
                estimated_tokens=500,
                confidence=intent_confidence
            )
        
        # 默认 -> STANDARD路由
        self.route_stats["route_distribution"][RouteType.STANDARD.value] += 1
        self.route_stats["complexity_distribution"][QueryComplexity.MEDIUM.value] += 1
        
        return RouteDecision(
            route_type=RouteType.STANDARD,
            complexity=QueryComplexity.MEDIUM,
            reasoning="默认标准路由",
            use_cache=False,
            use_retrieval=True,
            model_tier="standard",
            estimated_tokens=300,
            confidence=0.7
        )
    
    def _is_simple_greeting(self, query: str) -> bool:
        """判断是否为简单问候"""
        greetings = [
            "你好", "您好", "hello", "hi", "hey",
            "在吗", "在么", "在不在", "有人吗",
            "早上好", "下午好", "晚上好",
        ]
        
        query_clean = query.lower().strip()
        
        for greeting in greetings:
            if greeting in query_clean and len(query_clean) < 15:
                return True
        
        return False
    
    def record_metrics(self, metrics: RouteMetrics):
        """记录路由指标"""
        self.route_history.append(metrics)
    
    def get_route_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = self.route_stats["total_routes"]
        
        if total == 0:
            return self.route_stats
        
        # 计算路由分布百分比
        route_pct = {
            k: f"{v/total:.1%}" for k, v in self.route_stats["route_distribution"].items()
        }
        
        # 计算历史指标统计
        if self.route_history:
            avg_ttft = sum(m.ttft_ms for m in self.route_history) / len(self.route_history)
            avg_latency = sum(m.total_latency_ms for m in self.route_history) / len(self.route_history)
            avg_quality = sum(m.quality_score for m in self.route_history) / len(self.route_history)
            cache_hit_rate = sum(1 for m in self.route_history if m.cache_hit) / len(self.route_history)
        else:
            avg_ttft = avg_latency = avg_quality = cache_hit_rate = 0
        
        return {
            **self.route_stats,
            "route_distribution_pct": route_pct,
            "avg_ttft_ms": avg_ttft,
            "avg_total_latency_ms": avg_latency,
            "avg_quality_score": avg_quality,
            "cache_hit_rate": cache_hit_rate,
            "total_routed_queries": len(self.route_history),
        }
    
    def get_route_quality_by_type(self) -> Dict[str, Dict[str, float]]:
        """按路由类型获取质量统计"""
        quality_by_type = {}
        
        for route_type in RouteType:
            type_metrics = [m for m in self.route_history if m.route_type == route_type.value]
            
            if type_metrics:
                quality_by_type[route_type.value] = {
                    "count": len(type_metrics),
                    "avg_ttft_ms": sum(m.ttft_ms for m in type_metrics) / len(type_metrics),
                    "avg_latency_ms": sum(m.total_latency_ms for m in type_metrics) / len(type_metrics),
                    "avg_quality_score": sum(m.quality_score for m in type_metrics) / len(type_metrics),
                    "avg_token_usage": sum(m.token_usage for m in type_metrics) / len(type_metrics),
                }
        
        return quality_by_type


# 便捷函数
def create_model_router(cache_client=None) -> ModelRouter:
    """创建模型路由器"""
    return ModelRouter(cache_client=cache_client)


# 测试
if __name__ == "__main__":
    router = create_model_router()
    
    print("="*70)
    print("Model Router v1 Test")
    print("="*70)
    
    # 测试用例
    test_cases = [
        {
            "query": "你好",
            "query_type": "general",
            "intent": {"type": "greeting", "confidence": 0.9},
            "cache": None,
            "expected": RouteType.DIRECT
        },
        {
            "query": "我叫什么名字？",
            "query_type": "identity",
            "intent": {"type": "identity", "confidence": 0.8},
            "cache": None,
            "expected": RouteType.STANDARD
        },
        {
            "query": "我叫什么名字？",
            "query_type": "identity",
            "intent": {"type": "identity", "confidence": 0.8},
            "cache": type('Cache', (), {'response': 'Alice'}),  # 模拟缓存
            "expected": RouteType.FAST
        },
        {
            "query": "项目的目标是什么？",
            "query_type": "project_state",
            "intent": {"type": "fact", "confidence": 0.85},
            "cache": None,
            "expected": RouteType.STANDARD
        },
        {
            "query": "我们之前讨论过什么？",
            "query_type": "history_review",
            "intent": {"type": "history", "confidence": 0.7},
            "cache": None,
            "expected": RouteType.GOVERNANCE
        },
        {
            "query": "你确定吗？",
            "query_type": "general",
            "intent": {"type": "general", "confidence": 0.3},
            "cache": None,
            "expected": RouteType.GOVERNANCE
        },
    ]
    
    print("\n1. 路由决策测试")
    print("-"*50)
    
    for i, test in enumerate(test_cases, 1):
        decision = router.route(
            query=test["query"],
            query_type=test["query_type"],
            intent=test["intent"],
            conversation_history=[],
            cache_entry=test["cache"]
        )
        
        match = "✓" if decision.route_type == test["expected"] else "✗"
        
        print(f"\n测试{i}: {test['query']}")
        print(f"  类型: {test['query_type']}")
        print(f"  缓存: {'命中' if test['cache'] else '未命中'}")
        print(f"  决策: {decision.route_type.value} ({decision.model_tier})")
        print(f"  原因: {decision.reasoning}")
        print(f"  预期: {test['expected'].value} {match}")
    
    # 测试统计
    print("\n2. 路由统计")
    print("-"*50)
    
    stats = router.get_route_stats()
    print(f"总路由数: {stats['total_routes']}")
    print(f"路由分布:")
    for route_type, count in stats['route_distribution'].items():
        pct = stats['route_distribution_pct'].get(route_type, "0%")
        print(f"  {route_type}: {count} ({pct})")
    
    print(f"\n复杂度分布:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"  {complexity}: {count}")
    
    # 模拟记录一些指标
    print("\n3. 模拟路由指标记录")
    print("-"*50)
    
    router.record_metrics(RouteMetrics(
        route_type="fast",
        query_type="identity",
        ttft_ms=50,
        total_latency_ms=100,
        token_usage=50,
        quality_score=0.95,
        cache_hit=True,
        retrieval_count=0
    ))
    
    router.record_metrics(RouteMetrics(
        route_type="standard",
        query_type="project_state",
        ttft_ms=500,
        total_latency_ms=1500,
        token_usage=300,
        quality_score=0.9,
        cache_hit=False,
        retrieval_count=2
    ))
    
    router.record_metrics(RouteMetrics(
        route_type="governance",
        query_type="history_review",
        ttft_ms=800,
        total_latency_ms=2500,
        token_usage=500,
        quality_score=0.85,
        cache_hit=False,
        retrieval_count=3
    ))
    
    # 更新后的统计
    stats = router.get_route_stats()
    print(f"平均TTFT: {stats['avg_ttft_ms']:.0f}ms")
    print(f"平均总延迟: {stats['avg_total_latency_ms']:.0f}ms")
    print(f"平均质量分: {stats['avg_quality_score']:.2f}")
    print(f"缓存命中率: {stats['cache_hit_rate']:.1%}")
    
    # 按类型质量统计
    print("\n4. 按路由类型质量统计")
    print("-"*50)
    
    quality_stats = router.get_route_quality_by_type()
    for route_type, metrics in quality_stats.items():
        print(f"\n{route_type}:")
        print(f"  查询数: {metrics['count']}")
        print(f"  平均TTFT: {metrics['avg_ttft_ms']:.0f}ms")
        print(f"  平均延迟: {metrics['avg_latency_ms']:.0f}ms")
        print(f"  平均质量: {metrics['avg_quality_score']:.2f}")
        print(f"  平均Token: {metrics['avg_token_usage']:.0f}")
