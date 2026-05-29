"""
Retrieval Service V1 - 检索服务 V1

Phase 11 WP1 核心组件：
实现完整的记忆检索与知识检索服务

功能：
1. 内部记忆检索（短期/长期/永久层）
2. 外部知识检索
3. 分层索引管理
4. 多级缓存
5. 超时控制
6. 结果排序与过滤
"""

import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RetrievalType(Enum):
    """检索类型"""
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"


class MemoryLayer(Enum):
    """记忆层"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SHALLOW_PERMANENT = "shallow_permanent"
    DEEP_PERMANENT = "deep_permanent"


@dataclass
class RetrievalResult:
    """检索结果"""
    id: str
    content: str
    score: float
    source: str
    layer: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRequest:
    """检索请求"""
    query: str
    query_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    retrieval_type: RetrievalType = RetrievalType.HYBRID
    max_results: int = 10
    timeout_ms: int = 1000
    filters: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResponse:
    """检索响应"""
    query_id: str
    results: List[RetrievalResult]
    total_found: int
    latency_ms: float
    cache_hit: bool
    from_layer: str
    status: str
    error: Optional[str] = None


class CacheEntry:
    """缓存条目"""
    def __init__(self, results: List[RetrievalResult], ttl_seconds: int = 300):
        self.results = results
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl_seconds)
    
    def touch(self):
        """更新访问计数"""
        self.access_count += 1


class CacheLayer:
    """缓存层"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _make_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """生成缓存键"""
        key_content = f"{query}:{str(filters)}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[List[RetrievalResult]]:
        """获取缓存"""
        key = self._make_key(query, filters)
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats["misses"] += 1
            return None
        
        if entry.is_expired():
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        
        entry.touch()
        self.stats["hits"] += 1
        return entry.results
    
    def put(self, query: str, results: List[RetrievalResult], 
            filters: Optional[Dict] = None, ttl: Optional[int] = None):
        """写入缓存"""
        # 清理过期条目
        self._cleanup_expired()
        
        # 如果缓存已满，移除最少访问的条目
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        key = self._make_key(query, filters)
        self.cache[key] = CacheEntry(results, ttl or self.default_ttl)
    
    def _cleanup_expired(self):
        """清理过期条目"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_lru(self):
        """移除最少访问的条目"""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        del self.cache[lru_key]
        self.stats["evictions"] += 1
    
    def invalidate(self, pattern: Optional[str] = None):
        """失效缓存"""
        if pattern is None:
            self.cache.clear()
        else:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"]
        }


class IndexStore:
    """索引存储（模拟）"""
    
    def __init__(self, layer: MemoryLayer):
        self.layer = layer
        self.index: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            "queries": 0,
            "inserts": 0,
            "deletes": 0
        }
    
    def search(self, query: str, max_results: int = 10) -> List[RetrievalResult]:
        """搜索索引"""
        self.stats["queries"] += 1
        
        # 改进的搜索：支持中文子串匹配
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in self.index.items():
            content = doc.get("content", "").lower()
            
            # 计算匹配分数
            # 1. 完整查询匹配（最高优先级）
            if query_lower in content:
                score = 1.0
            else:
                # 2. 关键词匹配（按空格分割）
                query_terms = query_lower.split()
                if query_terms:
                    matches = sum(1 for term in query_terms if len(term) > 1 and term in content)
                    score = matches / len(query_terms) if matches > 0 else 0
                else:
                    score = 0
            
            if score > 0:
                results.append(RetrievalResult(
                    id=doc_id,
                    content=doc.get("content", ""),
                    score=score,
                    source=doc.get("source", "internal"),
                    layer=self.layer.value,
                    timestamp=doc.get("timestamp", datetime.now()),
                    metadata=doc.get("metadata", {})
                ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def insert(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """插入文档"""
        self.index[doc_id] = {
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.stats["inserts"] += 1
    
    def delete(self, doc_id: str):
        """删除文档"""
        if doc_id in self.index:
            del self.index[doc_id]
            self.stats["deletes"] += 1


class RetrievalService:
    """
    检索服务 V1
    
    职责：
    1. 内部记忆检索（多层索引）
    2. 外部知识检索
    3. 缓存管理
    4. 超时控制
    5. 结果合并与排序
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8081,
        default_timeout_ms: int = 1000
    ):
        self.host = host
        self.port = port
        self.default_timeout_ms = default_timeout_ms
        
        # 初始化缓存层
        self.cache = CacheLayer(max_size=10000, default_ttl=300)
        
        # 初始化多层索引
        self.indexes = {
            MemoryLayer.SHORT_TERM: IndexStore(MemoryLayer.SHORT_TERM),
            MemoryLayer.LONG_TERM: IndexStore(MemoryLayer.LONG_TERM),
            MemoryLayer.SHALLOW_PERMANENT: IndexStore(MemoryLayer.SHALLOW_PERMANENT),
            MemoryLayer.DEEP_PERMANENT: IndexStore(MemoryLayer.DEEP_PERMANENT)
        }
        
        # 外部知识源（模拟）
        self.external_knowledge: Dict[str, str] = {}
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "timeout_requests": 0,
            "error_requests": 0,
            "avg_latency_ms": 0
        }
        
        self.is_running = False
    
    async def start(self):
        """启动服务"""
        self.is_running = True
        print(f"Retrieval Service started on {self.host}:{self.port}")
    
    async def stop(self):
        """停止服务"""
        self.is_running = False
        print("Retrieval Service stopped")
    
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        执行检索
        
        流程：
        1. 检查缓存
        2. 根据类型选择检索策略
        3. 执行检索（带超时）
        4. 合并结果
        5. 更新缓存
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # 1. 检查缓存
            cached_results = self.cache.get(request.query, request.filters)
            if cached_results is not None:
                latency_ms = (time.time() - start_time) * 1000
                self._update_latency_stats(latency_ms)
                self.stats["successful_requests"] += 1
                
                return RetrievalResponse(
                    query_id=request.query_id,
                    results=cached_results,
                    total_found=len(cached_results),
                    latency_ms=latency_ms,
                    cache_hit=True,
                    from_layer="cache",
                    status="success"
                )
            
            # 2. 执行检索（带超时控制）
            results = await asyncio.wait_for(
                self._do_retrieve(request),
                timeout=request.timeout_ms / 1000
            )
            
            # 3. 更新缓存
            self.cache.put(request.query, results, request.filters)
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_stats(latency_ms)
            self.stats["successful_requests"] += 1
            
            return RetrievalResponse(
                query_id=request.query_id,
                results=results,
                total_found=len(results),
                latency_ms=latency_ms,
                cache_hit=False,
                from_layer=self._determine_source_layer(request.retrieval_type),
                status="success"
            )
            
        except asyncio.TimeoutError:
            self.stats["timeout_requests"] += 1
            latency_ms = (time.time() - start_time) * 1000
            
            return RetrievalResponse(
                query_id=request.query_id,
                results=[],
                total_found=0,
                latency_ms=latency_ms,
                cache_hit=False,
                from_layer="timeout",
                status="timeout",
                error="Retrieval timeout"
            )
            
        except Exception as e:
            self.stats["error_requests"] += 1
            latency_ms = (time.time() - start_time) * 1000
            
            return RetrievalResponse(
                query_id=request.query_id,
                results=[],
                total_found=0,
                latency_ms=latency_ms,
                cache_hit=False,
                from_layer="error",
                status="error",
                error=str(e)
            )
    
    async def _do_retrieve(self, request: RetrievalRequest) -> List[RetrievalResult]:
        """执行实际检索"""
        all_results = []
        
        if request.retrieval_type in [RetrievalType.MEMORY, RetrievalType.HYBRID]:
            # 检索内部记忆
            for layer in [MemoryLayer.DEEP_PERMANENT, MemoryLayer.SHALLOW_PERMANENT,
                         MemoryLayer.LONG_TERM, MemoryLayer.SHORT_TERM]:
                index = self.indexes[layer]
                results = index.search(request.query, request.max_results)
                all_results.extend(results)
        
        if request.retrieval_type in [RetrievalType.KNOWLEDGE, RetrievalType.HYBRID]:
            # 检索外部知识
            external_results = self._search_external(request.query, request.max_results)
            all_results.extend(external_results)
        
        # 合并、去重、排序
        all_results = self._merge_and_rank(all_results, request.max_results)
        
        return all_results
    
    def _search_external(self, query: str, max_results: int) -> List[RetrievalResult]:
        """搜索外部知识"""
        results = []
        query_lower = query.lower()
        
        for doc_id, content in self.external_knowledge.items():
            if query_lower in content.lower():
                results.append(RetrievalResult(
                    id=doc_id,
                    content=content,
                    score=0.8,  # 外部知识默认分数
                    source="external",
                    layer="external",
                    timestamp=datetime.now()
                ))
        
        return results[:max_results]
    
    def _merge_and_rank(self, results: List[RetrievalResult], max_results: int) -> List[RetrievalResult]:
        """合并结果并排序"""
        # 去重（基于 ID）
        seen_ids = set()
        unique_results = []
        for r in results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique_results.append(r)
        
        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:max_results]
    
    def _determine_source_layer(self, retrieval_type: RetrievalType) -> str:
        """确定来源层"""
        layer_map = {
            RetrievalType.MEMORY: "memory_layers",
            RetrievalType.KNOWLEDGE: "external",
            RetrievalType.HYBRID: "hybrid"
        }
        return layer_map.get(retrieval_type, "unknown")
    
    def _update_latency_stats(self, latency_ms: float):
        """更新延迟统计"""
        # 简单移动平均
        n = self.stats["successful_requests"]
        if n == 0:
            self.stats["avg_latency_ms"] = latency_ms
        else:
            self.stats["avg_latency_ms"] = (
                self.stats["avg_latency_ms"] * (n - 1) + latency_ms
            ) / n
    
    def insert_memory(self, layer: MemoryLayer, doc_id: str, content: str, 
                     metadata: Optional[Dict] = None):
        """插入记忆"""
        self.indexes[layer].insert(doc_id, content, metadata)
    
    def insert_knowledge(self, doc_id: str, content: str):
        """插入外部知识"""
        self.external_knowledge[doc_id] = content
    
    def invalidate_cache(self, pattern: Optional[str] = None):
        """失效缓存"""
        self.cache.invalidate(pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "service": "retrieval",
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "requests": self.stats,
            "cache": self.cache.get_stats(),
            "indexes": {
                layer.value: idx.stats
                for layer, idx in self.indexes.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }


async def demo_retrieval_service():
    """演示检索服务"""
    print("\n" + "="*70)
    print("Retrieval Service V1 - 演示")
    print("="*70)
    
    service = RetrievalService()
    await service.start()
    
    # 插入测试数据
    print("\n1. 准备测试数据")
    print("-" * 50)
    
    # 长期层数据
    service.insert_memory(
        MemoryLayer.LONG_TERM,
        "mem_001",
        "The quick brown fox jumps over the lazy dog",
        {"source": "training", "confidence": 0.9}
    )
    service.insert_memory(
        MemoryLayer.LONG_TERM,
        "mem_002",
        "Machine learning is a subset of artificial intelligence",
        {"source": "training", "confidence": 0.85}
    )
    
    # 永久层数据
    service.insert_memory(
        MemoryLayer.DEEP_PERMANENT,
        "perm_001",
        "Python is a high-level programming language",
        {"source": "core_knowledge", "confidence": 0.95}
    )
    
    # 外部知识
    service.insert_knowledge(
        "ext_001",
        "Neural networks are computing systems inspired by biological neural networks"
    )
    
    print("  ✓ 插入 3 条记忆 + 1 条外部知识")
    
    # 测试检索
    print("\n2. 检索测试")
    print("-" * 50)
    
    test_queries = [
        ("python programming", RetrievalType.MEMORY),
        ("neural networks", RetrievalType.KNOWLEDGE),
        ("machine learning", RetrievalType.HYBRID),
    ]
    
    for query, ret_type in test_queries:
        request = RetrievalRequest(
            query=query,
            query_id=f"q_{hash(query) % 10000}",
            retrieval_type=ret_type,
            max_results=5,
            timeout_ms=1000
        )
        
        response = await service.retrieve(request)
        
        print(f"\n  查询: '{query}' (类型: {ret_type.value})")
        print(f"  状态: {response.status}, 延迟: {response.latency_ms:.2f}ms, 缓存: {response.cache_hit}")
        print(f"  结果数: {len(response.results)}")
        
        for i, result in enumerate(response.results[:2], 1):
            print(f"    {i}. [{result.layer}] {result.content[:50]}... (score: {result.score:.2f})")
    
    # 测试缓存
    print("\n3. 缓存测试")
    print("-" * 50)
    
    request = RetrievalRequest(
        query="python programming",
        query_id="cache_test",
        retrieval_type=RetrievalType.MEMORY
    )
    
    # 第一次查询
    response1 = await service.retrieve(request)
    print(f"  第一次查询: 缓存命中 = {response1.cache_hit}")
    
    # 第二次查询（应该命中缓存）
    response2 = await service.retrieve(request)
    print(f"  第二次查询: 缓存命中 = {response2.cache_hit}")
    
    # 统计
    print("\n4. 服务统计")
    print("-" * 50)
    
    stats = service.get_stats()
    print(f"  总请求数: {stats['requests']['total_requests']}")
    print(f"  成功请求: {stats['requests']['successful_requests']}")
    print(f"  平均延迟: {stats['requests']['avg_latency_ms']:.2f}ms")
    print(f"  缓存命中率: {stats['cache']['hit_rate']:.2%}")
    print(f"  缓存大小: {stats['cache']['size']}/{stats['cache']['max_size']}")
    
    await service.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_retrieval_service())
