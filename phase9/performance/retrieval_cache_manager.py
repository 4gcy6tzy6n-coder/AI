"""
Retrieval Cache Manager - 检索缓存管理器

WP1 核心组件：
优化检索性能，减少重复计算

功能：
1. 多级缓存架构 (L1: 内存, L2: 本地磁盘, L3: 分布式)
2. 智能缓存策略 (LRU, LFU, TTL)
3. 缓存预热与预取
4. 缓存命中率监控
"""

import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """更新访问时间"""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """
    LRU (Least Recently Used) 缓存
    
    基于 OrderedDict 实现 O(1) 的 get 和 put
    """
    
    def __init__(self, capacity: int = 1000, max_size_bytes: int = 100 * 1024 * 1024):
        self.capacity = capacity
        self.max_size_bytes = max_size_bytes
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.lock = Lock()
        
        # 统计
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """生成缓存键"""
        key_data = f"{query}:{json.dumps(filters, sort_keys=True) if filters else ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> Optional[Any]:
        """获取缓存值"""
        key = self._generate_key(query, filters)
        
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # 检查过期
            if entry.is_expired():
                self._evict(key)
                self.misses += 1
                return None
            
            # 更新访问信息
            entry.touch()
            self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def put(
        self,
        query: str,
        value: Any,
        filters: Optional[Dict] = None,
        ttl_seconds: Optional[float] = None
    ):
        """设置缓存值"""
        key = self._generate_key(query, filters)
        
        # 估算大小
        size_bytes = len(json.dumps(value, default=str).encode())
        
        with self.lock:
            # 如果已存在，更新
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.size_bytes
            
            # 检查容量
            while (len(self.cache) >= self.capacity or 
                   self.current_size_bytes + size_bytes > self.max_size_bytes):
                if not self.cache:
                    break
                self._evict_lru()
            
            # 创建新条目
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.cache.move_to_end(key)
            self.current_size_bytes += size_bytes
    
    def _evict(self, key: str):
        """驱逐指定条目"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes
            self.evictions += 1
    
    def _evict_lru(self):
        """驱逐最久未使用的条目"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.current_size_bytes -= entry.size_bytes
            self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "utilization": len(self.cache) / self.capacity if self.capacity > 0 else 0,
            "size_bytes": self.current_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0


class MultiLevelCache:
    """
    多级缓存管理器
    
    架构：
    - L1: 内存缓存 (LRU, 最快)
    - L2: 本地磁盘缓存 (持久化)
    - L3: 远程缓存 (分布式)
    """
    
    def __init__(
        self,
        l1_capacity: int = 1000,
        l2_capacity: int = 10000,
        l2_ttl_seconds: float = 3600
    ):
        self.l1_cache = LRUCache(capacity=l1_capacity)
        self.l2_cache: Dict[str, CacheEntry] = {}
        self.l2_capacity = l2_capacity
        self.l2_ttl_seconds = l2_ttl_seconds
        
        # 统计
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.misses = 0
    
    def get(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> Optional[Any]:
        """多级缓存获取"""
        # L1: 内存缓存
        value = self.l1_cache.get(query, filters)
        if value is not None:
            self.l1_hits += 1
            return value
        
        # L2: 本地磁盘缓存 (简化版，使用内存模拟)
        key = self.l1_cache._generate_key(query, filters)
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if not entry.is_expired():
                entry.touch()
                self.l2_hits += 1
                
                # 回填 L1
                self.l1_cache.put(query, entry.value, filters)
                
                return entry.value
            else:
                del self.l2_cache[key]
        
        self.misses += 1
        return None
    
    def put(
        self,
        query: str,
        value: Any,
        filters: Optional[Dict] = None,
        l1_ttl: Optional[float] = None,
        l2_ttl: Optional[float] = None
    ):
        """多级缓存设置"""
        # 写入 L1
        self.l1_cache.put(query, value, filters, l1_ttl)
        
        # 写入 L2
        key = self.l1_cache._generate_key(query, filters)
        
        # 清理过期条目
        self._cleanup_l2()
        
        # 如果 L2 满了，移除最旧的
        while len(self.l2_cache) >= self.l2_capacity:
            if self.l2_cache:
                oldest = min(self.l2_cache.items(), key=lambda x: x[1].accessed_at)
                del self.l2_cache[oldest[0]]
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            ttl_seconds=l2_ttl or self.l2_ttl_seconds
        )
        self.l2_cache[key] = entry
    
    def _cleanup_l2(self):
        """清理 L2 过期条目"""
        expired_keys = [
            key for key, entry in self.l2_cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.l2_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取多级缓存统计"""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        
        return {
            "l1": self.l1_cache.get_stats(),
            "l2": {
                "size": len(self.l2_cache),
                "capacity": self.l2_capacity,
                "utilization": len(self.l2_cache) / self.l2_capacity if self.l2_capacity > 0 else 0
            },
            "hits": {
                "l1": self.l1_hits,
                "l2": self.l2_hits,
                "total": self.l1_hits + self.l2_hits
            },
            "misses": self.misses,
            "overall_hit_rate": (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0
        }


class CacheWarmer:
    """
    缓存预热器
    
    功能：
    1. 系统启动时预热高频查询
    2. 基于历史数据预取
    3. 智能预取策略
    """
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.hot_queries: List[str] = []
        self.warmup_complete = False
    
    def add_hot_query(self, query: str):
        """添加热点查询"""
        if query not in self.hot_queries:
            self.hot_queries.append(query)
    
    def warmup(
        self,
        query_func: Callable[[str], Any],
        filters: Optional[Dict] = None
    ):
        """执行预热"""
        print(f"  开始缓存预热，热点查询数: {len(self.hot_queries)}")
        
        warmed = 0
        for query in self.hot_queries:
            # 检查是否已在缓存中
            if self.cache.get(query, filters) is None:
                try:
                    # 执行查询并缓存
                    result = query_func(query)
                    self.cache.put(query, result, filters)
                    warmed += 1
                except Exception as e:
                    print(f"    预热失败 [{query}]: {e}")
        
        self.warmup_complete = True
        print(f"  预热完成，新增缓存: {warmed}")
        
        return warmed
    
    def get_warmup_status(self) -> Dict[str, Any]:
        """获取预热状态"""
        return {
            "hot_queries_count": len(self.hot_queries),
            "warmup_complete": self.warmup_complete
        }


def demo_cache_manager():
    """演示缓存管理器"""
    print("\n" + "="*70)
    print("Retrieval Cache Manager - 演示")
    print("="*70)
    
    # 创建多级缓存
    cache = MultiLevelCache(
        l1_capacity=100,
        l2_capacity=500,
        l2_ttl_seconds=3600
    )
    
    print("\n1. 基础缓存操作")
    print("-" * 50)
    
    # 写入缓存
    test_queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning",
        "What is AI?",
        "How to train a model?"
    ]
    
    for i, query in enumerate(test_queries):
        result = {
            "query": query,
            "answer": f"Answer to: {query}",
            "confidence": 0.8 + i * 0.02
        }
        cache.put(query, result)
        print(f"  缓存写入: {query[:30]}...")
    
    # 读取缓存
    print(f"\n  读取缓存:")
    for query in test_queries[:3]:
        value = cache.get(query)
        if value:
            print(f"    ✓ 命中: {query[:30]}...")
        else:
            print(f"    ✗ 未命中: {query[:30]}...")
    
    # 读取未缓存的
    miss_query = "Unknown query"
    value = cache.get(miss_query)
    print(f"    {'✓ 命中' if value else '✗ 未命中'}: {miss_query}")
    
    print("\n2. 缓存统计")
    print("-" * 50)
    
    stats = cache.get_stats()
    print(f"  L1 缓存:")
    print(f"    大小: {stats['l1']['size']}/{stats['l1']['capacity']}")
    print(f"    命中率: {stats['l1']['hit_rate']:.1%}")
    print(f"  L2 缓存:")
    print(f"    大小: {stats['l2']['size']}/{stats['l2']['capacity']}")
    print(f"  整体命中率: {stats['overall_hit_rate']:.1%}")
    
    print("\n3. 缓存预热")
    print("-" * 50)
    
    warmer = CacheWarmer(cache)
    
    # 添加热点查询
    for query in test_queries:
        warmer.add_hot_query(query)
    
    # 模拟预热
    def mock_query_func(query: str) -> Dict:
        return {"query": query, "answer": f"Mock answer for {query}"}
    
    warmed = warmer.warmup(mock_query_func)
    
    # 预热后统计
    stats = cache.get_stats()
    print(f"\n  预热后统计:")
    print(f"    L1 大小: {stats['l1']['size']}")
    print(f"    L2 大小: {stats['l2']['size']}")
    
    print("\n4. TTL 过期测试")
    print("-" * 50)
    
    # 写入带 TTL 的缓存
    cache.put("temp_query", {"data": "temporary"}, l2_ttl=0.1)
    print(f"  写入临时缓存 (TTL=0.1s)")
    
    # 立即读取
    value = cache.get("temp_query")
    print(f"  立即读取: {'✓ 命中' if value else '✗ 未命中'}")
    
    # 等待过期
    time.sleep(0.2)
    value = cache.get("temp_query")
    print(f"  过期后读取: {'✓ 命中' if value else '✗ 未命中'}")
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_cache_manager()
