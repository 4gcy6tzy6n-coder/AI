"""
Response Cache v1 - 响应缓存 v1

目标：
1. 缓存高频查询响应
2. 降低TTFT到<100ms
3. 监控Cache Hit Rate
"""

import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class CachePolicy(Enum):
    """缓存策略"""
    IDENTITY = "identity"      # 身份类 - 缓存1小时
    PROJECT_STATE = "project"  # 项目状态 - 缓存30分钟
    TECH_STACK = "tech"        # 技术栈 - 缓存1小时
    HISTORY = "history"        # 历史回顾 - 不缓存
    GENERAL = "general"        # 通用 - 缓存5分钟


@dataclass
class CacheEntry:
    """缓存条目"""
    query_hash: str
    response: str
    strategy: str
    confidence: float
    citations: List[str]
    created_at: float
    ttl_seconds: int
    query_type: str
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.created_at > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "response": self.response[:100] + "..." if len(self.response) > 100 else self.response,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "ttl_seconds": self.ttl_seconds,
            "query_type": self.query_type,
            "hit_count": self.hit_count,
            "is_expired": self.is_expired(),
        }


class ResponseCache:
    """响应缓存"""
    
    # 缓存策略配置
    POLICY_TTL = {
        CachePolicy.IDENTITY: 3600,      # 1小时
        CachePolicy.PROJECT_STATE: 1800, # 30分钟
        CachePolicy.TECH_STACK: 3600,    # 1小时
        CachePolicy.HISTORY: 0,          # 不缓存
        CachePolicy.GENERAL: 300,        # 5分钟
    }
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }
    
    def _generate_key(self, query: str, query_type: str) -> str:
        """生成缓存键"""
        # 规范化查询
        normalized = query.lower().strip().replace(" ", "")
        # 组合查询类型和内容
        key_content = f"{query_type}:{normalized}"
        return hashlib.md5(key_content.encode()).hexdigest()[:16]
    
    def _get_policy(self, query_type: str) -> CachePolicy:
        """根据查询类型获取缓存策略"""
        policy_map = {
            "identity": CachePolicy.IDENTITY,
            "project_state": CachePolicy.PROJECT_STATE,
            "tech_stack": CachePolicy.TECH_STACK,
            "history_review": CachePolicy.HISTORY,
            "general": CachePolicy.GENERAL,
        }
        return policy_map.get(query_type, CachePolicy.GENERAL)
    
    def get(self, query: str, query_type: str) -> Optional[CacheEntry]:
        """获取缓存"""
        self.stats["total_requests"] += 1
        
        policy = self._get_policy(query_type)
        
        # 不缓存的策略直接返回None
        if self.POLICY_TTL[policy] == 0:
            self.stats["cache_misses"] += 1
            return None
        
        key = self._generate_key(query, query_type)
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats["cache_misses"] += 1
            return None
        
        if entry.is_expired():
            # 过期删除
            del self.cache[key]
            self.stats["cache_misses"] += 1
            return None
        
        # 命中
        entry.hit_count += 1
        self.stats["cache_hits"] += 1
        return entry
    
    def set(
        self,
        query: str,
        query_type: str,
        response: str,
        strategy: str,
        confidence: float,
        citations: List[str]
    ) -> bool:
        """设置缓存"""
        policy = self._get_policy(query_type)
        ttl = self.POLICY_TTL[policy]
        
        # 不缓存的策略
        if ttl == 0:
            return False
        
        # 检查缓存大小，必要时淘汰
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._generate_key(query, query_type)
        
        entry = CacheEntry(
            query_hash=key,
            response=response,
            strategy=strategy,
            confidence=confidence,
            citations=citations,
            created_at=time.time(),
            ttl_seconds=ttl,
            query_type=query_type,
            hit_count=0
        )
        
        self.cache[key] = entry
        return True
    
    def _evict_oldest(self):
        """淘汰最旧的缓存"""
        if not self.cache:
            return
        
        # 找到最旧的条目
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        del self.cache[oldest_key]
        self.stats["evictions"] += 1
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.stats["total_requests"]
        if total == 0:
            return 0.0
        return self.stats["cache_hits"] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.stats["total_requests"]
        
        # 按类型统计命中率
        type_stats = {}
        for entry in self.cache.values():
            qt = entry.query_type
            if qt not in type_stats:
                type_stats[qt] = {"count": 0, "hits": 0}
            type_stats[qt]["count"] += 1
            type_stats[qt]["hits"] += entry.hit_count
        
        return {
            "total_requests": total,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": self.get_hit_rate(),
            "evictions": self.stats["evictions"],
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "type_stats": type_stats,
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }
    
    def get_cached_entries(self, limit: int = 10) -> List[Dict]:
        """获取缓存条目（用于监控）"""
        entries = sorted(
            self.cache.values(),
            key=lambda e: e.hit_count,
            reverse=True
        )
        return [e.to_dict() for e in entries[:limit]]


# 便捷函数
def create_response_cache(max_size: int = 1000) -> ResponseCache:
    """创建响应缓存"""
    return ResponseCache(max_size=max_size)


# 测试
if __name__ == "__main__":
    cache = create_response_cache()
    
    print("="*70)
    print("Response Cache Test")
    print("="*70)
    
    # 测试1: 身份类缓存
    print("\n1. 身份类缓存测试")
    print("-"*50)
    
    query1 = "我叫什么名字？"
    query_type1 = "identity"
    
    # 第一次查询 - 未命中
    result1 = cache.get(query1, query_type1)
    print(f"第一次查询: {'命中' if result1 else '未命中'}")
    
    # 设置缓存
    cache.set(query1, query_type1, "你的名字是Alice。", "RETRIEVAL_FIRST", 0.9, ["来源1"])
    print(f"设置缓存: 成功")
    
    # 第二次查询 - 命中
    result2 = cache.get(query1, query_type1)
    print(f"第二次查询: {'命中' if result2 else '未命中'}")
    if result2:
        print(f"  响应: {result2.response}")
        print(f"  命中次数: {result2.hit_count}")
    
    # 测试2: 历史类不缓存
    print("\n2. 历史类不缓存测试")
    print("-"*50)
    
    query2 = "我们之前讨论过什么？"
    query_type2 = "history_review"
    
    cache.set(query2, query_type2, "之前讨论过项目目标。", "RETRIEVAL_FIRST", 0.8, ["来源1"])
    result3 = cache.get(query2, query_type2)
    print(f"历史类查询: {'命中' if result3 else '未命中'} (预期不缓存)")
    
    # 测试3: 命中率统计
    print("\n3. 命中率统计")
    print("-"*50)
    
    # 多次查询身份类
    for i in range(5):
        cache.get("我叫什么名字？", "identity")
    
    # 多次查询技术栈类
    cache.set("技术栈是什么？", "tech_stack", "技术栈是Python+React。", "RETRIEVAL_FIRST", 0.9, ["来源1"])
    for i in range(3):
        cache.get("技术栈是什么？", "tech_stack")
    
    stats = cache.get_stats()
    print(f"总请求数: {stats['total_requests']}")
    print(f"缓存命中: {stats['cache_hits']}")
    print(f"缓存未命中: {stats['cache_misses']}")
    print(f"命中率: {stats['hit_rate']:.1%}")
    print(f"当前缓存大小: {stats['current_size']}/{stats['max_size']}")
    
    # 测试4: 按类型统计
    print("\n4. 按类型统计")
    print("-"*50)
    
    for query_type, type_stat in stats['type_stats'].items():
        print(f"{query_type}: {type_stat['count']}条缓存, {type_stat['hits']}次命中")
    
    # 测试5: 缓存内容查看
    print("\n5. 热门缓存条目")
    print("-"*50)
    
    entries = cache.get_cached_entries(5)
    for i, entry in enumerate(entries, 1):
        print(f"{i}. [{entry['query_type']}] 命中{entry['hit_count']}次 - {entry['response'][:50]}...")
