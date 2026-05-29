"""
Retrieval Service Prototype - 检索服务原型

WP3 核心组件：
实现检索服务独立运行模式

功能：
1. 独立检索服务
2. RPC 接口
3. 缓存管理
4. 索引管理
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class RetrievalRequest:
    """检索请求"""
    request_id: str
    query_vector: List[float]
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    use_cache: bool = True


@dataclass
class RetrievalResponse:
    """检索响应"""
    request_id: str
    results: List[Dict[str, Any]]
    total_found: int
    search_time_ms: float
    cache_hit: bool
    status: str
    error: Optional[str] = None


@dataclass
class IndexUpdateRequest:
    """索引更新请求"""
    operation: str  # add, update, delete
    units: List[Dict[str, Any]]


class MockIndex:
    """模拟索引"""
    
    def __init__(self):
        self.units: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, List[float]] = {}
    
    def add(self, unit_id: str, vector: List[float], metadata: Dict[str, Any]):
        """添加单元"""
        self.units[unit_id] = metadata
        self.vectors[unit_id] = vector
    
    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """搜索"""
        # 简化的相似度计算
        results = []
        for unit_id, vector in self.vectors.items():
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vector, vector)
            results.append({
                "unit_id": unit_id,
                "score": similarity,
                "metadata": self.units[unit_id]
            })
        
        # 排序并返回前 k 个
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(a * a for a in v2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class MockCache:
    """模拟缓存"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.stats = {"hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """设置缓存"""
        self.cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }


class RetrievalService:
    """
    检索服务
    
    功能：
    1. 处理检索请求
    2. 管理索引
    3. 管理缓存
    4. 提供服务状态
    """
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.index = MockIndex()
        self.cache = MockCache()
        self.is_running = False
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_search_time_ms": 0.0
        }
    
    async def start(self):
        """启动服务"""
        self.is_running = True
        print(f"  检索服务已启动: {self.host}:{self.port}")
        
        # 初始化索引
        await self._init_index()
    
    async def stop(self):
        """停止服务"""
        self.is_running = False
        print(f"  检索服务已停止")
    
    async def _init_index(self):
        """初始化索引"""
        # 添加一些模拟数据
        import random
        
        for i in range(100):
            unit_id = f"unit_{i:03d}"
            vector = [random.random() for _ in range(128)]
            metadata = {
                "type": random.choice(["concept", "relation", "rule"]),
                "quality": random.uniform(0.7, 0.95)
            }
            self.index.add(unit_id, vector, metadata)
        
        print(f"  索引已初始化: {len(self.index.units)} 个 Units")
    
    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """处理检索请求"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # 检查缓存
            cache_key = f"{request.query_vector[:5]}:{request.top_k}"
            cached_result = None
            
            if request.use_cache:
                cached_result = self.cache.get(cache_key)
            
            if cached_result:
                search_time = (time.time() - start_time) * 1000
                self.stats["successful_requests"] += 1
                
                return RetrievalResponse(
                    request_id=request.request_id,
                    results=cached_result,
                    total_found=len(cached_result),
                    search_time_ms=search_time,
                    cache_hit=True,
                    status="success"
                )
            
            # 执行搜索
            results = self.index.search(request.query_vector, request.top_k)
            
            # 更新缓存
            if request.use_cache:
                self.cache.set(cache_key, results)
            
            search_time = (time.time() - start_time) * 1000
            self.stats["successful_requests"] += 1
            
            # 更新平均搜索时间
            self.stats["avg_search_time_ms"] = (
                (self.stats["avg_search_time_ms"] * (self.stats["successful_requests"] - 1) + search_time)
                / self.stats["successful_requests"]
            )
            
            return RetrievalResponse(
                request_id=request.request_id,
                results=results,
                total_found=len(results),
                search_time_ms=search_time,
                cache_hit=False,
                status="success"
            )
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            
            return RetrievalResponse(
                request_id=request.request_id,
                results=[],
                total_found=0,
                search_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False,
                status="error",
                error=str(e)
            )
    
    async def update_index(self, request: IndexUpdateRequest) -> Dict[str, Any]:
        """更新索引"""
        updated = 0
        
        for unit in request.units:
            unit_id = unit.get("unit_id")
            vector = unit.get("vector")
            metadata = unit.get("metadata", {})
            
            if request.operation == "add":
                self.index.add(unit_id, vector, metadata)
                updated += 1
            elif request.operation == "delete":
                if unit_id in self.index.units:
                    del self.index.units[unit_id]
                    del self.index.vectors[unit_id]
                    updated += 1
        
        return {
            "status": "success",
            "operation": request.operation,
            "updated_count": updated,
            "total_units": len(self.index.units)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "service": "retrieval",
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "index_size": len(self.index.units),
            **self.stats,
            "cache": self.cache.get_stats()
        }


class RetrievalServiceClient:
    """
    检索服务客户端
    
    用于其他服务调用检索服务
    """
    
    def __init__(self, service: RetrievalService):
        self.service = service
    
    async def query_memory(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> RetrievalResponse:
        """查询记忆"""
        import uuid
        
        request = RetrievalRequest(
            request_id=str(uuid.uuid4())[:8],
            query_vector=query_vector,
            top_k=top_k
        )
        
        return await self.service.search(request)


async def demo_retrieval_service():
    """演示检索服务"""
    print("\n" + "="*70)
    print("Retrieval Service Prototype - 演示")
    print("="*70)
    
    # 创建服务
    service = RetrievalService(host="localhost", port=8001)
    client = RetrievalServiceClient(service)
    
    print("\n1. 启动检索服务")
    print("-" * 50)
    await service.start()
    
    print("\n2. 执行检索请求")
    print("-" * 50)
    
    import random
    
    # 执行多次检索
    for i in range(5):
        # 生成随机查询向量
        query_vector = [random.random() for _ in range(128)]
        
        response = await client.query_memory(query_vector, top_k=3)
        
        print(f"\n  请求 {i+1}:")
        print(f"    状态: {response.status}")
        print(f"    耗时: {response.search_time_ms:.2f} ms")
        print(f"    缓存命中: {response.cache_hit}")
        print(f"    结果数: {response.total_found}")
        
        if response.results:
            top_result = response.results[0]
            print(f"    最佳匹配: {top_result['unit_id']} (score: {top_result['score']:.3f})")
    
    print("\n3. 再次执行 (测试缓存)")
    print("-" * 50)
    
    # 使用相同的查询向量
    query_vector = [random.random() for _ in range(128)]
    
    # 第一次
    response1 = await client.query_memory(query_vector, top_k=3)
    print(f"  第一次: 耗时={response1.search_time_ms:.2f}ms, 缓存={response1.cache_hit}")
    
    # 第二次 (应该命中缓存)
    response2 = await client.query_memory(query_vector, top_k=3)
    print(f"  第二次: 耗时={response2.search_time_ms:.2f}ms, 缓存={response2.cache_hit}")
    
    print("\n4. 服务统计")
    print("-" * 50)
    
    stats = service.get_stats()
    print(f"  服务状态:")
    print(f"    运行中: {stats['is_running']}")
    print(f"    地址: {stats['host']}:{stats['port']}")
    print(f"  请求统计:")
    print(f"    总请求: {stats['total_requests']}")
    print(f"    成功: {stats['successful_requests']}")
    print(f"    失败: {stats['failed_requests']}")
    print(f"    平均耗时: {stats['avg_search_time_ms']:.2f} ms")
    print(f"  缓存统计:")
    print(f"    大小: {stats['cache']['size']}")
    print(f"    命中率: {stats['cache']['hit_rate']:.1%}")
    print(f"  索引统计:")
    print(f"    Units: {stats['index_size']}")
    
    print("\n5. 停止检索服务")
    print("-" * 50)
    await service.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_retrieval_service())
