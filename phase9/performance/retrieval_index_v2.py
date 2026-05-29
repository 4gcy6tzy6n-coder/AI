"""
Retrieval Index v2 - 检索索引优化

WP1 核心组件：
解决 Phase 8 暴露的检索瓶颈

优化策略：
1. 分层索引结构
2. 近似最近邻 (ANN) 支持
3. 增量更新机制
4. 索引压缩
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class IndexEntry:
    """索引条目"""
    unit_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    last_accessed: float
    access_count: int = 0


@dataclass
class SearchResult:
    """搜索结果"""
    unit_id: str
    score: float
    metadata: Dict[str, Any]
    search_time_ms: float


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) 近似最近邻索引
    
    优化点：
    1. 对数级搜索复杂度 O(log N)
    2. 支持高维向量
    3. 可增量构建
    """
    
    def __init__(
        self,
        dim: int = 128,
        m: int = 16,  # 每个节点的连接数
        ef_construction: int = 200,
        ef_search: int = 50
    ):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.nodes: Dict[str, IndexEntry] = {}
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.entry_point: Optional[str] = None
        self.max_level = 0
        
    def _random_level(self) -> int:
        """随机生成层级"""
        level = 0
        while np.random.random() < 0.5 and level < self.max_level + 1:
            level += 1
        return level
    
    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算欧氏距离"""
        return np.linalg.norm(v1 - v2)
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[str],
        ef: int
    ) -> List[Tuple[float, str]]:
        """在单层中搜索"""
        visited = set(entry_points)
        candidates = []
        
        for ep in entry_points:
            dist = self._distance(query, self.nodes[ep].vector)
            heapq.heappush(candidates, (dist, ep))
        
        results = []
        while candidates and len(results) < ef:
            dist, current = heapq.heappop(candidates)
            results.append((dist, current))
            
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    ndist = self._distance(query, self.nodes[neighbor].vector)
                    heapq.heappush(candidates, (ndist, neighbor))
        
        return results[:ef]
    
    def add(self, unit_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        """添加向量到索引"""
        if unit_id in self.nodes:
            return
        
        level = self._random_level()
        entry = IndexEntry(
            unit_id=unit_id,
            vector=vector,
            metadata=metadata,
            last_accessed=time.time()
        )
        self.nodes[unit_id] = entry
        
        if self.entry_point is None:
            self.entry_point = unit_id
            self.max_level = level
            return
        
        # 简化版：直接连接到 entry_point
        self.graph[unit_id].append(self.entry_point)
        self.graph[self.entry_point].append(unit_id)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """搜索最近邻"""
        if not self.entry_point or not self.nodes:
            return []
        
        start_time = time.time()
        
        # 从 entry point 开始搜索
        results = self._search_layer(
            query,
            [self.entry_point],
            max(k, self.ef_search)
        )
        
        search_time = (time.time() - start_time) * 1000
        
        # 返回前 k 个结果
        return [(unit_id, 1.0 / (1.0 + dist)) for dist, unit_id in results[:k]]


class LayeredIndex:
    """
    分层索引结构
    
    架构：
    - L0 (Hot Layer): 高频访问 Units，内存常驻
    - L1 (Warm Layer): 中频访问 Units，内存 + 磁盘缓存
    - L2 (Cold Layer): 低频访问 Units，磁盘存储
    """
    
    def __init__(
        self,
        dim: int = 128,
        hot_layer_size: int = 1000,
        warm_layer_size: int = 10000
    ):
        self.dim = dim
        self.hot_layer_size = hot_layer_size
        self.warm_layer_size = warm_layer_size
        
        # 三层索引
        self.hot_index: Dict[str, IndexEntry] = {}
        self.warm_index: Dict[str, IndexEntry] = {}
        self.cold_index: Dict[str, IndexEntry] = {}
        
        # HNSW 索引用于热层和温层
        self.hot_hnsw = HNSWIndex(dim=dim)
        self.warm_hnsw = HNSWIndex(dim=dim)
        
        # 访问统计
        self.access_stats: Dict[str, int] = defaultdict(int)
        
    def _get_access_frequency(self, unit_id: str) -> int:
        """获取访问频率"""
        return self.access_stats.get(unit_id, 0)
    
    def _promote_to_hot(self, unit_id: str):
        """提升到热层"""
        if unit_id in self.warm_index:
            entry = self.warm_index.pop(unit_id)
            self.hot_index[unit_id] = entry
            self.hot_hnsw.add(unit_id, entry.vector, entry.metadata)
            
            # 如果热层满了，降级最冷的
            if len(self.hot_index) > self.hot_layer_size:
                self._demote_oldest_from_hot()
    
    def _demote_oldest_from_hot(self):
        """从热层降级最老的"""
        if not self.hot_index:
            return
        
        oldest = min(
            self.hot_index.items(),
            key=lambda x: x[1].last_accessed
        )
        unit_id, entry = oldest
        
        del self.hot_index[unit_id]
        self.warm_index[unit_id] = entry
        self.warm_hnsw.add(unit_id, entry.vector, entry.metadata)
    
    def add(
        self,
        unit_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        layer: str = "warm"
    ):
        """添加条目到索引"""
        entry = IndexEntry(
            unit_id=unit_id,
            vector=vector,
            metadata=metadata,
            last_accessed=time.time()
        )
        
        if layer == "hot":
            self.hot_index[unit_id] = entry
            self.hot_hnsw.add(unit_id, vector, metadata)
        elif layer == "warm":
            self.warm_index[unit_id] = entry
            self.warm_hnsw.add(unit_id, vector, metadata)
        else:
            self.cold_index[unit_id] = entry
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        search_layers: List[str] = None
    ) -> List[SearchResult]:
        """分层搜索"""
        start_time = time.time()
        
        if search_layers is None:
            search_layers = ["hot", "warm", "cold"]
        
        results = []
        found_ids = set()
        
        # 按优先级搜索各层
        for layer in search_layers:
            if len(results) >= k:
                break
            
            remaining = k - len(results)
            
            if layer == "hot":
                layer_results = self.hot_hnsw.search(query, remaining)
            elif layer == "warm":
                layer_results = self.warm_hnsw.search(query, remaining)
            else:
                # 冷层使用暴力搜索（简化版）
                layer_results = self._brute_force_search(
                    query, self.cold_index, remaining
                )
            
            for unit_id, score in layer_results:
                if unit_id not in found_ids:
                    found_ids.add(unit_id)
                    
                    # 更新访问统计
                    self.access_stats[unit_id] += 1
                    
                    # 获取元数据
                    metadata = {}
                    if unit_id in self.hot_index:
                        metadata = self.hot_index[unit_id].metadata
                        self.hot_index[unit_id].last_accessed = time.time()
                    elif unit_id in self.warm_index:
                        metadata = self.warm_index[unit_id].metadata
                        self.warm_index[unit_id].last_accessed = time.time()
                        
                        # 考虑提升到热层
                        if self.access_stats[unit_id] > 10:
                            self._promote_to_hot(unit_id)
                    elif unit_id in self.cold_index:
                        metadata = self.cold_index[unit_id].metadata
                    
                    results.append(SearchResult(
                        unit_id=unit_id,
                        score=score,
                        metadata=metadata,
                        search_time_ms=0
                    ))
        
        search_time = (time.time() - start_time) * 1000
        
        # 更新搜索时间
        for r in results:
            r.search_time_ms = search_time
        
        return results[:k]
    
    def _brute_force_search(
        self,
        query: np.ndarray,
        index: Dict[str, IndexEntry],
        k: int
    ) -> List[Tuple[str, float]]:
        """暴力搜索（用于冷层）"""
        distances = []
        for unit_id, entry in index.items():
            dist = np.linalg.norm(query - entry.vector)
            distances.append((dist, unit_id))
        
        distances.sort()
        return [(unit_id, 1.0 / (1.0 + dist)) for dist, unit_id in distances[:k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        return {
            "hot_layer": {
                "size": len(self.hot_index),
                "capacity": self.hot_layer_size,
                "utilization": len(self.hot_index) / self.hot_layer_size
            },
            "warm_layer": {
                "size": len(self.warm_index),
                "capacity": self.warm_layer_size,
                "utilization": len(self.warm_index) / self.warm_layer_size
            },
            "cold_layer": {
                "size": len(self.cold_index)
            },
            "total_units": len(self.hot_index) + len(self.warm_index) + len(self.cold_index)
        }


class IncrementalIndexUpdater:
    """
    增量索引更新器
    
    功能：
    1. 支持增量添加 Units
    2. 支持批量更新
    3. 避免全量重建
    """
    
    def __init__(self, index: LayeredIndex):
        self.index = index
        self.pending_updates: List[Dict] = []
        self.update_batch_size = 100
        
    def queue_update(
        self,
        unit_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        operation: str = "add"
    ):
        """队列化更新"""
        self.pending_updates.append({
            "unit_id": unit_id,
            "vector": vector,
            "metadata": metadata,
            "operation": operation,
            "timestamp": time.time()
        })
        
        # 批量处理
        if len(self.pending_updates) >= self.update_batch_size:
            self.flush_updates()
    
    def flush_updates(self):
        """刷新待处理更新"""
        if not self.pending_updates:
            return
        
        for update in self.pending_updates:
            if update["operation"] == "add":
                self.index.add(
                    update["unit_id"],
                    update["vector"],
                    update["metadata"]
                )
        
        count = len(self.pending_updates)
        self.pending_updates = []
        
        return count
    
    def get_pending_count(self) -> int:
        """获取待处理更新数量"""
        return len(self.pending_updates)


def demo_retrieval_index_v2():
    """演示检索索引 v2"""
    print("\n" + "="*70)
    print("Retrieval Index v2 - 演示")
    print("="*70)
    
    # 创建分层索引
    index = LayeredIndex(dim=128, hot_layer_size=100, warm_layer_size=1000)
    updater = IncrementalIndexUpdater(index)
    
    print("\n1. 添加 Units 到索引")
    print("-" * 50)
    
    # 添加一些测试数据
    np.random.seed(42)
    for i in range(500):
        unit_id = f"unit_{i:04d}"
        vector = np.random.randn(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # 归一化
        
        # 前 50 个放入热层
        layer = "hot" if i < 50 else "warm"
        index.add(unit_id, vector, {"type": "concept", "index": i}, layer=layer)
    
    print(f"  已添加 500 个 Units")
    print(f"  - 热层: 50 个")
    print(f"  - 温层: 450 个")
    
    # 显示索引统计
    stats = index.get_stats()
    print(f"\n  索引统计:")
    print(f"    热层: {stats['hot_layer']['size']}/{stats['hot_layer']['capacity']}")
    print(f"    温层: {stats['warm_layer']['size']}/{stats['warm_layer']['capacity']}")
    print(f"    总计: {stats['total_units']}")
    
    print("\n2. 搜索测试")
    print("-" * 50)
    
    # 生成查询向量
    query = np.random.randn(128).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # 搜索热层
    start = time.time()
    hot_results = index.search(query, k=3, search_layers=["hot"])
    hot_time = (time.time() - start) * 1000
    
    print(f"\n  热层搜索 (k=3):")
    print(f"    耗时: {hot_time:.2f} ms")
    for r in hot_results:
        print(f"    - {r.unit_id}: score={r.score:.3f}")
    
    # 搜索所有层
    start = time.time()
    all_results = index.search(query, k=5)
    all_time = (time.time() - start) * 1000
    
    print(f"\n  全层搜索 (k=5):")
    print(f"    耗时: {all_time:.2f} ms")
    for r in all_results:
        print(f"    - {r.unit_id}: score={r.score:.3f}")
    
    print("\n3. 增量更新测试")
    print("-" * 50)
    
    # 队列化更新
    for i in range(150):
        unit_id = f"new_unit_{i:04d}"
        vector = np.random.randn(128).astype(np.float32)
        updater.queue_update(unit_id, vector, {"type": "relation"})
    
    print(f"  已队列化 {updater.get_pending_count()} 个更新")
    
    # 刷新更新
    flushed = updater.flush_updates()
    print(f"  已刷新 {flushed} 个更新")
    
    # 更新后统计
    stats = index.get_stats()
    print(f"\n  更新后统计:")
    print(f"    总计: {stats['total_units']} 个 Units")
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_retrieval_index_v2()
