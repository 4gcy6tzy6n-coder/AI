from collections import OrderedDict
from typing import Any, Optional


class RAMCache:
    """RAM 缓存管理器"""

    def __init__(self, max_size_gb: float = 64.0):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._size_bytes = 0
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """获取数据"""
        if key in self._cache:
            # 移动到末尾 (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]["data"]

        self._misses += 1
        return None

    def put(self, key: str, value: Any, ttl: int = 0) -> bool:
        """存储数据"""
        value_size = self._estimate_size(value)

        # 检查是否有足够空间
        if self._size_bytes + value_size > self.max_size_bytes:
            if not self._evict(value_size):
                return False

        self._cache[key] = {
            "data": value,
            "size": value_size,
            "ttl": ttl
        }
        self._size_bytes += value_size
        return True

    def invalidate(self, key: str) -> bool:
        """使缓存失效"""
        if key in self._cache:
            size = self._cache[key]["size"]
            del self._cache[key]
            self._size_bytes -= size
            return True
        return False

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._size_bytes = 0

    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def _evict(self, required_bytes: int) -> bool:
        """驱逐足够的空间"""
        freed = 0
        to_evict = []

        for key, entry in self._cache.items():
            to_evict.append(key)
            freed += entry["size"]
            if freed >= required_bytes:
                break

        for key in to_evict:
            self.invalidate(key)

        return freed >= required_bytes

    def _estimate_size(self, value: Any) -> int:
        """估算数据大小"""
        import sys
        return sys.getsizeof(value)

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计"""
        return {
            "tier": "ram",
            "size_bytes": self._size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "utilization": self._size_bytes / self.max_size_bytes,
            "item_count": len(self._cache),
            "hit_rate": self.get_hit_rate()
        }
