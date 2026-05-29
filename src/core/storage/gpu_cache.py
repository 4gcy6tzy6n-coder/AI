from typing import Any, Optional


class GPUCache:
    """GPU 缓存管理器"""

    def __init__(self, max_size_gb: float = 16.0):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self._cache: dict[str, Any] = {}
        self._size_bytes = 0

    def allocate(self, size_bytes: int) -> bool:
        """分配 GPU 内存"""
        if self._size_bytes + size_bytes > self.max_size_bytes:
            # 尝试驱逐
            if not self._evict(size_bytes):
                return False

        self._size_bytes += size_bytes
        return True

    def load_tensor(self, tensor_id: str, tensor: Any) -> bool:
        """加载张量到 GPU"""
        # 估算张量大小
        tensor_size = self._estimate_tensor_size(tensor)

        if not self.allocate(tensor_size):
            return False

        self._cache[tensor_id] = {
            "tensor": tensor,
            "size": tensor_size
        }
        return True

    def get_tensor(self, tensor_id: str) -> Optional[Any]:
        """获取张量"""
        entry = self._cache.get(tensor_id)
        if entry:
            return entry["tensor"]
        return None

    def evict(self, tensor_id: str) -> bool:
        """驱逐特定张量"""
        if tensor_id in self._cache:
            size = self._cache[tensor_id]["size"]
            del self._cache[tensor_id]
            self._size_bytes -= size
            return True
        return False

    def _evict(self, required_bytes: int) -> bool:
        """驱逐足够的空间"""
        # 简单的 LRU 驱逐
        freed = 0
        to_evict = []

        for tensor_id, entry in self._cache.items():
            to_evict.append(tensor_id)
            freed += entry["size"]
            if freed >= required_bytes:
                break

        for tensor_id in to_evict:
            self.evict(tensor_id)

        return freed >= required_bytes

    def _estimate_tensor_size(self, tensor: Any) -> int:
        """估算张量大小"""
        # 简化的估算
        # 实际应该根据张量形状和数据类型计算
        return 1024 * 1024  # 1MB 默认值

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计"""
        return {
            "tier": "gpu",
            "size_bytes": self._size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "utilization": self._size_bytes / self.max_size_bytes,
            "item_count": len(self._cache)
        }
