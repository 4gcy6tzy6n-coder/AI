import json
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

from ..unit.models import MemoryZone, Unit


class LongTermStore:
    """长期记忆存储 - 持久化的中期存储"""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_hours: int = 168,
        persistence_path: Optional[str] = None
    ):
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._store: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, datetime] = {}
        self._access_counts: dict[str, int] = {}

        if self.persistence_path:
            self._load_from_disk()

    def write(self, memory_id: str, content: Any, unit: Unit) -> bool:
        """写入长期记忆"""
        # 清理过期数据
        self._cleanup_expired()

        # 如果已满，移除最少访问的
        if len(self._store) >= self.max_size:
            self._evict_lfu()

        self._store[memory_id] = {
            "content": content,
            "unit_id": str(unit.unit_id),
            "zone": MemoryZone.LONG_TERM.value
        }
        self._timestamps[memory_id] = datetime.utcnow()
        self._access_counts[memory_id] = 0

        # 异步持久化
        self._persist_async()

        return True

    def read(self, memory_id: str) -> Optional[dict[str, Any]]:
        """读取长期记忆"""
        # 检查是否过期
        if self._is_expired(memory_id):
            self.delete(memory_id)
            return None

        if memory_id in self._store:
            # 更新访问计数
            self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
            return self._store[memory_id]

        return None

    def delete(self, memory_id: str) -> bool:
        """删除长期记忆"""
        if memory_id in self._store:
            del self._store[memory_id]
            if memory_id in self._timestamps:
                del self._timestamps[memory_id]
            if memory_id in self._access_counts:
                del self._access_counts[memory_id]
            self._persist_async()
            return True
        return False

    def query(self, filter_func: Optional[callable] = None) -> list[dict[str, Any]]:
        """查询长期记忆"""
        self._cleanup_expired()

        results = []
        for memory_id, data in self._store.items():
            if filter_func is None or filter_func(data):
                results.append({"memory_id": memory_id, **data})

        return results

    def _is_expired(self, memory_id: str) -> bool:
        """检查是否过期"""
        if memory_id not in self._timestamps:
            return True

        timestamp = self._timestamps[memory_id]
        return datetime.utcnow() - timestamp > timedelta(hours=self.ttl_hours)

    def _cleanup_expired(self):
        """清理过期数据"""
        expired = [
            memory_id for memory_id in list(self._store.keys())
            if self._is_expired(memory_id)
        ]
        for memory_id in expired:
            self.delete(memory_id)

    def _evict_lfu(self):
        """移除最少访问的数据"""
        if self._access_counts:
            lfu = min(self._access_counts, key=self._access_counts.get)
            self.delete(lfu)

    def _persist_async(self):
        """异步持久化到磁盘"""
        if self.persistence_path:
            # 简化的实现，实际应该使用异步 IO
            self._save_to_disk()

    def _save_to_disk(self):
        """保存到磁盘"""
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "store": self._store,
                "timestamps": {k: v.isoformat() for k, v in self._timestamps.items()},
                "access_counts": self._access_counts
            }
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f)

    def _load_from_disk(self):
        """从磁盘加载"""
        if self.persistence_path and self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                self._store = data.get("store", {})
                self._timestamps = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.get("timestamps", {}).items()
                }
                self._access_counts = data.get("access_counts", {})
            except Exception:
                pass

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计"""
        return {
            "zone": MemoryZone.LONG_TERM.value,
            "size": len(self._store),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_hours
        }
