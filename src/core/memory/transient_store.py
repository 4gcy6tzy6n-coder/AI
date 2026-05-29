"""
Transient Store - 瞬时层存储

第二阶段：所有输入必须先进入瞬时层
"""

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID, uuid4

from ..unit.models import Unit
from .memory_events import MemoryEvent, MemoryEventLog, MemoryEventType, MemoryZone


class TransientEntry:
    """瞬时层条目"""

    def __init__(
        self,
        unit: Unit,
        ttl_seconds: int = 86400  # 默认24小时
    ):
        self.unit = unit
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
        self.access_count = 0
        self.last_accessed = self.created_at
        self.content_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """计算内容哈希"""
        content = json.dumps(self.unit.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.utcnow() > self.expires_at

    def access(self) -> None:
        """记录访问"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "unit": self.unit.to_dict(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "content_hash": self.content_hash
        }


class TransientStore:
    """
    瞬时层存储

    第二阶段核心特性：
    1. 所有输入必须先进入瞬时层
    2. 支持TTL自动过期
    3. 记录访问统计
    4. 生成正式事件记录
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 86400,
        event_log: Optional[MemoryEventLog] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._store: OrderedDict[UUID, TransientEntry] = OrderedDict()
        self._event_log = event_log or MemoryEventLog()

    def write(
        self,
        unit: Unit,
        session_id: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        写入瞬时层

        返回: (成功, 原因)
        """
        # 清理过期条目
        self._cleanup_expired()

        # 检查容量
        if len(self._store) >= self.max_size:
            self._evict_oldest()

        # 创建条目
        entry = TransientEntry(unit, self.default_ttl)

        # 写入存储
        self._store[unit.unit_id] = entry
        self._store.move_to_end(unit.unit_id)

        # 记录事件
        event = MemoryEvent(
            event_type=MemoryEventType.WRITE_SUCCESS,
            unit_id=unit.unit_id,
            session_id=session_id,
            target_zone=MemoryZone.TRANSIENT,
            details={
                "content_hash": entry.content_hash,
                "ttl_seconds": self.default_ttl
            }
        )
        self._event_log.log(event)

        return True, "写入瞬时层成功"

    def read(self, unit_id: UUID) -> Optional[Unit]:
        """读取 Unit"""
        entry = self._store.get(unit_id)

        if not entry:
            return None

        if entry.is_expired():
            del self._store[unit_id]
            return None

        # 记录访问
        entry.access()
        self._store.move_to_end(unit_id)

        # 记录访问事件
        event = MemoryEvent(
            event_type=MemoryEventType.ACCESSED,
            unit_id=unit_id,
            details={"access_count": entry.access_count}
        )
        self._event_log.log(event)

        return entry.unit

    def exists(self, unit_id: UUID) -> bool:
        """检查 Unit 是否存在"""
        entry = self._store.get(unit_id)
        if not entry:
            return False
        if entry.is_expired():
            del self._store[unit_id]
            return False
        return True

    def get_stats(self, unit_id: UUID) -> Optional[dict[str, Any]]:
        """获取 Unit 统计信息"""
        entry = self._store.get(unit_id)
        if not entry:
            return None

        return {
            "unit_id": str(unit_id),
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat(),
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed.isoformat(),
            "content_hash": entry.content_hash,
            "is_expired": entry.is_expired()
        }

    def get_all_units(self) -> list[Unit]:
        """获取所有未过期的 Unit"""
        self._cleanup_expired()
        return [entry.unit for entry in self._store.values()]

    def get_entry_count(self) -> int:
        """获取条目数量"""
        self._cleanup_expired()
        return len(self._store)

    def delete(self, unit_id: UUID) -> bool:
        """删除 Unit"""
        if unit_id in self._store:
            del self._store[unit_id]
            return True
        return False

    def get_event_log(self) -> MemoryEventLog:
        """获取事件日志"""
        return self._event_log

    def _cleanup_expired(self) -> None:
        """清理过期条目"""
        expired_ids = [
            uid for uid, entry in self._store.items()
            if entry.is_expired()
        ]
        for uid in expired_ids:
            del self._store[uid]

    def _evict_oldest(self) -> None:
        """驱逐最旧的条目"""
        if self._store:
            oldest_id = next(iter(self._store))
            del self._store[oldest_id]

    def get_store_summary(self) -> dict[str, Any]:
        """获取存储摘要"""
        self._cleanup_expired()

        total_access = sum(e.access_count for e in self._store.values())

        return {
            "zone": "transient",
            "entry_count": len(self._store),
            "max_size": self.max_size,
            "utilization": len(self._store) / self.max_size,
            "total_access_count": total_access,
            "event_log_summary": self._event_log.get_summary()
        }
