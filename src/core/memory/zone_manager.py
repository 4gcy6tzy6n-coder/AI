"""
Zone Manager - 长期层分区管理器

第二阶段核心：
- 受审区 (Review Zone): 待审查的 Unit
- 隔离区 (Isolation Zone): 有冲突或问题的 Unit
- 错误区 (Error Zone): 已确认错误但有学习价值的 Unit
- 正常区 (Normal Zone): 通过审查的 Unit
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from ..unit.models import Unit
from .memory_events import MemoryEvent, MemoryEventLog, MemoryEventType, MemoryZone


@dataclass
class ZoneEntry:
    """区域条目"""
    unit: Unit
    entered_at: datetime = field(default_factory=datetime.utcnow)
    review_deadline: Optional[datetime] = None
    review_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": str(self.unit.unit_id),
            "entered_at": self.entered_at.isoformat(),
            "review_deadline": self.review_deadline.isoformat() if self.review_deadline else None,
            "review_count": self.review_count,
            "metadata": self.metadata
        }


class ZoneManager:
    """
    长期层分区管理器

    管理四个区域：
    - long_term_review: 受审区
    - long_term_isolation: 隔离区
    - long_term_error: 错误区
    - long_term_normal: 正常区
    """

    def __init__(
        self,
        review_ttl_days: int = 30,
        isolation_ttl_days: int = 90,
        error_ttl_days: int = 365,
        event_log: Optional[MemoryEventLog] = None
    ):
        self._zones: dict[MemoryZone, dict[UUID, ZoneEntry]] = {
            MemoryZone.LONG_TERM_REVIEW: {},
            MemoryZone.LONG_TERM_ISOLATION: {},
            MemoryZone.LONG_TERM_ERROR: {},
            MemoryZone.LONG_TERM_NORMAL: {}
        }

        self._ttl_config = {
            MemoryZone.LONG_TERM_REVIEW: timedelta(days=review_ttl_days),
            MemoryZone.LONG_TERM_ISOLATION: timedelta(days=isolation_ttl_days),
            MemoryZone.LONG_TERM_ERROR: timedelta(days=error_ttl_days),
            MemoryZone.LONG_TERM_NORMAL: None  # 正常区不过期
        }

        self._event_log = event_log

    def move_to_zone(
        self,
        unit: Unit,
        target_zone: MemoryZone,
        reason: str = "",
        session_id: Optional[str] = None
    ) -> bool:
        """
        将 Unit 移动到指定区域

        返回是否成功
        """
        # 清理过期条目
        self._cleanup_expired()

        # 确定源区域
        source_zone = self._find_unit_zone(unit.unit_id)

        # 从源区域移除
        if source_zone:
            del self._zones[source_zone][unit.unit_id]

        # 创建条目
        entry = ZoneEntry(unit=unit)

        # 设置审查期限
        ttl = self._ttl_config.get(target_zone)
        if ttl:
            entry.review_deadline = datetime.utcnow() + ttl

        # 添加到目标区域
        self._zones[target_zone][unit.unit_id] = entry

        # 记录事件
        if self._event_log:
            event = MemoryEvent(
                event_type=self._get_event_type_for_zone(target_zone),
                unit_id=unit.unit_id,
                session_id=session_id,
                source_zone=source_zone,
                target_zone=target_zone,
                details={"reason": reason}
            )
            self._event_log.log(event)

        return True

    def get_unit_location(self, unit_id: UUID) -> Optional[tuple[MemoryZone, ZoneEntry]]:
        """获取 Unit 所在位置和条目"""
        self._cleanup_expired()

        for zone, entries in self._zones.items():
            if unit_id in entries:
                return zone, entries[unit_id]

        return None

    def get_zone_contents(
        self,
        zone: MemoryZone
    ) -> list[tuple[Unit, ZoneEntry]]:
        """获取区域中的所有 Unit"""
        self._cleanup_expired()

        entries = self._zones.get(zone, {})
        return [(entry.unit, entry) for entry in entries.values()]

    def get_zone_stats(self, zone: MemoryZone) -> dict[str, Any]:
        """获取区域统计信息"""
        self._cleanup_expired()

        entries = self._zones.get(zone, {})

        return {
            "zone": zone.value,
            "unit_count": len(entries),
            "ttl_days": self._ttl_config[zone].days if self._ttl_config[zone] else None
        }

    def increment_review_count(self, unit_id: UUID) -> bool:
        """增加 Unit 的审查次数"""
        location = self.get_unit_location(unit_id)
        if location:
            zone, entry = location
            entry.review_count += 1
            return True
        return False

    def is_in_review_zone(self, unit_id: UUID) -> bool:
        """检查 Unit 是否在受审区"""
        location = self.get_unit_location(unit_id)
        return location is not None and location[0] == MemoryZone.LONG_TERM_REVIEW

    def is_in_isolation_zone(self, unit_id: UUID) -> bool:
        """检查 Unit 是否在隔离区"""
        location = self.get_unit_location(unit_id)
        return location is not None and location[0] == MemoryZone.LONG_TERM_ISOLATION

    def is_in_error_zone(self, unit_id: UUID) -> bool:
        """检查 Unit 是否在错误区"""
        location = self.get_unit_location(unit_id)
        return location is not None and location[0] == MemoryZone.LONG_TERM_ERROR

    def get_all_zones_summary(self) -> dict[str, Any]:
        """获取所有区域的摘要"""
        self._cleanup_expired()

        summary = {}
        for zone in self._zones.keys():
            summary[zone.value] = self.get_zone_stats(zone)

        return summary

    def get_unit_history_summary(self, unit_id: UUID) -> Optional[dict[str, Any]]:
        """获取 Unit 在历史中的摘要"""
        if not self._event_log:
            return None

        events = self._event_log.get_unit_history(unit_id)
        transitions = self._event_log.get_zone_transitions(unit_id)

        return {
            "unit_id": str(unit_id),
            "total_events": len(events),
            "zone_transitions": [
                {
                    "from": src.value if src else None,
                    "to": dst.value if dst else None,
                    "timestamp": ts.isoformat()
                }
                for src, dst, ts in transitions
            ],
            "current_zone": self._event_log.get_current_zone(unit_id).value if self._event_log.get_current_zone(unit_id) else None,
            "access_count": self._event_log.get_access_count(unit_id)
        }

    def _find_unit_zone(self, unit_id: UUID) -> Optional[MemoryZone]:
        """查找 Unit 所在的区域"""
        for zone, entries in self._zones.items():
            if unit_id in entries:
                return zone
        return None

    def _cleanup_expired(self) -> None:
        """清理过期条目"""
        now = datetime.utcnow()

        for zone, entries in self._zones.items():
            expired_ids = [
                uid for uid, entry in entries.items()
                if entry.review_deadline and now > entry.review_deadline
            ]
            for uid in expired_ids:
                del entries[uid]

    def _get_event_type_for_zone(self, zone: MemoryZone) -> MemoryEventType:
        """根据区域获取对应的事件类型"""
        mapping = {
            MemoryZone.LONG_TERM_REVIEW: MemoryEventType.PROMOTION_SUCCESS,
            MemoryZone.LONG_TERM_ISOLATION: MemoryEventType.ISOLATION,
            MemoryZone.LONG_TERM_ERROR: MemoryEventType.ERROR_ARCHIVAL,
            MemoryZone.LONG_TERM_NORMAL: MemoryEventType.REVIEW_COMPLETED
        }
        return mapping.get(zone, MemoryEventType.WRITE_SUCCESS)
