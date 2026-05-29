"""
Memory Events - 记忆事件系统

第二阶段：每次输入后都留下正式事件记录
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional
from uuid import UUID, uuid4


class MemoryEventType(Enum):
    """记忆事件类型"""
    # 写入事件
    WRITE_ATTEMPT = "write_attempt"
    WRITE_SUCCESS = "write_success"
    WRITE_REJECTED = "write_rejected"

    # 迁移事件
    PROMOTION_ATTEMPT = "promotion_attempt"
    PROMOTION_SUCCESS = "promotion_success"
    PROMOTION_REJECTED = "promotion_rejected"

    # 治理事件
    ISOLATION = "isolation"
    ERROR_ARCHIVAL = "error_archival"
    ROLLBACK = "rollback"
    SPLIT = "split"

    # 审查事件
    REVIEW_ASSIGNED = "review_assigned"
    REVIEW_COMPLETED = "review_completed"
    BACKFLOW_TRIGGERED = "backflow_triggered"

    # 访问事件
    ACCESSED = "accessed"
    ACCESSED_MULTI = "accessed_multi"  # 多次访问


class MemoryZone(Enum):
    """记忆区域"""
    TRANSIENT = "transient"
    LONG_TERM_REVIEW = "long_term_review"
    LONG_TERM_ISOLATION = "long_term_isolation"
    LONG_TERM_ERROR = "long_term_error"
    LONG_TERM_NORMAL = "long_term_normal"
    SHALLOW_PERMANENT = "shallow_permanent"
    DEEP_PERMANENT = "deep_permanent"


@dataclass
class MemoryEvent:
    """记忆事件"""
    event_id: UUID = field(default_factory=uuid4)
    event_type: MemoryEventType = MemoryEventType.WRITE_ATTEMPT
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # 关联对象
    unit_id: Optional[UUID] = None
    session_id: Optional[str] = None

    # 位置信息
    source_zone: Optional[MemoryZone] = None
    target_zone: Optional[MemoryZone] = None

    # 事件详情
    details: dict[str, Any] = field(default_factory=dict)

    # TSLA 相关
    tsla_action: Optional[str] = None
    tsla_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "unit_id": str(self.unit_id) if self.unit_id else None,
            "session_id": self.session_id,
            "source_zone": self.source_zone.value if self.source_zone else None,
            "target_zone": self.target_zone.value if self.target_zone else None,
            "details": self.details,
            "tsla_action": self.tsla_action,
            "tsla_scores": self.tsla_scores
        }


class MemoryEventLog:
    """记忆事件日志"""

    def __init__(self, max_events: int = 10000):
        self._events: list[MemoryEvent] = []
        self._unit_events: dict[UUID, list[UUID]] = {}  # unit_id -> event_ids
        self._max_events = max_events

    def log(self, event: MemoryEvent) -> None:
        """记录事件"""
        self._events.append(event)

        # 索引
        if event.unit_id:
            if event.unit_id not in self._unit_events:
                self._unit_events[event.unit_id] = []
            self._unit_events[event.unit_id].append(event.event_id)

        # 清理旧事件
        if len(self._events) > self._max_events:
            self._cleanup_old_events()

    def get_unit_history(self, unit_id: UUID) -> list[MemoryEvent]:
        """获取 Unit 的事件历史"""
        event_ids = self._unit_events.get(unit_id, [])
        return [
            e for e in self._events
            if e.event_id in event_ids
        ]

    def get_events_by_type(
        self,
        event_type: MemoryEventType,
        limit: int = 100
    ) -> list[MemoryEvent]:
        """按类型获取事件"""
        events = [
            e for e in self._events
            if e.event_type == event_type
        ]
        return events[-limit:]

    def get_zone_transitions(
        self,
        unit_id: UUID
    ) -> list[tuple[MemoryZone, MemoryZone, datetime]]:
        """获取 Unit 的区域迁移历史"""
        events = self.get_unit_history(unit_id)
        transitions = []

        for event in events:
            if event.source_zone and event.target_zone:
                transitions.append(
                    (event.source_zone, event.target_zone, event.timestamp)
                )

        return transitions

    def get_current_zone(self, unit_id: UUID) -> Optional[MemoryZone]:
        """获取 Unit 当前所在区域"""
        events = self.get_unit_history(unit_id)
        if not events:
            return None

        # 找到最后一个有目标区域的事件
        for event in reversed(events):
            if event.target_zone:
                return event.target_zone

        return None

    def get_access_count(self, unit_id: UUID) -> int:
        """获取 Unit 被访问次数"""
        events = self.get_unit_history(unit_id)
        return sum(
            1 for e in events
            if e.event_type in [MemoryEventType.ACCESSED, MemoryEventType.ACCESSED_MULTI]
        )

    def export(self, format: str = "json") -> str:
        """导出事件日志"""
        import json

        if format == "json":
            events_data = [e.to_dict() for e in self._events]
            return json.dumps(events_data, indent=2, ensure_ascii=False)

        return ""

    def _cleanup_old_events(self) -> None:
        """清理旧事件"""
        # 保留最近的事件
        cutoff = len(self._events) - self._max_events
        removed_events = self._events[:cutoff]
        self._events = self._events[cutoff:]

        # 更新索引
        removed_ids = {e.event_id for e in removed_events}
        for unit_id in list(self._unit_events.keys()):
            self._unit_events[unit_id] = [
                eid for eid in self._unit_events[unit_id]
                if eid not in removed_ids
            ]
            if not self._unit_events[unit_id]:
                del self._unit_events[unit_id]

    def get_summary(self) -> dict[str, Any]:
        """获取事件日志摘要"""
        from collections import Counter

        event_type_counts = Counter(e.event_type.value for e in self._events)

        return {
            "total_events": len(self._events),
            "unique_units": len(self._unit_events),
            "event_type_distribution": dict(event_type_counts),
            "zone_distribution": self._get_zone_distribution()
        }

    def _get_zone_distribution(self) -> dict[str, int]:
        """获取区域分布"""
        from collections import Counter

        zones = []
        for event in self._events:
            if event.target_zone:
                zones.append(event.target_zone.value)

        return dict(Counter(zones))
