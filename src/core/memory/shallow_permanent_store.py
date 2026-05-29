"""
Shallow Permanent Store - 浅层永久存储

第四阶段核心组件：
- 受保护存储（非普通KV存储）
- 记录完整元数据
- 禁止直接删除
- 支持保护回退

保护原则：
- 数量极少、质量极高、修改极慢、回退极严
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Any
import json


class PermanentStatus(Enum):
    """永久对象状态"""
    ACTIVE = "active"                    # 正常活跃
    CONFLICT_MARKED = "conflict_marked"  # 已标记冲突
    UNDER_REVIEW = "under_review"        # 正在复查
    DOWNGRADED = "downgraded"            # 已降级
    SPLIT_PENDING = "split_pending"      # 待拆分
    ARCHIVED = "archived"                # 已归档


class DowngradeTarget(Enum):
    """降级目标"""
    REVIEW = "review"        # 降级到受审区
    ISOLATION = "isolation"  # 降级到隔离区
    ERROR = "error"          # 降级到错误区


@dataclass
class EvidenceSummary:
    """支持证据摘要"""
    primary_sources: list[str] = field(default_factory=list)
    evidence_count: int = 0
    verification_status: str = ""
    last_verified: str = ""


@dataclass
class StabilityWindowSummary:
    """稳定性窗口摘要"""
    total_cycles: int = 0
    stable_cycles: int = 0
    avg_q_score: float = 0.0
    min_q_score: float = 0.0
    max_q_score: float = 0.0
    conflict_count: int = 0


@dataclass
class DowngradeRecord:
    """降级记录"""
    timestamp: str
    from_status: str
    to_target: str
    reason: str
    action_taken: str


@dataclass
class ShallowPermanentEntry:
    """浅层永久条目"""
    # 基础信息
    unit_id: str
    content: str
    core_meaning: str

    # 晋升记录
    promoted_at: str
    promoted_from: str
    promotion_scores: dict[str, float] = field(default_factory=dict)

    # 来源信息
    source_type: str = ""
    source_details: dict = field(default_factory=dict)

    # 元数据
    evidence_summary: EvidenceSummary = field(default_factory=EvidenceSummary)
    stability_summary: StabilityWindowSummary = field(default_factory=StabilityWindowSummary)

    # 状态管理
    status: str = PermanentStatus.ACTIVE.value
    downgrade_history: list[DowngradeRecord] = field(default_factory=list)

    # 时间戳
    last_accessed: str = ""
    last_reviewed: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # 保护标记
    protection_level: str = "high"  # high, critical
    modification_count: int = 0
    conflict_markers: list[dict] = field(default_factory=list)


class ShallowPermanentStore:
    """
    浅层永久存储

    特性：
    1. 受保护存储 - 禁止直接删除
    2. 完整元数据 - 记录进入时间、晋升来源、证据摘要
    3. 回退保护 - 优先修正/拆分/降级，不直接删除
    4. 状态追踪 - 记录回退历史
    """

    def __init__(self):
        self._storage: dict[str, ShallowPermanentEntry] = {}
        self._access_log: list[dict] = []

    def promote_to_permanent(
        self,
        unit_id: str,
        content: str,
        core_meaning: str,
        promotion_scores: dict[str, float],
        source_type: str,
        source_details: dict = None,
        evidence_summary: EvidenceSummary = None,
        stability_summary: StabilityWindowSummary = None
    ) -> ShallowPermanentEntry:
        """
        晋升对象到浅层永久

        注意：只能由长期正常区晋升，禁止跨层直写
        """
        if unit_id in self._storage:
            raise ValueError(f"对象 {unit_id} 已存在于永久层")

        entry = ShallowPermanentEntry(
            unit_id=unit_id,
            content=content,
            core_meaning=core_meaning,
            promoted_at=datetime.utcnow().isoformat(),
            promoted_from="normal",
            promotion_scores=promotion_scores,
            source_type=source_type,
            source_details=source_details or {},
            evidence_summary=evidence_summary or EvidenceSummary(),
            stability_summary=stability_summary or StabilityWindowSummary(),
            status=PermanentStatus.ACTIVE.value
        )

        self._storage[unit_id] = entry
        self._log_access(unit_id, "promote", "对象晋升到浅层永久")

        return entry

    def get(self, unit_id: str) -> Optional[ShallowPermanentEntry]:
        """获取永久对象（只读访问）"""
        entry = self._storage.get(unit_id)
        if entry:
            entry.last_accessed = datetime.utcnow().isoformat()
            self._log_access(unit_id, "read", "访问永久对象")
        return entry

    def mark_conflict(
        self,
        unit_id: str,
        conflict_type: str,
        conflict_details: dict,
        severity: str = "medium"
    ) -> bool:
        """
        标记冲突

        这是保护回退的第一步：先标记冲突，不直接删除
        """
        entry = self._storage.get(unit_id)
        if not entry:
            return False

        conflict_marker = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": conflict_type,
            "severity": severity,
            "details": conflict_details,
            "status": "open"
        }

        entry.conflict_markers.append(conflict_marker)
        entry.status = PermanentStatus.CONFLICT_MARKED.value

        self._log_access(unit_id, "mark_conflict", f"标记冲突: {conflict_type}")

        return True

    def attempt_local_fix(
        self,
        unit_id: str,
        fix_description: str,
        new_content: str = None
    ) -> dict:
        """
        尝试局部修正

        保护回退策略：先尝试修正，再考虑降级
        """
        entry = self._storage.get(unit_id)
        if not entry:
            return {"success": False, "error": "对象不存在"}

        if entry.status != PermanentStatus.CONFLICT_MARKED.value:
            return {"success": False, "error": "对象未标记冲突"}

        # 记录修改
        entry.modification_count += 1

        if new_content:
            # 保存旧版本（简化实现，实际应做版本控制）
            entry.content = new_content

        # 更新冲突标记状态
        for marker in entry.conflict_markers:
            if marker["status"] == "open":
                marker["status"] = "resolved"
                marker["resolution"] = f"局部修正: {fix_description}"

        entry.status = PermanentStatus.ACTIVE.value

        self._log_access(unit_id, "local_fix", f"局部修正: {fix_description}")

        return {
            "success": True,
            "modification_count": entry.modification_count,
            "status": entry.status
        }

    def initiate_split(
        self,
        unit_id: str,
        split_reason: str,
        split_plan: list[dict]
    ) -> dict:
        """
        发起拆分

        保护回退策略：对于多义混装或边界不稳的对象，拆分为子对象
        """
        entry = self._storage.get(unit_id)
        if not entry:
            return {"success": False, "error": "对象不存在"}

        entry.status = PermanentStatus.SPLIT_PENDING.value

        # 记录拆分计划
        split_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": split_reason,
            "plan": split_plan,
            "original_unit_id": unit_id
        }

        self._log_access(unit_id, "initiate_split", f"发起拆分: {split_reason}")

        return {
            "success": True,
            "split_record": split_record,
            "status": entry.status,
            "note": "原对象退出永久主位，子对象需重新进入长期治理链"
        }

    def downgrade(
        self,
        unit_id: str,
        target: DowngradeTarget,
        reason: str,
        action_taken: str = ""
    ) -> dict:
        """
        降级对象

        保护回退策略：降级到长期受审区或隔离区，不直接删除
        """
        entry = self._storage.get(unit_id)
        if not entry:
            return {"success": False, "error": "对象不存在"}

        # 创建降级记录
        downgrade_record = DowngradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            from_status=entry.status,
            to_target=target.value,
            reason=reason,
            action_taken=action_taken or f"降级到{target.value}"
        )

        entry.downgrade_history.append(downgrade_record)
        entry.status = PermanentStatus.DOWNGRADED.value

        self._log_access(unit_id, "downgrade", f"降级到{target.value}: {reason}")

        return {
            "success": True,
            "downgrade_record": asdict(downgrade_record),
            "total_downgrades": len(entry.downgrade_history)
        }

    def get_downgrade_history(self, unit_id: str) -> list[dict]:
        """获取对象的降级历史"""
        entry = self._storage.get(unit_id)
        if not entry:
            return []
        return [asdict(r) for r in entry.downgrade_history]

    def get_conflict_markers(self, unit_id: str) -> list[dict]:
        """获取对象的冲突标记"""
        entry = self._storage.get(unit_id)
        if not entry:
            return []
        return entry.conflict_markers

    def list_all(self, status_filter: str = None) -> list[ShallowPermanentEntry]:
        """列出所有永久对象"""
        entries = list(self._storage.values())
        if status_filter:
            entries = [e for e in entries if e.status == status_filter]
        return entries

    def get_statistics(self) -> dict:
        """获取存储统计"""
        total = len(self._storage)
        status_counts = {}
        for entry in self._storage.values():
            status = entry.status
            status_counts[status] = status_counts.get(status, 0) + 1

        total_downgrades = sum(
            len(e.downgrade_history) for e in self._storage.values()
        )

        return {
            "total_entries": total,
            "status_distribution": status_counts,
            "total_downgrades": total_downgrades,
            "total_access_logs": len(self._access_log)
        }

    def _log_access(self, unit_id: str, action: str, details: str):
        """记录访问日志"""
        self._access_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "unit_id": unit_id,
            "action": action,
            "details": details
        })

    def export_entry(self, unit_id: str) -> dict:
        """导出条目（用于备份或迁移）"""
        entry = self._storage.get(unit_id)
        if not entry:
            return {}
        return asdict(entry)

    def is_in_permanent(self, unit_id: str) -> bool:
        """检查对象是否在永久层"""
        return unit_id in self._storage

    def get_protection_summary(self, unit_id: str) -> dict:
        """获取保护摘要"""
        entry = self._storage.get(unit_id)
        if not entry:
            return {}

        return {
            "unit_id": entry.unit_id,
            "status": entry.status,
            "protection_level": entry.protection_level,
            "promoted_at": entry.promoted_at,
            "modification_count": entry.modification_count,
            "conflict_count": len(entry.conflict_markers),
            "downgrade_count": len(entry.downgrade_history),
            "can_delete": False,  # 永久层禁止直接删除
            "can_modify": entry.status == PermanentStatus.CONFLICT_MARKED.value
        }
