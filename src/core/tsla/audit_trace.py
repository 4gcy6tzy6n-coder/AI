import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4


class AuditRecord:
    """审计记录"""

    def __init__(
        self,
        unit_id: UUID,
        scores: dict[str, Any],
        action: dict[str, Any],
        context_hash: str
    ):
        self.record_id = uuid4()
        self.timestamp = datetime.utcnow()
        self.unit_id = unit_id
        self.scores = scores
        self.action = action
        self.context_hash = context_hash
        self.signature = self._generate_signature()

    def _generate_signature(self) -> str:
        """生成防篡改签名"""
        data = {
            "record_id": str(self.record_id),
            "timestamp": self.timestamp.isoformat(),
            "unit_id": str(self.unit_id),
            "scores": self.scores,
            "action": self.action,
            "context_hash": self.context_hash
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify(self) -> bool:
        """验证记录完整性"""
        return self.signature == self._generate_signature()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "record_id": str(self.record_id),
            "timestamp": self.timestamp.isoformat(),
            "unit_id": str(self.unit_id),
            "scores": self.scores,
            "action": self.action,
            "context_hash": self.context_hash,
            "signature": self.signature
        }


class AuditTrace:
    """审计追踪"""

    def __init__(self, storage_path: Optional[str] = None):
        self._records: list[AuditRecord] = []
        self.storage_path = storage_path

    def record(
        self,
        unit_id: UUID,
        scores: dict[str, Any],
        action: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> AuditRecord:
        """记录审计日志"""
        # 计算上下文哈希
        context_hash = self._hash_context(context or {})

        record = AuditRecord(
            unit_id=unit_id,
            scores=scores,
            action=action,
            context_hash=context_hash
        )

        self._records.append(record)

        # 持久化
        if self.storage_path:
            self._persist_record(record)

        return record

    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        unit_id: Optional[UUID] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100
    ) -> list[AuditRecord]:
        """查询审计记录"""
        results = self._records

        if start_time:
            results = [r for r in results if r.timestamp >= start_time]

        if end_time:
            results = [r for r in results if r.timestamp <= end_time]

        if unit_id:
            results = [r for r in results if r.unit_id == unit_id]

        if filters:
            # 应用额外过滤器
            pass

        return results[-limit:]

    def export(self, format: str = "json") -> str:
        """导出审计日志"""
        if format == "json":
            records = [r.to_dict() for r in self._records]
            return json.dumps(records, indent=2)
        elif format == "csv":
            # CSV 格式导出
            lines = ["record_id,timestamp,unit_id,action_type"]
            for r in self._records:
                action_type = r.action.get("action_type", "unknown")
                lines.append(f"{r.record_id},{r.timestamp.isoformat()},{r.unit_id},{action_type}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _hash_context(self, context: dict[str, Any]) -> str:
        """哈希上下文"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()

    def _persist_record(self, record: AuditRecord) -> None:
        """持久化记录"""
        # 简化的实现，实际应该写入数据库或文件
        pass

    def verify_all(self) -> list[AuditRecord]:
        """验证所有记录"""
        corrupted = []
        for record in self._records:
            if not record.verify():
                corrupted.append(record)
        return corrupted
