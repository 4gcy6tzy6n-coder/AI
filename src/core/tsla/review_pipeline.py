from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class ReviewType(str, Enum):
    AUTO = "auto"
    HUMAN = "human"
    HYBRID = "hybrid"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


class ReviewTicket:
    """审查工单"""

    def __init__(
        self,
        unit_id: UUID,
        review_type: ReviewType,
        priority: int = 5,
        deadline: Optional[datetime] = None
    ):
        self.ticket_id = uuid4()
        self.unit_id = unit_id
        self.review_type = review_type
        self.priority = priority
        self.deadline = deadline or datetime.utcnow() + timedelta(hours=24)
        self.status = ReviewStatus.PENDING
        self.created_at = datetime.utcnow()
        self.assigned_to: Optional[str] = None
        self.result: Optional[ReviewResult] = None


class ReviewResult:
    """审查结果"""

    def __init__(
        self,
        approved: bool,
        reviewer: str,
        comments: str = "",
        metadata: Optional[dict[str, Any]] = None
    ):
        self.approved = approved
        self.reviewer = reviewer
        self.comments = comments
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class ReviewPipeline:
    """审查流水线"""

    def __init__(self, auto_review_enabled: bool = True):
        self.auto_review_enabled = auto_review_enabled
        self._tickets: dict[UUID, ReviewTicket] = {}
        self._queue: list[UUID] = []

    def submit(
        self,
        unit_id: UUID,
        review_type: ReviewType = ReviewType.AUTO,
        priority: int = 5
    ) -> ReviewTicket:
        """提交审查"""
        ticket = ReviewTicket(
            unit_id=unit_id,
            review_type=review_type,
            priority=priority
        )
        self._tickets[ticket.ticket_id] = ticket
        self._queue.append(ticket.ticket_id)

        # 自动处理自动审查
        if review_type == ReviewType.AUTO and self.auto_review_enabled:
            self._process_auto_review(ticket)

        return ticket

    def get_result(self, ticket_id: UUID) -> Optional[ReviewResult]:
        """获取审查结果"""
        ticket = self._tickets.get(ticket_id)
        if ticket:
            return ticket.result
        return None

    def _process_auto_review(self, ticket: ReviewTicket) -> None:
        """处理自动审查"""
        # 简化的自动审查逻辑
        # 实际应该使用规则引擎或 ML 模型
        ticket.status = ReviewStatus.IN_PROGRESS

        # 模拟审查结果
        result = ReviewResult(
            approved=True,
            reviewer="auto_reviewer",
            comments="Auto-approved based on rule engine"
        )

        ticket.result = result
        ticket.status = ReviewStatus.COMPLETED

    def assign_to_human(
        self,
        ticket_id: UUID,
        reviewer: str
    ) -> bool:
        """分配给人工审查"""
        ticket = self._tickets.get(ticket_id)
        if ticket and ticket.status == ReviewStatus.PENDING:
            ticket.assigned_to = reviewer
            ticket.status = ReviewStatus.IN_PROGRESS
            return True
        return False

    def submit_human_review(
        self,
        ticket_id: UUID,
        result: ReviewResult
    ) -> bool:
        """提交人工审查结果"""
        ticket = self._tickets.get(ticket_id)
        if ticket and ticket.status == ReviewStatus.IN_PROGRESS:
            ticket.result = result
            ticket.status = ReviewStatus.COMPLETED
            return True
        return False

    def get_pending_tickets(self) -> list[ReviewTicket]:
        """获取待处理工单"""
        return [
            self._tickets[tid]
            for tid in self._queue
            if self._tickets[tid].status == ReviewStatus.PENDING
        ]

    def cleanup_expired(self) -> None:
        """清理过期工单"""
        now = datetime.utcnow()
        for ticket in self._tickets.values():
            if ticket.status == ReviewStatus.PENDING and ticket.deadline < now:
                ticket.status = ReviewStatus.EXPIRED
