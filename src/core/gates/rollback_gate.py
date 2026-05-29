from datetime import datetime
from typing import Any, Optional


class RollbackDecision:
    """回滚决策"""

    def __init__(
        self,
        rollback: bool,
        scope: str = "unit",
        replacement: Optional[dict[str, Any]] = None,
        audit_level: str = "normal",
        reason: str = ""
    ):
        self.rollback = rollback
        self.scope = scope  # unit, batch, cascade
        self.replacement = replacement
        self.audit_level = audit_level
        self.reason = reason


class RollbackGate:
    """回滚门 - 处理错误或有害记忆"""

    def __init__(
        self,
        auto_rollback_on_error: bool = True,
        user_feedback_threshold: float = 0.3
    ):
        self.auto_rollback_on_error = auto_rollback_on_error
        self.user_feedback_threshold = user_feedback_threshold
        self._error_log: list[dict[str, Any]] = []

    def evaluate(
        self,
        memory_id: str,
        error_detected: bool = False,
        harmful_detected: bool = False,
        user_feedback: Optional[float] = None
    ) -> RollbackDecision:
        """评估是否需要回滚"""
        # 错误检测
        if error_detected and self.auto_rollback_on_error:
            return RollbackDecision(
                rollback=True,
                scope="unit",
                audit_level="high",
                reason="Error detected in memory content"
            )

        # 有害内容检测
        if harmful_detected:
            return RollbackDecision(
                rollback=True,
                scope="cascade",
                audit_level="high",
                reason="Harmful content detected"
            )

        # 用户反馈
        if user_feedback is not None and user_feedback < self.user_feedback_threshold:
            return RollbackDecision(
                rollback=True,
                scope="unit",
                audit_level="normal",
                reason=f"User feedback below threshold: {user_feedback}"
            )

        return RollbackDecision(
            rollback=False,
            reason="No rollback conditions met"
        )

    def record_error(
        self,
        memory_id: str,
        error_type: str,
        error_message: str
    ) -> None:
        """记录错误"""
        self._error_log.append({
            "memory_id": memory_id,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.utcnow()
        })

    def get_error_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取错误历史"""
        return [
            entry for entry in self._error_log
            if entry["memory_id"] == memory_id
        ]

    def should_alert(self, memory_id: str) -> bool:
        """是否应该发出警报"""
        errors = self.get_error_history(memory_id)
        # 如果短时间内多次错误，发出警报
        if len(errors) >= 3:
            return True
        return False
