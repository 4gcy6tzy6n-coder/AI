from typing import Any, list

from .models import Unit, UnitStatus


class ValidationError(Exception):
    """验证错误"""
    pass


class UnitValidator:
    """验证 Unit 的完整性和有效性"""

    def validate(self, unit: Unit) -> list[str]:
        """验证 Unit，返回错误列表"""
        errors = []

        # 检查必填字段
        if not unit.unit_id:
            errors.append("unit_id is required")

        if not unit.unit_type:
            errors.append("unit_type is required")

        # 检查 content
        if unit.content is None:
            errors.append("content cannot be None")

        # 检查 priority
        if not 1 <= unit.priority <= 10:
            errors.append("priority must be between 1 and 10")

        # 检查 ttl
        if unit.ttl is not None and unit.ttl <= 0:
            errors.append("ttl must be positive")

        return errors

    def validate_strict(self, unit: Unit) -> None:
        """严格验证，失败时抛出异常"""
        errors = self.validate(unit)
        if errors:
            raise ValidationError(f"Unit validation failed: {', '.join(errors)}")

    def is_valid(self, unit: Unit) -> bool:
        """检查 Unit 是否有效"""
        return len(self.validate(unit)) == 0

    def validate_content(self, unit: Unit, required_fields: list[str]) -> list[str]:
        """验证 content 中是否包含必需字段"""
        errors = []
        content = unit.content or {}

        for field in required_fields:
            if field not in content:
                errors.append(f"content.{field} is required")

        return errors
