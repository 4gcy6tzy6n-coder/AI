from typing import Any

from .models import ThinkingUnit, Unit, UnitType


class UnitSplitter:
    """将复杂 Unit 拆分为子 Unit"""

    def split(self, unit: Unit, strategy: str = "auto") -> list[Unit]:
        """根据策略拆分 Unit"""
        if strategy == "auto":
            strategy = self._detect_strategy(unit)

        if strategy == "by_component":
            return self._split_by_component(unit)
        elif strategy == "by_step":
            return self._split_by_step(unit)
        elif strategy == "none":
            return [unit]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _detect_strategy(self, unit: Unit) -> str:
        """自动检测拆分策略"""
        if unit.unit_type == UnitType.THINKING:
            thinking_unit = unit
            if isinstance(thinking_unit, ThinkingUnit) and len(thinking_unit.steps) > 1:
                return "by_step"
        return "none"

    def _split_by_component(self, unit: Unit) -> list[Unit]:
        """按组件拆分"""
        # 默认不拆分
        return [unit]

    def _split_by_step(self, unit: ThinkingUnit) -> list[Unit]:
        """按推理步骤拆分"""
        sub_units = []

        for i, step in enumerate(unit.steps):
            sub_unit = ThinkingUnit(
                parent_id=unit.unit_id,
                session_id=unit.session_id,
                reasoning_type=unit.reasoning_type,
                steps=[step],
                content={"step_index": i, "total_steps": len(unit.steps)},
                priority=unit.priority
            )
            sub_units.append(sub_unit)

        return sub_units

    def estimate_complexity(self, unit: Unit) -> int:
        """估算 Unit 复杂度 (1-10)"""
        complexity = 1

        # 基于 content 大小
        content_size = len(str(unit.content))
        if content_size > 10000:
            complexity += 3
        elif content_size > 1000:
            complexity += 2
        elif content_size > 100:
            complexity += 1

        # 基于类型
        if unit.unit_type == UnitType.THINKING:
            complexity += 2

        return min(complexity, 10)
