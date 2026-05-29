import json
import random
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4


class TrainingCase:
    """训练案例"""

    def __init__(
        self,
        case_id: str,
        input_data: dict[str, Any],
        expected_output: dict[str, Any],
        expected_reasoning: list[dict[str, Any]],
        difficulty: int,
        tags: list[str],
        category: str,
        source: str
    ):
        self.case_id = case_id
        self.input_data = input_data
        self.expected_output = expected_output
        self.expected_reasoning = expected_reasoning
        self.difficulty = difficulty
        self.tags = tags
        self.category = category
        self.source = source
        self.created_at = datetime.utcnow()
        self.version = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "case_id": self.case_id,
            "input": self.input_data,
            "expected_output": self.expected_output,
            "expected_reasoning": self.expected_reasoning,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "category": self.category,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }


class CaseBuilder:
    """训练案例构建器"""

    def __init__(self, output_dir: str = "./data/goldens"):
        self.output_dir = output_dir
        self._cases: list[TrainingCase] = []

    def build_from_raw(
        self,
        raw_data: list[dict[str, Any]],
        template: dict[str, Any]
    ) -> list[TrainingCase]:
        """从原始数据构建案例"""
        cases = []

        for data in raw_data:
            case = self._create_case(data, template)
            cases.append(case)

        self._cases.extend(cases)
        return cases

    def build_synthetic(
        self,
        count: int,
        difficulty_range: tuple[int, int] = (1, 10),
        categories: Optional[list[str]] = None
    ) -> list[TrainingCase]:
        """构建合成案例"""
        cases = []
        categories = categories or ["general"]

        for i in range(count):
            case = TrainingCase(
                case_id=str(uuid4()),
                input_data={
                    "content": f"Synthetic input {i}",
                    "content_type": "text"
                },
                expected_output={
                    "content": f"Synthetic output {i}",
                    "content_type": "text"
                },
                expected_reasoning=[
                    {
                        "step_id": "step_1",
                        "description": "Analyze input",
                        "step_type": "analysis"
                    }
                ],
                difficulty=random.randint(*difficulty_range),
                tags=["synthetic"],
                category=random.choice(categories),
                source="synthetic_generator"
            )
            cases.append(case)

        self._cases.extend(cases)
        return cases

    def validate_case(self, case: TrainingCase) -> list[str]:
        """验证案例"""
        errors = []

        if not case.case_id:
            errors.append("case_id is required")

        if not case.input_data:
            errors.append("input_data is required")

        if not case.expected_output:
            errors.append("expected_output is required")

        if not 1 <= case.difficulty <= 10:
            errors.append("difficulty must be between 1 and 10")

        return errors

    def save_cases(self, filename: str = "training_cases.json") -> None:
        """保存案例到文件"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        filepath = os.path.join(self.output_dir, filename)
        data = [case.to_dict() for case in self._cases]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _create_case(
        self,
        data: dict[str, Any],
        template: dict[str, Any]
    ) -> TrainingCase:
        """根据模板创建案例"""
        return TrainingCase(
            case_id=str(uuid4()),
            input_data=data.get("input", {}),
            expected_output=data.get("output", {}),
            expected_reasoning=data.get("reasoning", []),
            difficulty=data.get("difficulty", 5),
            tags=data.get("tags", []),
            category=data.get("category", "general"),
            source=data.get("source", "raw_data")
        )
