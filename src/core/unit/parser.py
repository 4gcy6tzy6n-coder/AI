import re
from typing import Any, Optional

from .models import SourceType, Unit, UnitType


class ParseResult:
    """解析结果"""

    def __init__(
        self,
        units: list[Unit],
        has_information_gap: bool = False,
        gap_reason: str = ""
    ):
        self.units = units
        self.has_information_gap = has_information_gap
        self.gap_reason = gap_reason

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "units": [u.to_dict() for u in self.units],
            "has_information_gap": self.has_information_gap,
            "gap_reason": self.gap_reason
        }


class UnitParser:
    """
    中文输入解析器

    第一阶段：支持中文短句/短问题解析
    """

    def __init__(self):
        # 预定义的知识片段（模拟内部知识库）
        self._knowledge_base = {
            "苹果": {
                "core_meaning": "一种常见的水果，味甜多汁",
                "phonetic_form": "píng guǒ"
            },
            "记忆层": {
                "core_meaning": "系统中用于存储和管理信息的层次结构",
                "phonetic_form": "jì yì céng"
            },
            "永久层": {
                "core_meaning": "记忆系统中最高层级，存储经过验证的核心知识",
                "phonetic_form": "yǒng jiǔ céng"
            },
            "瞬态层": {
                "core_meaning": "记忆系统中最临时层级，存储当前会话上下文",
                "phonetic_form": "shùn tài céng"
            },
            "TSLA": {
                "core_meaning": "Trustworthiness, Safety, Liability, Accountability 评估框架",
                "phonetic_form": "T-S-L-A"
            },
            "Unit": {
                "core_meaning": "系统中的最小治理单元，包含语义三态和TSLA分数",
                "phonetic_form": "Yōu nì tè"
            }
        }

        # 疑问词列表
        self._question_words = ["什么", "怎么", "为什么", "多少", "哪里", "谁", "怎样"]

    def parse(self, text: str, session_id: Optional[str] = None) -> ParseResult:
        """
        解析中文输入

        返回：ParseResult 包含 Unit 候选列表和信息缺口判断
        """
        # 1. 创建主输入 Unit
        main_unit = Unit.from_input(text, session_id)
        main_unit.unit_type = UnitType.INPUT

        # 2. 提取关键对象（简化版：基于知识库匹配）
        key_objects = self._extract_key_objects(text)

        # 3. 为每个关键对象生成 Unit 候选
        candidate_units = [main_unit]
        has_gap = False
        gap_reasons = []

        for obj in key_objects:
            unit = self._create_unit_for_object(obj, text)
            candidate_units.append(unit)

            # 检查是否有语义定义
            if not unit.has_meaning():
                has_gap = True
                gap_reasons.append(f"对象 '{obj}' 缺少基本语义定义")

        # 4. 判断是否是问题（是否需要更多信息）
        is_question = self._is_question(text)
        if is_question and not key_objects:
            has_gap = True
            gap_reasons.append("问题中未识别出已知对象")

        # 5. 检查输入完整性
        if len(text) < 3:
            has_gap = True
            gap_reasons.append("输入过短，信息不足")

        return ParseResult(
            units=candidate_units,
            has_information_gap=has_gap,
            gap_reason="; ".join(gap_reasons) if gap_reasons else ""
        )

    def _extract_key_objects(self, text: str) -> list[str]:
        """提取关键对象（简化版）"""
        objects = []

        # 在知识库中查找匹配
        for keyword in self._knowledge_base.keys():
            if keyword in text:
                objects.append(keyword)

        # 简单的名词提取（基于常见模式）
        # 匹配 "X是什么" 中的 X
        match = re.search(r'(.+?)(?:是什么|有什么|怎么|如何)', text)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate not in objects:
                objects.append(candidate)

        return objects

    def _create_unit_for_object(self, obj: str, context: str) -> Unit:
        """为对象创建 Unit"""
        # 查找知识库
        knowledge = self._knowledge_base.get(obj, {})

        unit = Unit(
            unit_type=UnitType.THINKING,
            script_form=obj,
            phonetic_form=knowledge.get("phonetic_form", ""),
            core_meaning=knowledge.get("core_meaning", ""),
            source_type=SourceType.INFERRED,
            content={"object": obj, "context": context}
        )

        # 计算 legality_score：基于结构合法性
        unit.legality_score = 1.0 if unit.is_structurally_valid() else 0.0

        # 计算 conflict_cleanliness：第一阶段默认无冲突
        unit.conflict_cleanliness = 1.0

        # 如果有语义定义，提升分数
        if unit.has_meaning():
            unit.truth_score = 0.8
            unit.evidence_score = 0.7
        else:
            unit.truth_score = 0.3
            unit.evidence_score = 0.2

        return unit

    def _is_question(self, text: str) -> bool:
        """判断是否是问题"""
        # 检查疑问词
        for qword in self._question_words:
            if qword in text:
                return True

        # 检查问号
        if "？" in text or "?" in text:
            return True

        return False

    def add_knowledge(self, keyword: str, meaning: str, phonetic: str = "") -> None:
        """添加知识到知识库（用于扩展）"""
        self._knowledge_base[keyword] = {
            "core_meaning": meaning,
            "phonetic_form": phonetic
        }
