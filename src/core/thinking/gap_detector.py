from typing import Any

from ..unit.models import Unit


class GapDetectionResult:
    """缺口检测结果"""

    def __init__(
        self,
        has_gap: bool = False,
        gap_types: list[str] | None = None,
        reasons: list[str] | None = None,
        severity: str = "low"
    ):
        self.has_gap = has_gap
        self.gap_types = gap_types or []
        self.reasons = reasons or []
        self.severity = severity

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "has_gap": self.has_gap,
            "gap_types": self.gap_types,
            "reasons": self.reasons,
            "severity": self.severity
        }


class GapDetector:
    """
    缺口感知器

    判断何时需要检索、何时不能直接回答
    这是系统和普通聊天机器人最大的区别之一
    """

    def __init__(self):
        pass

    def detect(
        self,
        input_unit: Unit,
        activated_units: list[Unit],
        evidence_pack: list[dict[str, Any]]
    ) -> GapDetectionResult:
        """
        检测是否存在缺口

        判断 3 件事：
        1. 当前输入是否有核心对象没有基本义
        2. 当前候选 Unit 之间是否冲突
        3. 当前结论是否缺少支撑证据
        """
        gap_types = []
        reasons = []

        # 检查 1: 核心对象是否缺少基本语义定义
        missing_meaning = self._check_missing_meaning(activated_units)
        if missing_meaning:
            gap_types.append("missing_meaning")
            reasons.extend(missing_meaning)

        # 检查 2: 候选 Unit 之间是否冲突
        conflicts = self._check_conflicts(activated_units)
        if conflicts:
            gap_types.append("conflict")
            reasons.extend(conflicts)

        # 检查 3: 是否缺少支撑证据
        evidence_gap = self._check_evidence_gap(input_unit, evidence_pack)
        if evidence_gap:
            gap_types.append("insufficient_evidence")
            reasons.extend(evidence_gap)

        # 确定严重程度
        severity = self._calculate_severity(gap_types, len(reasons))

        has_gap = len(gap_types) > 0

        return GapDetectionResult(
            has_gap=has_gap,
            gap_types=gap_types,
            reasons=reasons,
            severity=severity
        )

    def _check_missing_meaning(self, units: list[Unit]) -> list[str]:
        """检查是否有对象缺少基本语义定义"""
        missing = []

        for unit in units:
            # 跳过输入 Unit
            if unit.unit_type.value == "input":
                continue

            # 检查是否有 core_meaning
            if not unit.has_meaning():
                missing.append(
                    f"对象 '{unit.script_form}' 缺少基本语义定义"
                )

        return missing

    def _check_conflicts(self, units: list[Unit]) -> list[str]:
        """检查候选 Unit 之间是否冲突"""
        conflicts = []

        # 检查是否有相同的 script_form 但不同的 core_meaning
        meaning_map: dict[str, list[str]] = {}

        for unit in units:
            if unit.unit_type.value == "input":
                continue

            form = unit.script_form
            meaning = unit.core_meaning

            if form not in meaning_map:
                meaning_map[form] = []

            if meaning and meaning not in meaning_map[form]:
                meaning_map[form].append(meaning)

        # 如果一个形式有多个不同含义，标记为冲突
        for form, meanings in meaning_map.items():
            if len(meanings) > 1:
                conflicts.append(
                    f"对象 '{form}' 存在语义冲突: {meanings}"
                )

        # 检查 conflict_cleanliness 分数
        for unit in units:
            if unit.conflict_cleanliness < 0.5:
                conflicts.append(
                    f"对象 '{unit.script_form}' 冲突清洁度低 ({unit.conflict_cleanliness})"
                )

        return conflicts

    def _check_evidence_gap(
        self,
        input_unit: Unit,
        evidence_pack: list[dict[str, Any]]
    ) -> list[str]:
        """检查是否缺少支撑证据"""
        gaps = []

        # 如果证据包为空，标记为证据不足
        if not evidence_pack:
            gaps.append("没有检索到相关证据")

        # 检查证据质量
        low_quality_evidence = 0
        for evidence in evidence_pack:
            relevance = evidence.get("relevance_score", 0)
            if relevance < 0.3:
                low_quality_evidence += 1

        if low_quality_evidence == len(evidence_pack) and evidence_pack:
            gaps.append("检索到的证据质量都较低")

        # 检查输入 Unit 的 evidence_score
        if hasattr(input_unit, 'evidence_score'):
            if input_unit.evidence_score < 0.3:
                gaps.append("输入证据分数过低")

        return gaps

    def _calculate_severity(
        self,
        gap_types: list[str],
        reason_count: int
    ) -> str:
        """计算缺口严重程度"""
        if "conflict" in gap_types:
            return "high"
        if "missing_meaning" in gap_types and len(gap_types) > 1:
            return "high"
        if reason_count >= 3:
            return "medium"
        if gap_types:
            return "low"
        return "none"
