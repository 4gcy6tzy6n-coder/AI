from typing import Any

from ..unit.models import CandidateConclusion, Unit
from .gap_detector import GapDetectionResult


class ConclusionBuilder:
    """
    候选结论生成器

    第一阶段简单版：组合输入、Unit、证据生成可审查对象
    """

    def __init__(self):
        pass

    def build(
        self,
        input_unit: Unit,
        activated_units: list[Unit],
        evidence_pack: list[dict[str, Any]],
        gap_result: GapDetectionResult | None = None
    ) -> CandidateConclusion:
        """
        构建候选结论

        组合：
        - 输入问题
        - 激活的 Unit
        - 检索到的 evidence
        - 当前冲突信息
        """
        # 提取输入文本
        input_text = str(input_unit.content) if input_unit.content else ""

        # 生成回答文本（简化版）
        answer_text = self._generate_answer_text(
            input_text,
            activated_units,
            evidence_pack,
            gap_result
        )

        # 识别缺失槽位
        missing_slots = []
        if gap_result and gap_result.has_gap:
            missing_slots = gap_result.reasons

        # 识别冲突标记
        conflict_flags = []
        for unit in activated_units:
            if unit.conflict_cleanliness < 1.0:
                conflict_flags.append(
                    f"Unit '{unit.script_form}' 冲突清洁度: {unit.conflict_cleanliness}"
                )

        # 计算置信度 stub
        confidence = self._calculate_confidence(
            activated_units,
            evidence_pack,
            gap_result
        )

        return CandidateConclusion(
            answer_text=answer_text,
            used_units=activated_units,
            used_evidence=evidence_pack,
            missing_slots=missing_slots,
            conflict_flags=conflict_flags,
            confidence_stub=confidence
        )

    def _generate_answer_text(
        self,
        input_text: str,
        units: list[Unit],
        evidence: list[dict[str, Any]],
        gap_result: GapDetectionResult | None
    ) -> str:
        """生成回答文本（简化版）"""
        # 如果有严重缺口，返回保留回答
        if gap_result and gap_result.has_gap and gap_result.severity == "high":
            return (
                f"关于'{input_text}'，系统检测到信息缺口，"
                f"暂时无法给出确切回答。缺口原因: {', '.join(gap_result.reasons)}"
            )

        # 如果有证据，基于证据生成回答
        if evidence:
            # 组合证据内容
            evidence_parts = []
            for ev in evidence[:2]:  # 最多使用2条证据
                content = ev.get("content", "")
                if content:
                    evidence_parts.append(content)

            if evidence_parts:
                return (
                    f"基于检索到的信息: "
                    f"{'; '.join(evidence_parts)}"
                )

        # 如果有 Unit 有语义定义，使用它
        for unit in units:
            if unit.core_meaning:
                return f"{unit.script_form} 是 {unit.core_meaning}"

        # 默认回复
        return f"收到关于'{input_text}'的查询，但相关信息不足。"

    def _calculate_confidence(
        self,
        units: list[Unit],
        evidence: list[dict[str, Any]],
        gap_result: GapDetectionResult | None
    ) -> float:
        """计算置信度 stub"""
        base_confidence = 0.5

        # 根据缺口调整
        if gap_result:
            if gap_result.has_gap:
                if gap_result.severity == "high":
                    base_confidence -= 0.3
                elif gap_result.severity == "medium":
                    base_confidence -= 0.2
                else:
                    base_confidence -= 0.1

        # 根据证据调整
        if evidence:
            avg_relevance = sum(
                ev.get("relevance_score", 0.5) for ev in evidence
            ) / len(evidence)
            base_confidence += avg_relevance * 0.2

        # 根据 Unit 质量调整
        for unit in units:
            if unit.truth_score > 0.7:
                base_confidence += 0.05
            if unit.evidence_score > 0.7:
                base_confidence += 0.05

        return max(0.0, min(1.0, base_confidence))
