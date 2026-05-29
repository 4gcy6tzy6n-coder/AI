from enum import Enum, auto
from typing import Any, Optional

from ..unit.models import CandidateConclusion, TSLAResult, Unit
from .conclusion_builder import ConclusionBuilder
from .gap_detector import GapDetectionResult, GapDetector


class ThinkingState(str, Enum):
    """思考状态 - 6个固定状态"""
    INPUT_PARSED = "InputParsed"
    UNITS_ACTIVATED = "UnitsActivated"
    GAP_DETECTED = "GapDetected"
    EVIDENCE_COLLECTED = "EvidenceCollected"
    CANDIDATE_BUILT = "CandidateBuilt"
    TSLA_REVIEWED = "TSLAReviewed"


class ThinkingEngine:
    """
    思考引擎 - 状态机实现

    把"思考"从抽象名词变成一串可观察状态转换
    """

    def __init__(self):
        self.state_trace: list[str] = []
        self.current_state: ThinkingState = ThinkingState.INPUT_PARSED
        self.gap_detector = GapDetector()
        self.conclusion_builder = ConclusionBuilder()

        # 中间状态存储
        self._units: list[Unit] = []
        self._gap_result: Optional[GapDetectionResult] = None
        self._evidence: list[dict[str, Any]] = []
        self._candidate: Optional[CandidateConclusion] = None
        self._tsla_result: Optional[TSLAResult] = None

    def think(
        self,
        input_unit: Unit,
        activated_units: list[Unit],
        evidence_pack: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        执行思考流程

        返回完整的状态转换轨迹和结果
        """
        self.state_trace = []
        self._units = [input_unit] + activated_units
        self._evidence = evidence_pack

        # 状态 1: InputParsed
        self._transition_to(ThinkingState.INPUT_PARSED)

        # 状态 2: UnitsActivated
        self._transition_to(ThinkingState.UNITS_ACTIVATED)

        # 状态 3: GapDetected (关键步骤)
        self._gap_result = self.gap_detector.detect(
            input_unit=input_unit,
            activated_units=activated_units,
            evidence_pack=evidence_pack
        )

        if self._gap_result.has_gap:
            self._transition_to(ThinkingState.GAP_DETECTED)
            # 如果有缺口，需要收集证据
            if evidence_pack:
                self._transition_to(ThinkingState.EVIDENCE_COLLECTED)
        else:
            # 没有缺口，直接到证据收集
            self._transition_to(ThinkingState.EVIDENCE_COLLECTED)

        # 状态 5: CandidateBuilt
        self._candidate = self.conclusion_builder.build(
            input_unit=input_unit,
            activated_units=activated_units,
            evidence_pack=evidence_pack,
            gap_result=self._gap_result
        )
        self._transition_to(ThinkingState.CANDIDATE_BUILT)

        # 状态 6: TSLAReviewed (由外部调用 TSLA 后设置)
        # 这里先标记为待审查
        self._transition_to(ThinkingState.TSLA_REVIEWED)

        return {
            "state_trace": self.state_trace,
            "current_state": self.current_state.value,
            "gap_result": self._gap_result.to_dict() if self._gap_result else None,
            "candidate_conclusion": self._candidate.to_dict() if self._candidate else None
        }

    def set_tsla_result(self, tsla_result: TSLAResult) -> None:
        """设置 TSLA 审查结果"""
        self._tsla_result = tsla_result

    def get_candidate(self) -> Optional[CandidateConclusion]:
        """获取候选结论"""
        return self._candidate

    def _transition_to(self, new_state: ThinkingState) -> None:
        """状态转换"""
        self.current_state = new_state
        self.state_trace.append(new_state.value)

    def get_state_summary(self) -> dict[str, Any]:
        """获取状态摘要"""
        return {
            "state_trace": self.state_trace,
            "current_state": self.current_state.value,
            "has_gap": self._gap_result.has_gap if self._gap_result else False,
            "has_candidate": self._candidate is not None,
            "has_tsla_result": self._tsla_result is not None
        }
