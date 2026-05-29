"""
TSLA Scorer - 统一评分框架

第二阶段核心：
完整评分字段输出（占位参数版本）：
T - 真实性 (Truthfulness)
S - 稳定性 (Stability)
E - 证据强度 (Evidence)
C - 冲突洁净度 (Conflict Cleanliness)
L - 结构合法性 (Legality)
R - 复用价值 (Reusability)
P - 来源风险修正 (Provenance Risk)
Q - 综合治理分 (Composite)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from ..unit.models import Unit


@dataclass
class TSLAScores:
    """TSLA 完整评分"""
    # 基础评分 (0-100)
    T: float = 50.0  # 真实性
    S: float = 50.0  # 稳定性
    E: float = 50.0  # 证据强度
    C: float = 100.0  # 冲突洁净度
    L: float = 100.0  # 结构合法性
    R: float = 50.0  # 复用价值
    P: float = 60.0  # 来源风险修正

    # 综合治理分
    Q: float = 50.0  # 综合治理分

    # 权重配置（占位参数）
    weights: dict[str, float] | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "T": 0.25,  # 真实性权重
                "S": 0.20,  # 稳定性权重
                "E": 0.20,  # 证据强度权重
                "C": 0.15,  # 冲突洁净度权重
                "L": 0.10,  # 结构合法性权重
                "R": 0.05,  # 复用价值权重
                "P": 0.05   # 来源风险修正权重
            }

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "T": round(self.T, 1),
            "S": round(self.S, 1),
            "E": round(self.E, 1),
            "C": round(self.C, 1),
            "L": round(self.L, 1),
            "R": round(self.R, 1),
            "P": round(self.P, 1),
            "Q": round(self.Q, 1),
            "weights": self.weights
        }

    def get_shortfall(self) -> list[dict[str, Any]]:
        """获取分数短板"""
        scores = {
            "T": (self.T, 50, "真实性"),
            "S": (self.S, 40, "稳定性"),
            "E": (self.E, 45, "证据强度"),
            "C": (self.C, 60, "冲突洁净度"),
            "L": (self.L, 70, "结构合法性"),
            "R": (self.R, 30, "复用价值"),
            "P": (self.P, 40, "来源风险")
        }

        shortfalls = []
        for key, (score, threshold, name) in scores.items():
            if score < threshold:
                shortfalls.append({
                    "dimension": key,
                    "name": name,
                    "score": score,
                    "threshold": threshold,
                    "gap": threshold - score
                })

        # 按差距排序
        shortfalls.sort(key=lambda x: x["gap"], reverse=True)
        return shortfalls


class TSLAScorer:
    """
    TSLA 评分器

    计算完整的 7 维度评分 + 综合治理分
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None
    ):
        self.weights = weights or {
            "T": 0.25,
            "S": 0.20,
            "E": 0.20,
            "C": 0.15,
            "L": 0.10,
            "R": 0.05,
            "P": 0.05
        }

    def calculate(
        self,
        unit: Unit,
        gap_detected: bool = False,
        gap_severity: str = "low",
        evidence_count: int = 0,
        access_count: int = 0
    ) -> TSLAScores:
        """
        计算完整评分

        参数：
        - unit: Unit 对象
        - gap_detected: 是否检测到缺口
        - gap_severity: 缺口严重程度
        - evidence_count: 证据数量
        - access_count: 访问次数

        返回：完整评分对象
        """
        # T - 真实性 (基于 Unit 的 truth_score)
        T = self._calculate_truth(unit)

        # S - 稳定性（基于访问次数和历史）
        S = self._calculate_stability(unit, access_count)

        # E - 证据强度
        E = self._calculate_evidence(unit, evidence_count, gap_detected, gap_severity)

        # C - 冲突洁净度
        C = self._calculate_conflict_cleanliness(unit)

        # L - 结构合法性
        L = self._calculate_legality(unit)

        # R - 复用价值
        R = self._calculate_reusability(unit, access_count)

        # P - 来源风险修正
        P = self._calculate_provenance(unit)

        # Q - 综合治理分
        Q = self._calculate_composite(T, S, E, C, L, R, P)

        return TSLAScores(
            T=T, S=S, E=E, C=C, L=L, R=R, P=P, Q=Q,
            weights=self.weights
        )

    def _calculate_truth(self, unit: Unit) -> float:
        """计算真实性分数"""
        # 基础分数来自 Unit
        base_score = unit.truth_score * 100

        # 如果有核心语义定义，加分
        if unit.core_meaning and len(unit.core_meaning) > 5:
            base_score += 10

        # 如果有语义三态完整，加分
        if unit.script_form and unit.phonetic_form and unit.core_meaning:
            base_score += 5

        return min(100.0, base_score)

    def _calculate_stability(self, unit: Unit, access_count: int) -> float:
        """计算稳定性分数"""
        # 基础分数
        base_score = unit.stability_score * 100

        # 访问次数影响稳定性
        if access_count > 5:
            base_score += 15
        elif access_count > 2:
            base_score += 8
        elif access_count > 0:
            base_score += 3

        return min(100.0, base_score)

    def _calculate_evidence(
        self,
        unit: Unit,
        evidence_count: int,
        gap_detected: bool,
        gap_severity: str
    ) -> float:
        """计算证据强度分数"""
        # 基础分数
        base_score = unit.evidence_score * 100

        # 证据数量影响
        if evidence_count >= 3:
            base_score += 15
        elif evidence_count == 2:
            base_score += 8
        elif evidence_count == 1:
            base_score += 3

        # 缺口惩罚
        if gap_detected:
            if gap_severity == "high":
                base_score -= 30
            elif gap_severity == "medium":
                base_score -= 15
            else:
                base_score -= 5

        return max(0.0, min(100.0, base_score))

    def _calculate_conflict_cleanliness(self, unit: Unit) -> float:
        """计算冲突洁净度"""
        # 直接使用 Unit 的分数
        return unit.conflict_cleanliness * 100

    def _calculate_legality(self, unit: Unit) -> float:
        """计算结构合法性"""
        # 直接使用 Unit 的分数
        return unit.legality_score * 100

    def _calculate_reusability(self, unit: Unit, access_count: int) -> float:
        """计算复用价值"""
        base_score = 50.0

        # 访问次数反映复用价值
        if access_count > 10:
            base_score += 30
        elif access_count > 5:
            base_score += 20
        elif access_count > 2:
            base_score += 10

        # 如果有完整语义定义，更有价值
        if unit.core_meaning:
            base_score += 10

        return min(100.0, base_score)

    def _calculate_provenance(self, unit: Unit) -> float:
        """计算来源风险修正"""
        # 根据来源类型调整
        source_type = unit.source_type.value if hasattr(unit.source_type, 'value') else str(unit.source_type)

        base_scores = {
            "system_generated": 85,
            "user_input": 70,
            "retrieved": 60,
            "inferred": 50
        }

        base_score = base_scores.get(source_type, 60)

        # 治理分数影响
        if unit.governance_score > 0.7:
            base_score += 10
        elif unit.governance_score < 0.3:
            base_score -= 10

        return max(0.0, min(100.0, base_score))

    def _calculate_composite(
        self,
        T: float, S: float, E: float, C: float,
        L: float, R: float, P: float
    ) -> float:
        """计算综合治理分"""
        Q = (
            T * self.weights["T"] +
            S * self.weights["S"] +
            E * self.weights["E"] +
            C * self.weights["C"] +
            L * self.weights["L"] +
            R * self.weights["R"] +
            P * self.weights["P"]
        )
        return round(Q, 1)

    def explain_score(self, scores: TSLAScores) -> dict[str, Any]:
        """解释评分结果"""
        shortfalls = scores.get_shortfall()

        return {
            "composite_score": scores.Q,
            "grade": self._get_grade(scores.Q),
            "shortfalls": shortfalls,
            "strengths": self._get_strengths(scores),
            "recommendation": self._get_recommendation(scores, shortfalls)
        }

    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def _get_strengths(self, scores: TSLAScores) -> list[dict[str, Any]]:
        """获取优势维度"""
        all_scores = [
            ("T", scores.T, "真实性"),
            ("S", scores.S, "稳定性"),
            ("E", scores.E, "证据强度"),
            ("C", scores.C, "冲突洁净度"),
            ("L", scores.L, "结构合法性"),
            ("R", scores.R, "复用价值"),
            ("P", scores.P, "来源风险")
        ]

        # 筛选高分维度
        strengths = [
            {"dimension": dim, "name": name, "score": score}
            for dim, score, name in all_scores
            if score >= 70
        ]

        # 按分数排序
        strengths.sort(key=lambda x: x["score"], reverse=True)
        return strengths

    def _get_recommendation(
        self,
        scores: TSLAScores,
        shortfalls: list[dict[str, Any]]
    ) -> str:
        """获取改进建议"""
        if scores.Q >= 80:
            return "评分优秀，可考虑晋升"
        elif scores.Q >= 60:
            if shortfalls:
                top_shortfall = shortfalls[0]
                return f"需改进: {top_shortfall['name']}({top_shortfall['score']:.0f})"
            return "评分良好，保持观察"
        elif scores.Q >= 40:
            if shortfalls:
                top_shortfall = shortfalls[0]
                return f"需重点改进: {top_shortfall['name']}({top_shortfall['score']:.0f})"
            return "评分一般，需要审查"
        else:
            return "评分不合格，建议隔离或排除"


# ============================================================================
# 第三阶段新增：历史序列评分与滚动统计
# ============================================================================

from datetime import datetime
import statistics


@dataclass
class ScoreSnapshot:
    """
    评分快照 - 单轮评分的完整记录

    用于积累历史评分序列，支持后续稳定性分析和定标实验
    """
    unit_id: str
    timestamp: str
    session_id: Optional[str] = None

    # 八维评分
    T: float = 0.0  # 真实性
    S: float = 0.0  # 稳定性
    E: float = 0.0  # 证据强度
    C: float = 0.0  # 冲突洁净度
    L: float = 0.0  # 结构合法性
    R: float = 0.0  # 复用价值
    P: float = 0.0  # 来源风险修正
    Q: float = 0.0  # 综合治理分

    # 评分理由摘要
    reasons: dict[str, str] = field(default_factory=dict)

    # 上下文信息
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "scores": {
                "T": round(self.T, 1),
                "S": round(self.S, 1),
                "E": round(self.E, 1),
                "C": round(self.C, 1),
                "L": round(self.L, 1),
                "R": round(self.R, 1),
                "P": round(self.P, 1),
                "Q": round(self.Q, 1)
            },
            "reasons": self.reasons,
            "context": self.context
        }


@dataclass
class RollingStats:
    """
    滚动统计 - 基于历史评分序列的统计指标

    为 stability_gate.py 提供数据支持
    """
    # Q分数统计
    q_mean: float = 0.0
    q_std: float = 0.0
    q_min: float = 0.0
    q_max: float = 0.0

    # S分数统计
    s_mean: float = 0.0
    s_std: float = 0.0
    s_min: float = 0.0
    s_max: float = 0.0

    # E分数统计
    e_mean: float = 0.0
    e_std: float = 0.0

    # 峰值追踪
    peak_q: float = 0.0
    peak_s: float = 0.0
    peak_timestamp: Optional[str] = None

    # 趋势（简单线性趋势）
    trend_q: str = "stable"  # "up", "down", "stable"
    trend_s: str = "stable"

    # 变化量
    delta_from_last: dict[str, float] = field(default_factory=dict)
    delta_from_peak: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_stats": {
                "mean": round(self.q_mean, 2),
                "std": round(self.q_std, 2),
                "min": round(self.q_min, 1),
                "max": round(self.q_max, 1)
            },
            "s_stats": {
                "mean": round(self.s_mean, 2),
                "std": round(self.s_std, 2),
                "min": round(self.s_min, 1),
                "max": round(self.s_max, 1)
            },
            "e_stats": {
                "mean": round(self.e_mean, 2),
                "std": round(self.e_std, 2)
            },
            "peaks": {
                "Q": round(self.peak_q, 1),
                "S": round(self.peak_s, 1),
                "timestamp": self.peak_timestamp
            },
            "trends": {
                "Q": self.trend_q,
                "S": self.trend_s
            },
            "delta_from_last": {k: round(v, 2) for k, v in self.delta_from_last.items()},
            "delta_from_peak": {k: round(v, 2) for k, v in self.delta_from_peak.items()}
        }


@dataclass
class ScoringResult:
    """
    完整评分结果 - 第三阶段输出格式

    包含当前评分、历史快照、滚动统计
    """
    current_scores: TSLAScores
    snapshot: ScoreSnapshot
    rolling_stats: RollingStats
    history_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_scores": self.current_scores.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "rolling_stats": self.rolling_stats.to_dict(),
            "history_count": self.history_count
        }


class TSLAScorerWithHistory:
    """
    带历史记录的 TSLA 评分器

    第三阶段核心组件，支持：
    - 每轮评分形成独立快照
    - 同一对象积累评分历史
    - 输出滚动统计和峰值差值
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        max_history: int = 100
    ):
        self.weights = weights or {
            "T": 0.25, "S": 0.20, "E": 0.20,
            "C": 0.15, "L": 0.10, "R": 0.05, "P": 0.05
        }
        self.max_history = max_history
        self._history: dict[str, list[ScoreSnapshot]] = {}  # unit_id -> snapshots

    def score_with_history(
        self,
        unit_id: str,
        unit: Unit,
        current_context: dict[str, Any],
        gap_detected: bool = False,
        gap_severity: str = "low",
        evidence_count: int = 0,
        access_count: int = 0,
        session_id: Optional[str] = None
    ) -> ScoringResult:
        """
        评分并记录历史

        Args:
            unit_id: 对象ID
            unit: Unit对象
            current_context: 当前上下文
            gap_detected: 是否检测到缺口
            gap_severity: 缺口严重程度
            evidence_count: 证据数量
            access_count: 访问次数
            session_id: 会话ID

        Returns:
            ScoringResult: 包含当前评分、历史快照、滚动统计
        """
        # 计算当前评分
        base_scorer = TSLAScorer(self.weights)
        scores = base_scorer.calculate(
            unit, gap_detected, gap_severity, evidence_count, access_count
        )

        # 生成评分理由
        reasons = self._generate_reasons(
            scores, unit, gap_detected, evidence_count
        )

        # 创建快照
        snapshot = ScoreSnapshot(
            unit_id=unit_id,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            T=scores.T,
            S=scores.S,
            E=scores.E,
            C=scores.C,
            L=scores.L,
            R=scores.R,
            P=scores.P,
            Q=scores.Q,
            reasons=reasons,
            context={
                "gap_detected": gap_detected,
                "gap_severity": gap_severity,
                "evidence_count": evidence_count,
                "access_count": access_count,
                "source_type": str(unit.source_type)
            }
        )

        # 保存到历史
        if unit_id not in self._history:
            self._history[unit_id] = []
        self._history[unit_id].append(snapshot)

        # 限制历史长度
        if len(self._history[unit_id]) > self.max_history:
            self._history[unit_id] = self._history[unit_id][-self.max_history:]

        # 计算滚动统计
        rolling_stats = self._compute_rolling_stats(unit_id)

        return ScoringResult(
            current_scores=scores,
            snapshot=snapshot,
            rolling_stats=rolling_stats,
            history_count=len(self._history[unit_id])
        )

    def _generate_reasons(
        self,
        scores: TSLAScores,
        unit: Unit,
        gap_detected: bool,
        evidence_count: int
    ) -> dict[str, str]:
        """生成每项分数的理由摘要"""
        reasons = {}

        # T - 真实性
        if unit.truth_score > 0.8:
            reasons["T"] = "direct evidence matched"
        elif unit.truth_score > 0.5:
            reasons["T"] = "partial evidence support"
        else:
            reasons["T"] = "insufficient verification"

        # S - 稳定性
        if scores.S >= 80:
            reasons["S"] = "multi-cycle consistent"
        elif scores.S >= 60:
            reasons["S"] = "moderately stable"
        else:
            reasons["S"] = "high volatility observed"

        # E - 证据强度
        if evidence_count >= 3:
            reasons["E"] = f"strong evidence base ({evidence_count} sources)"
        elif evidence_count >= 1:
            reasons["E"] = f"limited evidence ({evidence_count} source)"
        else:
            reasons["E"] = "no direct evidence"

        # C - 冲突洁净度
        if unit.conflict_cleanliness >= 0.9:
            reasons["C"] = "clean, no conflicts"
        elif unit.conflict_cleanliness >= 0.7:
            reasons["C"] = "minor unresolved conflict"
        else:
            reasons["C"] = "significant conflict remains"

        # L - 结构合法性
        if unit.legality_score >= 0.9:
            reasons["L"] = "fully compliant"
        elif unit.legality_score >= 0.7:
            reasons["L"] = "minor structural issues"
        else:
            reasons["L"] = "structural concerns"

        # R - 复用价值
        if scores.R >= 70:
            reasons["R"] = "highly reusable"
        elif scores.R >= 50:
            reasons["R"] = "moderate reuse potential"
        else:
            reasons["R"] = "limited reuse value"

        # P - 来源风险
        source_type = str(unit.source_type)
        if "system" in source_type:
            reasons["P"] = "low risk, system verified"
        elif "user" in source_type:
            reasons["P"] = "medium risk, user input"
        else:
            reasons["P"] = "elevated risk, inferred/retrieved"

        # Q - 综合
        if scores.Q >= 80:
            reasons["Q"] = "excellent overall quality"
        elif scores.Q >= 60:
            reasons["Q"] = "acceptable quality"
        elif scores.Q >= 40:
            reasons["Q"] = "marginal quality"
        else:
            reasons["Q"] = "poor quality, needs review"

        return reasons

    def _compute_rolling_stats(self, unit_id: str) -> RollingStats:
        """计算滚动统计"""
        history = self._history.get(unit_id, [])

        if not history:
            return RollingStats()

        if len(history) == 1:
            # 只有一条记录
            snap = history[0]
            return RollingStats(
                q_mean=snap.Q, q_min=snap.Q, q_max=snap.Q,
                s_mean=snap.S, s_min=snap.S, s_max=snap.S,
                e_mean=snap.E,
                peak_q=snap.Q, peak_s=snap.S,
                peak_timestamp=snap.timestamp
            )

        # 提取序列
        q_values = [s.Q for s in history]
        s_values = [s.S for s in history]
        e_values = [s.E for s in history]

        # 计算峰值
        peak_q = max(q_values)
        peak_s = max(s_values)
        peak_idx = q_values.index(peak_q)
        peak_timestamp = history[peak_idx].timestamp

        # 当前值
        current = history[-1]
        last = history[-2] if len(history) >= 2 else current

        # 计算趋势（简单比较最近3个）
        trend_q = self._compute_trend(q_values[-3:])
        trend_s = self._compute_trend(s_values[-3:])

        return RollingStats(
            q_mean=statistics.mean(q_values),
            q_std=statistics.stdev(q_values) if len(q_values) > 1 else 0,
            q_min=min(q_values),
            q_max=max(q_values),
            s_mean=statistics.mean(s_values),
            s_std=statistics.stdev(s_values) if len(s_values) > 1 else 0,
            s_min=min(s_values),
            s_max=max(s_values),
            e_mean=statistics.mean(e_values),
            e_std=statistics.stdev(e_values) if len(e_values) > 1 else 0,
            peak_q=peak_q,
            peak_s=peak_s,
            peak_timestamp=peak_timestamp,
            trend_q=trend_q,
            trend_s=trend_s,
            delta_from_last={
                "Q": current.Q - last.Q,
                "S": current.S - last.S,
                "T": current.T - last.T
            },
            delta_from_peak={
                "Q": current.Q - peak_q,
                "S": current.S - peak_s
            }
        )

    def _compute_trend(self, values: list[float]) -> str:
        """计算简单趋势"""
        if len(values) < 2:
            return "stable"

        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        avg_diff = sum(diffs) / len(diffs)

        if avg_diff > 3:
            return "up"
        elif avg_diff < -3:
            return "down"
        return "stable"

    def get_history(self, unit_id: str) -> list[ScoreSnapshot]:
        """获取对象的历史评分记录"""
        return self._history.get(unit_id, [])

    def get_score_history_for_stability_gate(
        self,
        unit_id: str,
        window_size: int = 3
    ) -> list[dict]:
        """
        为 stability_gate.py 提供格式化的评分历史

        返回最近N轮的评分字典列表
        """
        history = self._history.get(unit_id, [])
        recent = history[-window_size:] if len(history) >= window_size else history

        return [
            {
                "T": s.T, "S": s.S, "E": s.E, "C": s.C,
                "L": s.L, "R": s.R, "P": s.P, "Q": s.Q,
                "timestamp": s.timestamp
            }
            for s in recent
        ]

    def clear_history(self, unit_id: Optional[str] = None):
        """清除历史记录"""
        if unit_id:
            self._history.pop(unit_id, None)
        else:
            self._history.clear()
