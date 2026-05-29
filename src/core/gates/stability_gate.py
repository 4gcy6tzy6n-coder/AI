"""
Stability Gate - 稳定性门控

第三阶段核心：
长期内部稳定门 - 判断对象在多轮之后是否真的稳定

职责：
- 基于多轮历史而非单轮当前值做判决
- 输出稳定性状态标签（stable/pseudo_stable/unstable/degrading/error_prone）
- 返回稳定门候选决策（stay_review/move_normal_candidate/downgrade等）

判决顺序：
1. 重大异常优先（硬否决/冲突/回流）
2. 长期平稳可进入正常区候选
3. 证据强但波动大 → 隔离
4. 峰值后明显下滑 → 降级
5. 反复失败且价值低 → 排除/错误归档
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

import statistics


class StabilityState(Enum):
    """稳定性状态标签"""
    STABLE = "stable"              # 稳定：可进入晋升候选
    PSEUDO_STABLE = "pseudo_stable"  # 假稳定：表面连续但不可靠
    UNSTABLE = "unstable"          # 不稳定：继续受审或隔离
    DEGRADING = "degrading"        # 下滑：历史高位后下滑
    ERROR_PRONE = "error_prone"    # 易错：多轮失败，接近错误区


class StabilityDecision(Enum):
    """稳定门判决结果"""
    STAY_REVIEW = "stay_review"                    # 继续受审
    MOVE_NORMAL_CANDIDATE = "move_normal_candidate"  # 进入正常区候选
    STAY_ISOLATION = "stay_isolation"              # 继续隔离
    DOWNGRADE = "downgrade"                        # 降级
    REVIEW_BACKFLOW = "review_backflow"            # 回流重审
    ERROR_ARCHIVE_CANDIDATE = "error_archive_candidate"  # 错误归档候选
    EXCLUDE_CANDIDATE = "exclude_candidate"        # 排除候选


@dataclass
class StabilitySnapshot:
    """稳定性快照 - 从历史记录提取的判决窗口"""
    unit_id: str
    window_size: int                           # 窗口大小
    recent_events: list[dict] = field(default_factory=list)  # 最近N轮事件
    recent_scores: list[dict] = field(default_factory=list)  # 最近N轮分数
    historical_peak: dict[str, float] = field(default_factory=dict)  # 历史峰值
    conflict_count: int = 0                    # 冲突次数
    backflow_count: int = 0                    # 回流次数
    isolation_count: int = 0                   # 隔离次数
    recent_actions: list[str] = field(default_factory=list)  # 最近动作序列
    hard_veto_hits: int = 0                    # 硬否决命中次数


@dataclass
class VolatilityMetrics:
    """波动性指标"""
    q_range: float = 0.0           # Q分数范围
    q_std: float = 0.0             # Q分数标准差
    q_mean: float = 0.0            # Q分数均值
    s_range: float = 0.0           # S分数范围
    s_std: float = 0.0             # S分数标准差
    s_mean: float = 0.0            # S分数均值
    e_range: float = 0.0           # E分数范围
    e_std: float = 0.0             # E分数标准差
    conflict_rate: float = 0.0     # 冲突率
    backflow_rate: float = 0.0     # 回流率
    hard_veto_rate: float = 0.0    # 硬否决率


@dataclass
class StabilityResult:
    """稳定性门判决结果"""
    unit_id: str
    current_zone: str
    stability_state: StabilityState
    decision: StabilityDecision
    reason: str
    confidence: float
    metrics: VolatilityMetrics
    snapshot: StabilitySnapshot
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "current_zone": self.current_zone,
            "stability_state": self.stability_state.value,
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "metrics": {
                "q_range": round(self.metrics.q_range, 2),
                "q_std": round(self.metrics.q_std, 2),
                "q_mean": round(self.metrics.q_mean, 2),
                "s_range": round(self.metrics.s_range, 2),
                "s_std": round(self.metrics.s_std, 2),
                "s_mean": round(self.metrics.s_mean, 2),
                "conflict_rate": round(self.metrics.conflict_rate, 2),
                "backflow_rate": round(self.metrics.backflow_rate, 2),
                "hard_veto_rate": round(self.metrics.hard_veto_rate, 2)
            },
            "snapshot": {
                "window_size": self.snapshot.window_size,
                "conflict_count": self.snapshot.conflict_count,
                "backflow_count": self.snapshot.backflow_count,
                "isolation_count": self.snapshot.isolation_count,
                "hard_veto_hits": self.snapshot.hard_veto_hits,
                "historical_peak": {k: round(v, 1) for k, v in self.snapshot.historical_peak.items()}
            },
            "recommendations": self.recommendations
        }


class StabilityGate:
    """
    稳定性门控

    基于多轮历史记录判断对象的稳定性状态
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化稳定性门

        Args:
            config: 配置参数
                - window_size: 稳定窗口长度（默认3）
                - stable_min_q: Q最低稳定线（默认70）
                - stable_min_s: S最低稳定线（默认65）
                - max_q_std: Q标准差上限（默认10）
                - max_conflict_rate: 最大冲突率（默认0.2）
                - max_backflow_count: 最大回流次数（默认1）
                - peak_drop_q: Q峰值回落阈值（默认15）
                - peak_drop_s: S峰值回落阈值（默认15）
        """
        self.config = config or {}
        self.window_size = self.config.get("window_size", 3)
        self.stable_min_q = self.config.get("stable_min_q", 70)
        self.stable_min_s = self.config.get("stable_min_s", 65)
        self.max_q_std = self.config.get("max_q_std", 10)
        self.max_conflict_rate = self.config.get("max_conflict_rate", 0.2)
        self.max_backflow_count = self.config.get("max_backflow_count", 1)
        self.peak_drop_q = self.config.get("peak_drop_q", 15)
        self.peak_drop_s = self.config.get("peak_drop_s", 15)

    def evaluate(
        self,
        unit_id: str,
        current_zone: str,
        current_scores: dict[str, float],
        history_events: list[dict],
        score_history: list[dict],
        config: Optional[dict] = None
    ) -> StabilityResult:
        """
        评估对象稳定性

        Args:
            unit_id: 对象ID
            current_zone: 当前区位（review/normal/isolation/error）
            current_scores: 当前轮评分（T/S/E/C/L/R/P/Q）
            history_events: 历史事件序列
            score_history: 历史评分序列
            config: 运行时配置（覆盖初始化配置）

        Returns:
            StabilityResult: 稳定性判决结果
        """
        # 使用运行时配置（如有）
        cfg = config or self.config
        window_size = cfg.get("window_size", self.window_size)

        # Step 1: 构建稳定性快照
        snapshot = self._build_stability_snapshot(
            unit_id, history_events, score_history, window_size
        )

        # Step 2: 计算波动性指标
        metrics = self._compute_volatility_metrics(snapshot, score_history)

        # Step 3: 检测峰值下滑
        peak_drop = self._detect_peak_drop(current_scores, snapshot, cfg)

        # Step 4: 分类稳定性状态
        state = self._classify_stability_state(
            snapshot, metrics, peak_drop, current_scores, cfg
        )

        # Step 5: 做出最终判决
        decision, reason, confidence = self._make_gate_decision(
            current_zone, state, snapshot, metrics, peak_drop, current_scores, cfg
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            state, decision, metrics, current_scores
        )

        return StabilityResult(
            unit_id=unit_id,
            current_zone=current_zone,
            stability_state=state,
            decision=decision,
            reason=reason,
            confidence=confidence,
            metrics=metrics,
            snapshot=snapshot,
            recommendations=recommendations
        )

    def _build_stability_snapshot(
        self,
        unit_id: str,
        history_events: list[dict],
        score_history: list[dict],
        window_size: int
    ) -> StabilitySnapshot:
        """
        构建稳定性快照 - 从历史记录提取判决窗口
        """
        # 获取最近N轮
        recent_scores = score_history[-window_size:] if len(score_history) >= window_size else score_history
        recent_events = history_events[-window_size:] if len(history_events) >= window_size else history_events

        # 计算历史峰值
        historical_peak = {}
        if score_history:
            for key in ["T", "S", "E", "C", "L", "Q"]:
                values = [s.get(key, 0) for s in score_history if key in s]
                if values:
                    historical_peak[key] = max(values)

        # 统计冲突、回流、隔离次数
        conflict_count = sum(1 for e in history_events if e.get("has_conflict", False))
        backflow_count = sum(1 for e in history_events if "backflow" in e.get("action", "").lower())
        isolation_count = sum(1 for e in history_events if "isolate" in e.get("action", "").lower())
        hard_veto_hits = sum(1 for e in history_events if e.get("hard_veto_triggered", False))

        # 提取最近动作序列
        recent_actions = [e.get("action", "unknown") for e in recent_events]

        return StabilitySnapshot(
            unit_id=unit_id,
            window_size=len(recent_scores),
            recent_events=recent_events,
            recent_scores=recent_scores,
            historical_peak=historical_peak,
            conflict_count=conflict_count,
            backflow_count=backflow_count,
            isolation_count=isolation_count,
            recent_actions=recent_actions,
            hard_veto_hits=hard_veto_hits
        )

    def _compute_volatility_metrics(
        self,
        snapshot: StabilitySnapshot,
        score_history: list[dict]
    ) -> VolatilityMetrics:
        """
        计算波动性指标
        """
        metrics = VolatilityMetrics()

        if not score_history:
            return metrics

        # Q分数统计
        q_values = [s.get("Q", 0) for s in score_history if "Q" in s]
        if q_values:
            metrics.q_range = max(q_values) - min(q_values)
            metrics.q_mean = statistics.mean(q_values)
            metrics.q_std = statistics.stdev(q_values) if len(q_values) > 1 else 0

        # S分数统计
        s_values = [s.get("S", 0) for s in score_history if "S" in s]
        if s_values:
            metrics.s_range = max(s_values) - min(s_values)
            metrics.s_mean = statistics.mean(s_values)
            metrics.s_std = statistics.stdev(s_values) if len(s_values) > 1 else 0

        # E分数统计
        e_values = [s.get("E", 0) for s in score_history if "E" in s]
        if e_values:
            metrics.e_range = max(e_values) - min(e_values)
            metrics.e_std = statistics.stdev(e_values) if len(e_values) > 1 else 0

        # 比率统计
        total_events = len(snapshot.recent_events) if snapshot.recent_events else 1
        metrics.conflict_rate = snapshot.conflict_count / total_events
        metrics.backflow_rate = snapshot.backflow_count / total_events
        metrics.hard_veto_rate = snapshot.hard_veto_hits / total_events

        return metrics

    def _detect_peak_drop(
        self,
        current_scores: dict[str, float],
        snapshot: StabilitySnapshot,
        config: dict
    ) -> dict[str, Any]:
        """
        检测历史峰值后下滑
        """
        peak_drop_q = config.get("peak_drop_q", self.peak_drop_q)
        peak_drop_s = config.get("peak_drop_s", self.peak_drop_s)

        current_q = current_scores.get("Q", 0)
        current_s = current_scores.get("S", 0)
        peak_q = snapshot.historical_peak.get("Q", current_q)
        peak_s = snapshot.historical_peak.get("S", current_s)

        q_drop = peak_q - current_q if peak_q > current_q else 0
        s_drop = peak_s - current_s if peak_s > current_s else 0

        return {
            "detected": q_drop >= peak_drop_q or s_drop >= peak_drop_s,
            "q_drop": q_drop,
            "s_drop": s_drop,
            "peak_q": peak_q,
            "peak_s": peak_s
        }

    def _classify_stability_state(
        self,
        snapshot: StabilitySnapshot,
        metrics: VolatilityMetrics,
        peak_drop: dict[str, Any],
        current_scores: dict[str, float],
        config: dict
    ) -> StabilityState:
        """
        分类稳定性状态
        """
        # 规则1：重大异常 → ERROR_PRONE
        if snapshot.hard_veto_hits >= 2:
            return StabilityState.ERROR_PRONE

        if snapshot.backflow_count > config.get("max_backflow_count", self.max_backflow_count):
            return StabilityState.ERROR_PRONE

        if metrics.conflict_rate > 0.5:
            return StabilityState.ERROR_PRONE

        # 规则2：峰值后明显下滑 → DEGRADING
        if peak_drop["detected"]:
            return StabilityState.DEGRADING

        # 规则3：检查是否稳定
        stable_min_q = config.get("stable_min_q", self.stable_min_q)
        stable_min_s = config.get("stable_min_s", self.stable_min_s)
        max_q_std = config.get("max_q_std", self.max_q_std)
        max_conflict_rate = config.get("max_conflict_rate", self.max_conflict_rate)

        current_q = current_scores.get("Q", 0)
        current_s = current_scores.get("S", 0)

        # 基本条件检查
        is_q_stable = current_q >= stable_min_q and metrics.q_std <= max_q_std
        is_s_stable = current_s >= stable_min_s and metrics.s_std <= max_q_std
        is_conflict_low = metrics.conflict_rate <= max_conflict_rate
        is_backflow_clean = snapshot.backflow_count == 0

        if is_q_stable and is_s_stable and is_conflict_low and is_backflow_clean:
            return StabilityState.STABLE

        # 规则4：表面连续但有问题 → PSEUDO_STABLE
        if current_q >= stable_min_q and current_s >= stable_min_s:
            if metrics.q_std > max_q_std or metrics.conflict_rate > 0:
                return StabilityState.PSEUDO_STABLE

        # 规则5：默认不稳定
        return StabilityState.UNSTABLE

    def _make_gate_decision(
        self,
        current_zone: str,
        state: StabilityState,
        snapshot: StabilitySnapshot,
        metrics: VolatilityMetrics,
        peak_drop: dict[str, Any],
        current_scores: dict[str, float],
        config: dict
    ) -> tuple[StabilityDecision, str, float]:
        """
        做出最终稳定门判决
        """
        # 规则1：重大异常优先
        if state == StabilityState.ERROR_PRONE:
            if snapshot.isolation_count >= 2:
                return (StabilityDecision.ERROR_ARCHIVE_CANDIDATE,
                        "多次隔离且反复失败，建议错误归档", 0.8)
            return (StabilityDecision.EXCLUDE_CANDIDATE,
                    "反复失败且价值低，建议排除", 0.75)

        # 规则2：峰值后下滑
        if state == StabilityState.DEGRADING:
            if peak_drop["q_drop"] >= config.get("peak_drop_q", self.peak_drop_q) * 1.5:
                return (StabilityDecision.REVIEW_BACKFLOW,
                        f"Q分数从峰值{peak_drop['peak_q']:.1f}下降{peak_drop['q_drop']:.1f}，建议回流重审", 0.8)
            return (StabilityDecision.DOWNGRADE,
                    f"历史峰值后下滑(Q↓{peak_drop['q_drop']:.1f}, S↓{peak_drop['s_drop']:.1f})，建议降级", 0.75)

        # 规则3：假稳定
        if state == StabilityState.PSEUDO_STABLE:
            return (StabilityDecision.STAY_REVIEW,
                    "表面连续但波动/支撑不可靠，继续受审", 0.7)

        # 规则4：不稳定
        if state == StabilityState.UNSTABLE:
            current_e = current_scores.get("E", 0)
            if current_e >= 75:  # 证据强但不够稳定
                return (StabilityDecision.STAY_ISOLATION,
                        "证据强但稳定性不足，继续隔离", 0.7)
            return (StabilityDecision.STAY_REVIEW,
                    "稳定性不足，继续受审", 0.65)

        # 规则5：稳定 → 进入正常区候选
        if state == StabilityState.STABLE:
            if current_zone == "isolation":
                return (StabilityDecision.MOVE_NORMAL_CANDIDATE,
                        "稳定性验证通过，从隔离区进入正常区候选", 0.85)
            return (StabilityDecision.MOVE_NORMAL_CANDIDATE,
                    "稳定性验证通过，进入正常区候选", 0.85)

        # 默认
        return (StabilityDecision.STAY_REVIEW, "默认继续受审", 0.5)

    def _generate_recommendations(
        self,
        state: StabilityState,
        decision: StabilityDecision,
        metrics: VolatilityMetrics,
        current_scores: dict[str, float]
    ) -> list[str]:
        """
        生成改进建议
        """
        recommendations = []

        if state == StabilityState.UNSTABLE:
            if metrics.s_std > 10:
                recommendations.append("建议增加稳定性验证轮次")
            if metrics.conflict_rate > 0:
                recommendations.append("建议清理未解决冲突")

        if state == StabilityState.PSEUDO_STABLE:
            recommendations.append("建议检查支撑来源独立性")
            recommendations.append("建议观察隐藏冲突")

        if state == StabilityState.DEGRADING:
            recommendations.append("建议审查近期变化原因")
            recommendations.append("建议重新验证核心证据")

        if decision == StabilityDecision.MOVE_NORMAL_CANDIDATE:
            q = current_scores.get("Q", 0)
            if q < 75:
                recommendations.append("虽通过稳定门，但Q分数仍有提升空间")

        return recommendations
