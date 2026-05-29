"""
Promotion Gate - 晋升门控

第三阶段核心：
长期层内部晋升 - 受审区/隔离区 → 长期正常区

职责：
- 只负责长期层内部晋升，不碰永久层
- 所有晋升必须经过稳定门前置判断
- 晋升失败时能说明失败原因

晋升路径：
review/isolation → normal

晋升条件：
1. 通过TSLA初判
2. 已过强审查（最小轮数）
3. 稳定门输出 move_normal_candidate
4. 达到长期正常区阈值
5. 不存在硬否决信号
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PromotionDecision(Enum):
    """晋升门判决结果"""
    PROMOTE_TO_NORMAL = "promote_to_normal"      # 晋升到正常区
    STAY_REVIEW = "stay_review"                  # 继续受审
    STAY_ISOLATION = "stay_isolation"            # 继续隔离
    BLOCKED_BY_THRESHOLD = "blocked_by_threshold"  # 被阈值阻挡
    BLOCKED_BY_STABILITY = "blocked_by_stability"  # 被稳定性阻挡
    BLOCKED_BY_CONFLICT = "blocked_by_conflict"    # 被冲突阻挡
    BLOCKED_BY_VETO = "blocked_by_veto"            # 被硬否决阻挡
    BLOCKED_BY_REVIEW = "blocked_by_review"        # 被审查不足阻挡


@dataclass
class ThresholdCheck:
    """阈值检查结果"""
    passed: bool
    dimension: str
    required: float
    actual: float
    gap: float


@dataclass
class PromotionResult:
    """晋升门判决结果"""
    unit_id: str
    current_zone: str
    decision: PromotionDecision
    reason: str
    confidence: float
    stability_decision: str
    threshold_checks: list[ThresholdCheck]
    blockers: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "current_zone": self.current_zone,
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "stability_decision": self.stability_decision,
            "threshold_checks": [
                {
                    "dimension": tc.dimension,
                    "required": tc.required,
                    "actual": round(tc.actual, 1),
                    "passed": tc.passed,
                    "gap": round(tc.gap, 1)
                }
                for tc in self.threshold_checks
            ],
            "blockers": self.blockers,
            "recommendations": self.recommendations
        }


class PromotionGate:
    """
    晋升门控

    负责长期层内部晋升决策
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化晋升门

        Args:
            config: 配置参数
                - min_review_rounds: 最小审查轮数（默认3）
                - review_zone_thresholds: 受审区阈值
                - normal_zone_thresholds: 正常区阈值
        """
        self.config = config or {}

        # 最小审查轮数
        self.min_review_rounds = self.config.get("min_review_rounds", 3)

        # 受审区阈值（较低门槛）
        self.review_thresholds = self.config.get("review_zone_thresholds", {
            "Q": 50,
            "T": 55,
            "L": 60
        })

        # 正常区阈值（较高门槛）
        self.normal_thresholds = self.config.get("normal_zone_thresholds", {
            "Q": 72,
            "T": 75,
            "S": 70,
            "E": 65,
            "C": 75,
            "L": 75
        })

    def evaluate(
        self,
        unit_id: str,
        current_zone: str,
        current_scores: dict[str, float],
        stability_decision: str,
        latest_tsla_action: str,
        review_rounds: int,
        has_conflict: bool = False,
        hard_veto_hit: bool = False,
        recent_backflow: bool = False,
        config: Optional[dict] = None
    ) -> PromotionResult:
        """
        评估晋升资格

        Args:
            unit_id: 对象ID
            current_zone: 当前区位（review/isolation/normal/error）
            current_scores: 当前轮评分
            stability_decision: 稳定门判决（如 move_normal_candidate）
            latest_tsla_action: 最近TSLA动作
            review_rounds: 已完成审查轮数
            has_conflict: 是否存在未解决冲突
            hard_veto_hit: 是否命中过硬否决
            recent_backflow: 最近是否回流
            config: 运行时配置

        Returns:
            PromotionResult: 晋升判决结果
        """
        cfg = config or self.config
        blockers = []
        threshold_checks = []

        # Step 1: 检查基本前提
        prerequisite_passed, prerequisite_blockers = self._check_prerequisites(
            current_zone, stability_decision, review_rounds, recent_backflow, cfg
        )
        blockers.extend(prerequisite_blockers)

        if not prerequisite_passed:
            decision = self._determine_block_decision(blockers, current_zone)
            return PromotionResult(
                unit_id=unit_id,
                current_zone=current_zone,
                decision=decision,
                reason=f"前提检查失败: {', '.join(blockers)}",
                confidence=0.7,
                stability_decision=stability_decision,
                threshold_checks=[],
                blockers=blockers,
                recommendations=self._generate_recommendations(decision, blockers, current_scores)
            )

        # Step 2: 检查阈值
        thresholds_passed, threshold_checks, threshold_blockers = self._check_thresholds(
            current_scores, cfg
        )
        blockers.extend(threshold_blockers)

        if not thresholds_passed:
            return PromotionResult(
                unit_id=unit_id,
                current_zone=current_zone,
                decision=PromotionDecision.BLOCKED_BY_THRESHOLD,
                reason=f"未达到正常区阈值: {', '.join(threshold_blockers)}",
                confidence=0.75,
                stability_decision=stability_decision,
                threshold_checks=threshold_checks,
                blockers=blockers,
                recommendations=self._generate_recommendations(
                    PromotionDecision.BLOCKED_BY_THRESHOLD, threshold_blockers, current_scores
                )
            )

        # Step 3: 检查一票否决项
        veto_passed, veto_blockers = self._check_blockers(
            hard_veto_hit, has_conflict, latest_tsla_action, cfg
        )
        blockers.extend(veto_blockers)

        if not veto_passed:
            decision = PromotionDecision.BLOCKED_BY_VETO if hard_veto_hit else PromotionDecision.BLOCKED_BY_CONFLICT
            return PromotionResult(
                unit_id=unit_id,
                current_zone=current_zone,
                decision=decision,
                reason=f"一票否决: {', '.join(veto_blockers)}",
                confidence=0.8,
                stability_decision=stability_decision,
                threshold_checks=threshold_checks,
                blockers=blockers,
                recommendations=self._generate_recommendations(decision, veto_blockers, current_scores)
            )

        # Step 4: 通过所有检查，允许晋升
        return PromotionResult(
            unit_id=unit_id,
            current_zone=current_zone,
            decision=PromotionDecision.PROMOTE_TO_NORMAL,
            reason="通过所有晋升检查，允许进入长期正常区",
            confidence=0.85,
            stability_decision=stability_decision,
            threshold_checks=threshold_checks,
            blockers=[],
            recommendations=["建议持续监控稳定性", "建议定期复查冲突状态"]
        )

    def _check_prerequisites(
        self,
        current_zone: str,
        stability_decision: str,
        review_rounds: int,
        recent_backflow: bool,
        config: dict
    ) -> tuple[bool, list[str]]:
        """
        检查基本前提
        """
        blockers = []

        # 检查当前区位是否允许晋升
        if current_zone not in ["review", "isolation"]:
            blockers.append(f"当前区位'{current_zone}'不允许晋升")

        # 检查稳定门是否输出正向候选
        if stability_decision != "move_normal_candidate":
            blockers.append(f"稳定门未通过({stability_decision})")

        # 检查审查轮数
        min_rounds = config.get("min_review_rounds", self.min_review_rounds)
        if review_rounds < min_rounds:
            blockers.append(f"审查轮数不足({review_rounds}/{min_rounds})")

        # 检查最近是否回流
        if recent_backflow:
            blockers.append("最近发生过回流")

        return len(blockers) == 0, blockers

    def _check_thresholds(
        self,
        current_scores: dict[str, float],
        config: dict
    ) -> tuple[bool, list[ThresholdCheck], list[str]]:
        """
        检查是否达到正常区阈值
        """
        normal_thresholds = config.get("normal_zone_thresholds", self.normal_thresholds)
        checks = []
        blockers = []

        for dimension, required in normal_thresholds.items():
            actual = current_scores.get(dimension, 0)
            gap = required - actual
            passed = actual >= required

            checks.append(ThresholdCheck(
                passed=passed,
                dimension=dimension,
                required=required,
                actual=actual,
                gap=gap
            ))

            if not passed:
                blockers.append(f"{dimension}不足({actual:.1f}/{required})")

        return len(blockers) == 0, checks, blockers

    def _check_blockers(
        self,
        hard_veto_hit: bool,
        has_conflict: bool,
        latest_tsla_action: str,
        config: dict
    ) -> tuple[bool, list[str]]:
        """
        检查一票否决项
        """
        blockers = []

        if hard_veto_hit:
            blockers.append("命中硬否决")

        if has_conflict:
            blockers.append("存在未解决冲突")

        if latest_tsla_action in ["review_backflow", "exclude", "error_archive"]:
            blockers.append(f"最近动作为{latest_tsla_action}")

        return len(blockers) == 0, blockers

    def _determine_block_decision(self, blockers: list[str], current_zone: str) -> PromotionDecision:
        """
        根据阻挡原因确定判决
        """
        if any("审查轮数不足" in b for b in blockers):
            return PromotionDecision.BLOCKED_BY_REVIEW
        if any("稳定门未通过" in b for b in blockers):
            return PromotionDecision.BLOCKED_BY_STABILITY
        if current_zone == "isolation":
            return PromotionDecision.STAY_ISOLATION
        return PromotionDecision.STAY_REVIEW

    def _generate_recommendations(
        self,
        decision: PromotionDecision,
        blockers: list[str],
        current_scores: dict[str, float]
    ) -> list[str]:
        """
        生成改进建议
        """
        recommendations = []

        if decision == PromotionDecision.BLOCKED_BY_THRESHOLD:
            # 找出最大短板
            if current_scores.get("Q", 0) < 72:
                gap = 72 - current_scores.get("Q", 0)
                recommendations.append(f"Q分数需提升{gap:.1f}分")
            if current_scores.get("S", 0) < 70:
                recommendations.append("建议增加稳定性验证")
            if current_scores.get("C", 0) < 75:
                recommendations.append("建议清理冲突")

        if decision == PromotionDecision.BLOCKED_BY_STABILITY:
            recommendations.append("建议观察多轮稳定性表现")
            recommendations.append("建议检查支撑来源独立性")

        if decision == PromotionDecision.BLOCKED_BY_CONFLICT:
            recommendations.append("建议优先解决未决冲突")

        if decision == PromotionDecision.BLOCKED_BY_REVIEW:
            recommendations.append("建议继续积累审查轮次")

        if decision == PromotionDecision.STAY_ISOLATION:
            recommendations.append("建议改善稳定性指标")
            recommendations.append("建议检查证据独立性")

        return recommendations

    def get_threshold_gap_analysis(
        self,
        current_scores: dict[str, float],
        target_zone: str = "normal"
    ) -> dict[str, Any]:
        """
        获取阈值差距分析

        用于诊断距离晋升还有多远
        """
        if target_zone == "normal":
            thresholds = self.normal_thresholds
        elif target_zone == "review":
            thresholds = self.review_thresholds
        else:
            return {"error": f"未知目标区域: {target_zone}"}

        analysis = {
            "target_zone": target_zone,
            "overall_ready": True,
            "dimensions": []
        }

        for dimension, required in thresholds.items():
            actual = current_scores.get(dimension, 0)
            gap = required - actual
            passed = actual >= required

            if not passed:
                analysis["overall_ready"] = False

            analysis["dimensions"].append({
                "dimension": dimension,
                "required": required,
                "actual": round(actual, 1),
                "gap": round(gap, 1),
                "passed": passed,
                "priority": "high" if gap > 10 else "medium" if gap > 5 else "low"
            })

        # 按差距排序
        analysis["dimensions"].sort(key=lambda x: x["gap"], reverse=True)

        # 找出最大短板
        if not analysis["overall_ready"]:
            failed = [d for d in analysis["dimensions"] if not d["passed"]]
            analysis["bottleneck"] = failed[0]["dimension"] if failed else None

        return analysis
