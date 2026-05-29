"""
Calibration Pipeline - 定标实验管道

第三阶段核心组件：
- 阈值扫描
- 参数敏感性分析
- 假阳性/假阴性统计
- 最优参数推荐
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gates.stability_gate import StabilityGate, StabilityState, StabilityDecision
from core.gates.promotion_gate import PromotionGate, PromotionDecision


@dataclass
class ThresholdConfig:
    """阈值配置"""
    stable_min_q: float = 70.0
    stable_min_s: float = 65.0
    max_q_std: float = 10.0
    max_conflict_rate: float = 0.2
    peak_drop_q: float = 15.0
    peak_drop_s: float = 15.0


@dataclass
class CalibrationResult:
    """定标结果"""
    config: ThresholdConfig
    total_cases: int = 0
    correct_predictions: int = 0
    false_positives: int = 0  # 假阳性：不该晋升但被晋升
    false_negatives: int = 0  # 假阴性：该晋升但被阻止
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    case_results: list[dict] = field(default_factory=list)


class CalibrationPipeline:
    """
    定标实验管道

    功能：
    1. 加载测试样本
    2. 批量阈值扫描
    3. 统计假阳性/假阴性
    4. 推荐最优参数组合
    """

    def __init__(self, cases_path: Optional[str] = None):
        self.cases_path = cases_path or Path(__file__).parent.parent.parent / "tests" / "fixtures" / "phase3_cases.json"
        self.cases = []
        self.results = []

    def load_cases(self) -> list[dict]:
        """加载测试样本"""
        with open(self.cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cases = data["cases"]
        return self.cases

    def run_single_config(self, config: ThresholdConfig) -> CalibrationResult:
        """运行单组阈值配置"""
        result = CalibrationResult(config=config)

        stability_gate = StabilityGate(config={
            "window_size": 3,
            "stable_min_q": config.stable_min_q,
            "stable_min_s": config.stable_min_s,
            "max_q_std": config.max_q_std,
            "max_conflict_rate": config.max_conflict_rate,
            "peak_drop_q": config.peak_drop_q,
            "peak_drop_s": config.peak_drop_s
        })

        promotion_gate = PromotionGate(config={
            "min_review_rounds": 3,
            "normal_zone_thresholds": {
                "Q": 72, "T": 75, "S": 70, "E": 65, "C": 75, "L": 75
            }
        })

        for case in self.cases:
            case_result = self._evaluate_case(case, stability_gate, promotion_gate)
            result.case_results.append(case_result)

            # 统计
            expected_promotion = case["expected"]["promotion_decision"] == "promote_to_normal"
            actual_promotion = case_result["promotion_decision"] == "promote_to_normal"

            if expected_promotion and actual_promotion:
                result.correct_predictions += 1
            elif not expected_promotion and not actual_promotion:
                result.correct_predictions += 1
            elif not expected_promotion and actual_promotion:
                result.false_positives += 1  # 假阳性
            else:
                result.false_negatives += 1  # 假阴性

        result.total_cases = len(self.cases)
        result.accuracy = result.correct_predictions / result.total_cases if result.total_cases > 0 else 0

        # 计算精确率和召回率
        true_positives = sum(1 for c in result.case_results
                           if c["expected_promotion"] and c["promotion_decision"] == "promote_to_normal")
        false_pos = result.false_positives
        false_neg = result.false_negatives

        result.precision = true_positives / (true_positives + false_pos) if (true_positives + false_pos) > 0 else 0
        result.recall = true_positives / (true_positives + false_neg) if (true_positives + false_neg) > 0 else 0

        if result.precision + result.recall > 0:
            result.f1_score = 2 * (result.precision * result.recall) / (result.precision + result.recall)

        return result

    def _evaluate_case(self, case: dict, stability_gate: StabilityGate,
                      promotion_gate: PromotionGate) -> dict:
        """评估单个用例"""
        current_scores = case["history_scores"][-1]
        history_events = case["history_events"]
        score_history = case["history_scores"]

        # 运行稳定门
        stability_result = stability_gate.evaluate(
            unit_id=case["case_id"],
            current_zone=case["current_zone"],
            current_scores=current_scores,
            history_events=history_events,
            score_history=score_history
        )

        # 运行晋升门
        latest_action = case["history_events"][-1]["action"]
        has_conflict = case["history_events"][-1].get("has_conflict", False)
        hard_veto = case["history_events"][-1].get("hard_veto_triggered", False)

        promotion_result = promotion_gate.evaluate(
            unit_id=case["case_id"],
            current_zone=case["current_zone"],
            current_scores=current_scores,
            stability_decision=stability_result.decision.value,
            latest_tsla_action=latest_action,
            review_rounds=case["review_rounds"],
            has_conflict=has_conflict,
            hard_veto_hit=hard_veto,
            recent_backflow=latest_action == "review_backflow"
        )

        expected_promotion = case["expected"]["promotion_decision"] == "promote_to_normal"

        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "stability_state": stability_result.stability_state.value,
            "stability_decision": stability_result.decision.value,
            "promotion_decision": promotion_result.decision.value,
            "expected_promotion": expected_promotion,
            "correct": (promotion_result.decision.value == "promote_to_normal") == expected_promotion
        }

    def grid_search(self, param_grid: dict[str, list]) -> list[CalibrationResult]:
        """
        网格搜索最优阈值

        Args:
            param_grid: 参数网格，如 {
                "stable_min_q": [68, 70, 72],
                "stable_min_s": [63, 65, 67],
                ...
            }
        """
        if not self.cases:
            self.load_cases()

        results = []

        # 生成所有参数组合
        import itertools
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        for combo in itertools.product(*values):
            config_dict = dict(zip(keys, combo))
            config = ThresholdConfig(**config_dict)

            result = self.run_single_config(config)
            results.append(result)

        # 按F1分数排序
        results.sort(key=lambda x: x.f1_score, reverse=True)
        self.results = results

        return results

    def get_best_config(self, metric: str = "f1_score") -> Optional[CalibrationResult]:
        """获取最优配置"""
        if not self.results:
            return None

        if metric == "accuracy":
            return max(self.results, key=lambda x: x.accuracy)
        elif metric == "precision":
            return max(self.results, key=lambda x: x.precision)
        elif metric == "recall":
            return max(self.results, key=lambda x: x.recall)
        else:
            return max(self.results, key=lambda x: x.f1_score)

    def export_results(self, output_path: str):
        """导出定标结果"""
        export_data = []
        for result in self.results:
            export_data.append({
                "config": {
                    "stable_min_q": result.config.stable_min_q,
                    "stable_min_s": result.config.stable_min_s,
                    "max_q_std": result.config.max_q_std,
                    "max_conflict_rate": result.config.max_conflict_rate,
                    "peak_drop_q": result.config.peak_drop_q,
                    "peak_drop_s": result.config.peak_drop_s
                },
                "metrics": {
                    "accuracy": round(result.accuracy, 4),
                    "precision": round(result.precision, 4),
                    "recall": round(result.recall, 4),
                    "f1_score": round(result.f1_score, 4),
                    "false_positives": result.false_positives,
                    "false_negatives": result.false_negatives
                }
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data

    def analyze_sensitivity(self, param_name: str, param_range: list) -> list[dict]:
        """分析单个参数的敏感性"""
        if not self.cases:
            self.load_cases()

        base_config = ThresholdConfig()
        results = []

        for value in param_range:
            config_dict = {
                "stable_min_q": base_config.stable_min_q,
                "stable_min_s": base_config.stable_min_s,
                "max_q_std": base_config.max_q_std,
                "max_conflict_rate": base_config.max_conflict_rate,
                "peak_drop_q": base_config.peak_drop_q,
                "peak_drop_s": base_config.peak_drop_s
            }
            config_dict[param_name] = value

            config = ThresholdConfig(**config_dict)
            result = self.run_single_config(config)

            results.append({
                "param_value": value,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "false_positives": result.false_positives,
                "false_negatives": result.false_negatives
            })

        return results
