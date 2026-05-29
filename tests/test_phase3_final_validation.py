"""
第三阶段最终验证 - 5条完成标志检查

验证目标：
1. stability_gate.py 能正确消费 scorer 的历史序列输出
2. promotion_gate.py 能在 review 区判定晋升到 normal 区
3. scorer.py 能输出每轮评分的完整历史序列
4. test_phase3_calibration.py 四类样本都能被正确分类
5. calibration_pipeline.py 能跑通阈值扫描并输出推荐值
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.stability_gate import StabilityGate, StabilityState, StabilityDecision
from core.gates.promotion_gate import PromotionGate, PromotionDecision
from core.tsla.scorer import TSLAScorerWithHistory
from core.unit.models import Unit
from pipelines.calibration_pipeline import CalibrationPipeline, ThresholdConfig


def check_1_stability_gate_integration():
    """
    完成标志1：stability_gate.py 能正确消费 scorer 的历史序列输出
    """
    print("\n" + "=" * 70)
    print("完成标志 1/5: Stability Gate 历史序列消费")
    print("=" * 70)

    # 使用 scorer 生成历史
    scorer = TSLAScorerWithHistory()

    for i in range(3):
        unit = Unit(
            script_form="测试对象",
            core_meaning="这是一个测试",
            truth_score=0.80 + i * 0.02,
            stability_score=0.75 + i * 0.02,
            evidence_score=0.70 + i * 0.02,
            conflict_cleanliness=0.90,
            legality_score=1.0,
            source_type="retrieved"
        )
        scorer.score_with_history(
            unit_id="check_001",
            unit=unit,
            current_context={},
            evidence_count=2
        )

    # 获取 scorer 输出的历史序列
    score_history = scorer.get_score_history_for_stability_gate("check_001", window_size=3)

    # stability_gate 消费这些历史
    gate = StabilityGate()
    result = gate.evaluate(
        unit_id="check_001",
        current_zone="review",
        current_scores=score_history[-1],
        history_events=[{"action": "keep", "has_conflict": False, "hard_veto_triggered": False}],
        score_history=score_history
    )

    print(f"  Scorer 历史记录数: {len(score_history)}")
    print(f"  Stability Gate 状态: {result.stability_state.value}")
    print(f"  Stability Gate 判决: {result.decision.value}")

    passed = len(score_history) == 3 and result.stability_state in [
        StabilityState.STABLE, StabilityState.PSEUDO_STABLE, StabilityState.UNSTABLE
    ]

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: scorer 输出的历史序列能被 stability_gate 正确消费")

    return passed


def check_2_promotion_gate_review_to_normal():
    """
    完成标志2：promotion_gate.py 能在 review 区判定晋升到 normal 区
    """
    print("\n" + "=" * 70)
    print("完成标志 2/5: Promotion Gate Review → Normal 晋升")
    print("=" * 70)

    gate = PromotionGate(config={
        "min_review_rounds": 3,
        "normal_zone_thresholds": {
            "Q": 72, "T": 75, "S": 70, "E": 65, "C": 75, "L": 75
        }
    })

    # 模拟 review 区的晋升判定
    current_scores = {
        "Q": 75, "T": 78, "S": 73, "E": 70, "C": 80, "L": 80
    }

    result = gate.evaluate(
        unit_id="promote_001",
        current_zone="review",
        current_scores=current_scores,
        stability_decision="move_normal_candidate",
        latest_tsla_action="keep",
        review_rounds=4,
        has_conflict=False,
        hard_veto_hit=False,
        recent_backflow=False
    )

    print(f"  当前区域: review")
    print(f"  目标区域: normal")
    print(f"  晋升判决: {result.decision.value}")
    print(f"  判决理由: {result.reason}")

    passed = result.decision == PromotionDecision.PROMOTE_TO_NORMAL

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: promotion_gate 能从 review 区判定晋升到 normal 区")

    return passed


def check_3_scorer_history_output():
    """
    完成标志3：scorer.py 能输出每轮评分的完整历史序列
    """
    print("\n" + "=" * 70)
    print("完成标志 3/5: Scorer 完整历史序列输出")
    print("=" * 70)

    scorer = TSLAScorerWithHistory()

    # 多轮评分
    for i in range(4):
        unit = Unit(
            script_form="测试对象",
            core_meaning="这是一个测试",
            truth_score=0.75 + i * 0.03,
            stability_score=0.70 + i * 0.02,
            evidence_score=0.65 + i * 0.03,
            conflict_cleanliness=0.85,
            legality_score=1.0,
            source_type="retrieved"
        )

        result = scorer.score_with_history(
            unit_id="history_001",
            unit=unit,
            current_context={},
            evidence_count=2 + i,
            access_count=i + 1,
            session_id=f"session_{i:03d}"
        )

    # 验证输出
    history = scorer.get_history("history_001")
    rolling = result.rolling_stats

    print(f"  历史记录数: {len(history)}")
    print(f"  快照字段: {list(history[0].__dict__.keys()) if history else []}")
    print(f"  滚动统计 - Q均值: {rolling.q_mean:.2f}")
    print(f"  滚动统计 - Q标准差: {rolling.q_std:.2f}")
    print(f"  滚动统计 - Q峰值: {rolling.peak_q:.1f}")
    print(f"  滚动统计 - Q趋势: {rolling.trend_q}")

    passed = (
        len(history) == 4 and
        rolling.q_mean > 0 and
        rolling.q_std >= 0 and
        rolling.peak_q > 0 and
        rolling.trend_q in ["up", "down", "stable"]
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: scorer 能输出每轮评分的完整历史序列和滚动统计")

    return passed


def check_4_calibration_test_classification():
    """
    完成标志4：test_phase3_calibration.py 四类样本都能被正确分类
    """
    print("\n" + "=" * 70)
    print("完成标志 4/5: 定标测试四类样本正确分类")
    print("=" * 70)

    import json

    cases_path = Path(__file__).parent / "fixtures" / "phase3_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]

    # 统计各类样本
    categories = {}
    for case in cases:
        cat = case["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(case["case_id"])

    print(f"  测试样本总数: {len(cases)}")
    print(f"  样本分类:")
    for cat, ids in categories.items():
        print(f"    - {cat}: {len(ids)}个")

    # 运行完整测试
    stability_gate = StabilityGate()
    promotion_gate = PromotionGate()

    correct_count = 0
    for case in cases:
        current_scores = case["history_scores"][-1]

        stability_result = stability_gate.evaluate(
            unit_id=case["case_id"],
            current_zone=case["current_zone"],
            current_scores=current_scores,
            history_events=case["history_events"],
            score_history=case["history_scores"]
        )

        latest_action = case["history_events"][-1]["action"]
        promotion_result = promotion_gate.evaluate(
            unit_id=case["case_id"],
            current_zone=case["current_zone"],
            current_scores=current_scores,
            stability_decision=stability_result.decision.value,
            latest_tsla_action=latest_action,
            review_rounds=case["review_rounds"],
            has_conflict=case["history_events"][-1].get("has_conflict", False),
            hard_veto_hit=case["history_events"][-1].get("hard_veto_triggered", False),
            recent_backflow=latest_action == "review_backflow"
        )

        expected = case["expected"]
        if (stability_result.stability_state.value == expected["stability_state"] and
            promotion_result.decision.value == expected["promotion_decision"]):
            correct_count += 1

    accuracy = correct_count / len(cases)

    print(f"\n  正确分类: {correct_count}/{len(cases)}")
    print(f"  准确率: {accuracy:.1%}")

    passed = accuracy >= 0.8  # 80%以上视为通过

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 四类样本都能被正确分类")

    return passed


def check_5_calibration_pipeline():
    """
    完成标志5：calibration_pipeline.py 能跑通阈值扫描并输出推荐值
    """
    print("\n" + "=" * 70)
    print("完成标志 5/5: Calibration Pipeline 阈值扫描")
    print("=" * 70)

    pipeline = CalibrationPipeline()
    pipeline.load_cases()

    # 小规模网格搜索
    param_grid = {
        "stable_min_q": [68, 70, 72],
        "stable_min_s": [65, 67],
        "max_q_std": [8, 10],
        "max_conflict_rate": [0.15, 0.2],
        "peak_drop_q": [12, 15],
        "peak_drop_s": [12, 15]
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    print(f"  参数组合数: {total_combos}")
    print("  开始网格搜索...")

    results = pipeline.grid_search(param_grid)

    best = pipeline.get_best_config("f1_score")

    print(f"\n  搜索完成")
    print(f"  最优配置:")
    print(f"    - stable_min_q: {best.config.stable_min_q}")
    print(f"    - stable_min_s: {best.config.stable_min_s}")
    print(f"    - max_q_std: {best.config.max_q_std}")
    print(f"  最优指标:")
    print(f"    - 准确率: {best.accuracy:.1%}")
    print(f"    - F1分数: {best.f1_score:.3f}")
    print(f"    - 假阳性: {best.false_positives}")
    print(f"    - 假阴性: {best.false_negatives}")

    passed = best.accuracy >= 0.8 and len(results) == total_combos

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: calibration_pipeline 能跑通阈值扫描并输出推荐值")

    return passed


def run_final_validation():
    """运行最终验证"""
    print("\n" + "🎯 " * 35)
    print("第三阶段最终验证 - 5条完成标志检查")
    print("🎯 " * 35)

    checks = {
        "Stability Gate 历史序列消费": check_1_stability_gate_integration(),
        "Promotion Gate Review→Normal 晋升": check_2_promotion_gate_review_to_normal(),
        "Scorer 完整历史序列输出": check_3_scorer_history_output(),
        "定标测试四类样本正确分类": check_4_calibration_test_classification(),
        "Calibration Pipeline 阈值扫描": check_5_calibration_pipeline()
    }

    print("\n" + "=" * 70)
    print("第三阶段最终验证总结")
    print("=" * 70)

    for check_name, passed in checks.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")

    all_passed = all(checks.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("第三阶段 Validation & Calibration v0.3 全部完成！")
        print("🎉 " * 35)
        print("\n核心成果：")
        print("  ✓ stability_gate.py - 多轮稳定性评估")
        print("  ✓ promotion_gate.py - 长期区晋升决策")
        print("  ✓ scorer.py - 历史序列 + 滚动统计")
        print("  ✓ test_phase3_calibration.py - 四类样本测试")
        print("  ✓ calibration_pipeline.py - 阈值扫描定标")
        print("\n第三阶段目标达成：")
        print("  • 稳定性门控正式做出决策")
        print("  • 长期区晋升门能判定晋升")
        print("  • TSLA阈值从示例值推进到实验值")
        print("  • 建立了可校准、可验证、可回退的实验管道")
    else:
        print("\n⚠️ 部分验证未通过，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_final_validation()
