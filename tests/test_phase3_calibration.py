"""
第三阶段测试 - 定标验证

测试类型：
A. 稳定晋升样本 - 确认能进入正常区
B. 假稳定样本 - 确认不会误晋升
C. 高证据低稳定样本 - 确认会被隔离
D. 历史峰值下滑样本 - 确认触发降级/回流

通过标准：
- 四类样本都能被正确分类
- 稳定晋升样本能进正常区
- 假稳定样本不会误晋升
- 高证据低稳定样本会被隔离
- 历史峰值下滑样本能触发降级/回流
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.stability_gate import StabilityGate, StabilityState, StabilityDecision
from core.gates.promotion_gate import PromotionGate, PromotionDecision
from core.unit.models import Unit


def load_phase3_cases() -> list[dict]:
    """加载第三阶段测试样本"""
    cases_path = Path(__file__).parent / "fixtures" / "phase3_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def create_unit_from_case(case: dict) -> Unit:
    """从测试用例创建 Unit"""
    unit_data = case["unit"]
    return Unit(
        script_form=unit_data["script_form"],
        core_meaning=unit_data["core_meaning"],
        truth_score=0.75,
        stability_score=0.70,
        evidence_score=0.70,
        conflict_cleanliness=0.80,
        legality_score=0.85,
        source_type=unit_data.get("source_type", "retrieved")
    )


def run_stability_gate(case: dict) -> dict:
    """运行稳定门测试"""
    gate = StabilityGate(config={
        "window_size": 3,
        "stable_min_q": 70,
        "stable_min_s": 65,
        "max_q_std": 10,
        "max_conflict_rate": 0.2,
        "peak_drop_q": 15,
        "peak_drop_s": 15
    })

    current_scores = case["history_scores"][-1]
    history_events = case["history_events"]
    score_history = case["history_scores"]

    result = gate.evaluate(
        unit_id=case["case_id"],
        current_zone=case["current_zone"],
        current_scores=current_scores,
        history_events=history_events,
        score_history=score_history
    )

    return {
        "state": result.stability_state.value,
        "decision": result.decision.value,
        "confidence": result.confidence,
        "reason": result.reason
    }


def run_promotion_gate(case: dict, stability_decision: str) -> dict:
    """运行晋升门测试"""
    gate = PromotionGate(config={
        "min_review_rounds": 3,
        "normal_zone_thresholds": {
            "Q": 72, "T": 75, "S": 70, "E": 65, "C": 75, "L": 75
        }
    })

    current_scores = case["history_scores"][-1]
    latest_action = case["history_events"][-1]["action"]
    has_conflict = case["history_events"][-1].get("has_conflict", False)
    hard_veto = case["history_events"][-1].get("hard_veto_triggered", False)

    result = gate.evaluate(
        unit_id=case["case_id"],
        current_zone=case["current_zone"],
        current_scores=current_scores,
        stability_decision=stability_decision,
        latest_tsla_action=latest_action,
        review_rounds=case["review_rounds"],
        has_conflict=has_conflict,
        hard_veto_hit=hard_veto,
        recent_backflow=latest_action == "review_backflow"
    )

    return {
        "decision": result.decision.value,
        "confidence": result.confidence,
        "reason": result.reason,
        "blockers": result.blockers
    }


def test_stable_promotion_cases():
    """A. 稳定晋升样本测试"""
    print("\n" + "=" * 60)
    print("A. 稳定晋升样本测试")
    print("=" * 60)

    cases = load_phase3_cases()
    stable_cases = [c for c in cases if c["category"] == "稳定晋升样本"]

    results = []
    for case in stable_cases:
        print(f"\n  测试: {case['case_id']}")
        print(f"  描述: {case['description']}")

        # 运行稳定门
        stability_result = run_stability_gate(case)
        print(f"  稳定门状态: {stability_result['state']}")
        print(f"  稳定门判决: {stability_result['decision']}")

        # 运行晋升门
        promotion_result = run_promotion_gate(case, stability_result['decision'])
        print(f"  晋升门判决: {promotion_result['decision']}")

        # 验证
        expected = case["expected"]
        state_ok = stability_result['state'] == expected['stability_state']
        stability_decision_ok = stability_result['decision'] == expected['stability_decision']
        promotion_decision_ok = promotion_result['decision'] == expected['promotion_decision']

        passed = state_ok and stability_decision_ok and promotion_decision_ok
        results.append(passed)

        status = "✅" if passed else "❌"
        print(f"  结果: {status}")

    all_passed = all(results)
    print(f"\n  稳定晋升样本: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
    return all_passed


def test_pseudo_stable_cases():
    """B. 假稳定样本测试"""
    print("\n" + "=" * 60)
    print("B. 假稳定样本测试")
    print("=" * 60)

    cases = load_phase3_cases()
    pseudo_cases = [c for c in cases if c["category"] == "假稳定样本"]

    results = []
    for case in pseudo_cases:
        print(f"\n  测试: {case['case_id']}")
        print(f"  描述: {case['description']}")

        stability_result = run_stability_gate(case)
        print(f"  稳定门状态: {stability_result['state']}")
        print(f"  稳定门判决: {stability_result['decision']}")

        promotion_result = run_promotion_gate(case, stability_result['decision'])
        print(f"  晋升门判决: {promotion_result['decision']}")

        # 验证
        expected = case["expected"]
        state_ok = stability_result['state'] == expected['stability_state']
        stability_decision_ok = stability_result['decision'] == expected['stability_decision']
        promotion_decision_ok = promotion_result['decision'] == expected['promotion_decision']

        passed = state_ok and stability_decision_ok and promotion_decision_ok
        results.append(passed)

        status = "✅" if passed else "❌"
        print(f"  结果: {status}")

    all_passed = all(results)
    print(f"\n  假稳定样本: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
    return all_passed


def test_high_e_low_s_cases():
    """C. 高证据低稳定样本测试"""
    print("\n" + "=" * 60)
    print("C. 高证据低稳定样本测试")
    print("=" * 60)

    cases = load_phase3_cases()
    high_e_cases = [c for c in cases if c["category"] == "高证据低稳定样本"]

    results = []
    for case in high_e_cases:
        print(f"\n  测试: {case['case_id']}")
        print(f"  描述: {case['description']}")

        stability_result = run_stability_gate(case)
        print(f"  稳定门状态: {stability_result['state']}")
        print(f"  稳定门判决: {stability_result['decision']}")

        # 验证
        expected = case["expected"]
        state_ok = stability_result['state'] == expected['stability_state']
        decision_ok = stability_result['decision'] == expected['stability_decision']

        passed = state_ok and decision_ok
        results.append(passed)

        status = "✅" if passed else "❌"
        print(f"  结果: {status}")

    all_passed = all(results)
    print(f"\n  高证据低稳定样本: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
    return all_passed


def test_peak_drop_cases():
    """D. 历史峰值下滑样本测试"""
    print("\n" + "=" * 60)
    print("D. 历史峰值下滑样本测试")
    print("=" * 60)

    cases = load_phase3_cases()
    peak_drop_cases = [c for c in cases if c["category"] == "历史峰值下滑样本"]

    results = []
    for case in peak_drop_cases:
        print(f"\n  测试: {case['case_id']}")
        print(f"  描述: {case['description']}")

        stability_result = run_stability_gate(case)
        print(f"  稳定门状态: {stability_result['state']}")
        print(f"  稳定门判决: {stability_result['decision']}")

        # 验证：应该触发降级或回流
        expected = case["expected"]
        is_degrading = stability_result['state'] == 'degrading'
        has_downgrade_action = stability_result['decision'] in ['downgrade', 'review_backflow']

        passed = is_degrading and has_downgrade_action
        results.append(passed)

        status = "✅" if passed else "❌"
        print(f"  结果: {status} (下滑状态: {is_degrading}, 降级动作: {has_downgrade_action})")

    all_passed = all(results)
    print(f"\n  历史峰值下滑样本: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
    return all_passed


def run_all_tests():
    """运行所有第三阶段测试"""
    print("\n" + "🧪 " * 30)
    print("第三阶段定标测试套件")
    print("🧪 " * 30)

    results = {
        "稳定晋升样本": test_stable_promotion_cases(),
        "假稳定样本": test_pseudo_stable_cases(),
        "高证据低稳定样本": test_high_e_low_s_cases(),
        "历史峰值下滑样本": test_peak_drop_cases()
    }

    print("\n" + "=" * 60)
    print("第三阶段测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 第三阶段定标测试全部通过！")
        print("\n完成标志:")
        print("  ✓ 稳定晋升样本能进入正常区")
        print("  ✓ 假稳定样本不会误晋升")
        print("  ✓ 高证据低稳定样本会被隔离")
        print("  ✓ 历史峰值下滑样本能触发降级/回流")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现或调整阈值。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
