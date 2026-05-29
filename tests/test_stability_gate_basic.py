"""
Stability Gate 基础自测

验证：
1. 历史窗口提取
2. 波动性计算
3. 稳定状态分类
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.stability_gate import StabilityGate, StabilityState, StabilityDecision


def test_stable_case():
    """测试稳定晋升样本"""
    print("\n" + "=" * 60)
    print("测试 1: 稳定晋升样本")
    print("=" * 60)

    gate = StabilityGate(config={
        "window_size": 3,
        "stable_min_q": 70,
        "stable_min_s": 65,
        "max_q_std": 10
    })

    # 稳定晋升样本
    score_history = [
        {"T": 74, "S": 68, "E": 66, "C": 78, "L": 80, "Q": 71},
        {"T": 76, "S": 71, "E": 67, "C": 79, "L": 80, "Q": 73},
        {"T": 77, "S": 73, "E": 69, "C": 81, "L": 82, "Q": 75}
    ]

    current_scores = {"T": 77, "S": 73, "E": 69, "C": 81, "L": 82, "Q": 75}
    history_events = [{"action": "keep"}, {"action": "keep"}, {"action": "keep"}]

    result = gate.evaluate(
        unit_id="stable_001",
        current_zone="review",
        current_scores=current_scores,
        history_events=history_events,
        score_history=score_history
    )

    print(f"稳定性状态: {result.stability_state.value}")
    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"Q标准差: {result.metrics.q_std:.2f}")
    print(f"S标准差: {result.metrics.s_std:.2f}")

    passed = (
        result.stability_state == StabilityState.STABLE and
        result.decision == StabilityDecision.MOVE_NORMAL_CANDIDATE
    )
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: stable + move_normal_candidate")

    return passed


def test_pseudo_stable_case():
    """测试假稳定样本"""
    print("\n" + "=" * 60)
    print("测试 2: 假稳定样本")
    print("=" * 60)

    gate = StabilityGate(config={
        "window_size": 3,
        "stable_min_q": 70,
        "stable_min_s": 65,
        "max_q_std": 10
    })

    # 假稳定样本 - 表面连续但C下降
    score_history = [
        {"T": 78, "S": 72, "E": 82, "C": 76, "L": 84, "Q": 76},
        {"T": 79, "S": 70, "E": 83, "C": 71, "L": 84, "Q": 75},
        {"T": 72, "S": 61, "E": 80, "C": 58, "L": 84, "Q": 67}
    ]

    current_scores = {"T": 72, "S": 61, "E": 80, "C": 58, "L": 84, "Q": 67}
    history_events = [
        {"action": "keep"},
        {"action": "keep"},
        {"action": "review_backflow", "has_conflict": True}
    ]

    result = gate.evaluate(
        unit_id="pseudo_001",
        current_zone="review",
        current_scores=current_scores,
        history_events=history_events,
        score_history=score_history
    )

    print(f"稳定性状态: {result.stability_state.value}")
    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"Q标准差: {result.metrics.q_std:.2f}")
    print(f"冲突率: {result.metrics.conflict_rate:.2f}")

    passed = result.stability_state in [StabilityState.PSEUDO_STABLE, StabilityState.UNSTABLE]
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: pseudo_stable 或 unstable（不应是 stable）")

    return passed


def test_high_e_low_s_case():
    """测试高证据低稳定样本"""
    print("\n" + "=" * 60)
    print("测试 3: 高证据低稳定样本")
    print("=" * 60)

    gate = StabilityGate(config={
        "window_size": 3,
        "stable_min_q": 70,
        "stable_min_s": 65,
        "max_q_std": 10
    })

    # 高证据低稳定样本
    score_history = [
        {"T": 81, "S": 50, "E": 90, "C": 73, "L": 86, "Q": 70},
        {"T": 83, "S": 54, "E": 91, "C": 74, "L": 86, "Q": 71},
        {"T": 82, "S": 48, "E": 89, "C": 72, "L": 86, "Q": 69}
    ]

    current_scores = {"T": 82, "S": 48, "E": 89, "C": 72, "L": 86, "Q": 69}
    history_events = [
        {"action": "isolate"},
        {"action": "isolate"},
        {"action": "isolate"}
    ]

    result = gate.evaluate(
        unit_id="high_e_low_s_001",
        current_zone="isolation",
        current_scores=current_scores,
        history_events=history_events,
        score_history=score_history
    )

    print(f"稳定性状态: {result.stability_state.value}")
    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"E分数: {current_scores['E']}")
    print(f"S分数: {current_scores['S']}")

    passed = result.decision == StabilityDecision.STAY_ISOLATION
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: stay_isolation")

    return passed


def test_peak_drop_case():
    """测试历史峰值下滑样本"""
    print("\n" + "=" * 60)
    print("测试 4: 历史峰值下滑样本")
    print("=" * 60)

    gate = StabilityGate(config={
        "window_size": 3,
        "peak_drop_q": 15,
        "peak_drop_s": 15
    })

    # 历史峰值下滑样本
    score_history = [
        {"T": 85, "S": 82, "E": 80, "C": 87, "L": 88, "Q": 84},
        {"T": 86, "S": 83, "E": 80, "C": 88, "L": 88, "Q": 85},
        {"T": 74, "S": 61, "E": 73, "C": 66, "L": 84, "Q": 68}
    ]

    current_scores = {"T": 74, "S": 61, "E": 73, "C": 66, "L": 84, "Q": 68}
    history_events = [
        {"action": "keep"},
        {"action": "keep"},
        {"action": "review_backflow"}
    ]

    result = gate.evaluate(
        unit_id="peak_drop_001",
        current_zone="normal",
        current_scores=current_scores,
        history_events=history_events,
        score_history=score_history
    )

    print(f"稳定性状态: {result.stability_state.value}")
    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"历史峰值Q: {result.snapshot.historical_peak.get('Q', 0):.1f}")
    print(f"当前Q: {current_scores['Q']}")

    passed = (
        result.stability_state == StabilityState.DEGRADING and
        result.decision in [StabilityDecision.DOWNGRADE, StabilityDecision.REVIEW_BACKFLOW]
    )
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: degrading + downgrade/review_backflow")

    return passed


def run_all_tests():
    """运行所有基础测试"""
    print("\n" + "🧪 " * 30)
    print("Stability Gate 基础自测")
    print("🧪 " * 30)

    results = {
        "稳定晋升样本": test_stable_case(),
        "假稳定样本": test_pseudo_stable_case(),
        "高证据低稳定样本": test_high_e_low_s_case(),
        "历史峰值下滑样本": test_peak_drop_case()
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 Stability Gate 基础功能验证通过！")
        print("\n完成标志:")
        print("  ✓ 能基于多轮历史输出稳定性状态")
        print("  ✓ 能识别 stable/pseudo_stable/unstable/degrading")
        print("  ✓ 能输出稳定门候选决策")
        print("  ✓ 判决依赖历史窗口而非单轮值")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
