"""
Promotion Gate 基础自测

验证：
1. 前提检查（稳定门、审查轮数、区位）
2. 阈值检查（正常区门槛）
3. 一票否决（硬否决、冲突）
4. 晋升成功路径
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.promotion_gate import PromotionGate, PromotionDecision


def test_successful_promotion():
    """测试成功晋升"""
    print("\n" + "=" * 60)
    print("测试 1: 成功晋升到正常区")
    print("=" * 60)

    gate = PromotionGate(config={
        "min_review_rounds": 3,
        "normal_zone_thresholds": {
            "Q": 72, "T": 75, "S": 70, "E": 65, "C": 75, "L": 75
        }
    })

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

    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"阻挡项: {result.blockers}")

    passed = result.decision == PromotionDecision.PROMOTE_TO_NORMAL
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: promote_to_normal")

    return passed


def test_blocked_by_stability():
    """测试被稳定性阻挡"""
    print("\n" + "=" * 60)
    print("测试 2: 被稳定性阻挡")
    print("=" * 60)

    gate = PromotionGate()

    current_scores = {
        "Q": 75, "T": 78, "S": 73, "E": 70, "C": 80, "L": 80
    }

    result = gate.evaluate(
        unit_id="blocked_001",
        current_zone="review",
        current_scores=current_scores,
        stability_decision="stay_review",  # 稳定门未通过
        latest_tsla_action="keep",
        review_rounds=4,
        has_conflict=False,
        hard_veto_hit=False,
        recent_backflow=False
    )

    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"阻挡项: {result.blockers}")

    passed = result.decision == PromotionDecision.BLOCKED_BY_STABILITY
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: blocked_by_stability")

    return passed


def test_blocked_by_threshold():
    """测试被阈值阻挡"""
    print("\n" + "=" * 60)
    print("测试 3: 被阈值阻挡")
    print("=" * 60)

    gate = PromotionGate()

    # S分数不足
    current_scores = {
        "Q": 75, "T": 78, "S": 65, "E": 70, "C": 80, "L": 80
    }

    result = gate.evaluate(
        unit_id="threshold_001",
        current_zone="review",
        current_scores=current_scores,
        stability_decision="move_normal_candidate",
        latest_tsla_action="keep",
        review_rounds=4,
        has_conflict=False,
        hard_veto_hit=False,
        recent_backflow=False
    )

    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"阈值检查:")
    for tc in result.threshold_checks:
        status = "✓" if tc.passed else "✗"
        print(f"  {status} {tc.dimension}: {tc.actual:.1f}/{tc.required} (差距{tc.gap:+.1f})")

    passed = result.decision == PromotionDecision.BLOCKED_BY_THRESHOLD
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: blocked_by_threshold")

    return passed


def test_blocked_by_conflict():
    """测试被冲突阻挡"""
    print("\n" + "=" * 60)
    print("测试 4: 被冲突阻挡")
    print("=" * 60)

    gate = PromotionGate()

    current_scores = {
        "Q": 75, "T": 78, "S": 73, "E": 70, "C": 80, "L": 80
    }

    result = gate.evaluate(
        unit_id="conflict_001",
        current_zone="review",
        current_scores=current_scores,
        stability_decision="move_normal_candidate",
        latest_tsla_action="keep",
        review_rounds=4,
        has_conflict=True,  # 存在冲突
        hard_veto_hit=False,
        recent_backflow=False
    )

    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"阻挡项: {result.blockers}")
    print(f"建议: {result.recommendations}")

    passed = result.decision == PromotionDecision.BLOCKED_BY_CONFLICT
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: blocked_by_conflict")

    return passed


def test_isolation_to_normal():
    """测试从隔离区晋升"""
    print("\n" + "=" * 60)
    print("测试 5: 从隔离区晋升")
    print("=" * 60)

    gate = PromotionGate()

    current_scores = {
        "Q": 76, "T": 79, "S": 75, "E": 72, "C": 82, "L": 81
    }

    result = gate.evaluate(
        unit_id="isolation_001",
        current_zone="isolation",  # 从隔离区
        current_scores=current_scores,
        stability_decision="move_normal_candidate",
        latest_tsla_action="isolate",
        review_rounds=5,
        has_conflict=False,
        hard_veto_hit=False,
        recent_backflow=False
    )

    print(f"判决: {result.decision.value}")
    print(f"原因: {result.reason}")
    print(f"当前区位: {result.current_zone}")

    passed = result.decision == PromotionDecision.PROMOTE_TO_NORMAL
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: promote_to_normal (隔离区也可晋升)")

    return passed


def test_threshold_gap_analysis():
    """测试阈值差距分析"""
    print("\n" + "=" * 60)
    print("测试 6: 阈值差距分析")
    print("=" * 60)

    gate = PromotionGate()

    # 接近但未达标
    current_scores = {
        "Q": 70, "T": 73, "S": 68, "E": 64, "C": 73, "L": 74
    }

    analysis = gate.get_threshold_gap_analysis(current_scores, target_zone="normal")

    print(f"目标区域: {analysis['target_zone']}")
    print(f"整体就绪: {analysis['overall_ready']}")
    print(f"各维度检查:")
    for dim in analysis['dimensions']:
        status = "✓" if dim['passed'] else "✗"
        print(f"  {status} {dim['dimension']}: {dim['actual']}/{dim['required']} "
              f"(差距{dim['gap']:+.1f}, 优先级{dim['priority']})")

    print(f"最大短板: {analysis.get('bottleneck', '无')}")

    passed = (
        not analysis['overall_ready'] and
        analysis.get('bottleneck') is not None
    )
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 识别出未达标项和最大短板")

    return passed


def run_all_tests():
    """运行所有基础测试"""
    print("\n" + "🧪 " * 30)
    print("Promotion Gate 基础自测")
    print("🧪 " * 30)

    results = {
        "成功晋升": test_successful_promotion(),
        "被稳定性阻挡": test_blocked_by_stability(),
        "被阈值阻挡": test_blocked_by_threshold(),
        "被冲突阻挡": test_blocked_by_conflict(),
        "从隔离区晋升": test_isolation_to_normal(),
        "阈值差距分析": test_threshold_gap_analysis()
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 Promotion Gate 基础功能验证通过！")
        print("\n完成标志:")
        print("  ✓ 受审区对象可以进入正常区")
        print("  ✓ 隔离区对象在条件改善后也可以进入正常区")
        print("  ✓ 所有晋升都必须经过稳定门前置判断")
        print("  ✓ 晋升失败时能说明失败原因")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
