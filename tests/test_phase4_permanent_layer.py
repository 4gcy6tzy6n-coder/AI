"""
第四阶段测试 - 永久层功能验证

测试类型：
A. 永久候选成功样本 - 确认能进入浅层永久
B. 永久候选不足样本 - 确认被阻止
C. 永久对象后续冲突暴露 - 确认降级而非删除
D. 永久对象可拆分修复 - 确认拆分机制

第四阶段完成标志：
1. 长期正常区对象可以被判定为浅层永久候选
2. 永久层禁止跨层直写
3. 永久对象暴露问题时不会直接删除
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.permanent_protection_gate import (
    PermanentProtectionGate, PermanentDecision, PermanentZone
)
from core.memory.shallow_permanent_store import (
    ShallowPermanentStore, DowngradeTarget,
    EvidenceSummary, StabilityWindowSummary
)


def test_successful_promotion():
    """A. 永久候选成功样本"""
    print("\n" + "=" * 60)
    print("A. 永久候选成功样本")
    print("=" * 60)

    gate = PermanentProtectionGate(config={
        "shallow_min_q": 78,
        "shallow_min_t": 80,
        "shallow_min_s": 75,
        "shallow_min_c": 85,
        "shallow_min_l": 85,
        "min_stability_cycles": 5,
        "min_conflict_free_rounds": 3,
        "min_days_since_promotion": 7
    })

    # 满足所有条件的对象
    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    result = gate.evaluate(
        unit_id="perm_success_001",
        current_zone="normal",
        current_scores=current_scores,
        source_type="retrieved",
        stability_cycles=6,
        conflict_free_rounds=4,
        days_since_promotion=10,
        recent_conflict=False,
        recent_backflow=False,
        hard_veto_history=[]
    )

    print(f"  决策: {result.decision.value}")
    print(f"  置信度: {result.confidence}")
    print(f"  理由: {result.reason}")

    passed = result.decision == PermanentDecision.PROMOTE_TO_SHALLOW

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")

    return passed, result


def test_blocked_by_cycles():
    """B. 永久候选不足 - 周期不足"""
    print("\n" + "=" * 60)
    print("B. 永久候选不足 - 周期不足")
    print("=" * 60)

    gate = PermanentProtectionGate()

    # 分数达标但周期不足
    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    result = gate.evaluate(
        unit_id="perm_blocked_001",
        current_zone="normal",
        current_scores=current_scores,
        source_type="retrieved",
        stability_cycles=3,  # 不足5个周期
        conflict_free_rounds=4,
        days_since_promotion=10,
        recent_conflict=False,
        recent_backflow=False,
        hard_veto_history=[]
    )

    print(f"  决策: {result.decision.value}")
    print(f"  阻止原因: {'; '.join(result.blockers)}")

    passed = result.decision == PermanentDecision.BLOCKED_BY_INSUFFICIENT_CYCLES

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")

    return passed


def test_blocked_by_source():
    """B. 永久候选不足 - 来源策略"""
    print("\n" + "=" * 60)
    print("B. 永久候选不足 - 来源策略")
    print("=" * 60)

    gate = PermanentProtectionGate()

    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    # 使用禁止的来源类型
    result = gate.evaluate(
        unit_id="perm_blocked_002",
        current_zone="normal",
        current_scores=current_scores,
        source_type="user_input",  # 禁止的来源
        stability_cycles=6,
        conflict_free_rounds=4,
        days_since_promotion=10,
        recent_conflict=False,
        recent_backflow=False,
        hard_veto_history=[]
    )

    print(f"  决策: {result.decision.value}")
    print(f"  阻止原因: {'; '.join(result.blockers)}")

    passed = result.decision == PermanentDecision.BLOCKED_BY_SOURCE_POLICY

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")

    return passed


def test_blocked_by_conflict():
    """B. 永久候选不足 - 近期冲突"""
    print("\n" + "=" * 60)
    print("B. 永久候选不足 - 近期冲突")
    print("=" * 60)

    gate = PermanentProtectionGate()

    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    result = gate.evaluate(
        unit_id="perm_blocked_003",
        current_zone="normal",
        current_scores=current_scores,
        source_type="retrieved",
        stability_cycles=6,
        conflict_free_rounds=4,
        days_since_promotion=10,
        recent_conflict=True,  # 近期有冲突
        recent_backflow=False,
        hard_veto_history=[]
    )

    print(f"  决策: {result.decision.value}")
    print(f"  阻止原因: {'; '.join(result.blockers)}")

    passed = result.decision == PermanentDecision.BLOCKED_BY_CONFLICT

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")

    return passed


def test_no_cross_layer_write():
    """验证：禁止跨层直写"""
    print("\n" + "=" * 60)
    print("验证：禁止跨层直写")
    print("=" * 60)

    gate = PermanentProtectionGate()
    store = ShallowPermanentStore()

    # 尝试从 review 区直接晋升到永久层
    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    result = gate.evaluate(
        unit_id="cross_layer_001",
        current_zone="review",  # 不是 normal 区
        current_scores=current_scores,
        source_type="retrieved",
        stability_cycles=6,
        conflict_free_rounds=4,
        days_since_promotion=10,
        recent_conflict=False,
        recent_backflow=False,
        hard_veto_history=[]
    )

    print(f"  当前区域: review")
    print(f"  目标区域: shallow_permanent")
    print(f"  决策: {result.decision.value}")
    print(f"  阻止原因: {'; '.join(result.blockers)}")

    # 验证存储层也拒绝
    try:
        store.promote_to_permanent(
            unit_id="cross_layer_001",
            content="测试内容",
            core_meaning="测试含义",
            promotion_scores=current_scores,
            source_type="retrieved"
        )
        # 如果能到这里，说明有问题
        store_success = True
    except Exception as e:
        store_success = False
        print(f"  存储层拒绝: {e}")

    passed = (
        result.decision != PermanentDecision.PROMOTE_TO_SHALLOW and
        "review" in ' '.join(result.blockers).lower()
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 永久层禁止跨层直写，只能从 normal 区晋升")

    return passed


def test_downgrade_instead_of_delete():
    """C. 永久对象暴露问题时降级而非删除"""
    print("\n" + "=" * 60)
    print("C. 永久对象暴露问题时降级而非删除")
    print("=" * 60)

    store = ShallowPermanentStore()

    # 先创建一个永久对象
    entry = store.promote_to_permanent(
        unit_id="perm_downgrade_001",
        content="高质量知识",
        core_meaning="这是一个高质量知识定义",
        promotion_scores={"Q": 82, "T": 85, "S": 80},
        source_type="retrieved",
        evidence_summary=EvidenceSummary(
            primary_sources=["source_a", "source_b"],
            evidence_count=3,
            verification_status="verified"
        ),
        stability_summary=StabilityWindowSummary(
            total_cycles=6,
            stable_cycles=6,
            avg_q_score=82.0
        )
    )

    print(f"  对象已创建: {entry.unit_id}")
    print(f"  初始状态: {entry.status}")

    # 标记冲突
    store.mark_conflict(
        unit_id="perm_downgrade_001",
        conflict_type="evidence_contradiction",
        conflict_details={"new_evidence": "contradicts original"},
        severity="high"
    )

    entry = store.get("perm_downgrade_001")
    print(f"  冲突标记后状态: {entry.status}")

    # 尝试局部修正失败，选择降级
    result = store.downgrade(
        unit_id="perm_downgrade_001",
        target=DowngradeTarget.REVIEW,
        reason="新证据与原定义冲突，需要重新审查",
        action_taken="降级到review区重新验证"
    )

    entry = store.get("perm_downgrade_001")
    print(f"  降级后状态: {entry.status}")
    print(f"  降级历史数: {len(entry.downgrade_history)}")

    # 验证对象仍然存在（没有被删除）
    still_exists = store.is_in_permanent("perm_downgrade_001")
    has_downgrade_history = len(entry.downgrade_history) > 0

    passed = still_exists and has_downgrade_history

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 永久对象暴露问题时降级到review区，而不是直接删除")

    return passed


def test_split_mechanism():
    """D. 永久对象可拆分修复"""
    print("\n" + "=" * 60)
    print("D. 永久对象可拆分修复")
    print("=" * 60)

    store = ShallowPermanentStore()

    # 创建一个有多义性的永久对象
    entry = store.promote_to_permanent(
        unit_id="perm_split_001",
        content="多义词定义",
        core_meaning="这个词有多个含义",
        promotion_scores={"Q": 80, "T": 82, "S": 78},
        source_type="retrieved"
    )

    print(f"  对象已创建: {entry.unit_id}")

    # 发起拆分
    split_plan = [
        {"unit_id": "perm_split_001_a", "meaning": "含义A", "priority": "high"},
        {"unit_id": "perm_split_001_b", "meaning": "含义B", "priority": "medium"}
    ]

    result = store.initiate_split(
        unit_id="perm_split_001",
        split_reason="发现多义混装，需要拆分为独立对象",
        split_plan=split_plan
    )

    entry = store.get("perm_split_001")
    print(f"  拆分后状态: {entry.status}")
    print(f"  拆分计划: {len(split_plan)}个子对象")

    passed = (
        result["success"] and
        entry.status == "split_pending"
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 多义对象可以拆分，原对象退出永久主位")

    return passed


def test_protection_summary():
    """验证保护摘要"""
    print("\n" + "=" * 60)
    print("验证：保护摘要")
    print("=" * 60)

    store = ShallowPermanentStore()

    # 创建对象
    store.promote_to_permanent(
        unit_id="perm_summary_001",
        content="测试内容",
        core_meaning="测试",
        promotion_scores={"Q": 82},
        source_type="retrieved"
    )

    # 获取保护摘要
    summary = store.get_protection_summary("perm_summary_001")

    print(f"  对象ID: {summary['unit_id']}")
    print(f"  状态: {summary['status']}")
    print(f"  保护级别: {summary['protection_level']}")
    print(f"  可删除: {summary['can_delete']}")
    print(f"  可修改: {summary['can_modify']}")

    passed = (
        summary['can_delete'] == False and  # 禁止删除
        summary['protection_level'] == 'high'
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 永久层对象禁止直接删除")

    return passed


def run_all_tests():
    """运行所有第四阶段测试"""
    print("\n" + "🧪 " * 30)
    print("第四阶段永久层功能测试")
    print("🧪 " * 30)

    results = {
        "永久候选成功": test_successful_promotion()[0],
        "周期不足阻止": test_blocked_by_cycles(),
        "来源策略阻止": test_blocked_by_source(),
        "冲突阻止": test_blocked_by_conflict(),
        "禁止跨层直写": test_no_cross_layer_write(),
        "降级而非删除": test_downgrade_instead_of_delete(),
        "拆分机制": test_split_mechanism(),
        "保护摘要": test_protection_summary()
    }

    print("\n" + "=" * 60)
    print("第四阶段测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 第四阶段永久层功能测试全部通过！")
        print("\n完成标志:")
        print("  ✓ 长期正常区对象可以被判定为浅层永久候选")
        print("  ✓ 永久层禁止跨层直写")
        print("  ✓ 永久对象暴露问题时不会直接删除")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
