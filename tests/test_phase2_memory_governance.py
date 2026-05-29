"""
第二阶段测试 - 记忆治理与跨轮次回放

新增测试类型：
1. 重复出现测试 - 同一对象多次出现
2. 后续冲突暴露测试 - 第一次合理，第二次加入反证
3. 多轮解释稳定性测试 - 同一对象换说法后是否一致
4. 错误归档测试 - 已确认错误但有学习价值
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.write_gate import WriteGate
from core.memory.memory_events import MemoryEventLog, MemoryZone
from core.memory.transient_store import TransientStore
from core.memory.zone_manager import ZoneManager
from core.tsla.action_router import ActionRouter, TSLAActionType
from core.tsla.scorer import TSLAScorer
from core.unit.models import Unit


def test_repeated_appearance():
    """1. 重复出现测试"""
    print("\n" + "=" * 60)
    print("测试 1: 重复出现测试")
    print("=" * 60)

    # 初始化组件
    event_log = MemoryEventLog()
    transient_store = TransientStore(event_log=event_log)
    zone_manager = ZoneManager(event_log=event_log)
    write_gate = WriteGate(event_log=event_log)
    scorer = TSLAScorer()

    # 创建一个高质量 Unit
    unit = Unit(
        script_form="苹果",
        core_meaning="一种常见的水果",
        truth_score=0.85,
        stability_score=0.80,
        evidence_score=0.75,
        legality_score=1.0,
        conflict_cleanliness=1.0,
        source_type="retrieved"  # 明确来源类型
    )

    # 模拟多次出现
    session_ids = ["session_001", "session_002", "session_003", "session_004", "session_005"]

    for i, session_id in enumerate(session_ids):
        # 写入瞬时层
        transient_store.write(unit, session_id)

        # 模拟访问
        for _ in range(i + 1):  # 递增访问次数
            transient_store.read(unit.unit_id)

        # 计算评分
        access_count = event_log.get_access_count(unit.unit_id)
        scores = scorer.calculate(
            unit,
            gap_detected=False,
            evidence_count=2,
            access_count=access_count
        )

        # 评估是否可进入长期层
        decision = write_gate.evaluate(unit, scores.Q, session_id)

        if decision.allowed:
            zone_manager.move_to_zone(
                unit,
                MemoryZone.LONG_TERM_REVIEW,
                reason=f"第{i+1}次出现，评分达标",
                session_id=session_id
            )

    # 验证结果
    location = zone_manager.get_unit_location(unit.unit_id)
    history = zone_manager.get_unit_history_summary(unit.unit_id)

    print(f"\nUnit '{unit.script_form}' 最终位置: {location[0].value if location else '未迁移'}")
    print(f"访问次数: {history['access_count']}")
    print(f"事件历史数: {history['total_events']}")
    print(f"区域迁移: {len(history['zone_transitions'])}")

    # 验证：应该进入长期受审区
    passed = location is not None and location[0] == MemoryZone.LONG_TERM_REVIEW
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 多次出现后应进入长期受审区")

    return passed


def test_conflict_exposure():
    """2. 后续冲突暴露测试"""
    print("\n" + "=" * 60)
    print("测试 2: 后续冲突暴露测试")
    print("=" * 60)

    event_log = MemoryEventLog()
    zone_manager = ZoneManager(event_log=event_log)
    action_router = ActionRouter()
    scorer = TSLAScorer()

    # 第一轮：看起来合理（高质量 Unit）
    unit_v1 = Unit(
        script_form="某概念",
        core_meaning="定义A",
        truth_score=0.85,
        stability_score=0.80,
        evidence_score=0.80,
        conflict_cleanliness=1.0,
        legality_score=1.0,
        source_type="retrieved"
    )

    scores_v1 = scorer.calculate(unit_v1, gap_detected=False, evidence_count=2)
    decision_v1 = action_router.route(
        unit_v1, scores_v1.Q,
        gap_detected=False,
        conflict_detected=False
    )

    print(f"\n第一轮:")
    print(f"  动作: {decision_v1.action.value}")
    print(f"  目标区域: {decision_v1.target_zone.value}")

    # 进入长期受审区
    zone_manager.move_to_zone(unit_v1, MemoryZone.LONG_TERM_REVIEW, "第一轮评估")

    # 第二轮：加入反证（冲突暴露）
    unit_v2 = Unit(
        script_form="某概念",
        core_meaning="定义B（与A冲突）",
        truth_score=0.70,
        stability_score=0.65,
        evidence_score=0.60,
        conflict_cleanliness=0.3,  # 冲突清洁度下降
        legality_score=1.0,
        source_type="retrieved",
        metadata={"conflict_with": str(unit_v1.unit_id)}
    )

    scores_v2 = scorer.calculate(unit_v2, gap_detected=False, evidence_count=1)
    decision_v2 = action_router.route(
        unit_v2, scores_v2.Q,
        gap_detected=False,
        conflict_detected=True,  # 检测到冲突
        current_zone=MemoryZone.LONG_TERM_REVIEW
    )

    print(f"\n第二轮（加入反证）:")
    print(f"  动作: {decision_v2.action.value}")
    print(f"  目标区域: {decision_v2.target_zone.value}")
    print(f"  硬否决: {decision_v2.hard_veto_triggered}")

    # 执行动作
    if decision_v2.action in [TSLAActionType.ISOLATE]:
        zone_manager.move_to_zone(
            unit_v2,
            MemoryZone.LONG_TERM_ISOLATION,
            "检测到冲突，需要隔离"
        )

    # 验证结果
    location = zone_manager.get_unit_location(unit_v2.unit_id)
    passed = location is not None and location[0] == MemoryZone.LONG_TERM_ISOLATION

    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 冲突暴露后应进入隔离区")

    return passed


def test_multi_round_stability():
    """3. 多轮解释稳定性测试"""
    print("\n" + "=" * 60)
    print("测试 3: 多轮解释稳定性测试")
    print("=" * 60)

    scorer = TSLAScorer()

    # 同一对象的不同表述
    descriptions = [
        "苹果是一种水果",
        "苹果属于蔷薇科植物",
        "苹果是常见的红色或绿色水果",
        "苹果富含维生素C"
    ]

    scores_history = []

    for desc in descriptions:
        unit = Unit(
            script_form="苹果",
            core_meaning=desc,
            truth_score=0.80,
            stability_score=0.75,
            evidence_score=0.70
        )

        scores = scorer.calculate(unit, gap_detected=False, evidence_count=2)
        scores_history.append(scores.Q)

        print(f"  '{desc[:20]}...' -> Q={scores.Q:.1f}")

    # 检查稳定性
    score_variance = max(scores_history) - min(scores_history)
    stable = score_variance < 15  # 变化小于15分认为稳定

    print(f"\n  分数范围: {min(scores_history):.1f} - {max(scores_history):.1f}")
    print(f"  方差: {score_variance:.1f}")

    passed = stable
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 同一对象不同表述应保持相对稳定")

    return passed


def test_error_archival():
    """4. 错误归档测试"""
    print("\n" + "=" * 60)
    print("测试 4: 错误归档测试")
    print("=" * 60)

    event_log = MemoryEventLog()
    zone_manager = ZoneManager(event_log=event_log)
    action_router = ActionRouter()
    scorer = TSLAScorer()

    # 已确认错误但有学习价值的 Unit
    unit = Unit(
        script_form="错误概念",
        core_meaning="这是一个错误的定义",
        truth_score=0.20,  # 低真实性
        stability_score=0.30,
        evidence_score=0.25,
        legality_score=0.80,
        metadata={
            "confirmed_error": True,
            "learning_value": True,  # 有学习价值
            "error_reason": "概念混淆"
        }
    )

    scores = scorer.calculate(unit, gap_detected=True, gap_severity="medium")

    print(f"\nUnit 信息:")
    print(f"  名称: {unit.script_form}")
    print(f"  已确认错误: {unit.metadata.get('confirmed_error')}")
    print(f"  有学习价值: {unit.metadata.get('learning_value')}")
    print(f"  综合治理分: {scores.Q:.1f}")

    decision = action_router.route(
        unit, scores.Q,
        gap_detected=True,
        gap_severity="medium"
    )

    print(f"\nTSLA 决策:")
    print(f"  动作: {decision.action.value}")
    print(f"  目标区域: {decision.target_zone.value}")
    print(f"  原因: {decision.reason}")

    # 执行动作
    if decision.action == TSLAActionType.ERROR_ARCHIVE:
        zone_manager.move_to_zone(
            unit,
            MemoryZone.LONG_TERM_ERROR,
            "已确认错误但有学习价值，归档到错误区"
        )

    # 验证结果
    location = zone_manager.get_unit_location(unit.unit_id)
    passed = location is not None and location[0] == MemoryZone.LONG_TERM_ERROR

    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 有学习价值的错误应进入错误区而非直接删除")

    return passed


def test_full_memory_flow():
    """5. 完整记忆流程测试"""
    print("\n" + "=" * 60)
    print("测试 5: 完整记忆流程测试")
    print("=" * 60)

    # 初始化所有组件
    event_log = MemoryEventLog()
    transient_store = TransientStore(event_log=event_log)
    zone_manager = ZoneManager(event_log=event_log)
    write_gate = WriteGate(event_log=event_log)
    action_router = ActionRouter()
    scorer = TSLAScorer()

    # 测试场景：一个高质量 Unit 的完整生命周期
    unit = Unit(
        script_form="高质量知识",
        core_meaning="这是一个经过验证的知识",
        truth_score=0.90,
        stability_score=0.85,
        evidence_score=0.80,
        legality_score=1.0,
        conflict_cleanliness=1.0,
        source_type="retrieved"
    )

    print(f"\n完整流程:")

    # Step 1: 写入瞬时层
    print("\n  Step 1: 写入瞬时层")
    transient_store.write(unit, "test_session")
    print(f"    ✓ 已写入瞬时层")

    # Step 2: 计算评分
    print("\n  Step 2: 计算 TSLA 评分")
    scores = scorer.calculate(unit, gap_detected=False, evidence_count=3)
    print(f"    T={scores.T:.1f}, S={scores.S:.1f}, E={scores.E:.1f}")
    print(f"    C={scores.C:.1f}, L={scores.L:.1f}, R={scores.R:.1f}, P={scores.P:.1f}")
    print(f"    Q(综合)={scores.Q:.1f}")

    # Step 3: 写入门评估
    print("\n  Step 3: 写入门评估")
    write_decision = write_gate.evaluate(unit, scores.Q, "test_session")
    print(f"    允许进入长期层: {write_decision.allowed}")
    print(f"    原因: {write_decision.reason}")

    # Step 4: TSLA 动作路由
    print("\n  Step 4: TSLA 动作路由")
    action_decision = action_router.route(
        unit, scores.Q,
        gap_detected=False,
        conflict_detected=False
    )
    print(f"    动作: {action_decision.action.value}")
    print(f"    目标区域: {action_decision.target_zone.value}")

    # Step 5: 执行迁移
    print("\n  Step 5: 执行区域迁移")
    if action_decision.action != TSLAActionType.EXCLUDE:
        zone_manager.move_to_zone(
            unit,
            action_decision.target_zone,
            f"TSLA动作: {action_decision.action.value}"
        )
        print(f"    ✓ 已迁移到 {action_decision.target_zone.value}")
    else:
        print(f"    ✗ 被排除，不进入长期层")

    # Step 6: 验证最终状态
    print("\n  Step 6: 验证最终状态")
    location = zone_manager.get_unit_location(unit.unit_id)
    history = zone_manager.get_unit_history_summary(unit.unit_id)

    print(f"    当前区域: {location[0].value if location else '未知'}")
    print(f"    总事件数: {history['total_events']}")
    print(f"    迁移次数: {len(history['zone_transitions'])}")

    # 验证
    passed = (
        location is not None and
        location[0] == MemoryZone.LONG_TERM_REVIEW and
        history['total_events'] >= 2  # 至少写入和迁移两个事件
    )

    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 高质量 Unit 应成功进入长期受审区")

    return passed


def run_all_tests():
    """运行所有第二阶段测试"""
    print("\n" + "🧪 " * 30)
    print("第二阶段记忆治理测试套件")
    print("🧪 " * 30)

    results = {
        "重复出现测试": test_repeated_appearance(),
        "冲突暴露测试": test_conflict_exposure(),
        "稳定性测试": test_multi_round_stability(),
        "错误归档测试": test_error_archival(),
        "完整流程测试": test_full_memory_flow()
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 所有第二阶段测试通过！记忆治理原型验证成功。")
        print("\n完成标志:")
        print("  ✓ 输入后一定先写入瞬时层")
        print("  ✓ 部分对象可通过写入门进入长期受审区")
        print("  ✓ TSLA 能输出正式治理动作")
        print("  ✓ 支持回流/隔离/错误归档真实迁移")
        print("  ✓ 跨轮次回放时状态变化可追踪")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
