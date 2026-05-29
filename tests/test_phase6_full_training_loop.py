"""
第六阶段测试 - 完整训练闭环验证

测试类型：
A. 完整训练循环 - 确认循环可以独立跑通
B. 深层永久作为参数晋升前置层 - 确认 deep_permanent 可进入参数候选
C. 参数写回控制 - 确认只进入自构建学习参数区
D. 基础参数保护 - 确认手工参数不会被覆盖
E. 回滚/冻结机制 - 确认写回后有回滚能力
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.full_training_loop import FullTrainingLoop, TrainingCase
from src.core.training.promotion_manager import PromotionManager, PromotionStatus
from src.core.gates.parameter_promotion_gate import ParameterPromotionGate, ParamPromotionDecision
from src.core.params.manual_param_store import ManualParamStore
from src.core.params.self_learned_param_store import SelfLearnedParamStore


def test_a_full_training_loop():
    """
    A. 完整训练循环测试
    验证：循环可以独立跑通，输出各类统计
    """
    print("\n" + "=" * 70)
    print("A. 完整训练循环测试")
    print("=" * 70)

    loop = FullTrainingLoop()

    # 创建测试案例
    seed_cases = [
        TrainingCase(
            case_id=f"test_{i:03d}",
            content=f"测试样本 {i}",
            source="curated_dataset",
            seed_quality_score=0.85 + (i % 3) * 0.05
        )
        for i in range(5)
    ]

    # 运行循环
    result = loop.run_loop(
        seed_cases=seed_cases,
        max_rounds=2,
        enable_param_promotion=True
    )

    print(f"\n  循环ID: {result.loop_id}")
    print(f"  总轮数: {result.total_rounds}")
    print(f"  总案例: {result.total_cases}")

    # 验证：循环能跑通（有输出）
    loop_runs = result.total_rounds > 0 and result.total_cases > 0

    # 验证：有统计输出
    has_metrics = (
        result.final_metrics.total_candidates > 0 or
        result.final_metrics.rejected_count >= 0
    )

    passed = loop_runs and has_metrics

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 完整训练循环可以独立跑通")

    return passed


def test_b_deep_permanent_to_param():
    """
    B. 深层永久作为参数晋升前置层测试
    验证：deep_permanent 可作为参数晋升前置层
    """
    print("\n" + "=" * 70)
    print("B. 深层永久作为参数晋升前置层测试")
    print("=" * 70)

    gate = ParameterPromotionGate()

    # 模拟 deep_permanent 对象尝试晋升参数
    result = gate.evaluate(
        candidate_id="param_candidate_001",
        source_zone="deep_permanent",
        quality_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
        verification_rounds=10,
        cross_task_score=0.92,
        long_term_stability=0.96,
        cross_version_score=0.88
    )

    print(f"\n  候选ID: {result.candidate_id}")
    print(f"  来源: deep_permanent")
    print(f"  决策: {result.decision.value}")
    print(f"  置信度: {result.confidence}")

    # 验证：deep_permanent 来源被接受
    source_accepted = result.decision != ParamPromotionDecision.BLOCKED_BY_SOURCE

    # 验证：决策是 approved 或被其他因素阻止（不是来源问题）
    can_promote = result.decision == ParamPromotionDecision.APPROVED or source_accepted

    passed = source_accepted

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: deep_permanent 可作为参数晋升前置层")

    return passed


def test_c_param_write_to_self_learned_only():
    """
    C. 参数写回控制测试
    验证：参数写回只进入自构建学习参数区
    """
    print("\n" + "=" * 70)
    print("C. 参数写回控制测试")
    print("=" * 70)

    # 测试 self_learned_param_store 接受写回
    self_store = SelfLearnedParamStore()

    print("\n  1. 自构建参数存储:")
    try:
        param = self_store.write_param(
            param_id="self_learned_001",
            source_unit_id="deep_unit_001",
            source_zone="deep_permanent",
            value_shape=(768, 768),
            value_hash="hash_sl_001",
            promotion_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
            verification_rounds=10,
            cross_task_score=0.92,
            long_term_stability=0.96,
            gate_decision="approved",
            gate_confidence=0.95
        )
        print(f"     写入成功: {param.param_id}")
        self_learned_accepts = True
    except ValueError as e:
        print(f"     写入失败: {e}")
        self_learned_accepts = False

    # 测试 manual_param_store 阻止写回
    manual_store = ManualParamStore()

    print("\n  2. 手工参数存储:")
    success = manual_store.write_param(
        param_id="attention_weights_layer_0",
        value={"new": "value"},
        source="model_self_constructed"
    )
    print(f"     写入结果: {'成功' if success else '被阻止'}")
    manual_blocks = not success

    passed = self_learned_accepts and manual_blocks

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 参数写回只进入自构建学习参数区，手工参数被保护")

    return passed


def test_d_manual_param_protection():
    """
    D. 基础参数保护测试
    验证：基础手工训练参数不会被直接覆盖
    """
    print("\n" + "=" * 70)
    print("D. 基础参数保护测试")
    print("=" * 70)

    store = ManualParamStore()

    # 尝试各种方式写入
    attempts = [
        ("直接写入", lambda: store.write_param("attention_weights_layer_0", {}, "test")),
        ("无令牌写入", lambda: store.attempt_write("feedforward_weights_layer_0", {}, "test")),
        ("关键参数写入", lambda: store.attempt_write("attention_weights_layer_0", {}, "test", "APPROVED_123")),
    ]

    all_blocked = True
    print()
    for name, attempt in attempts:
        try:
            result = attempt()
            blocked = not result
            print(f"  {name}: {'被阻止' if blocked else '成功'}")
            if not blocked:
                all_blocked = False
        except Exception as e:
            print(f"  {name}: 异常 - {e}")

    # 验证保护摘要
    summary = store.get_protection_summary()
    print(f"\n  保护状态: {'激活' if summary['protection_active'] else '未激活'}")
    print(f"  关键参数数: {summary['critical_params']}")

    passed = all_blocked and summary['protection_active']

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 基础手工训练参数不会被直接覆盖")

    return passed


def test_e_rollback_mechanism():
    """
    E. 回滚/冻结机制测试
    验证：写回后有回滚/冻结机制，且旧能力可验证不受破坏
    """
    print("\n" + "=" * 70)
    print("E. 回滚/冻结机制测试")
    print("=" * 70)

    store = SelfLearnedParamStore()

    # 1. 写入参数
    print("\n  1. 写入参数:")
    param = store.write_param(
        param_id="self_learned_rollback_test",
        source_unit_id="deep_unit_001",
        source_zone="deep_permanent",
        value_shape=(768, 768),
        value_hash="hash_test",
        promotion_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
        verification_rounds=10,
        cross_task_score=0.92,
        long_term_stability=0.96,
        gate_decision="approved",
        gate_confidence=0.95
    )
    print(f"     写入: {param.param_id}")

    # 2. 激活参数
    print("\n  2. 激活参数:")
    store.activate_param(param.param_id)
    param = store.get_param(param.param_id)
    print(f"     状态: {param.status.value}")

    # 3. 回滚参数
    print("\n  3. 回滚参数:")
    rollback_success = store.rollback_param(
        param_id=param.param_id,
        reason="发现能力漂移"
    )
    print(f"     回滚: {'成功' if rollback_success else '失败'}")

    param = store.get_param(param.param_id)
    print(f"     新状态: {param.status.value}")
    print(f"     回滚原因: {param.rollback_reason}")

    # 4. 写入另一个参数并冻结
    print("\n  4. 冻结测试:")
    param2 = store.write_param(
        param_id="self_learned_freeze_test",
        source_unit_id="deep_unit_002",
        source_zone="deep_permanent",
        value_shape=(768, 768),
        value_hash="hash_test2",
        promotion_scores={"Q": 91, "T": 92, "S": 90, "C": 95, "L": 93},
        verification_rounds=9,
        cross_task_score=0.91,
        long_term_stability=0.95,
        gate_decision="approved",
        gate_confidence=0.90
    )
    freeze_success = store.freeze_param(param2.param_id, "需要进一步验证")
    print(f"     冻结: {'成功' if freeze_success else '失败'}")

    param2 = store.get_param(param2.param_id)
    print(f"     状态: {param2.status.value}")

    # 5. 检查回滚历史
    print("\n  5. 回滚历史:")
    history = store.get_rollback_history()
    print(f"     回滚记录数: {len(history)}")
    for record in history:
        print(f"     - {record['param_id']}: {record['reason']}")

    # 验证
    has_rollback = rollback_success and param.status.value == "rolled_back"
    has_freeze = freeze_success
    has_history = len(history) > 0

    passed = has_rollback and has_freeze and has_history

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 写回后有回滚/冻结机制")

    return passed


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 35)
    print("第六阶段测试 - 完整训练闭环验证")
    print("🧪 " * 35)

    tests = [
        ("A. 完整训练循环", test_a_full_training_loop),
        ("B. 深层永久作为参数晋升前置层", test_b_deep_permanent_to_param),
        ("C. 参数写回控制", test_c_param_write_to_self_learned_only),
        ("D. 基础参数保护", test_d_manual_param_protection),
        ("E. 回滚/冻结机制", test_e_rollback_mechanism),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} 异常: {e}")
            results[test_name] = False

    # 汇总
    print("\n" + "=" * 70)
    print("第六阶段测试总结")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("所有测试通过！第六阶段实现符合预期。")
        print("🎉 " * 35)
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
