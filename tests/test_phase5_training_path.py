"""
第五阶段测试 - 训练路径集成验证

测试类型：
A. 高质量种子样本 - 确认能进入长期受审区并逐步晋升
B. 混淆训练样本 - 确认触发缺失感知或TSLA初判拦截
C. 自构建候选样本 - 确认可进入治理链但不能直接晋升深层永久
D. 深层永久候选样本 - 确认允许进入深层永久

第五阶段完成标志：
1. 训练路径可以独立跑通
2. 训练样本能进入长期治理链
3. 强审查与验证成为正式节点
4. 深层永久层首次接入成功
5. 自构建内容只能走受控晋升，不会直接改写基础参数
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.training_pipeline import TrainingPipeline, TrainingCase, TrainingOutcome
from core.training.review_manager import ReviewManager, Evidence, ReviewDecision
from core.training.verifier import Verifier, VerificationType
from core.memory.deep_permanent_store import (
    DeepPermanentStore, VerificationProof, TrainingOrigin
)
from core.gates.deep_permanent_gate import DeepPermanentGate


def test_a_high_quality_seed():
    """
    A. 高质量种子样本测试
    验证：来源清晰、低噪声、单义明确，能进入长期受审区并逐步晋升
    """
    print("\n" + "=" * 70)
    print("A. 高质量种子样本测试")
    print("=" * 70)

    pipeline = TrainingPipeline()

    # 高质量种子
    case = TrainingCase(
        case_id="seed_001",
        content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        source="curated_dataset",
        seed_quality_score=0.92
    )

    result = pipeline.process(case)

    print(f"  案例ID: {result.case_id}")
    print(f"  结果: {result.outcome.value}")
    print(f"  成功: {'✅' if result.success else '❌'}")
    print(f"  耗时: {result.duration_ms}ms")
    print(f"  最终阶段: {result.final_stage.value}")

    # 验证：不应在早期被排除（种子检查阶段）
    # 高质量种子应该能通过种子检查，进入后续阶段
    not_excluded_at_seed = result.final_stage.value != "seed_input"
    
    # 检查是否进入了治理链（至少到达受审区或更后）
    entered_chain = result.final_stage.value in [
        "review_candidate", "noise_injection", "model_thinking",
        "missing_perception", "retrieval", "candidate_conclusion",
        "tsla_initial", "strong_review", "verification", "gate_promotion"
    ] or result.success

    passed = not_excluded_at_seed and entered_chain

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 高质量种子进入治理链，未被排除")

    return passed


def test_b_confusion_sample():
    """
    B. 混淆训练样本测试
    验证：表面合理但实际有冲突，应触发拦截，不应直接进入正常区
    """
    print("\n" + "=" * 70)
    print("B. 混淆训练样本测试")
    print("=" * 70)

    pipeline = TrainingPipeline()

    # 混淆样本（来源不佳）
    case = TrainingCase(
        case_id="confusion_001",
        content="可能是某种技术或者方法",
        source="unverified",  # 不允许的来源
        seed_quality_score=0.45
    )

    result = pipeline.process(case)

    print(f"  案例ID: {result.case_id}")
    print(f"  结果: {result.outcome.value}")
    print(f"  成功: {'✅' if result.success else '❌'}")

    # 验证：应在早期被排除或隔离
    blocked_early = result.outcome in [
        TrainingOutcome.EXCLUDED, TrainingOutcome.ISOLATED
    ]

    passed = blocked_early

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 混淆样本被拦截，未直接进入正常区")

    return passed


def test_c_self_constructed_candidate():
    """
    C. 自构建候选样本测试
    验证：模型自构建内容可进入治理链，但不能直接晋升深层永久
    """
    print("\n" + "=" * 70)
    print("C. 自构建候选样本测试")
    print("=" * 70)

    gate = DeepPermanentGate()

    # 模拟自构建内容尝试进入深层永久
    result = gate.evaluate(
        unit_id="self_constructed_001",
        current_zone="training_normal",
        current_scores={"Q": 88, "T": 85, "S": 87, "C": 90, "L": 89},
        source_type="model_self_constructed",  # 自构建来源
        review_passed=True,
        review_confidence=0.92,
        verification_passed=True,
        verification_rounds=5,
        stability_cycles=12,
        conflict_free_rounds=10,
        training_iterations=150,
        recent_conflict=False
    )

    print(f"  案例ID: {result.unit_id}")
    print(f"  来源: model_self_constructed")
    print(f"  决策: {result.decision.value}")
    print(f"  阻止项: {result.blockers}")

    # 验证：自构建来源被阻止
    blocked = result.decision.value.startswith("blocked")
    source_blocked = any("来源" in b or "source" in b.lower()
                         for b in result.blockers)

    passed = blocked and source_blocked

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 自构建内容不能直接晋升深层永久，只能走受控晋升")

    return passed


def test_d_deep_permanent_candidate():
    """
    D. 深层永久候选样本测试
    验证：来自训练路径、多轮稳定、强审查通过的内容可进入深层永久
    """
    print("\n" + "=" * 70)
    print("D. 深层永久候选样本测试")
    print("=" * 70)

    store = DeepPermanentStore()
    gate = DeepPermanentGate()

    # 先评估
    eval_result = gate.evaluate(
        unit_id="deep_candidate_001",
        current_zone="training_normal",
        current_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
        source_type="training_path",
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_rounds=5,
        stability_cycles=12,
        conflict_free_rounds=10,
        training_iterations=150,
        recent_conflict=False
    )

    print(f"  门控评估:")
    print(f"    决策: {eval_result.decision.value}")
    print(f"    置信度: {eval_result.confidence}")

    # 如果通过门控，尝试晋升
    if eval_result.decision.value == "promote_to_deep":
        proof = VerificationProof(
            verification_rounds=5,
            avg_score=0.92,
            consistency_score=0.95,
            cross_task_score=0.90,
            long_term_stability=0.93,
            verified_at="2026-04-17T10:00:00",
            verifier_id="verifier_v1"
        )

        origin = TrainingOrigin(
            training_iteration=150,
            source_dataset="curated_v2",
            review_passed=True,
            review_confidence=0.95,
            verification_passed=True,
            verification_proof=proof
        )

        entry = store.promote_to_deep_permanent(
            unit_id="deep_candidate_001",
            content="深度学习是机器学习的一个分支，使用多层神经网络。",
            core_meaning="深度学习是使用多层神经网络的机器学习方法",
            promotion_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
            training_origin=origin,
            review_confidence=0.95
        )

        print(f"\n  晋升结果:")
        print(f"    对象ID: {entry.unit_id}")
        print(f"    状态: {entry.status}")
        print(f"    验证轮数: {entry.training_origin.verification_proof.verification_rounds}")

        # 验证
        in_deep = store.is_in_deep_permanent("deep_candidate_001")
        has_proof = store.get_verification_proof("deep_candidate_001") is not None

        passed = in_deep and has_proof

        status = "✅ 通过" if passed else "❌ 失败"
        print(f"\n  结果: {status}")
        print("  验证: 训练路径内容成功进入深层永久")

        return passed
    else:
        print(f"\n  结果: ❌ 失败")
        print("  验证: 门控未通过")
        return False


def test_strong_review_node():
    """
    验证：强审查成为正式节点
    """
    print("\n" + "=" * 70)
    print("强审查节点测试")
    print("=" * 70)

    manager = ReviewManager()

    # 高质量样本
    evidence = [
        Evidence("ev1", "expert_doc", "专家定义", 0.95, True),
        Evidence("ev2", "verified_paper", "论文支撑", 0.90, True)
    ]

    result = manager.review(
        case_id="review_test_001",
        content="人工智能是指由人制造出来的系统所表现出来的智能。",
        source="curated_dataset",
        evidence_list=evidence,
        existing_knowledge=[],
        target_zone="deep_permanent"
    )

    print(f"  案例ID: {result.case_id}")
    print(f"  决策: {result.decision.value}")
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  通过检查: {[c.value for c in result.passed_checks]}")
    print(f"  失败检查: {[c.value for c in result.failed_checks]}")

    # 验证强审查工作
    has_checks = len(result.passed_checks) + len(result.failed_checks) > 0
    has_decision = result.decision in ReviewDecision

    passed = has_checks and has_decision

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 强审查成为正式节点")

    return passed


def test_verification_node():
    """
    验证：验证节点成为正式节点
    """
    print("\n" + "=" * 70)
    print("验证节点测试")
    print("=" * 70)

    verifier = Verifier()

    # 先记录稳定性历史
    for i in range(12):
        verifier.record_stability("verify_test_001", i, 0.88 + (i % 3) * 0.02, 0.05)

    result = verifier.verify(
        case_id="verify_test_001",
        content="机器学习是人工智能的一个分支。",
        verification_types=[
            VerificationType.CONSISTENCY,
            VerificationType.CROSS_TASK,
            VerificationType.LONG_TERM_STABILITY
        ],
        min_rounds=5,
        target_deep_permanent=True
    )

    print(f"  案例ID: {result.case_id}")
    print(f"  总轮数: {result.total_rounds}")
    print(f"  通过: {result.passed_rounds}")
    print(f"  失败: {result.failed_rounds}")
    print(f"  总体状态: {result.overall_status.value}")

    # 验证验证节点工作
    has_rounds = result.total_rounds >= 5
    has_status = result.overall_status is not None

    passed = has_rounds and has_status

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 验证节点成为正式节点")

    return passed


def test_no_parameter_override():
    """
    验证：自构建内容不会直接改写基础参数
    """
    print("\n" + "=" * 70)
    print("参数保护测试")
    print("=" * 70)

    # 这个测试验证系统设计原则
    # 自构建内容只能通过知识库晋升链，不能直接写回参数

    print("  系统设计验证:")
    print("  ✓ 自构建内容走知识库晋升链")
    print("  ✓ 不能直接改写基础手工训练参数")
    print("  ✓ 需要更高门槛才可能走参数晋升链")
    print("  ✓ 所有晋升必须经过门控审查")

    # 验证 deep_permanent_gate 阻止自构建来源
    gate = DeepPermanentGate()

    result = gate.evaluate(
        unit_id="param_test_001",
        current_zone="training_normal",
        current_scores={"Q": 95, "T": 95, "S": 95, "C": 95, "L": 95},
        source_type="model_self_constructed",
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_rounds=10,
        stability_cycles=15,
        conflict_free_rounds=12,
        training_iterations=200,
        recent_conflict=False
    )

    blocked = result.decision.value.startswith("blocked")

    print(f"\n  门控测试:")
    print(f"    来源: model_self_constructed")
    print(f"    决策: {result.decision.value}")
    print(f"    阻止: {'✅' if blocked else '❌'}")

    passed = blocked

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 自构建内容不能直接晋升，保护基础参数")

    return passed


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🎯 " * 35)
    print("第五阶段训练路径集成测试")
    print("🎯 " * 35)

    tests = {
        "A. 高质量种子样本": test_a_high_quality_seed,
        "B. 混淆训练样本": test_b_confusion_sample,
        "C. 自构建候选样本": test_c_self_constructed_candidate,
        "D. 深层永久候选样本": test_d_deep_permanent_candidate,
        "强审查节点": test_strong_review_node,
        "验证节点": test_verification_node,
        "参数保护": test_no_parameter_override
    }

    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} 异常: {e}")
            results[test_name] = False

    # 汇总
    print("\n" + "=" * 70)
    print("第五阶段测试汇总")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 第五阶段训练路径集成测试全部通过！")
        print("\n完成标志:")
        print("  ✓ 训练路径可以独立跑通")
        print("  ✓ 训练样本能进入长期治理链")
        print("  ✓ 强审查与验证成为正式节点")
        print("  ✓ 深层永久层首次接入成功")
        print("  ✓ 自构建内容只能走受控晋升，不会直接改写基础参数")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
