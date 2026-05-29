"""
第五阶段最终验证 - 5条完成标志检查

验证目标：
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
from core.training.review_manager import ReviewManager, Evidence
from core.training.verifier import Verifier, VerificationType
from core.memory.deep_permanent_store import (
    DeepPermanentStore, VerificationProof, TrainingOrigin
)
from core.gates.deep_permanent_gate import DeepPermanentGate


def check_1_training_path_runs():
    """
    完成标志1：训练路径可以独立跑通
    """
    print("\n" + "=" * 70)
    print("完成标志 1/5: 训练路径独立跑通")
    print("=" * 70)

    pipeline = TrainingPipeline()

    # 创建训练案例
    case = TrainingCase(
        case_id="check_train_001",
        content="测试训练内容",
        source="curated_dataset",
        seed_quality_score=0.85
    )

    result = pipeline.process(case)

    print(f"  案例ID: {result.case_id}")
    print(f"  最终阶段: {result.final_stage.value}")
    print(f"  结果: {result.outcome.value}")
    print(f"  日志数: {len(result.logs)}")

    # 验证：训练路径能完整执行（有日志，到达某个阶段）
    has_logs = len(result.logs) > 0
    reached_stage = result.final_stage.value != "seed_input"

    passed = has_logs and reached_stage

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 训练路径可以独立跑通")

    return passed


def check_2_training_enters_governance():
    """
    完成标志2：训练样本能进入长期治理链
    """
    print("\n" + "=" * 70)
    print("完成标志 2/5: 训练样本进入长期治理链")
    print("=" * 70)

    pipeline = TrainingPipeline()

    # 高质量训练样本
    case = TrainingCase(
        case_id="check_gov_001",
        content="机器学习是人工智能的一个分支。",
        source="curated_dataset",
        seed_quality_score=0.90
    )

    result = pipeline.process(case)

    print(f"  案例ID: {result.case_id}")
    print(f"  最终阶段: {result.final_stage.value}")

    # 验证：进入治理链（至少到达受审区或更后）
    governance_stages = [
        "review_candidate", "noise_injection", "model_thinking",
        "missing_perception", "retrieval", "candidate_conclusion",
        "tsla_initial", "strong_review", "verification", "gate_promotion"
    ]

    entered_governance = result.final_stage.value in governance_stages

    passed = entered_governance

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 训练样本能进入长期治理链")

    return passed


def check_3_review_and_verification_nodes():
    """
    完成标志3：强审查与验证成为正式节点
    """
    print("\n" + "=" * 70)
    print("完成标志 3/5: 强审查与验证成为正式节点")
    print("=" * 70)

    # 测试强审查节点
    manager = ReviewManager()
    evidence = [
        Evidence("ev1", "expert_doc", "专家定义", 0.95, True),
        Evidence("ev2", "verified_paper", "论文支撑", 0.90, True)
    ]

    review_result = manager.review(
        case_id="check_review_001",
        content="人工智能是指由人制造出来的系统所表现出来的智能。",
        source="curated_dataset",
        evidence_list=evidence,
        existing_knowledge=[],
        target_zone="deep_permanent"
    )

    print(f"  强审查节点:")
    print(f"    决策: {review_result.decision.value}")
    print(f"    检查项: {len(review_result.passed_checks) + len(review_result.failed_checks)}")

    review_works = len(review_result.passed_checks) + len(review_result.failed_checks) > 0

    # 测试验证节点
    verifier = Verifier()

    # 先记录稳定性历史
    for i in range(12):
        verifier.record_stability("check_verify_001", i, 0.88 + (i % 3) * 0.02, 0.05)

    verify_result = verifier.verify(
        case_id="check_verify_001",
        content="测试内容",
        verification_types=[
            VerificationType.CONSISTENCY,
            VerificationType.CROSS_TASK,
            VerificationType.LONG_TERM_STABILITY
        ],
        min_rounds=5,
        target_deep_permanent=True
    )

    print(f"  验证节点:")
    print(f"    总轮数: {verify_result.total_rounds}")
    print(f"    状态: {verify_result.overall_status.value}")

    verify_works = verify_result.total_rounds >= 5

    passed = review_works and verify_works

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 强审查与验证成为正式节点")

    return passed


def check_4_deep_permanent_integration():
    """
    完成标志4：深层永久层首次接入成功
    """
    print("\n" + "=" * 70)
    print("完成标志 4/5: 深层永久层首次接入成功")
    print("=" * 70)

    store = DeepPermanentStore()
    gate = DeepPermanentGate()

    # 评估
    eval_result = gate.evaluate(
        unit_id="check_deep_001",
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

    # 晋升
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
            unit_id="check_deep_001",
            content="深度学习是机器学习的一个分支。",
            core_meaning="深度学习是使用多层神经网络的机器学习方法",
            promotion_scores={"Q": 92, "T": 90, "S": 88, "C": 95, "L": 93},
            training_origin=origin,
            review_confidence=0.95
        )

        print(f"  晋升结果:")
        print(f"    对象ID: {entry.unit_id}")
        print(f"    状态: {entry.status}")

        # 验证
        in_deep = store.is_in_deep_permanent("check_deep_001")
        has_origin = store.get_training_origin("check_deep_001") is not None

        passed = in_deep and has_origin

        status = "✅ 通过" if passed else "❌ 失败"
        print(f"\n  结果: {status}")
        print("  验证: 深层永久层首次接入成功")

        return passed
    else:
        print(f"\n  结果: ❌ 失败")
        print("  验证: 门控未通过")
        return False


def check_5_no_parameter_override():
    """
    完成标志5：自构建内容只能走受控晋升，不会直接改写基础参数
    """
    print("\n" + "=" * 70)
    print("完成标志 5/5: 自构建内容保护基础参数")
    print("=" * 70)

    gate = DeepPermanentGate()

    # 自构建内容尝试进入深层永久
    result = gate.evaluate(
        unit_id="check_self_001",
        current_zone="training_normal",
        current_scores={"Q": 95, "T": 95, "S": 95, "C": 95, "L": 95},
        source_type="model_self_constructed",  # 自构建来源
        review_passed=True,
        review_confidence=0.95,
        verification_passed=True,
        verification_rounds=10,
        stability_cycles=15,
        conflict_free_rounds=12,
        training_iterations=200,
        recent_conflict=False
    )

    print(f"  自构建内容评估:")
    print(f"    来源: model_self_constructed")
    print(f"    决策: {result.decision.value}")
    print(f"    阻止项: {result.blockers}")

    # 验证：自构建来源被阻止
    blocked = result.decision.value.startswith("blocked")

    passed = blocked

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 自构建内容不能直接晋升，保护基础参数")

    return passed


def run_final_validation():
    """运行最终验证"""
    print("\n" + "🎯 " * 35)
    print("第五阶段最终验证 - 5条完成标志检查")
    print("🎯 " * 35)

    checks = {
        "训练路径独立跑通": check_1_training_path_runs,
        "训练样本进入治理链": check_2_training_enters_governance,
        "强审查与验证节点": check_3_review_and_verification_nodes,
        "深层永久层接入": check_4_deep_permanent_integration,
        "自构建内容保护": check_5_no_parameter_override
    }

    results = {}
    for check_name, check_func in checks.items():
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ {check_name} 异常: {e}")
            results[check_name] = False

    # 汇总
    print("\n" + "=" * 70)
    print("第五阶段最终验证总结")
    print("=" * 70)

    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("第五阶段 Training Path Integration v0.5 全部完成！")
        print("🎉 " * 35)
        print("\n核心成果：")
        print("  ✓ training_pipeline.py - 训练流水线")
        print("  ✓ review_manager.py - 强审查节点")
        print("  ✓ verifier.py - 验证节点")
        print("  ✓ deep_permanent_store.py - 深层永久存储")
        print("  ✓ deep_permanent_gate.py - 深层永久门控")
        print("  ✓ test_phase5_training_path.py - 训练路径测试")
        print("\n第五阶段目标达成：")
        print("  • 训练路径可以独立跑通")
        print("  • 训练样本能进入长期治理链")
        print("  • 强审查与验证成为正式节点")
        print("  • 深层永久层首次接入成功")
        print("  • 自构建内容只能走受控晋升，不会直接改写基础参数")
        print("\n下一阶段准备:")
        print("  Phase 6: Full Training Loop")
        print("  - 参数写回机制")
        print("  - 完整自训练循环")
        print("  - 大规模训练数据管线")
        print("  - 深层永久和参数层联动更新")
    else:
        print("\n⚠️ 部分验证未通过，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_final_validation()
