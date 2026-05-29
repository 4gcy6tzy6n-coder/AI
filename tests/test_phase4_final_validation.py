"""
第四阶段最终验证 - 5条完成标志检查

验证目标：
1. 长期正常区对象可以被判定为浅层永久候选
2. 永久层禁止跨层直写
3. 永久对象暴露问题时不会直接删除
4. 最小实验路线图正式形成
5. 至少跑通一组"浅层永久晋升 + 回退保护"实验
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gates.permanent_protection_gate import PermanentProtectionGate, PermanentDecision
from core.memory.shallow_permanent_store import (
    ShallowPermanentStore, DowngradeTarget,
    EvidenceSummary, StabilityWindowSummary
)
from pipelines.evaluation_pipeline import EvaluationPipeline


def check_1_normal_to_permanent_promotion():
    """
    完成标志1：长期正常区对象可以被判定为浅层永久候选
    """
    print("\n" + "=" * 70)
    print("完成标志 1/5: 长期正常区 → 浅层永久晋升")
    print("=" * 70)

    gate = PermanentProtectionGate()

    # 长期正常区的完美候选
    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    result = gate.evaluate(
        unit_id="check_promote_001",
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

    print(f"  当前区域: normal")
    print(f"  目标区域: shallow_permanent")
    print(f"  决策: {result.decision.value}")
    print(f"  置信度: {result.confidence}")

    passed = result.decision == PermanentDecision.PROMOTE_TO_SHALLOW

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 长期正常区对象可以被判定为浅层永久候选")

    return passed


def check_2_no_cross_layer_write():
    """
    完成标志2：永久层禁止跨层直写
    """
    print("\n" + "=" * 70)
    print("完成标志 2/5: 永久层禁止跨层直写")
    print("=" * 70)

    gate = PermanentProtectionGate()

    # 尝试从 review 区直接晋升
    current_scores = {
        "Q": 82, "T": 85, "S": 80, "E": 78, "C": 88, "L": 90
    }

    result = gate.evaluate(
        unit_id="check_cross_001",
        current_zone="review",  # 不是 normal
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

    # 验证被阻止
    blocked = result.decision != PermanentDecision.PROMOTE_TO_SHALLOW
    zone_check = any("review" in b.lower() or "正常区" in b for b in result.blockers)

    passed = blocked and zone_check

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 永久层禁止跨层直写，只能从 normal 区晋升")

    return passed


def check_3_no_delete_on_conflict():
    """
    完成标志3：永久对象暴露问题时不会直接删除
    """
    print("\n" + "=" * 70)
    print("完成标志 3/5: 永久对象暴露问题时不会直接删除")
    print("=" * 70)

    store = ShallowPermanentStore()

    # 创建永久对象
    entry = store.promote_to_permanent(
        unit_id="check_no_delete_001",
        content="高质量知识",
        core_meaning="这是一个高质量知识",
        promotion_scores={"Q": 82, "T": 85, "S": 80},
        source_type="retrieved",
        evidence_summary=EvidenceSummary(
            primary_sources=["source_a", "source_b"],
            evidence_count=3
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
        unit_id="check_no_delete_001",
        conflict_type="evidence_contradiction",
        conflict_details={"new_evidence": "contradicts"},
        severity="high"
    )

    # 降级而非删除
    store.downgrade(
        unit_id="check_no_delete_001",
        target=DowngradeTarget.REVIEW,
        reason="新证据冲突，需要重新审查",
        action_taken="降级到review区"
    )

    # 验证对象仍然存在
    still_exists = store.is_in_permanent("check_no_delete_001")
    entry = store.get("check_no_delete_001")

    print(f"  降级后状态: {entry.status}")
    print(f"  降级历史数: {len(entry.downgrade_history)}")
    print(f"  对象仍存在: {still_exists}")

    # 获取保护摘要
    summary = store.get_protection_summary("check_no_delete_001")
    can_delete = summary.get("can_delete", True)

    passed = still_exists and not can_delete

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 永久对象暴露问题时降级到review区，而不是直接删除")

    return passed


def check_4_experiment_plan_formalized():
    """
    完成标志4：最小实验路线图正式形成
    """
    print("\n" + "=" * 70)
    print("完成标志 4/5: 最小实验路线图正式形成")
    print("=" * 70)

    plan_path = Path(__file__).parent.parent / "docs" / "04_experiment_protocols" / "prototype_eval_plan.md"

    if not plan_path.exists():
        print(f"  错误: 实验计划文档不存在")
        return False

    with open(plan_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键内容
    checks = {
        "实验A": "实验 A: 瞬时层拦截能力" in content,
        "实验B": "实验 B: 长期层治理能力" in content,
        "实验C": "实验 C: 浅层永久晋升能力" in content,
        "实验D": "实验 D: 永久层回退保护能力" in content,
        "通过标准": "通过标准" in content,
        "评估指标": "评估指标" in content,
        "4个实验包": content.count("实验 ") >= 4
    }

    print(f"  文档路径: {plan_path}")
    print(f"  文档大小: {len(content)} 字符")
    print(f"\n  内容检查:")
    for check_name, exists in checks.items():
        status = "✅" if exists else "❌"
        print(f"    {status} {check_name}")

    passed = all(checks.values())

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 最小实验路线图已正式形成")

    return passed


def check_5_permanent_promotion_and_rollback():
    """
    完成标志5：至少跑通一组"浅层永久晋升 + 回退保护"实验
    """
    print("\n" + "=" * 70)
    print("完成标志 5/5: 浅层永久晋升 + 回退保护实验")
    print("=" * 70)

    # 使用评估管道运行实验
    pipeline = EvaluationPipeline()

    # 运行实验C（浅层永久晋升）
    result_c = pipeline.run_single_experiment("C")

    # 运行实验D（回退保护）
    result_d = pipeline.run_single_experiment("D")

    print(f"\n  实验C (浅层永久晋升):")
    print(f"    通过率: {result_c.passed_cases}/{result_c.total_cases}")
    print(f"    结果: {'✅ 通过' if result_c.passed else '❌ 失败'}")

    print(f"\n  实验D (回退保护):")
    print(f"    通过率: {result_d.passed_cases}/{result_d.total_cases}")
    print(f"    结果: {'✅ 通过' if result_d.passed else '❌ 失败'}")

    # 检查报告文件
    report_path = Path(__file__).parent.parent / "outputs" / "evaluation_report.json"
    report_exists = report_path.exists()

    if report_exists:
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        overall_passed = report.get("overall_passed", False)
    else:
        overall_passed = False

    passed = result_c.passed and result_d.passed and overall_passed

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print("  验证: 跑通浅层永久晋升 + 回退保护实验")

    return passed


def run_final_validation():
    """运行最终验证"""
    print("\n" + "🎯 " * 35)
    print("第四阶段最终验证 - 5条完成标志检查")
    print("🎯 " * 35)

    checks = {
        "长期正常区→浅层永久晋升": check_1_normal_to_permanent_promotion(),
        "永久层禁止跨层直写": check_2_no_cross_layer_write(),
        "永久对象不直接删除": check_3_no_delete_on_conflict(),
        "最小实验路线图形成": check_4_experiment_plan_formalized(),
        "晋升+回退保护实验": check_5_permanent_promotion_and_rollback()
    }

    print("\n" + "=" * 70)
    print("第四阶段最终验证总结")
    print("=" * 70)

    for check_name, passed in checks.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")

    all_passed = all(checks.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("第四阶段 Permanent Memory & Minimal Experiment Route v0.4 全部完成！")
        print("🎉 " * 35)
        print("\n核心成果：")
        print("  ✓ permanent_protection_gate.py - 永久层保护门")
        print("  ✓ shallow_permanent_store.py - 浅层永久存储")
        print("  ✓ test_phase4_permanent_layer.py - 永久层功能测试")
        print("  ✓ prototype_eval_plan.md - 最小实验路线")
        print("  ✓ evaluation_pipeline.py - 统一评估管道")
        print("\n第四阶段目标达成：")
        print("  • 长期正常区对象可以被判定为浅层永久候选")
        print("  • 永久层禁止跨层直写")
        print("  • 永久对象暴露问题时不会直接删除")
        print("  • 最小实验路线图正式形成")
        print("  • 跑通浅层永久晋升 + 回退保护实验")
        print("\n下一阶段准备:")
        print("  Phase 5: Training Path Integration")
        print("  - 训练路径接入")
        print("  - 深层永久层")
        print("  - 模型自构建双链")
    else:
        print("\n⚠️ 部分验证未通过，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_final_validation()
