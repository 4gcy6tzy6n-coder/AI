"""
第一阶段原型测试

测试4组样本：
A. 正常样本 - 能生成候选结论，TSLA不直接拒绝
B. 缺失样本 - 必须触发 GapDetected，不能直接高置信回答
C. 冲突样本 - 必须产生 conflict_flags，TSLA返回 review 或 isolate
D. 多义样本 - 触发 split 或标记结构非法
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.user_pipeline import UserPipeline
from core.unit.models import TSLAAction


def test_normal_case():
    """A. 正常样本测试"""
    print("\n" + "=" * 60)
    print("测试 A: 正常样本")
    print("=" * 60)

    pipeline = UserPipeline()

    test_cases = [
        "苹果是什么",
        "永久层是什么"
    ]

    all_passed = True
    for test_input in test_cases:
        result = pipeline.process(test_input)

        # 验证：能生成候选结论
        has_candidate = result['candidate_conclusion'] is not None

        # 验证：TSLA 不直接拒绝
        tsla_action = result['tsla_action']['action']
        not_rejected = tsla_action != TSLAAction.REJECT.value

        passed = has_candidate and not_rejected
        status = "✅ 通过" if passed else "❌ 失败"

        print(f"\n输入: {test_input}")
        print(f"  有候选结论: {has_candidate}")
        print(f"  TSLA动作: {tsla_action}")
        print(f"  结果: {status}")

        if not passed:
            all_passed = False

    return all_passed


def test_gap_case():
    """B. 缺失样本测试"""
    print("\n" + "=" * 60)
    print("测试 B: 缺失样本")
    print("=" * 60)

    pipeline = UserPipeline()

    test_cases = [
        "某个没定义的新词是什么意思",
        "XYZABC是什么东西"
    ]

    all_passed = True
    for test_input in test_cases:
        result = pipeline.process(test_input)

        # 验证：必须触发 GapDetected
        gap_result = result['gap_result']
        has_gap = gap_result and gap_result['has_gap']

        # 验证：不能直接高置信回答
        candidate = result['candidate_conclusion']
        if candidate:
            low_confidence = candidate['confidence_stub'] < 0.7
        else:
            low_confidence = True

        passed = has_gap and low_confidence
        status = "✅ 通过" if passed else "❌ 失败"

        print(f"\n输入: {test_input}")
        print(f"  检测到缺口: {has_gap}")
        if gap_result:
            print(f"  缺口原因: {gap_result.get('reasons', [])}")
        print(f"  结果: {status}")

        if not passed:
            all_passed = False

    return all_passed


def test_conflict_case():
    """C. 冲突样本测试"""
    print("\n" + "=" * 60)
    print("测试 C: 冲突样本")
    print("=" * 60)

    pipeline = UserPipeline()

    # 手动创建冲突场景
    from core.unit.models import Unit, UnitType, SourceType
    from core.tsla.reviewer import TSLAReviewer
    from core.thinking.gap_detector import GapDetectionResult
    from core.unit.models import CandidateConclusion

    # 创建一个有冲突的候选结论
    candidate = CandidateConclusion(
        answer_text="测试回答",
        used_units=[],
        used_evidence=[],
        missing_slots=[],
        conflict_flags=["模拟冲突: 测试对象有两个不同含义"],
        confidence_stub=0.5
    )

    # 运行 TSLA 审查
    reviewer = TSLAReviewer()
    tsla_result = reviewer.review(
        candidate=candidate,
        gap_result=GapDetectionResult(has_gap=False),
        units=[]
    )

    # 验证：TSLA 返回 review 或 isolate
    valid_actions = [TSLAAction.REVIEW.value, TSLAAction.ISOLATE.value]
    action_valid = tsla_result.action.value in valid_actions

    passed = action_valid
    status = "✅ 通过" if passed else "❌ 失败"

    print(f"\n输入: 测试对象（模拟冲突）")
    print(f"  TSLA动作: {tsla_result.action.value}")
    print(f"  结果: {status}")

    return passed


def test_ambiguity_case():
    """D. 多义样本测试"""
    print("\n" + "=" * 60)
    print("测试 D: 多义样本")
    print("=" * 60)

    pipeline = UserPipeline()

    # 测试一个可能有多个含义的输入
    test_input = "苹果"  # 可以是水果，也可以是公司

    result = pipeline.process(test_input)

    # 验证：应该触发 split 或至少标记需要审查
    tsla_action = result['tsla_action']['action']

    # 在多义情况下，应该返回 split 或 review
    valid_actions = [TSLAAction.SPLIT.value, TSLAAction.REVIEW.value]

    # 检查是否有多个 Unit 被激活
    units = result['units']
    multiple_units = len(units) > 1

    passed = tsla_action in valid_actions or multiple_units
    status = "✅ 通过" if passed else "❌ 失败"

    print(f"\n输入: {test_input}")
    print(f"  生成的Units数量: {len(units)}")
    print(f"  TSLA动作: {tsla_action}")
    print(f"  结果: {status}")

    return passed


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 30)
    print("第一阶段原型测试套件")
    print("🧪 " * 30)

    results = {
        "正常样本": test_normal_case(),
        "缺失样本": test_gap_case(),
        "冲突样本": test_conflict_case(),
        "多义样本": test_ambiguity_case()
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 所有测试通过！第一阶段原型验证成功。")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
