"""
Scorer 历史序列功能自测

验证：
1. 评分快照记录
2. 历史积累
3. 滚动统计（均值、标准差、峰值、趋势）
4. 与 stability_gate 的数据接口
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tsla.scorer import TSLAScorerWithHistory
from core.unit.models import Unit


def test_score_snapshot():
    """测试评分快照"""
    print("\n" + "=" * 60)
    print("测试 1: 评分快照记录")
    print("=" * 60)

    scorer = TSLAScorerWithHistory()

    unit = Unit(
        script_form="测试对象",
        core_meaning="这是一个测试",
        truth_score=0.85,
        stability_score=0.80,
        evidence_score=0.75,
        conflict_cleanliness=0.90,
        legality_score=1.0,
        source_type="retrieved"
    )

    result = scorer.score_with_history(
        unit_id="test_001",
        unit=unit,
        current_context={},
        evidence_count=2,
        access_count=3,
        session_id="session_001"
    )

    print(f"当前Q分数: {result.current_scores.Q:.1f}")
    print(f"历史记录数: {result.history_count}")
    print(f"快照时间戳: {result.snapshot.timestamp}")
    print(f"评分理由:")
    for dim, reason in result.snapshot.reasons.items():
        print(f"  {dim}: {reason}")

    passed = (
        result.history_count == 1 and
        result.snapshot.unit_id == "test_001" and
        len(result.snapshot.reasons) == 8
    )
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")

    return passed


def test_history_accumulation():
    """测试历史积累"""
    print("\n" + "=" * 60)
    print("测试 2: 历史积累")
    print("=" * 60)

    scorer = TSLAScorerWithHistory()

    # 模拟多轮评分
    for i in range(5):
        unit = Unit(
            script_form="测试对象",
            core_meaning="这是一个测试",
            truth_score=0.80 + i * 0.02,  # 逐渐提升
            stability_score=0.75 + i * 0.01,
            evidence_score=0.70 + i * 0.02,
            conflict_cleanliness=0.90,
            legality_score=1.0,
            source_type="retrieved"
        )

        result = scorer.score_with_history(
            unit_id="accum_001",
            unit=unit,
            current_context={},
            evidence_count=2 + i,
            access_count=i + 1,
            session_id=f"session_{i:03d}"
        )

    print(f"最终历史记录数: {result.history_count}")
    print(f"最终Q分数: {result.current_scores.Q:.1f}")

    # 获取完整历史
    history = scorer.get_history("accum_001")
    print(f"获取历史记录数: {len(history)}")

    passed = len(history) == 5 and result.history_count == 5
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 积累5条历史记录")

    return passed


def test_rolling_stats():
    """测试滚动统计"""
    print("\n" + "=" * 60)
    print("测试 3: 滚动统计")
    print("=" * 60)

    scorer = TSLAScorerWithHistory()

    # 模拟波动数据
    scores_pattern = [
        {"truth": 0.75, "stability": 0.70, "evidence": 0.65},  # 较低
        {"truth": 0.80, "stability": 0.75, "evidence": 0.70},  # 提升
        {"truth": 0.85, "stability": 0.80, "evidence": 0.75},  # 峰值
        {"truth": 0.82, "stability": 0.78, "evidence": 0.72},  # 回落
    ]

    for i, pattern in enumerate(scores_pattern):
        unit = Unit(
            script_form="测试对象",
            core_meaning="这是一个测试",
            truth_score=pattern["truth"],
            stability_score=pattern["stability"],
            evidence_score=pattern["evidence"],
            conflict_cleanliness=0.90,
            legality_score=1.0,
            source_type="retrieved"
        )

        result = scorer.score_with_history(
            unit_id="rolling_001",
            unit=unit,
            current_context={},
            evidence_count=2,
            access_count=i + 1
        )

    stats = result.rolling_stats

    print(f"Q均值: {stats.q_mean:.2f}")
    print(f"Q标准差: {stats.q_std:.2f}")
    print(f"Q范围: {stats.q_min:.1f} - {stats.q_max:.1f}")
    print(f"Q峰值: {stats.peak_q:.1f}")
    print(f"Q趋势: {stats.trend_q}")
    print(f"与上次差值Q: {stats.delta_from_last.get('Q', 0):.2f}")
    print(f"与峰值差值Q: {stats.delta_from_peak.get('Q', 0):.2f}")

    passed = (
        stats.q_mean > 0 and
        stats.q_std > 0 and
        stats.peak_q > 0 and
        stats.trend_q in ["up", "down", "stable"]
    )
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 有有效的滚动统计值")

    return passed


def test_stability_gate_interface():
    """测试与 stability_gate 的数据接口"""
    print("\n" + "=" * 60)
    print("测试 4: Stability Gate 数据接口")
    print("=" * 60)

    scorer = TSLAScorerWithHistory()

    # 积累3轮数据
    for i in range(3):
        unit = Unit(
            script_form="测试对象",
            core_meaning="这是一个测试",
            truth_score=0.80 + i * 0.02,
            stability_score=0.75 + i * 0.02,
            evidence_score=0.70 + i * 0.02,
            conflict_cleanliness=0.90,
            legality_score=1.0,
            source_type="retrieved"
        )

        scorer.score_with_history(
            unit_id="interface_001",
            unit=unit,
            current_context={},
            evidence_count=2
        )

    # 获取 stability_gate 格式数据
    history_for_gate = scorer.get_score_history_for_stability_gate(
        "interface_001",
        window_size=3
    )

    print(f"返回历史记录数: {len(history_for_gate)}")
    for i, record in enumerate(history_for_gate):
        print(f"  记录{i+1}: T={record['T']:.1f}, S={record['S']:.1f}, Q={record['Q']:.1f}")

    passed = len(history_for_gate) == 3 and all("Q" in r for r in history_for_gate)
    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print("预期: 返回3条格式化历史记录")

    return passed


def test_peak_detection():
    """测试峰值检测"""
    print("\n" + "=" * 60)
    print("测试 5: 峰值检测")
    print("=" * 60)

    scorer = TSLAScorerWithHistory()

    # 模拟峰值后下滑
    scores_pattern = [
        {"truth": 0.85, "stability": 0.82},  # 高
        {"truth": 0.86, "stability": 0.83},  # 峰值
        {"truth": 0.75, "stability": 0.70},  # 明显下滑
    ]

    for pattern in scores_pattern:
        unit = Unit(
            script_form="测试对象",
            core_meaning="这是一个测试",
            truth_score=pattern["truth"],
            stability_score=pattern["stability"],
            evidence_score=0.70,
            conflict_cleanliness=0.90,
            legality_score=1.0,
            source_type="retrieved"
        )

        result = scorer.score_with_history(
            unit_id="peak_001",
            unit=unit,
            current_context={}
        )

    stats = result.rolling_stats

    print(f"Q峰值: {stats.peak_q:.1f}")
    print(f"当前Q: {result.current_scores.Q:.1f}")
    print(f"与峰值差值: {stats.delta_from_peak.get('Q', 0):.2f}")

    # 应该检测到下滑
    q_drop = stats.delta_from_peak.get("Q", 0)
    passed = q_drop < -5  # 下滑超过5分

    print(f"\n结果: {'✅ 通过' if passed else '❌ 失败'}")
    print(f"预期: 检测到明显下滑 (实际下滑: {abs(q_drop):.1f}分)")

    return passed


def run_all_tests():
    """运行所有基础测试"""
    print("\n" + "🧪 " * 30)
    print("Scorer 历史序列功能自测")
    print("🧪 " * 30)

    results = {
        "评分快照": test_score_snapshot(),
        "历史积累": test_history_accumulation(),
        "滚动统计": test_rolling_stats(),
        "Stability Gate 接口": test_stability_gate_interface(),
        "峰值检测": test_peak_detection()
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 Scorer 历史序列功能验证通过！")
        print("\n完成标志:")
        print("  ✓ 每轮评分都能形成独立快照")
        print("  ✓ 同一对象能积累评分历史")
        print("  ✓ 能输出滚动统计和峰值差值")
        print("  ✓ stability_gate.py 可以直接消费这些数据")
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
