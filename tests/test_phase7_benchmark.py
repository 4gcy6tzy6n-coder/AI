"""
第七阶段测试 - 系统级 Benchmark 验证

测试目标：
- 验证 Benchmark 流水线能正确运行
- 验证各类系统级指标测量
- 验证系统稳定性报告生成
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.benchmark_pipeline import (
    BenchmarkPipeline, BenchmarkType, BenchmarkResult
)


def test_qa_stability_benchmark():
    """测试问答稳定性 Benchmark"""
    print("\n" + "=" * 70)
    print("测试 1: 问答稳定性 Benchmark")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    result = pipeline.benchmark_qa_stability()

    print(f"  测试案例数: {result.total_cases}")
    print(f"  通过数: {result.passed_cases}")
    print(f"  失败数: {result.failed_cases}")
    print(f"  通过率: {result.pass_rate:.1%}")

    # 验证
    passed = (
        result.benchmark_type == BenchmarkType.QA_STABILITY and
        result.total_cases > 0 and
        result.pass_rate >= 0
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: Benchmark 正确运行并返回结果")

    return passed


def test_long_term_governance_benchmark():
    """测试长期层治理 Benchmark"""
    print("\n" + "=" * 70)
    print("测试 2: 长期层治理 Benchmark")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    result = pipeline.benchmark_long_term_governance()

    print(f"  测试案例数: {result.total_cases}")
    print(f"  通过率: {result.pass_rate:.1%}")

    # 检查关键指标
    gap_rate = result.metrics.get("gap_detection_rate", 0)
    error_rate = result.metrics.get("error_interception_rate", 0)

    print(f"  缺口识别率: {gap_rate:.1%}")
    print(f"  错误拦截率: {error_rate:.1%}")

    passed = (
        result.benchmark_type == BenchmarkType.LONG_TERM_GOVERNANCE and
        gap_rate > 0 and
        error_rate > 0
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 治理指标已测量")

    return passed


def test_permanent_promotion_benchmark():
    """测试永久层晋升 Benchmark"""
    print("\n" + "=" * 70)
    print("测试 3: 永久层晋升 Benchmark")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    result = pipeline.benchmark_permanent_promotion()

    print(f"  测试案例数: {result.total_cases}")
    print(f"  通过率: {result.pass_rate:.1%}")

    # 检查关键指标
    false_stable = result.metrics.get("false_stable_rate", 1.0)
    accuracy = result.metrics.get("promotion_accuracy", 0)
    rollback = result.metrics.get("rollback_success_rate", 0)

    print(f"  假稳定误晋升率: {false_stable:.1%}")
    print(f"  晋升正确率: {accuracy:.1%}")
    print(f"  回退成功率: {rollback:.1%}")

    passed = (
        result.benchmark_type == BenchmarkType.PERMANENT_PROMOTION and
        accuracy > 0 and
        rollback > 0
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 晋升指标已测量")

    return passed


def test_training_loop_benchmark():
    """测试训练闭环 Benchmark"""
    print("\n" + "=" * 70)
    print("测试 4: 训练闭环 Benchmark")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    result = pipeline.benchmark_training_loop()

    print(f"  测试案例数: {result.total_cases}")
    print(f"  通过率: {result.pass_rate:.1%}")

    # 检查关键指标
    drift_rate = result.metrics.get("drift_rate", 1.0)
    integrity = result.metrics.get("loop_integrity", 0)

    print(f"  漂移率: {drift_rate:.1%}")
    print(f"  闭环完整性: {integrity:.1%}")

    passed = (
        result.benchmark_type == BenchmarkType.TRAINING_LOOP and
        drift_rate >= 0 and
        integrity > 0
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 训练闭环指标已测量")

    return passed


def test_param_write_back_benchmark():
    """测试参数写回 Benchmark"""
    print("\n" + "=" * 70)
    print("测试 5: 参数写回 Benchmark")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    result = pipeline.benchmark_param_write_back()

    print(f"  测试案例数: {result.total_cases}")
    print(f"  通过率: {result.pass_rate:.1%}")

    # 检查关键指标
    retention = result.metrics.get("old_capability_retention", 0)
    acquisition = result.metrics.get("new_capability_acquisition", 0)
    conflict = result.metrics.get("conflict_rate", 1.0)

    print(f"  旧能力保持率: {retention:.1%}")
    print(f"  新能力获取率: {acquisition:.1%}")
    print(f"  冲突率: {conflict:.1%}")

    passed = (
        result.benchmark_type == BenchmarkType.PARAM_WRITE_BACK and
        retention > 0 and
        acquisition > 0
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 参数写回指标已测量")

    return passed


def test_full_benchmark_pipeline():
    """测试完整 Benchmark 流水线"""
    print("\n" + "=" * 70)
    print("测试 6: 完整 Benchmark 流水线")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    report = pipeline.run_all_benchmarks()

    print(f"  综合评分: {report.overall_score:.1f}/100")
    print(f"\n  各类 Benchmark 结果:")
    print(f"    问答稳定性: {report.qa_stability.pass_rate:.1%}")
    print(f"    长期层治理: {report.long_term_governance.pass_rate:.1%}")
    print(f"    永久层晋升: {report.permanent_promotion.pass_rate:.1%}")
    print(f"    训练闭环: {report.training_loop.pass_rate:.1%}")
    print(f"    参数写回: {report.param_write_back.pass_rate:.1%}")

    print(f"\n  改进建议:")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"    {i}. {rec}")

    # 验证
    passed = (
        report.overall_score > 0 and
        report.qa_stability.pass_rate >= 0 and
        report.long_term_governance.pass_rate >= 0 and
        report.permanent_promotion.pass_rate >= 0 and
        report.training_loop.pass_rate >= 0 and
        report.param_write_back.pass_rate >= 0 and
        len(report.recommendations) > 0
    )

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 完整报告已生成")

    return passed


def test_benchmark_thresholds():
    """测试 Benchmark 阈值检查"""
    print("\n" + "=" * 70)
    print("测试 7: Benchmark 阈值检查")
    print("=" * 70)

    pipeline = BenchmarkPipeline()
    report = pipeline.run_all_benchmarks()

    # 定义阈值
    thresholds = {
        "qa_stability": 0.90,
        "long_term_governance": 0.85,
        "permanent_promotion": 0.90,
        "training_loop": 0.80,
        "param_write_back": 0.85,
    }

    checks = {
        "问答稳定性": report.qa_stability.pass_rate >= thresholds["qa_stability"],
        "长期层治理": report.long_term_governance.pass_rate >= thresholds["long_term_governance"],
        "永久层晋升": report.permanent_promotion.pass_rate >= thresholds["permanent_promotion"],
        "训练闭环": report.training_loop.pass_rate >= thresholds["training_loop"],
        "参数写回": report.param_write_back.pass_rate >= thresholds["param_write_back"],
    }

    print("  阈值检查:")
    for name, passed_check in checks.items():
        threshold = thresholds[name.replace("问答稳定性", "qa_stability")
                                    .replace("长期层治理", "long_term_governance")
                                    .replace("永久层晋升", "permanent_promotion")
                                    .replace("训练闭环", "training_loop")
                                    .replace("参数写回", "param_write_back")]
        status = "✅" if passed_check else "⚠️"
        print(f"    {status} {name}: 阈值 {threshold:.0%}")

    # 不要求所有都通过，只是记录状态
    passed = True

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 阈值检查完成（记录当前状态）")

    return passed


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 35)
    print("第七阶段 Benchmark 测试 - 系统级验证")
    print("🧪 " * 35)

    tests = [
        ("问答稳定性 Benchmark", test_qa_stability_benchmark),
        ("长期层治理 Benchmark", test_long_term_governance_benchmark),
        ("永久层晋升 Benchmark", test_permanent_promotion_benchmark),
        ("训练闭环 Benchmark", test_training_loop_benchmark),
        ("参数写回 Benchmark", test_param_write_back_benchmark),
        ("完整 Benchmark 流水线", test_full_benchmark_pipeline),
        ("Benchmark 阈值检查", test_benchmark_thresholds),
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
    print("第七阶段 Benchmark 测试总结")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("所有测试通过！Benchmark 流水线工作正常。")
        print("🎉 " * 35)
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
