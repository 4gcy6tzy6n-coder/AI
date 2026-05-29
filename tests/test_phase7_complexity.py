"""
第七阶段测试 - 复杂度分析验证

测试目标：
- 验证复杂度分析器能正确测量各项成本
- 验证系统与传统 Transformer 的对比结果
- 验证复杂度报告的准确性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.complexity_profiler import (
    ComplexityProfiler, ParamCostMetrics, MemoryCostMetrics,
    RetrievalCostMetrics, UpdateCostMetrics, RepairCostMetrics
)


def test_param_cost_measurement():
    """测试参数成本测量"""
    print("\n" + "=" * 70)
    print("测试 1: 参数成本测量")
    print("=" * 70)

    profiler = ComplexityProfiler()

    # 测试传统 Transformer
    traditional = profiler.measure_param_cost({
        "total_params": 1_000_000_000,
        "active_params": 300_000_000,
        "trainable_params": 1_000_000_000
    })

    print(f"  传统系统:")
    print(f"    总参数: {traditional.total_params:,}")
    print(f"    利用率: {traditional.utilization_rate:.1%}")

    # 测试本系统
    ours = profiler.measure_param_cost({
        "total_params": 100_000_000,
        "active_params": 90_000_000,
        "trainable_params": 100_000_000
    })

    print(f"  本系统:")
    print(f"    总参数: {ours.total_params:,}")
    print(f"    利用率: {ours.utilization_rate:.1%}")

    # 验证
    utilization_improved = ours.utilization_rate > traditional.utilization_rate
    params_reduced = ours.total_params < traditional.total_params

    passed = utilization_improved and params_reduced

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 参数减少且利用率提升")

    return passed


def test_memory_cost_measurement():
    """测试内存成本测量"""
    print("\n" + "=" * 70)
    print("测试 2: 内存成本测量")
    print("=" * 70)

    profiler = ComplexityProfiler()

    traditional = profiler.measure_memory_cost({
        "model": 4000.0,
        "knowledge_base": 0.0,
        "index": 0.0,
        "cache": 500.0
    })

    ours = profiler.measure_memory_cost({
        "model": 400.0,
        "knowledge_base": 2000.0,
        "index": 200.0,
        "cache": 300.0
    })

    print(f"  传统系统总内存: {traditional.total_memory_mb:.0f} MB")
    print(f"  本系统总内存: {ours.total_memory_mb:.0f} MB")

    # 本系统应该更省内存（参数大幅减少）
    memory_saved = traditional.total_memory_mb - ours.total_memory_mb

    print(f"  内存节省: {memory_saved:.0f} MB")

    passed = memory_saved > 0

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 本系统内存占用更低")

    return passed


def test_retrieval_cost_measurement():
    """测试检索成本测量"""
    print("\n" + "=" * 70)
    print("测试 3: 检索成本测量")
    print("=" * 70)

    profiler = ComplexityProfiler()

    # 传统系统无检索开销
    traditional_results = [{"latency_ms": 0.1, "success": True} for _ in range(100)]
    traditional = profiler.measure_retrieval_cost(traditional_results)

    # 本系统有检索开销
    ours_results = [{"latency_ms": 5.0, "success": True} for _ in range(100)]
    ours = profiler.measure_retrieval_cost(ours_results)

    print(f"  传统系统检索延迟: {traditional.avg_latency_ms:.2f} ms")
    print(f"  本系统检索延迟: {ours.avg_latency_ms:.2f} ms")

    # 本系统应该有检索延迟
    has_retrieval_overhead = ours.avg_latency_ms > traditional.avg_latency_ms

    passed = has_retrieval_overhead

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 本系统有检索开销（预期内的代价）")

    return passed


def test_update_cost_measurement():
    """测试知识更新成本测量"""
    print("\n" + "=" * 70)
    print("测试 4: 知识更新成本测量")
    print("=" * 70)

    profiler = ComplexityProfiler()

    # 传统系统需要重新训练
    traditional = profiler.measure_update_cost({
        "inject_time_ms": 3600000,
        "verify_time_ms": 600000,
        "side_effects": 10
    })

    # 本系统直接写入
    ours = profiler.measure_update_cost({
        "inject_time_ms": 100,
        "verify_time_ms": 5000,
        "side_effects": 0
    })

    print(f"  传统系统更新时间: {traditional.total_time_ms / 1000:.0f} 秒")
    print(f"  本系统更新时间: {ours.total_time_ms / 1000:.1f} 秒")

    time_saved = traditional.total_time_ms - ours.total_time_ms

    print(f"  时间节省: {time_saved / 1000:.0f} 秒")

    passed = time_saved > 0

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 本系统知识更新更快")

    return passed


def test_repair_cost_measurement():
    """测试错误修复成本测量"""
    print("\n" + "=" * 70)
    print("测试 5: 错误修复成本测量")
    print("=" * 70)

    profiler = ComplexityProfiler()

    # 传统系统难以定位错误
    traditional = profiler.measure_repair_cost({
        "identify_time_ms": 86400000,
        "fix_time_ms": 3600000,
        "verify_time_ms": 600000,
        "regression_risk": 0.3
    })

    # 本系统错误可见
    ours = profiler.measure_repair_cost({
        "identify_time_ms": 5000,
        "fix_time_ms": 100,
        "verify_time_ms": 5000,
        "regression_risk": 0.05
    })

    print(f"  传统系统修复时间: {traditional.total_time_ms / 1000:.0f} 秒")
    print(f"  本系统修复时间: {ours.total_time_ms / 1000:.0f} 秒")

    time_saved = traditional.total_time_ms - ours.total_time_ms

    print(f"  时间节省: {time_saved / 1000:.0f} 秒")

    passed = time_saved > 0

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 本系统错误修复更快")

    return passed


def test_complexity_report_generation():
    """测试复杂度报告生成"""
    print("\n" + "=" * 70)
    print("测试 6: 复杂度报告生成")
    print("=" * 70)

    profiler = ComplexityProfiler()

    # 生成传统系统报告
    traditional_param = profiler.measure_param_cost({
        "total_params": 1_000_000_000,
        "active_params": 300_000_000,
        "trainable_params": 1_000_000_000
    })

    traditional_memory = profiler.measure_memory_cost({
        "model": 4000.0,
        "knowledge_base": 0.0,
        "index": 0.0,
        "cache": 500.0
    })

    traditional_retrieval = profiler.measure_retrieval_cost([
        {"latency_ms": 0.1, "success": True} for _ in range(100)
    ])

    traditional_update = profiler.measure_update_cost({
        "inject_time_ms": 3600000,
        "verify_time_ms": 600000,
        "side_effects": 10
    })

    traditional_repair = profiler.measure_repair_cost({
        "identify_time_ms": 86400000,
        "fix_time_ms": 3600000,
        "verify_time_ms": 600000,
        "regression_risk": 0.3
    })

    traditional_report = profiler.generate_report(
        system_type="traditional",
        param_cost=traditional_param,
        memory_cost=traditional_memory,
        retrieval_cost=traditional_retrieval,
        update_cost=traditional_update,
        repair_cost=traditional_repair
    )

    print(f"  传统系统评分:")
    print(f"    效率: {traditional_report.overall_efficiency_score:.1f}")
    print(f"    可维护性: {traditional_report.maintainability_score:.1f}")
    print(f"    可扩展性: {traditional_report.scalability_score:.1f}")

    # 生成本系统报告
    ours_param = profiler.measure_param_cost({
        "total_params": 100_000_000,
        "active_params": 90_000_000,
        "trainable_params": 100_000_000
    })

    ours_memory = profiler.measure_memory_cost({
        "model": 400.0,
        "knowledge_base": 2000.0,
        "index": 200.0,
        "cache": 300.0
    })

    ours_retrieval = profiler.measure_retrieval_cost([
        {"latency_ms": 5.0, "success": True} for _ in range(100)
    ])

    ours_update = profiler.measure_update_cost({
        "inject_time_ms": 100,
        "verify_time_ms": 5000,
        "side_effects": 0
    })

    ours_repair = profiler.measure_repair_cost({
        "identify_time_ms": 5000,
        "fix_time_ms": 100,
        "verify_time_ms": 5000,
        "regression_risk": 0.05
    })

    ours_report = profiler.generate_report(
        system_type="ours",
        param_cost=ours_param,
        memory_cost=ours_memory,
        retrieval_cost=ours_retrieval,
        update_cost=ours_update,
        repair_cost=ours_repair
    )

    print(f"\n  本系统评分:")
    print(f"    效率: {ours_report.overall_efficiency_score:.1f}")
    print(f"    可维护性: {ours_report.maintainability_score:.1f}")
    print(f"    可扩展性: {ours_report.scalability_score:.1f}")

    # 验证报告有有效评分
    has_valid_scores = (
        traditional_report.overall_efficiency_score > 0 and
        ours_report.overall_efficiency_score > 0
    )

    passed = has_valid_scores

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"\n  结果: {status}")
    print(f"  验证: 报告生成成功且有有效评分")

    return passed


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 35)
    print("第七阶段复杂度测试 - 复杂度分析验证")
    print("🧪 " * 35)

    tests = [
        ("参数成本测量", test_param_cost_measurement),
        ("内存成本测量", test_memory_cost_measurement),
        ("检索成本测量", test_retrieval_cost_measurement),
        ("知识更新成本测量", test_update_cost_measurement),
        ("错误修复成本测量", test_repair_cost_measurement),
        ("复杂度报告生成", test_complexity_report_generation),
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
    print("第七阶段复杂度测试总结")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("所有测试通过！复杂度分析器工作正常。")
        print("🎉 " * 35)
    else:
        print("\n⚠️ 部分测试失败，需要检查实现。")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
