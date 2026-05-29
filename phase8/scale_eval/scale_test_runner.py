"""
Scale Test Runner - 规模测试运行器

WP2 核心组件：
执行多层级规模测试，验证系统在不同数据规模下的表现

测试层级：
- Small (S): 1K Units, 10 QPS
- Medium (M): 10K Units, 50 QPS
- Large (L): 100K Units, 200 QPS
- Stress (X): 1M Units, 1000 QPS
"""

import time
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ScaleTier:
    """规模层级定义"""
    id: str
    description: str
    max_units: int
    expected_qps: int
    knowledge_injection_intensity: str
    noise_ratio: float
    conflict_ratio: float
    retrieval_pressure: str


@dataclass
class TestMetrics:
    """测试指标"""
    # 延迟指标
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # 吞吐量指标
    actual_qps: float = 0.0
    throughput_ratio: float = 0.0
    
    # 质量指标
    qt_score: float = 0.0
    sl_score: float = 0.0
    
    # 稳定性指标
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # 资源指标
    memory_mb: float = 0.0
    cpu_percent: float = 0.0


@dataclass
class ScaleTestResult:
    """规模测试结果"""
    tier: ScaleTier
    metrics: TestMetrics
    passed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: List[str] = field(default_factory=list)


class ScaleTestRunner:
    """
    规模测试运行器
    
    功能：
    1. 执行四级规模测试（S/M/L/X）
    2. 测量延迟、吞吐量、质量、稳定性
    3. 生成规模扩展趋势报告
    """
    
    def __init__(self):
        # 定义规模层级
        self.scale_tiers = {
            "S": ScaleTier(
                id="S",
                description="小规模基准测试",
                max_units=1000,
                expected_qps=10,
                knowledge_injection_intensity="low",
                noise_ratio=0.05,
                conflict_ratio=0.05,
                retrieval_pressure="light"
            ),
            "M": ScaleTier(
                id="M",
                description="中等规模测试",
                max_units=10000,
                expected_qps=50,
                knowledge_injection_intensity="medium",
                noise_ratio=0.10,
                conflict_ratio=0.10,
                retrieval_pressure="moderate"
            ),
            "L": ScaleTier(
                id="L",
                description="大规模测试",
                max_units=100000,
                expected_qps=200,
                knowledge_injection_intensity="high",
                noise_ratio=0.15,
                conflict_ratio=0.15,
                retrieval_pressure="heavy"
            ),
            "X": ScaleTier(
                id="X",
                description="极限压力测试",
                max_units=1000000,
                expected_qps=1000,
                knowledge_injection_intensity="extreme",
                noise_ratio=0.25,
                conflict_ratio=0.20,
                retrieval_pressure="extreme"
            )
        }
        
        # 基准延迟（小规模下的目标）
        self.baseline_latency_ms = 50.0
        
        # 延迟增长阈值
        self.latency_growth_threshold = 2.0  # 2x
        
        # 质量衰减阈值
        self.quality_degradation_threshold = 0.1  # 10%
    
    def _simulate_processing(self, tier: ScaleTier) -> List[float]:
        """
        模拟处理延迟
        
        模拟逻辑：
        - 延迟随规模增长而增加
        - 增加随机噪声模拟真实环境
        - 高压场景增加超时概率
        """
        latencies = []
        
        # 基础延迟因子
        scale_factor = tier.max_units / 1000  # 相对于小规模
        
        # 检索压力影响
        pressure_multiplier = {
            "light": 1.0,
            "moderate": 1.3,
            "heavy": 1.8,
            "extreme": 2.5
        }[tier.retrieval_pressure]
        
        # 模拟请求
        num_requests = min(tier.expected_qps * 10, 1000)  # 最多1000个请求
        
        for _ in range(num_requests):
            # 基础延迟 + 规模增长 + 随机噪声
            base_latency = self.baseline_latency_ms * (1 + 0.1 * (scale_factor - 1))
            latency = base_latency * pressure_multiplier
            
            # 添加随机噪声（5-15%）
            noise = random.uniform(0.05, 0.15)
            latency *= (1 + noise)
            
            # 高压场景增加超时概率
            if tier.retrieval_pressure == "extreme" and random.random() < 0.05:
                latency *= 3  # 超时请求
            
            latencies.append(latency)
        
        return latencies
    
    def _calculate_metrics(self, tier: ScaleTier, latencies: List[float]) -> TestMetrics:
        """计算测试指标"""
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # 延迟指标
        avg_latency = statistics.mean(latencies)
        p50_latency = sorted_latencies[int(n * 0.50)]
        p95_latency = sorted_latencies[int(n * 0.95)]
        p99_latency = sorted_latencies[int(n * 0.99)]
        
        # 吞吐量计算
        total_time_sec = sum(latencies) / 1000
        actual_qps = len(latencies) / total_time_sec if total_time_sec > 0 else 0
        throughput_ratio = actual_qps / tier.expected_qps if tier.expected_qps > 0 else 0
        
        # 质量指标（模拟）
        # 质量随噪声和冲突比例下降
        base_quality = 1.0
        noise_penalty = tier.noise_ratio * 0.5
        conflict_penalty = tier.conflict_ratio * 0.3
        qt_score = max(0.0, base_quality - noise_penalty - conflict_penalty)
        
        # 稳定性指标
        timeout_threshold = self.baseline_latency_ms * 3
        timeout_count = sum(1 for l in latencies if l > timeout_threshold)
        timeout_rate = timeout_count / len(latencies)
        error_rate = timeout_rate * 0.5  # 假设50%的超时导致错误
        
        # 资源指标（模拟）
        memory_mb = 100 + (tier.max_units / 1000) * 10  # 每1K Units 10MB
        cpu_percent = 20 + tier.expected_qps / 10  # QPS越高CPU越高
        
        return TestMetrics(
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            actual_qps=actual_qps,
            throughput_ratio=throughput_ratio,
            qt_score=qt_score,
            sl_score=1.0 - error_rate,  # 稳定性 = 1 - 错误率
            error_rate=error_rate,
            timeout_rate=timeout_rate,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent
        )
    
    def _evaluate_result(self, tier: ScaleTier, metrics: TestMetrics, baseline: Optional[TestMetrics]) -> tuple[bool, List[str]]:
        """评估测试结果"""
        passed = True
        notes = []
        
        # 检查延迟增长
        if baseline:
            latency_growth = metrics.avg_latency_ms / baseline.avg_latency_ms
            if latency_growth > self.latency_growth_threshold:
                passed = False
                notes.append(f"延迟增长超标: {latency_growth:.2f}x (阈值: {self.latency_growth_threshold}x)")
            else:
                notes.append(f"延迟增长正常: {latency_growth:.2f}x")
        
        # 检查吞吐量
        if metrics.throughput_ratio < 0.8:
            passed = False
            notes.append(f"吞吐量不足: {metrics.throughput_ratio:.1%} (目标: 80%)")
        else:
            notes.append(f"吞吐量达标: {metrics.throughput_ratio:.1%}")
        
        # 检查质量
        if metrics.qt_score < 0.7:
            passed = False
            notes.append(f"质量分数过低: {metrics.qt_score:.2f} (阈值: 0.7)")
        else:
            notes.append(f"质量分数达标: {metrics.qt_score:.2f}")
        
        # 检查错误率
        if metrics.error_rate > 0.05:
            passed = False
            notes.append(f"错误率过高: {metrics.error_rate:.1%} (阈值: 5%)")
        else:
            notes.append(f"错误率正常: {metrics.error_rate:.1%}")
        
        # 检查超时率
        if metrics.timeout_rate > 0.10:
            passed = False
            notes.append(f"超时率过高: {metrics.timeout_rate:.1%} (阈值: 10%)")
        else:
            notes.append(f"超时率正常: {metrics.timeout_rate:.1%}")
        
        return passed, notes
    
    def run_tier_test(self, tier_id: str, baseline: Optional[TestMetrics] = None) -> ScaleTestResult:
        """运行单一层级测试"""
        tier = self.scale_tiers[tier_id]
        
        print(f"\n{'='*70}")
        print(f"运行规模测试: {tier.id} - {tier.description}")
        print(f"{'='*70}")
        print(f"  Units: {tier.max_units:,}")
        print(f"  预期 QPS: {tier.expected_qps}")
        print(f"  噪声比例: {tier.noise_ratio:.0%}")
        print(f"  冲突比例: {tier.conflict_ratio:.0%}")
        print(f"  检索压力: {tier.retrieval_pressure}")
        
        # 模拟处理
        latencies = self._simulate_processing(tier)
        
        # 计算指标
        metrics = self._calculate_metrics(tier, latencies)
        
        # 评估结果
        passed, notes = self._evaluate_result(tier, metrics, baseline)
        
        # 打印结果
        print(f"\n  测试结果:")
        print(f"    平均延迟: {metrics.avg_latency_ms:.2f}ms")
        print(f"    P95延迟: {metrics.p95_latency_ms:.2f}ms")
        print(f"    P99延迟: {metrics.p99_latency_ms:.2f}ms")
        print(f"    实际 QPS: {metrics.actual_qps:.1f}")
        print(f"    吞吐量比: {metrics.throughput_ratio:.1%}")
        print(f"    QT分数: {metrics.qt_score:.2f}")
        print(f"    错误率: {metrics.error_rate:.1%}")
        print(f"    超时率: {metrics.timeout_rate:.1%}")
        print(f"    内存占用: {metrics.memory_mb:.1f}MB")
        print(f"    CPU占用: {metrics.cpu_percent:.1f}%")
        
        status = "✅ 通过" if passed else "❌ 未通过"
        print(f"\n  状态: {status}")
        for note in notes:
            print(f"    - {note}")
        
        return ScaleTestResult(
            tier=tier,
            metrics=metrics,
            passed=passed,
            notes=notes
        )
    
    def run_all_tests(self) -> Dict[str, ScaleTestResult]:
        """运行所有层级测试"""
        print("\n" + "🔬 " * 35)
        print("Scale Test Suite - 规模测试套件")
        print("🔬 " * 35)
        
        results = {}
        baseline = None
        
        # 按顺序运行 S -> M -> L -> X
        for tier_id in ["S", "M", "L", "X"]:
            result = self.run_tier_test(tier_id, baseline)
            results[tier_id] = result
            
            # 小规模作为基准
            if tier_id == "S":
                baseline = result.metrics
        
        return results
    
    def generate_report(self, results: Dict[str, ScaleTestResult]) -> Dict[str, Any]:
        """生成测试报告"""
        print("\n" + "="*70)
        print("规模测试报告")
        print("="*70)
        
        # 汇总
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)
        
        print(f"\n  测试层级: {total_tests}")
        print(f"  通过: {passed_tests}")
        print(f"  未通过: {total_tests - passed_tests}")
        print(f"  总体通过率: {passed_tests/total_tests:.1%}")
        
        # 延迟趋势
        print("\n  延迟趋势:")
        baseline_latency = results["S"].metrics.avg_latency_ms
        for tier_id in ["S", "M", "L", "X"]:
            latency = results[tier_id].metrics.avg_latency_ms
            growth = latency / baseline_latency
            print(f"    {tier_id}: {latency:.2f}ms ({growth:.2f}x)")
        
        # 吞吐量趋势
        print("\n  吞吐量趋势:")
        for tier_id in ["S", "M", "L", "X"]:
            qps = results[tier_id].metrics.actual_qps
            expected = results[tier_id].tier.expected_qps
            ratio = results[tier_id].metrics.throughput_ratio
            print(f"    {tier_id}: {qps:.1f} QPS / {expected} 预期 ({ratio:.1%})")
        
        # 质量趋势
        print("\n  质量趋势:")
        baseline_qt = results["S"].metrics.qt_score
        for tier_id in ["S", "M", "L", "X"]:
            qt = results[tier_id].metrics.qt_score
            degradation = baseline_qt - qt
            print(f"    {tier_id}: QT={qt:.2f} (衰减: {degradation:.2f})")
        
        # 结论
        if passed_tests == total_tests:
            print("\n" + "🎉 " * 35)
            print("所有规模测试通过！系统具备良好的扩展性。")
            print("🎉 " * 35)
        elif passed_tests >= total_tests * 0.5:
            print("\n⚠️ 部分测试通过，系统在特定规模下需要优化。")
        else:
            print("\n❌ 大部分测试未通过，系统扩展性存在问题。")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "results": {
                tier_id: {
                    "passed": r.passed,
                    "avg_latency_ms": r.metrics.avg_latency_ms,
                    "actual_qps": r.metrics.actual_qps,
                    "qt_score": r.metrics.qt_score,
                    "error_rate": r.metrics.error_rate
                }
                for tier_id, r in results.items()
            }
        }


def demo_scale_test():
    """演示规模测试"""
    runner = ScaleTestRunner()
    results = runner.run_all_tests()
    report = runner.generate_report(results)
    return report


if __name__ == "__main__":
    demo_scale_test()
