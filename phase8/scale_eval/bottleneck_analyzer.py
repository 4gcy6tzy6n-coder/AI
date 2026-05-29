"""
Bottleneck Analyzer - 瓶颈分析器

WP2 核心组件：
分析规模测试中的性能瓶颈，识别限制扩展性的关键因素

分析维度：
1. 检索瓶颈（Retrieval Bottleneck）
2. 内存瓶颈（Memory Bottleneck）
3. 计算瓶颈（Computation Bottleneck）
4. 并发瓶颈（Concurrency Bottleneck）
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase8.scale_eval.scale_test_runner import ScaleTestRunner, ScaleTier, TestMetrics


class BottleneckType(Enum):
    """瓶颈类型"""
    RETRIEVAL = "retrieval"  # 检索瓶颈
    MEMORY = "memory"  # 内存瓶颈
    COMPUTATION = "computation"  # 计算瓶颈
    CONCURRENCY = "concurrency"  # 并发瓶颈
    NETWORK = "network"  # 网络瓶颈
    UNKNOWN = "unknown"  # 未知


@dataclass
class BottleneckAnalysis:
    """瓶颈分析结果"""
    tier_id: str
    primary_bottleneck: BottleneckType
    secondary_bottlenecks: List[BottleneckType]
    severity: str  # critical/high/medium/low
    description: str
    recommendations: List[str]
    metrics: Dict[str, float]


class BottleneckAnalyzer:
    """
    瓶颈分析器
    
    功能：
    1. 分析各规模层级的性能瓶颈
    2. 识别主要和次要瓶颈
    3. 提供优化建议
    """
    
    def __init__(self):
        # 阈值定义
        self.thresholds = {
            "retrieval_latency_ms": 500,  # 检索延迟阈值
            "memory_per_unit_mb": 0.1,  # 每Unit内存占用
            "cpu_saturation": 80,  # CPU饱和阈值
            "error_rate_critical": 0.20,  # 严重错误率
            "error_rate_high": 0.10,  # 高错误率
            "timeout_rate_critical": 0.30,  # 严重超时率
        }
    
    def _analyze_retrieval_bottleneck(self, tier: ScaleTier, metrics: TestMetrics) -> tuple[float, str]:
        """分析检索瓶颈"""
        score = 0.0
        details = []
        
        # 延迟增长分析
        if metrics.p95_latency_ms > self.thresholds["retrieval_latency_ms"]:
            score += 0.4
            details.append(f"P95延迟过高: {metrics.p95_latency_ms:.1f}ms")
        
        # 检索压力影响
        pressure_factor = {
            "light": 0.0,
            "moderate": 0.2,
            "heavy": 0.4,
            "extreme": 0.6
        }[tier.retrieval_pressure]
        score += pressure_factor
        
        if pressure_factor > 0:
            details.append(f"检索压力: {tier.retrieval_pressure}")
        
        # 超时率
        if metrics.timeout_rate > 0.05:
            score += 0.3
            details.append(f"超时率: {metrics.timeout_rate:.1%}")
        
        return score, "; ".join(details) if details else "无显著检索瓶颈"
    
    def _analyze_memory_bottleneck(self, tier: ScaleTier, metrics: TestMetrics) -> tuple[float, str]:
        """分析内存瓶颈"""
        score = 0.0
        details = []
        
        # 每Unit内存占用
        memory_per_unit = metrics.memory_mb / tier.max_units
        if memory_per_unit > self.thresholds["memory_per_unit_mb"]:
            score += 0.5
            details.append(f"每Unit内存占用过高: {memory_per_unit*1000:.2f}KB")
        
        # 总内存占用
        if metrics.memory_mb > 1000:  # 1GB
            score += 0.3
            details.append(f"总内存占用: {metrics.memory_mb:.0f}MB")
        
        # 规模因子
        scale_factor = tier.max_units / 1000
        if scale_factor > 100:  # 100K+
            score += 0.2
            details.append(f"大规模数据: {tier.max_units:,} Units")
        
        return score, "; ".join(details) if details else "无显著内存瓶颈"
    
    def _analyze_computation_bottleneck(self, tier: ScaleTier, metrics: TestMetrics) -> tuple[float, str]:
        """分析计算瓶颈"""
        score = 0.0
        details = []
        
        # CPU占用
        if metrics.cpu_percent > self.thresholds["cpu_saturation"]:
            score += 0.5
            details.append(f"CPU饱和: {metrics.cpu_percent:.1f}%")
        
        # 平均延迟
        if metrics.avg_latency_ms > 1000:  # 1秒
            score += 0.3
            details.append(f"高延迟: {metrics.avg_latency_ms:.1f}ms")
        
        # 吞吐量不足
        if metrics.throughput_ratio < 0.5:
            score += 0.2
            details.append(f"吞吐量不足: {metrics.throughput_ratio:.1%}")
        
        return score, "; ".join(details) if details else "无显著计算瓶颈"
    
    def _analyze_concurrency_bottleneck(self, tier: ScaleTier, metrics: TestMetrics) -> tuple[float, str]:
        """分析并发瓶颈"""
        score = 0.0
        details = []
        
        # 错误率
        if metrics.error_rate > self.thresholds["error_rate_critical"]:
            score += 0.5
            details.append(f"严重错误率: {metrics.error_rate:.1%}")
        elif metrics.error_rate > self.thresholds["error_rate_high"]:
            score += 0.3
            details.append(f"高错误率: {metrics.error_rate:.1%}")
        
        # 超时率
        if metrics.timeout_rate > self.thresholds["timeout_rate_critical"]:
            score += 0.4
            details.append(f"严重超时率: {metrics.timeout_rate:.1%}")
        
        # QPS与预期差距
        qps_gap = 1 - metrics.throughput_ratio
        if qps_gap > 0.8:
            score += 0.3
            details.append(f"QPS差距过大: {qps_gap:.1%}")
        
        return score, "; ".join(details) if details else "无显著并发瓶颈"
    
    def analyze_tier(self, tier: ScaleTier, metrics: TestMetrics) -> BottleneckAnalysis:
        """分析单个层级的瓶颈"""
        # 分析各类瓶颈
        retrieval_score, retrieval_details = self._analyze_retrieval_bottleneck(tier, metrics)
        memory_score, memory_details = self._analyze_memory_bottleneck(tier, metrics)
        computation_score, computation_details = self._analyze_computation_bottleneck(tier, metrics)
        concurrency_score, concurrency_details = self._analyze_concurrency_bottleneck(tier, metrics)
        
        # 排序瓶颈
        bottleneck_scores = [
            (BottleneckType.RETRIEVAL, retrieval_score, retrieval_details),
            (BottleneckType.MEMORY, memory_score, memory_details),
            (BottleneckType.COMPUTATION, computation_score, computation_details),
            (BottleneckType.CONCURRENCY, concurrency_score, concurrency_details)
        ]
        bottleneck_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 确定主要瓶颈
        primary = bottleneck_scores[0]
        primary_bottleneck = primary[0] if primary[1] > 0.3 else BottleneckType.UNKNOWN
        
        # 确定次要瓶颈
        secondary_bottlenecks = [
            b[0] for b in bottleneck_scores[1:]
            if b[1] > 0.2
        ]
        
        # 确定严重程度
        max_score = max(retrieval_score, memory_score, computation_score, concurrency_score)
        if max_score > 0.8:
            severity = "critical"
        elif max_score > 0.6:
            severity = "high"
        elif max_score > 0.4:
            severity = "medium"
        else:
            severity = "low"
        
        # 生成描述
        if primary_bottleneck == BottleneckType.UNKNOWN:
            description = "未发现显著性能瓶颈"
        else:
            description = f"主要瓶颈: {primary_bottleneck.value} ({primary[1]:.2f}) - {primary[2]}"
        
        # 生成建议
        recommendations = self._generate_recommendations(
            primary_bottleneck, secondary_bottlenecks, tier, metrics
        )
        
        return BottleneckAnalysis(
            tier_id=tier.id,
            primary_bottleneck=primary_bottleneck,
            secondary_bottlenecks=secondary_bottlenecks,
            severity=severity,
            description=description,
            recommendations=recommendations,
            metrics={
                "retrieval_score": retrieval_score,
                "memory_score": memory_score,
                "computation_score": computation_score,
                "concurrency_score": concurrency_score
            }
        )
    
    def _generate_recommendations(
        self,
        primary: BottleneckType,
        secondary: List[BottleneckType],
        tier: ScaleTier,
        metrics: TestMetrics
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 根据主要瓶颈生成建议
        if primary == BottleneckType.RETRIEVAL:
            recommendations.extend([
                "优化检索索引结构，考虑使用分层索引或近似最近邻算法",
                "实施检索结果缓存机制，减少重复查询",
                f"针对{tier.retrieval_pressure}检索压力场景优化检索策略"
            ])
        
        elif primary == BottleneckType.MEMORY:
            recommendations.extend([
                "实施内存分页和懒加载策略",
                "优化Unit数据结构，减少内存占用",
                f"考虑将{tier.max_units:,} Units分批加载"
            ])
        
        elif primary == BottleneckType.COMPUTATION:
            recommendations.extend([
                "并行化处理流程，利用多核CPU",
                "优化核心算法的时间复杂度",
                "考虑使用GPU加速计算密集型任务"
            ])
        
        elif primary == BottleneckType.CONCURRENCY:
            recommendations.extend([
                "实施请求队列和流量控制",
                "增加超时处理和重试机制",
                "优化并发控制策略，减少资源竞争"
            ])
        
        # 通用建议
        if tier.max_units > 100000:
            recommendations.append("考虑实施分布式架构以支持大规模数据处理")
        
        if metrics.qt_score < 0.8:
            recommendations.append("优化质量控制机制，提高QT分数")
        
        return recommendations
    
    def analyze_all_tiers(self, test_results: Dict[str, Any]) -> Dict[str, BottleneckAnalysis]:
        """分析所有层级的瓶颈"""
        print("\n" + "🔍 " * 35)
        print("Bottleneck Analysis - 瓶颈分析")
        print("🔍 " * 35)
        
        analyses = {}
        
        for tier_id in ["S", "M", "L", "X"]:
            if tier_id in test_results:
                result = test_results[tier_id]
                tier = result.tier
                metrics = result.metrics
                
                analysis = self.analyze_tier(tier, metrics)
                analyses[tier_id] = analysis
                
                # 打印分析结果
                print(f"\n{'='*70}")
                print(f"层级 {tier_id} - {tier.description}")
                print(f"{'='*70}")
                
                severity_icon = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }[analysis.severity]
                
                print(f"\n  严重程度: {severity_icon} {analysis.severity.upper()}")
                print(f"  主要瓶颈: {analysis.primary_bottleneck.value}")
                
                if analysis.secondary_bottlenecks:
                    secondary_names = [b.value for b in analysis.secondary_bottlenecks]
                    print(f"  次要瓶颈: {', '.join(secondary_names)}")
                
                print(f"\n  分析详情:")
                print(f"    {analysis.description}")
                
                print(f"\n  瓶颈评分:")
                for metric_name, score in analysis.metrics.items():
                    bar = "█" * int(score * 20)
                    print(f"    {metric_name}: {score:.2f} {bar}")
                
                print(f"\n  优化建议:")
                for i, rec in enumerate(analysis.recommendations, 1):
                    print(f"    {i}. {rec}")
        
        return analyses
    
    def generate_summary(self, analyses: Dict[str, BottleneckAnalysis]) -> Dict[str, Any]:
        """生成分析总结"""
        print("\n" + "="*70)
        print("瓶颈分析总结")
        print("="*70)
        
        # 统计
        severity_count = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        bottleneck_count: Dict[str, int] = {}
        
        for analysis in analyses.values():
            severity_count[analysis.severity] += 1
            
            primary = analysis.primary_bottleneck.value
            bottleneck_count[primary] = bottleneck_count.get(primary, 0) + 1
            
            for secondary in analysis.secondary_bottlenecks:
                secondary_name = secondary.value
                bottleneck_count[secondary_name] = bottleneck_count.get(secondary_name, 0) + 1
        
        print(f"\n  严重程度分布:")
        for severity, count in severity_count.items():
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[severity]
            print(f"    {icon} {severity}: {count}")
        
        print(f"\n  瓶颈类型分布:")
        for bottleneck, count in sorted(bottleneck_count.items(), key=lambda x: x[1], reverse=True):
            print(f"    {bottleneck}: {count}")
        
        # 关键发现
        print(f"\n  关键发现:")
        critical_count = severity_count["critical"] + severity_count["high"]
        if critical_count > 0:
            print(f"    ⚠️ 发现 {critical_count} 个严重瓶颈，需要优先解决")
        
        if bottleneck_count.get("retrieval", 0) >= 2:
            print(f"    📊 检索瓶颈在多个层级出现，建议优先优化索引结构")
        
        if bottleneck_count.get("memory", 0) >= 2:
            print(f"    💾 内存瓶颈在多个层级出现，建议优化数据结构和加载策略")
        
        if bottleneck_count.get("concurrency", 0) >= 2:
            print(f"    🔄 并发瓶颈在多个层级出现，建议优化并发控制机制")
        
        return {
            "severity_distribution": severity_count,
            "bottleneck_distribution": bottleneck_count,
            "critical_issues": critical_count
        }


def demo_bottleneck_analysis():
    """演示瓶颈分析"""
    # 先运行规模测试
    runner = ScaleTestRunner()
    test_results = runner.run_all_tests()
    
    # 分析瓶颈
    analyzer = BottleneckAnalyzer()
    analyses = analyzer.analyze_all_tiers(test_results)
    summary = analyzer.generate_summary(analyses)
    
    return {
        "test_results": test_results,
        "analyses": analyses,
        "summary": summary
    }


if __name__ == "__main__":
    demo_bottleneck_analysis()
