"""
Complexity Profiler - 复杂度分析器

第七阶段核心组件：
量化分析系统复杂度，包括：
- 参数量成本
- 显存占用
- 内存占用
- 检索开销
- 知识更新成本
- 错误修复成本

用于对比本系统与传统 Transformer 路线的复杂度差异。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class CostCategory(Enum):
    """成本类别"""
    PARAM = "parameter"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    UPDATE = "update"
    REPAIR = "repair"


@dataclass
class ParamCostMetrics:
    """参数成本指标"""
    total_params: int = 0
    effective_params: int = 0
    utilization_rate: float = 0.0
    trainable_params: int = 0
    frozen_params: int = 0


@dataclass
class MemoryCostMetrics:
    """内存成本指标"""
    model_memory_mb: float = 0.0
    kb_memory_mb: float = 0.0
    index_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    total_memory_mb: float = 0.0


@dataclass
class RetrievalCostMetrics:
    """检索成本指标"""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    accuracy: float = 0.0


@dataclass
class UpdateCostMetrics:
    """知识更新成本指标"""
    inject_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0
    side_effect_count: int = 0


@dataclass
class RepairCostMetrics:
    """错误修复成本指标"""
    identify_time_ms: float = 0.0
    fix_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    regression_risk: float = 0.0
    
    @property
    def total_time_ms(self) -> float:
        """总修复时间"""
        return self.identify_time_ms + self.fix_time_ms + self.verify_time_ms


@dataclass
class ComplexityReport:
    """复杂度报告"""
    timestamp: str
    system_type: str  # "traditional" 或 "ours"
    
    param_cost: ParamCostMetrics
    memory_cost: MemoryCostMetrics
    retrieval_cost: RetrievalCostMetrics
    update_cost: UpdateCostMetrics
    repair_cost: RepairCostMetrics
    
    # 综合评分
    overall_efficiency_score: float = 0.0
    maintainability_score: float = 0.0
    scalability_score: float = 0.0


class ComplexityProfiler:
    """
    复杂度分析器
    
    功能：
    1. 测量参数成本（总参数、有效参数、利用率）
    2. 测量内存成本（模型、知识库、索引、缓存）
    3. 测量检索成本（延迟、吞吐量、准确率）
    4. 测量知识更新成本（注入时间、验证时间）
    5. 测量错误修复成本（识别时间、修复时间）
    6. 生成综合复杂度报告
    
    对比维度：
    - 传统 Transformer 路线（统一参数记忆）
    - 本系统（分层治理 + 外部知识 + 受控参数晋升）
    """
    
    def __init__(self):
        self.measurements: List[Dict] = []
    
    def measure_param_cost(
        self,
        model_info: Dict[str, Any]
    ) -> ParamCostMetrics:
        """
        测量参数成本
        
        Args:
            model_info: 模型信息字典
                - total_params: 总参数量
                - active_params: 活跃参数量
                - trainable_params: 可训练参数量
        """
        total = model_info.get("total_params", 0)
        effective = model_info.get("active_params", total)
        trainable = model_info.get("trainable_params", total)
        
        utilization = effective / total if total > 0 else 0.0
        
        metrics = ParamCostMetrics(
            total_params=total,
            effective_params=effective,
            utilization_rate=utilization,
            trainable_params=trainable,
            frozen_params=total - trainable
        )
        
        self._log_measurement("param", metrics)
        return metrics
    
    def measure_memory_cost(
        self,
        memory_info: Dict[str, float]
    ) -> MemoryCostMetrics:
        """
        测量内存成本
        
        Args:
            memory_info: 内存信息字典（单位：MB）
                - model: 模型内存
                - knowledge_base: 知识库内存
                - index: 索引内存
                - cache: 缓存内存
        """
        model_mb = memory_info.get("model", 0.0)
        kb_mb = memory_info.get("knowledge_base", 0.0)
        index_mb = memory_info.get("index", 0.0)
        cache_mb = memory_info.get("cache", 0.0)
        
        total_mb = model_mb + kb_mb + index_mb + cache_mb
        
        metrics = MemoryCostMetrics(
            model_memory_mb=model_mb,
            kb_memory_mb=kb_mb,
            index_memory_mb=index_mb,
            cache_memory_mb=cache_mb,
            total_memory_mb=total_mb
        )
        
        self._log_measurement("memory", metrics)
        return metrics
    
    def measure_retrieval_cost(
        self,
        retrieval_results: List[Dict]
    ) -> RetrievalCostMetrics:
        """
        测量检索成本
        
        Args:
            retrieval_results: 检索结果列表
                每个元素包含：
                - latency_ms: 检索延迟
                - success: 是否成功
        """
        if not retrieval_results:
            return RetrievalCostMetrics()
        
        latencies = [r["latency_ms"] for r in retrieval_results]
        successes = [r["success"] for r in retrieval_results]
        
        latencies.sort()
        
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = latencies[len(latencies) // 2]
        p99_latency = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]
        
        total_time = sum(latencies) / 1000  # 转换为秒
        throughput = len(latencies) / total_time if total_time > 0 else 0.0
        
        accuracy = sum(successes) / len(successes) if successes else 0.0
        
        metrics = RetrievalCostMetrics(
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p99_latency_ms=p99_latency,
            throughput_qps=throughput,
            accuracy=accuracy
        )
        
        self._log_measurement("retrieval", metrics)
        return metrics
    
    def measure_update_cost(
        self,
        update_info: Dict[str, Any]
    ) -> UpdateCostMetrics:
        """
        测量知识更新成本
        
        Args:
            update_info: 更新信息字典
                - inject_time_ms: 注入时间
                - verify_time_ms: 验证时间
                - side_effects: 副作用数量
        """
        inject_time = update_info.get("inject_time_ms", 0.0)
        verify_time = update_info.get("verify_time_ms", 0.0)
        side_effects = update_info.get("side_effects", 0)
        
        metrics = UpdateCostMetrics(
            inject_time_ms=inject_time,
            verify_time_ms=verify_time,
            total_time_ms=inject_time + verify_time,
            side_effect_count=side_effects
        )
        
        self._log_measurement("update", metrics)
        return metrics
    
    def measure_repair_cost(
        self,
        repair_info: Dict[str, Any]
    ) -> RepairCostMetrics:
        """
        测量错误修复成本
        
        Args:
            repair_info: 修复信息字典
                - identify_time_ms: 识别时间
                - fix_time_ms: 修复时间
                - verify_time_ms: 验证时间
                - regression_risk: 回归风险
        """
        metrics = RepairCostMetrics(
            identify_time_ms=repair_info.get("identify_time_ms", 0.0),
            fix_time_ms=repair_info.get("fix_time_ms", 0.0),
            verify_time_ms=repair_info.get("verify_time_ms", 0.0),
            regression_risk=repair_info.get("regression_risk", 0.0)
        )
        
        self._log_measurement("repair", metrics)
        return metrics
    
    def generate_report(
        self,
        system_type: str,
        param_cost: ParamCostMetrics,
        memory_cost: MemoryCostMetrics,
        retrieval_cost: RetrievalCostMetrics,
        update_cost: UpdateCostMetrics,
        repair_cost: RepairCostMetrics
    ) -> ComplexityReport:
        """
        生成复杂度报告
        
        计算综合评分：
        - 效率评分：参数利用率、检索吞吐量
        - 可维护性：更新成本、修复成本
        - 可扩展性：内存增长、参数增长
        """
        # 效率评分
        efficiency_score = (
            param_cost.utilization_rate * 0.4 +
            min(retrieval_cost.throughput_qps / 1000, 1.0) * 0.3 +
            retrieval_cost.accuracy * 0.3
        ) * 100
        
        # 可维护性评分（成本越低分数越高）
        maintainability_score = (
            max(0, 1 - update_cost.total_time_ms / 1000) * 0.4 +
            max(0, 1 - repair_cost.total_time_ms / 1000) * 0.4 +
            max(0, 1 - repair_cost.regression_risk) * 0.2
        ) * 100
        
        # 可扩展性评分
        scalability_score = (
            max(0, 1 - memory_cost.total_memory_mb / 10000) * 0.5 +
            max(0, 1 - param_cost.total_params / 1e9) * 0.5
        ) * 100
        
        report = ComplexityReport(
            timestamp=datetime.utcnow().isoformat(),
            system_type=system_type,
            param_cost=param_cost,
            memory_cost=memory_cost,
            retrieval_cost=retrieval_cost,
            update_cost=update_cost,
            repair_cost=repair_cost,
            overall_efficiency_score=efficiency_score,
            maintainability_score=maintainability_score,
            scalability_score=scalability_score
        )
        
        return report
    
    def compare_systems(
        self,
        traditional_report: ComplexityReport,
        our_report: ComplexityReport
    ) -> Dict[str, Any]:
        """
        对比两个系统的复杂度
        
        返回对比结果，包括优势和代价
        """
        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "param_efficiency": {
                "traditional_utilization": traditional_report.param_cost.utilization_rate,
                "our_utilization": our_report.param_cost.utilization_rate,
                "improvement": our_report.param_cost.utilization_rate - traditional_report.param_cost.utilization_rate
            },
            "memory_overhead": {
                "traditional_total_mb": traditional_report.memory_cost.total_memory_mb,
                "our_total_mb": our_report.memory_cost.total_memory_mb,
                "overhead_mb": our_report.memory_cost.total_memory_mb - traditional_report.memory_cost.total_memory_mb
            },
            "retrieval_cost": {
                "traditional_latency_ms": traditional_report.retrieval_cost.avg_latency_ms,
                "our_latency_ms": our_report.retrieval_cost.avg_latency_ms,
                "additional_latency_ms": our_report.retrieval_cost.avg_latency_ms - traditional_report.retrieval_cost.avg_latency_ms
            },
            "update_efficiency": {
                "traditional_time_ms": traditional_report.update_cost.total_time_ms,
                "our_time_ms": our_report.update_cost.total_time_ms,
                "improvement_ms": traditional_report.update_cost.total_time_ms - our_report.update_cost.total_time_ms
            },
            "repair_efficiency": {
                "traditional_time_ms": traditional_report.repair_cost.identify_time_ms + traditional_report.repair_cost.fix_time_ms,
                "our_time_ms": our_report.repair_cost.identify_time_ms + our_report.repair_cost.fix_time_ms,
                "improvement_ms": (traditional_report.repair_cost.identify_time_ms + traditional_report.repair_cost.fix_time_ms) -
                                  (our_report.repair_cost.identify_time_ms + our_report.repair_cost.fix_time_ms)
            },
            "overall_scores": {
                "traditional": {
                    "efficiency": traditional_report.overall_efficiency_score,
                    "maintainability": traditional_report.maintainability_score,
                    "scalability": traditional_report.scalability_score
                },
                "ours": {
                    "efficiency": our_report.overall_efficiency_score,
                    "maintainability": our_report.maintainability_score,
                    "scalability": our_report.scalability_score
                }
            }
        }
        
        return comparison
    
    def _log_measurement(self, category: str, metrics):
        """记录测量结果"""
        self.measurements.append({
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "metrics": metrics
        })
    
    def get_measurements(self) -> List[Dict]:
        """获取所有测量记录"""
        return self.measurements.copy()


def demo_complexity_profiler():
    """演示复杂度分析器"""
    print("\n" + "=" * 70)
    print("Complexity Profiler Demo - 复杂度分析器演示")
    print("=" * 70)
    
    profiler = ComplexityProfiler()
    
    # 模拟传统 Transformer 系统
    print("\n1. 测量传统 Transformer 系统")
    traditional_param = profiler.measure_param_cost({
        "total_params": 1_000_000_000,  # 10亿参数
        "active_params": 300_000_000,   # 30%活跃
        "trainable_params": 1_000_000_000
    })
    print(f"   总参数: {traditional_param.total_params:,}")
    print(f"   利用率: {traditional_param.utilization_rate:.1%}")
    
    traditional_memory = profiler.measure_memory_cost({
        "model": 4000.0,  # 4GB
        "knowledge_base": 0.0,
        "index": 0.0,
        "cache": 500.0
    })
    print(f"   内存占用: {traditional_memory.total_memory_mb:.0f} MB")
    
    traditional_retrieval = profiler.measure_retrieval_cost([
        {"latency_ms": 0.1, "success": True} for _ in range(100)
    ])
    print(f"   检索延迟: {traditional_retrieval.avg_latency_ms:.2f} ms")
    
    traditional_update = profiler.measure_update_cost({
        "inject_time_ms": 3600000,  # 1小时训练
        "verify_time_ms": 600000,   # 10分钟验证
        "side_effects": 10
    })
    print(f"   更新时间: {traditional_update.total_time_ms / 1000:.0f} 秒")
    
    traditional_repair = profiler.measure_repair_cost({
        "identify_time_ms": 86400000,  # 1天定位
        "fix_time_ms": 3600000,        # 1小时修复
        "verify_time_ms": 600000,
        "regression_risk": 0.3
    })
    print(f"   修复时间: {(traditional_repair.identify_time_ms + traditional_repair.fix_time_ms) / 1000:.0f} 秒")
    
    # 模拟本系统
    print("\n2. 测量本系统")
    our_param = profiler.measure_param_cost({
        "total_params": 100_000_000,   # 1亿参数
        "active_params": 90_000_000,   # 90%活跃
        "trainable_params": 100_000_000
    })
    print(f"   总参数: {our_param.total_params:,}")
    print(f"   利用率: {our_param.utilization_rate:.1%}")
    
    our_memory = profiler.measure_memory_cost({
        "model": 400.0,      # 400MB
        "knowledge_base": 2000.0,  # 2GB
        "index": 200.0,      # 200MB
        "cache": 300.0       # 300MB
    })
    print(f"   内存占用: {our_memory.total_memory_mb:.0f} MB")
    
    our_retrieval = profiler.measure_retrieval_cost([
        {"latency_ms": 5.0, "success": True} for _ in range(100)
    ])
    print(f"   检索延迟: {our_retrieval.avg_latency_ms:.2f} ms")
    
    our_update = profiler.measure_update_cost({
        "inject_time_ms": 100,   # 100ms写入
        "verify_time_ms": 5000,  # 5秒验证
        "side_effects": 0
    })
    print(f"   更新时间: {our_update.total_time_ms:.0f} ms")
    
    our_repair = profiler.measure_repair_cost({
        "identify_time_ms": 5000,   # 5秒定位
        "fix_time_ms": 100,         # 100ms修复
        "verify_time_ms": 5000,
        "regression_risk": 0.05
    })
    print(f"   修复时间: {(our_repair.identify_time_ms + our_repair.fix_time_ms):.0f} ms")
    
    # 生成报告
    print("\n3. 生成复杂度报告")
    traditional_report = profiler.generate_report(
        system_type="traditional",
        param_cost=traditional_param,
        memory_cost=traditional_memory,
        retrieval_cost=traditional_retrieval,
        update_cost=traditional_update,
        repair_cost=traditional_repair
    )
    
    our_report = profiler.generate_report(
        system_type="ours",
        param_cost=our_param,
        memory_cost=our_memory,
        retrieval_cost=our_retrieval,
        update_cost=our_update,
        repair_cost=our_repair
    )
    
    print(f"\n   传统系统评分:")
    print(f"     效率: {traditional_report.overall_efficiency_score:.1f}")
    print(f"     可维护性: {traditional_report.maintainability_score:.1f}")
    print(f"     可扩展性: {traditional_report.scalability_score:.1f}")
    
    print(f"\n   本系统评分:")
    print(f"     效率: {our_report.overall_efficiency_score:.1f}")
    print(f"     可维护性: {our_report.maintainability_score:.1f}")
    print(f"     可扩展性: {our_report.scalability_score:.1f}")
    
    # 对比
    print("\n4. 系统对比")
    comparison = profiler.compare_systems(traditional_report, our_report)
    
    print(f"\n   参数效率提升: {comparison['param_efficiency']['improvement']:.1%}")
    print(f"   内存开销增加: {comparison['memory_overhead']['overhead_mb']:.0f} MB")
    print(f"   检索延迟增加: {comparison['retrieval_cost']['additional_latency_ms']:.2f} ms")
    print(f"   更新效率提升: {comparison['update_efficiency']['improvement_ms'] / 1000:.0f} 秒")
    print(f"   修复效率提升: {comparison['repair_efficiency']['improvement_ms'] / 1000:.0f} 秒")


if __name__ == "__main__":
    demo_complexity_profiler()
