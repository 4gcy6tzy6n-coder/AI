"""
Latency Profiler v1 - 延迟分析器 v1

目标：分解每段耗时，找出最大瓶颈
"""

import time
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LatencyBreakdown:
    """延迟分解"""
    total_ms: float
    
    # 各环节耗时
    intent_analysis_ms: float = 0
    query_rewrite_ms: float = 0
    retrieval_trigger_ms: float = 0
    retrieval_search_ms: float = 0
    governance_review_ms: float = 0
    prompt_build_ms: float = 0
    llm_generation_ms: float = 0
    response_build_ms: float = 0
    
    # 其他信息
    retrieval_count: int = 0
    llm_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "intent_analysis_ms": self.intent_analysis_ms,
            "query_rewrite_ms": self.query_rewrite_ms,
            "retrieval_trigger_ms": self.retrieval_trigger_ms,
            "retrieval_search_ms": self.retrieval_search_ms,
            "governance_review_ms": self.governance_review_ms,
            "prompt_build_ms": self.prompt_build_ms,
            "llm_generation_ms": self.llm_generation_ms,
            "response_build_ms": self.response_build_ms,
            "retrieval_count": self.retrieval_count,
            "llm_tokens": self.llm_tokens,
        }


class LatencyProfiler:
    """延迟分析器"""
    
    def __init__(self):
        self.measurements: List[LatencyBreakdown] = []
        self.current: Dict[str, float] = {}
    
    def start(self, stage: str):
        """开始计时某个阶段"""
        self.current[stage] = time.time()
    
    def end(self, stage: str) -> float:
        """结束计时某个阶段，返回耗时（毫秒）"""
        if stage not in self.current:
            return 0
        elapsed = (time.time() - self.current[stage]) * 1000
        del self.current[stage]
        return elapsed
    
    def record(self, breakdown: LatencyBreakdown):
        """记录一次完整测量的延迟分解"""
        self.measurements.append(breakdown)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取延迟统计摘要"""
        if not self.measurements:
            return {}
        
        # 计算各环节平均耗时
        stages = [
            "intent_analysis_ms", "query_rewrite_ms", "retrieval_trigger_ms",
            "retrieval_search_ms", "governance_review_ms", "prompt_build_ms",
            "llm_generation_ms", "response_build_ms"
        ]
        
        summary = {
            "sample_count": len(self.measurements),
            "total_avg_ms": sum(m.total_ms for m in self.measurements) / len(self.measurements),
            "stages": {},
            "bottlenecks": [],
        }
        
        for stage in stages:
            values = [getattr(m, stage) for m in self.measurements]
            avg = sum(values) / len(values)
            max_val = max(values)
            min_val = min(values)
            
            summary["stages"][stage] = {
                "avg_ms": avg,
                "max_ms": max_val,
                "min_ms": min_val,
                "percentage": (avg / summary["total_avg_ms"] * 100) if summary["total_avg_ms"] > 0 else 0,
            }
        
        # 识别瓶颈（耗时占比 > 30% 或绝对值 > 2000ms）
        for stage, stats in summary["stages"].items():
            if stats["percentage"] > 30 or stats["avg_ms"] > 2000:
                summary["bottlenecks"].append({
                    "stage": stage,
                    "avg_ms": stats["avg_ms"],
                    "percentage": stats["percentage"],
                })
        
        # 按耗时排序瓶颈
        summary["bottlenecks"].sort(key=lambda x: x["avg_ms"], reverse=True)
        
        return summary
    
    def print_report(self):
        """打印延迟分析报告"""
        summary = self.get_summary()
        
        if not summary:
            print("暂无延迟数据")
            return
        
        print("\n" + "="*70)
        print("延迟分析报告")
        print("="*70)
        
        print(f"\n样本数: {summary['sample_count']}")
        print(f"平均总延迟: {summary['total_avg_ms']:.0f}ms")
        
        print(f"\n各环节耗时分布:")
        print("-"*70)
        print(f"{'环节':<25} {'平均(ms)':<12} {'占比':<8} {'范围(ms)':<20}")
        print("-"*70)
        
        for stage, stats in sorted(summary["stages"].items(), key=lambda x: x[1]["avg_ms"], reverse=True):
            stage_name = stage.replace("_ms", "").replace("_", " ")
            print(f"{stage_name:<25} {stats['avg_ms']:<12.0f} {stats['percentage']:<8.1f}% {stats['min_ms']:.0f}-{stats['max_ms']:.0f}")
        
        if summary["bottlenecks"]:
            print(f"\n⚠ 识别到的瓶颈:")
            for b in summary["bottlenecks"]:
                print(f"  • {b['stage']}: {b['avg_ms']:.0f}ms ({b['percentage']:.1f}%)")
        
        print("="*70)


# 便捷函数
def create_latency_profiler() -> LatencyProfiler:
    """创建延迟分析器"""
    return LatencyProfiler()


# 测试
if __name__ == "__main__":
    profiler = LatencyProfiler()
    
    # 模拟几次测量
    for i in range(3):
        breakdown = LatencyBreakdown(
            total_ms=4800 + i * 200,
            intent_analysis_ms=10,
            query_rewrite_ms=5,
            retrieval_trigger_ms=15,
            retrieval_search_ms=50,
            governance_review_ms=20,
            prompt_build_ms=30,
            llm_generation_ms=4500 + i * 200,
            response_build_ms=10,
            retrieval_count=2,
            llm_tokens=150
        )
        profiler.record(breakdown)
    
    profiler.print_report()
