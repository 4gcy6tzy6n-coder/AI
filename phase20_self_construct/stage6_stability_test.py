"""
Stage 6 Stability Test Suite

Phase 3 长期稳定性测试套件

测试内容:
1. 100+ 步连续晋升测试 (LongRunningTest)
2. 内存泄漏检查 (MemoryLeakTest)
3. 性能衰减检查 (PerformanceDecayTest)

验收标准:
- 100 步无崩溃
- 内存增长 < 10%
- 执行时间衰减 < 20%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import time
import psutil
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from stage6_system_orchestrator import Stage6SystemOrchestrator


# ==================== 测试结果 ====================

@dataclass
class StabilityMetrics:
    """稳定性指标"""
    total_steps: int
    completed_steps: int
    crashed: bool
    crash_step: int = -1
    crash_reason: str = ""
    
    # 能力指标
    target_gain_trajectory: List[float] = field(default_factory=list)
    old_ability_drop_trajectory: List[float] = field(default_factory=list)
    final_target_gain: float = 0.0
    max_old_ability_drop: float = 0.0
    
    # 内存指标
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_growth_percent: float = 0.0
    
    # 性能指标
    execution_times: List[float] = field(default_factory=list)
    avg_execution_time: float = 0.0
    time_decay_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'crashed': self.crashed,
            'crash_step': self.crash_step,
            'crash_reason': self.crash_reason,
            'final_target_gain': self.final_target_gain,
            'max_old_ability_drop': self.max_old_ability_drop,
            'memory_start_mb': self.memory_start_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_growth_percent': self.memory_growth_percent,
            'avg_execution_time': self.avg_execution_time,
            'time_decay_percent': self.time_decay_percent,
        }


@dataclass
class StabilityReport:
    """稳定性测试报告"""
    experiment_id: str
    timestamp: str
    
    # 100步测试
    long_running_pass: bool = False
    long_running_metrics: Optional[StabilityMetrics] = None
    
    # 内存测试
    memory_pass: bool = False
    memory_growth_percent: float = 0.0
    
    # 性能测试
    performance_pass: bool = False
    time_decay_percent: float = 0.0
    
    # 总体
    overall_pass: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'long_running': {
                'pass': self.long_running_pass,
                'metrics': self.long_running_metrics.to_dict() if self.long_running_metrics else {},
            },
            'memory': {
                'pass': self.memory_pass,
                'growth_percent': self.memory_growth_percent,
            },
            'performance': {
                'pass': self.performance_pass,
                'decay_percent': self.time_decay_percent,
            },
            'overall_pass': self.overall_pass,
        }


# ==================== 内存监控器 ====================

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.measurements = []
    
    def get_current_memory_mb(self) -> float:
        """获取当前内存使用 (MB)"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    def record(self):
        """记录当前内存"""
        current = self.get_current_memory_mb()
        self.measurements.append(current)
        self.peak_memory = max(self.peak_memory, current)
        return current
    
    def get_stats(self) -> Dict:
        """获取内存统计"""
        if not self.measurements:
            return {}
        
        start = self.measurements[0]
        end = self.measurements[-1]
        growth = ((end - start) / start * 100) if start > 0 else 0
        
        return {
            'start_mb': start,
            'end_mb': end,
            'peak_mb': self.peak_memory,
            'growth_percent': growth,
        }


# ==================== 测试套件 ====================

class LongRunningTest:
    """100+ 步连续晋升测试"""
    
    TARGET_STEPS = 100
    TARGET_GAIN_THRESHOLD = 0.10  # 10%
    OLD_DROP_THRESHOLD = 0.15  # 15%
    
    def __init__(self, system: Stage6SystemOrchestrator):
        self.system = system
        self.metrics = StabilityMetrics(total_steps=self.TARGET_STEPS, completed_steps=0, crashed=False)
        self.memory_monitor = MemoryMonitor()
    
    def run(self) -> StabilityMetrics:
        """运行长期测试"""
        print("\n" + "="*70)
        print(f"长期稳定性测试 ({self.TARGET_STEPS} 步)")
        print("="*70)
        
        # 记录初始内存
        self.metrics.memory_start_mb = self.memory_monitor.record()
        print(f"初始内存: {self.metrics.memory_start_mb:.1f} MB")
        
        # 生成测试查询
        queries = [f"长期测试查询 {i+1}: 系统稳定性验证" for i in range(self.TARGET_STEPS)]
        
        try:
            for i, query in enumerate(queries, 1):
                # 记录执行时间
                step_start_time = time.time()
                
                # 执行单步
                trace = self.system._execute_step(i, query)
                
                # 记录执行时间
                execution_time = time.time() - step_start_time
                self.metrics.execution_times.append(execution_time)
                
                # 记录指标
                self.metrics.target_gain_trajectory.append(trace.target_improvement)
                self.metrics.old_ability_drop_trajectory.append(trace.old_ability_drop)
                
                # 每10步记录内存
                if i % 10 == 0:
                    current_memory = self.memory_monitor.record()
                    print(f"  Step {i}/{self.TARGET_STEPS}: "
                          f"target={trace.target_improvement:+.1%}, "
                          f"old_drop={trace.old_ability_drop:.1%}, "
                          f"memory={current_memory:.1f} MB")
                
                self.metrics.completed_steps = i
                
        except Exception as e:
            self.metrics.crashed = True
            self.metrics.crash_step = self.metrics.completed_steps
            self.metrics.crash_reason = str(e)
            print(f"\n✗ 测试中断: {e}")
        
        # 记录最终内存
        self.metrics.memory_end_mb = self.memory_monitor.record()
        self.metrics.memory_peak_mb = self.memory_monitor.peak_memory
        
        # 计算内存增长
        if self.metrics.memory_start_mb > 0:
            self.metrics.memory_growth_percent = (
                (self.metrics.memory_end_mb - self.metrics.memory_start_mb) / 
                self.metrics.memory_start_mb * 100
            )
        
        # 计算最终指标
        if self.metrics.target_gain_trajectory:
            self.metrics.final_target_gain = self.metrics.target_gain_trajectory[-1]
            self.metrics.max_old_ability_drop = max(self.metrics.old_ability_drop_trajectory)
        
        # 计算平均执行时间
        if self.metrics.execution_times:
            self.metrics.avg_execution_time = sum(self.metrics.execution_times) / len(self.metrics.execution_times)
        
        return self.metrics
    
    def evaluate(self) -> bool:
        """评估测试结果"""
        print("\n" + "-"*70)
        print("长期运行评估")
        print("-"*70)
        
        checks = []
        
        # 检查1: 无崩溃
        check1 = not self.metrics.crashed
        checks.append(("无崩溃", check1, f"crashed={self.metrics.crashed}"))
        print(f"  无崩溃: {'✓' if check1 else '✗'} (crashed={self.metrics.crashed})")
        
        # 检查2: 完成目标步数
        check2 = self.metrics.completed_steps >= self.TARGET_STEPS
        checks.append(("完成步数", check2, f"{self.metrics.completed_steps}/{self.TARGET_STEPS}"))
        print(f"  完成步数: {'✓' if check2 else '✗'} ({self.metrics.completed_steps}/{self.TARGET_STEPS})")
        
        # 检查3: 目标能力持续提升
        check3 = self.metrics.final_target_gain >= self.TARGET_GAIN_THRESHOLD
        checks.append(("目标提升", check3, f"{self.metrics.final_target_gain:+.1%}"))
        print(f"  目标提升: {'✓' if check3 else '✗'} ({self.metrics.final_target_gain:+.1%})")
        
        # 检查4: 旧能力保持在阈值内
        check4 = self.metrics.max_old_ability_drop <= self.OLD_DROP_THRESHOLD
        checks.append(("旧能力保护", check4, f"{self.metrics.max_old_ability_drop:.1%}"))
        print(f"  旧能力保护: {'✓' if check4 else '✗'} ({self.metrics.max_old_ability_drop:.1%})")
        
        # 总体通过
        passed = all(c[1] for c in checks)
        print(f"\n  总体: {'✓ 通过' if passed else '✗ 未通过'}")
        
        return passed


class MemoryLeakTest:
    """内存泄漏检查"""
    
    GROWTH_THRESHOLD = 10.0  # 10%
    
    def __init__(self, metrics: StabilityMetrics):
        self.metrics = metrics
    
    def evaluate(self) -> bool:
        """评估内存泄漏"""
        print("\n" + "-"*70)
        print("内存泄漏评估")
        print("-"*70)
        
        print(f"  初始内存: {self.metrics.memory_start_mb:.1f} MB")
        print(f"  最终内存: {self.metrics.memory_end_mb:.1f} MB")
        print(f"  峰值内存: {self.metrics.memory_peak_mb:.1f} MB")
        print(f"  内存增长: {self.metrics.memory_growth_percent:+.1f}%")
        
        passed = abs(self.metrics.memory_growth_percent) < self.GROWTH_THRESHOLD
        print(f"\n  评估: {'✓ 通过' if passed else '✗ 未通过'} (阈值: <{self.GROWTH_THRESHOLD}%)")
        
        return passed


class PerformanceDecayTest:
    """性能衰减检查"""
    
    DECAY_THRESHOLD = 20.0  # 20%
    
    def __init__(self, metrics: StabilityMetrics):
        self.metrics = metrics
    
    def evaluate(self) -> bool:
        """评估性能衰减"""
        print("\n" + "-"*70)
        print("性能衰减评估")
        print("-"*70)
        
        if len(self.metrics.execution_times) < 10:
            print("  数据不足，跳过评估")
            return True
        
        # 分段比较: 前10步 vs 后10步
        first_10_avg = sum(self.metrics.execution_times[:10]) / 10
        last_10_avg = sum(self.metrics.execution_times[-10:]) / 10
        
        decay = ((last_10_avg - first_10_avg) / first_10_avg * 100) if first_10_avg > 0 else 0
        
        print(f"  前10步平均: {first_10_avg:.3f}s")
        print(f"  后10步平均: {last_10_avg:.3f}s")
        print(f"  性能衰减: {decay:+.1f}%")
        
        passed = decay < self.DECAY_THRESHOLD
        print(f"\n  评估: {'✓ 通过' if passed else '✗ 未通过'} (阈值: <{self.DECAY_THRESHOLD}%)")
        
        return passed


# ==================== 测试执行器 ====================

class StabilityTestRunner:
    """稳定性测试执行器"""
    
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id or f"stability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.report = StabilityReport(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_full_suite(self) -> StabilityReport:
        """运行完整稳定性测试套件"""
        print("\n" + "="*70)
        print("Stage 6 Phase 3 - 长期稳定性测试")
        print("="*70)
        print(f"实验ID: {self.experiment_id}")
        print(f"开始时间: {self.report.timestamp}")
        
        # 创建系统
        system = Stage6SystemOrchestrator(experiment_id=self.experiment_id)
        
        # 建立基线
        system.establish_baseline()
        
        # 1. 100步长期运行测试
        long_test = LongRunningTest(system)
        long_metrics = long_test.run()
        self.report.long_running_pass = long_test.evaluate()
        self.report.long_running_metrics = long_metrics
        
        # 2. 内存泄漏检查
        memory_test = MemoryLeakTest(long_metrics)
        self.report.memory_pass = memory_test.evaluate()
        self.report.memory_growth_percent = long_metrics.memory_growth_percent
        
        # 3. 性能衰减检查
        perf_test = PerformanceDecayTest(long_metrics)
        self.report.performance_pass = perf_test.evaluate()
        self.report.time_decay_percent = long_metrics.time_decay_percent
        
        # 总体判断
        self.report.overall_pass = (
            self.report.long_running_pass and
            self.report.memory_pass and
            self.report.performance_pass
        )
        
        return self.report
    
    def export_report(self, filepath: str = None):
        """导出报告"""
        filepath = filepath or f"eval/{self.experiment_id}_stability_report.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        
        print(f"\n✓ 报告已导出: {filepath}")
        return filepath
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*70)
        print("稳定性测试摘要")
        print("="*70)
        
        print(f"\n【100步长期运行】")
        print(f"  完成: {self.report.long_running_metrics.completed_steps if self.report.long_running_metrics else 0}/100 步")
        print(f"  崩溃: {'是' if self.report.long_running_metrics and self.report.long_running_metrics.crashed else '否'}")
        print(f"  最终目标提升: {self.report.long_running_metrics.final_target_gain:+.1%}" if self.report.long_running_metrics else "  最终目标提升: N/A")
        print(f"  最大旧能力掉落: {self.report.long_running_metrics.max_old_ability_drop:.1%}" if self.report.long_running_metrics else "  最大旧能力掉落: N/A")
        print(f"  状态: {'✓ 通过' if self.report.long_running_pass else '✗ 未通过'}")
        
        print(f"\n【内存泄漏】")
        print(f"  内存增长: {self.report.memory_growth_percent:+.1f}%")
        print(f"  状态: {'✓ 通过' if self.report.memory_pass else '✗ 未通过'}")
        
        print(f"\n【性能衰减】")
        print(f"  时间衰减: {self.report.time_decay_percent:+.1f}%")
        print(f"  状态: {'✓ 通过' if self.report.performance_pass else '✗ 未通过'}")
        
        print(f"\n【总体】")
        print(f"  状态: {'✓ 全部通过' if self.report.overall_pass else '✗ 部分未通过'}")
        
        print("\n" + "="*70)


# ==================== 便捷函数 ====================

def run_stability_tests() -> StabilityReport:
    """运行稳定性测试"""
    runner = StabilityTestRunner()
    report = runner.run_full_suite()
    runner.print_summary()
    runner.export_report()
    return report


# ==================== 快速测试 ====================

def run_quick_stability_test(num_steps: int = 20) -> StabilityReport:
    """运行快速稳定性测试 (用于验证)"""
    print("\n" + "="*70)
    print(f"快速稳定性测试 ({num_steps} 步)")
    print("="*70)
    
    # 临时修改目标步数
    original_target = LongRunningTest.TARGET_STEPS
    LongRunningTest.TARGET_STEPS = num_steps
    
    try:
        report = run_stability_tests()
    finally:
        LongRunningTest.TARGET_STEPS = original_target
    
    return report


if __name__ == "__main__":
    # 默认运行快速测试
    run_quick_stability_test(num_steps=20)
