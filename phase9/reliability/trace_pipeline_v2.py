"""
Trace Pipeline V2 - 全链路追踪 v2

WP4 核心组件：
实现全链路追踪系统

功能：
1. 请求追踪
2. 阶段记录
3. 性能指标
4. 错误追踪
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TraceStage(Enum):
    """追踪阶段"""
    INPUT_SNAPSHOT = "input_snapshot"
    GAP_IDENTIFICATION = "gap_identification"
    RETRIEVAL = "retrieval"
    TSLA = "tsla"
    GATEWAY = "gateway"
    WRITEBACK = "writeback"
    ERROR_FIX = "error_fix"


class TraceStatus(Enum):
    """追踪状态"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class StageRecord:
    """阶段记录"""
    stage: TraceStage
    status: TraceStatus
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    """追踪记录"""
    trace_id: str
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0
    status: TraceStatus = TraceStatus.STARTED
    stages: List[StageRecord] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class TracePipeline:
    """
    全链路追踪管道
    
    功能：
    1. 创建追踪记录
    2. 记录各阶段
    3. 性能指标收集
    4. 错误追踪
    5. 报告生成
    """
    
    def __init__(self, max_traces: int = 10000):
        self.max_traces = max_traces
        self._traces: Dict[str, TraceRecord] = {}
        self._current_traces: Dict[str, str] = {}  # request_id -> trace_id
    
    def start_trace(
        self,
        request_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """开始追踪"""
        trace_id = str(uuid.uuid4())[:12]
        
        trace = TraceRecord(
            trace_id=trace_id,
            request_id=request_id,
            start_time=time.time(),
            context=context or {}
        )
        
        self._traces[trace_id] = trace
        self._current_traces[request_id] = trace_id
        
        return trace_id
    
    def start_stage(
        self,
        trace_id: str,
        stage: TraceStage,
        inputs: Optional[Dict[str, Any]] = None
    ) -> str:
        """开始阶段"""
        trace = self._traces.get(trace_id)
        if not trace:
            return None
        
        stage_record = StageRecord(
            stage=stage,
            status=TraceStatus.STARTED,
            start_time=time.time(),
            inputs=inputs or {}
        )
        
        trace.stages.append(stage_record)
        
        return f"{trace_id}:{stage.value}"
    
    def end_stage(
        self,
        trace_id: str,
        stage: TraceStage,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """结束阶段"""
        trace = self._traces.get(trace_id)
        if not trace:
            return
        
        # 找到对应的阶段记录
        for stage_record in reversed(trace.stages):
            if stage_record.stage == stage and stage_record.status == TraceStatus.STARTED:
                stage_record.end_time = time.time()
                stage_record.duration_ms = (stage_record.end_time - stage_record.start_time) * 1000
                stage_record.outputs = outputs or {}
                
                if error:
                    stage_record.status = TraceStatus.FAILED
                    stage_record.error = error
                else:
                    stage_record.status = TraceStatus.COMPLETED
                
                break
    
    def end_trace(
        self,
        trace_id: str,
        status: TraceStatus = TraceStatus.COMPLETED
    ):
        """结束追踪"""
        trace = self._traces.get(trace_id)
        if not trace:
            return
        
        trace.end_time = time.time()
        trace.total_duration_ms = (trace.end_time - trace.start_time) * 1000
        trace.status = status
        
        # 清理当前追踪映射
        if trace.request_id in self._current_traces:
            del self._current_traces[trace.request_id]
    
    def get_trace(self, trace_id: str) -> Optional[TraceRecord]:
        """获取追踪记录"""
        return self._traces.get(trace_id)
    
    def get_trace_by_request(self, request_id: str) -> Optional[TraceRecord]:
        """通过请求 ID 获取追踪记录"""
        trace_id = self._current_traces.get(request_id)
        if trace_id:
            return self._traces.get(trace_id)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_traces = len(self._traces)
        completed = sum(1 for t in self._traces.values() if t.status == TraceStatus.COMPLETED)
        failed = sum(1 for t in self._traces.values() if t.status == TraceStatus.FAILED)
        
        # 各阶段平均耗时
        stage_durations: Dict[str, List[float]] = {}
        for trace in self._traces.values():
            for stage in trace.stages:
                stage_name = stage.stage.value
                if stage_name not in stage_durations:
                    stage_durations[stage_name] = []
                stage_durations[stage_name].append(stage.duration_ms)
        
        avg_stage_durations = {
            name: sum(durations) / len(durations) if durations else 0
            for name, durations in stage_durations.items()
        }
        
        # 平均总耗时
        avg_total_duration = (
            sum(t.total_duration_ms for t in self._traces.values()) / total_traces
            if total_traces > 0 else 0
        )
        
        return {
            "total_traces": total_traces,
            "completed": completed,
            "failed": failed,
            "in_progress": len(self._current_traces),
            "avg_total_duration_ms": avg_total_duration,
            "avg_stage_durations_ms": avg_stage_durations
        }
    
    def generate_report(self, trace_id: str) -> str:
        """生成追踪报告"""
        trace = self._traces.get(trace_id)
        if not trace:
            return f"Trace {trace_id} not found"
        
        report = []
        report.append("=" * 70)
        report.append(f"Trace Report: {trace_id}")
        report.append("=" * 70)
        report.append(f"Request ID: {trace.request_id}")
        report.append(f"Status: {trace.status.value}")
        report.append(f"Total Duration: {trace.total_duration_ms:.2f} ms")
        report.append(f"Start Time: {datetime.fromtimestamp(trace.start_time).isoformat()}")
        if trace.end_time:
            report.append(f"End Time: {datetime.fromtimestamp(trace.end_time).isoformat()}")
        report.append("")
        
        report.append("Stages:")
        report.append("-" * 70)
        
        for i, stage in enumerate(trace.stages, 1):
            report.append(f"\n  {i}. {stage.stage.value}")
            report.append(f"     Status: {stage.status.value}")
            report.append(f"     Duration: {stage.duration_ms:.2f} ms")
            
            if stage.inputs:
                report.append(f"     Inputs: {json.dumps(stage.inputs, indent=2)[:100]}...")
            
            if stage.outputs:
                report.append(f"     Outputs: {json.dumps(stage.outputs, indent=2)[:100]}...")
            
            if stage.error:
                report.append(f"     Error: {stage.error}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


class TraceContext:
    """
    追踪上下文管理器
    
    用法:
    with TraceContext(pipeline, request_id) as trace_id:
        # 执行操作
        pass
    """
    
    def __init__(
        self,
        pipeline: TracePipeline,
        request_id: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.pipeline = pipeline
        self.request_id = request_id
        self.context = context or {}
        self.trace_id = None
    
    def __enter__(self):
        self.trace_id = self.pipeline.start_trace(self.request_id, self.context)
        return self.trace_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.pipeline.end_trace(self.trace_id, TraceStatus.FAILED)
        else:
            self.pipeline.end_trace(self.trace_id, TraceStatus.COMPLETED)


def demo_trace_pipeline():
    """演示追踪管道"""
    print("\n" + "="*70)
    print("Trace Pipeline V2 - 演示")
    print("="*70)
    
    # 创建追踪管道
    pipeline = TracePipeline()
    
    print("\n1. 模拟请求追踪")
    print("-" * 50)
    
    # 模拟 3 个请求
    for i in range(3):
        request_id = f"req_{i:03d}"
        trace_id = pipeline.start_trace(request_id, {"user_id": f"user_{i}"})
        
        print(f"\n  请求 {i+1}:")
        print(f"    Trace ID: {trace_id}")
        
        # 输入快照阶段
        pipeline.start_stage(trace_id, TraceStage.INPUT_SNAPSHOT, {"text": f"Query {i}"})
        time.sleep(0.01)
        pipeline.end_stage(trace_id, TraceStage.INPUT_SNAPSHOT, {"parsed": True})
        print(f"    ✓ Input Snapshot")
        
        # 缺口识别阶段
        pipeline.start_stage(trace_id, TraceStage.GAP_IDENTIFICATION, {"text": f"Query {i}"})
        time.sleep(0.02)
        has_gap = i % 2 == 0
        pipeline.end_stage(trace_id, TraceStage.GAP_IDENTIFICATION, {"has_gap": has_gap})
        print(f"    ✓ Gap Identification (has_gap={has_gap})")
        
        if has_gap:
            # 检索阶段
            pipeline.start_stage(trace_id, TraceStage.RETRIEVAL, {"query": f"Query {i}"})
            time.sleep(0.05)
            pipeline.end_stage(trace_id, TraceStage.RETRIEVAL, {"results": 3})
            print(f"    ✓ Retrieval")
        
        # TSLA 阶段
        pipeline.start_stage(trace_id, TraceStage.TSLA, {"unit_type": "concept"})
        time.sleep(0.03)
        pipeline.end_stage(trace_id, TraceStage.TSLA, {"action": "promote"})
        print(f"    ✓ TSLA")
        
        # 门控阶段
        pipeline.start_stage(trace_id, TraceStage.GATEWAY, {"scores": {"QT": 0.9}})
        time.sleep(0.02)
        
        # 模拟一个失败
        if i == 1:
            pipeline.end_stage(trace_id, TraceStage.GATEWAY, error="Isolation threshold exceeded")
            print(f"    ✗ Gateway (failed)")
            pipeline.end_trace(trace_id, TraceStatus.FAILED)
        else:
            pipeline.end_stage(trace_id, TraceStage.GATEWAY, {"decision": "pass"})
            print(f"    ✓ Gateway")
            
            # 写回阶段
            pipeline.start_stage(trace_id, TraceStage.WRITEBACK, {"unit_id": f"unit_{i}"})
            time.sleep(0.01)
            pipeline.end_stage(trace_id, TraceStage.WRITEBACK, {"success": True})
            print(f"    ✓ Writeback")
            
            pipeline.end_trace(trace_id, TraceStatus.COMPLETED)
    
    print("\n2. 追踪统计")
    print("-" * 50)
    
    stats = pipeline.get_stats()
    print(f"  总追踪数: {stats['total_traces']}")
    print(f"  完成: {stats['completed']}")
    print(f"  失败: {stats['failed']}")
    print(f"  进行中: {stats['in_progress']}")
    print(f"  平均总耗时: {stats['avg_total_duration_ms']:.2f} ms")
    
    print(f"\n  各阶段平均耗时:")
    for stage, duration in stats['avg_stage_durations_ms'].items():
        print(f"    {stage}: {duration:.2f} ms")
    
    print("\n3. 详细报告示例")
    print("-" * 50)
    
    # 获取第一个完成的追踪
    for trace_id, trace in pipeline._traces.items():
        if trace.status == TraceStatus.COMPLETED:
            report = pipeline.generate_report(trace_id)
            print(report)
            break
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_trace_pipeline()
