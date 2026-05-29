"""
Load Test Suite V1 - 负载测试套件 V1

Phase 11 WP3 核心组件：
实现系统负载测试、压力测试和稳定性验证

功能：
1. 并发请求测试
2. 压力测试
3. 长时间稳定性测试
4. 性能指标收集
"""

import asyncio
import time
import random
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, RetrievalRequest, RetrievalType
)
from phase11.core_services.governance_service_v1 import (
    GovernanceService, UnitInfo
)
from phase11.core_services.memory_service_v1 import (
    MemoryService, WritebackRequest, MemoryOperation
)


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    latency_ms_list: List[float]
    errors: List[str]
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_ms_list:
            return 0
        return statistics.mean(self.latency_ms_list)
    
    @property
    def p50_latency_ms(self) -> float:
        if not self.latency_ms_list:
            return 0
        return statistics.median(self.latency_ms_list)
    
    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms_list:
            return 0
        sorted_latencies = sorted(self.latency_ms_list)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def p99_latency_ms(self) -> float:
        if not self.latency_ms_list:
            return 0
        sorted_latencies = sorted(self.latency_ms_list)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[:10]  # 只保留前 10 个错误
        }


class LoadTestSuite:
    """负载测试套件"""
    
    def __init__(self):
        self.retrieval_service: RetrievalService = None
        self.governance_service: GovernanceService = None
        self.memory_service: MemoryService = None
        
        self.results: List[TestResult] = []
    
    async def setup(self):
        """初始化服务"""
        print("初始化测试环境...")
        
        self.retrieval_service = RetrievalService()
        self.governance_service = GovernanceService()
        self.memory_service = MemoryService()
        
        await self.retrieval_service.start()
        await self.governance_service.start()
        await self.memory_service.start()
        
        # 准备测试数据
        self._prepare_test_data()
        
        print("✓ 测试环境初始化完成\n")
    
    def _prepare_test_data(self):
        """准备测试数据"""
        # 插入测试记忆
        test_knowledge = [
            ("python", "Python is a high-level programming language"),
            ("machine_learning", "Machine learning is a subset of AI"),
            ("neural_networks", "Neural networks are computing systems"),
            ("deep_learning", "Deep learning uses multiple layers"),
            ("nlp", "Natural language processing enables computers to understand text"),
            ("computer_vision", "Computer vision enables computers to interpret images"),
            ("reinforcement_learning", "Reinforcement learning learns through trial and error"),
            ("transformer", "Transformer is a deep learning model architecture"),
        ]
        
        for i, (key, content) in enumerate(test_knowledge):
            from phase11.core_services.retrieval_service_v1 import MemoryLayer
            self.retrieval_service.insert_memory(
                MemoryLayer.LONG_TERM if i < 4 else MemoryLayer.DEEP_PERMANENT,
                f"knowledge_{key}",
                content,
                {"category": "ai", "confidence": 0.8 + i * 0.02}
            )
        
        print(f"  ✓ 插入 {len(test_knowledge)} 条测试记忆")
    
    async def teardown(self):
        """清理服务"""
        print("\n清理测试环境...")
        
        await self.retrieval_service.stop()
        await self.governance_service.stop()
        await self.memory_service.stop()
        
        print("✓ 测试环境清理完成")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("="*70)
        print("Load Test Suite V1 - 负载测试套件")
        print("="*70)
        
        await self.setup()
        
        # 执行测试
        tests = [
            ("并发检索测试", self.test_concurrent_retrieval),
            ("压力测试 (1000 QPS)", self.test_stress_1000),
            ("稳定性测试 (60s)", self.test_stability_60s),
            ("混合负载测试", self.test_mixed_load),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{test_name}")
            print("-" * 50)
            
            try:
                result = await test_func()
                self.results.append(result)
                self._print_result(result)
                    
            except Exception as e:
                print(f"  ✗ 测试异常: {e}")
        
        await self.teardown()
        
        # 打印汇总
        self._print_summary()
    
    def _print_result(self, result: TestResult):
        """打印测试结果"""
        print(f"  总请求: {result.total_requests}")
        print(f"  成功率: {result.success_rate:.2%}")
        print(f"  平均延迟: {result.avg_latency_ms:.2f}ms")
        print(f"  P50: {result.p50_latency_ms:.2f}ms")
        print(f"  P95: {result.p95_latency_ms:.2f}ms")
        print(f"  P99: {result.p99_latency_ms:.2f}ms")
        print(f"  持续时间: {result.duration_seconds:.2f}s")
        
        if result.errors:
            print(f"  错误数: {len(result.errors)}")
    
    async def test_concurrent_retrieval(self) -> TestResult:
        """并发检索测试"""
        start_time = datetime.now()
        
        total_requests = 100
        concurrency = 10
        
        latency_list = []
        successful = 0
        failed = 0
        errors = []
        
        async def make_request(query_id: int):
            nonlocal successful, failed
            
            try:
                request = RetrievalRequest(
                    query=random.choice(["python", "machine learning", "neural networks"]),
                    query_id=f"concurrent_{query_id}",
                    retrieval_type=RetrievalType.MEMORY,
                    max_results=5
                )
                
                start = time.time()
                response = await self.retrieval_service.retrieve(request)
                latency_ms = (time.time() - start) * 1000
                
                latency_list.append(latency_ms)
                
                if response.status == "success":
                    successful += 1
                else:
                    failed += 1
                    errors.append(response.error or "Unknown error")
                    
            except Exception as e:
                failed += 1
                errors.append(str(e))
        
        # 并发执行
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(query_id: int):
            async with semaphore:
                await make_request(query_id)
        
        await asyncio.gather(*[bounded_request(i) for i in range(total_requests)])
        
        return TestResult(
            test_name="concurrent_retrieval",
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            latency_ms_list=latency_list,
            errors=errors,
            start_time=start_time,
            end_time=datetime.now()
        )
    
    async def test_stress_1000(self) -> TestResult:
        """压力测试 - 1000 请求"""
        start_time = datetime.now()
        
        total_requests = 1000
        concurrency = 50
        
        latency_list = []
        successful = 0
        failed = 0
        errors = []
        
        async def make_request(query_id: int):
            nonlocal successful, failed
            
            try:
                request = RetrievalRequest(
                    query=random.choice([
                        "python programming",
                        "machine learning algorithms",
                        "deep learning frameworks",
                        "natural language processing"
                    ]),
                    query_id=f"stress_{query_id}",
                    retrieval_type=RetrievalType.HYBRID,
                    max_results=10,
                    timeout_ms=2000
                )
                
                start = time.time()
                response = await self.retrieval_service.retrieve(request)
                latency_ms = (time.time() - start) * 1000
                
                latency_list.append(latency_ms)
                
                if response.status == "success":
                    successful += 1
                else:
                    failed += 1
                    if response.error:
                        errors.append(response.error)
                    
            except Exception as e:
                failed += 1
                errors.append(str(e))
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(query_id: int):
            async with semaphore:
                await make_request(query_id)
        
        await asyncio.gather(*[bounded_request(i) for i in range(total_requests)])
        
        return TestResult(
            test_name="stress_1000",
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            latency_ms_list=latency_list,
            errors=errors,
            start_time=start_time,
            end_time=datetime.now()
        )
    
    async def test_stability_60s(self) -> TestResult:
        """稳定性测试 - 持续 60 秒"""
        start_time = datetime.now()
        
        duration_seconds = 60
        qps_target = 20
        
        latency_list = []
        successful = 0
        failed = 0
        errors = []
        total_requests = 0
        
        end_time = start_time + timedelta(seconds=duration_seconds)
        
        while datetime.now() < end_time:
            batch_start = time.time()
            
            # 发送一批请求
            tasks = []
            for i in range(qps_target):
                request = RetrievalRequest(
                    query=random.choice(["python", "ml", "ai", "nlp"]),
                    query_id=f"stability_{total_requests}_{i}",
                    retrieval_type=RetrievalType.MEMORY
                )
                tasks.append(self._make_single_request(request, latency_list, errors))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                total_requests += 1
                if isinstance(result, Exception):
                    failed += 1
                    errors.append(str(result))
                elif result:
                    successful += 1
                else:
                    failed += 1
            
            # 控制 QPS
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
        
        return TestResult(
            test_name="stability_60s",
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            latency_ms_list=latency_list,
            errors=errors,
            start_time=start_time,
            end_time=datetime.now()
        )
    
    async def _make_single_request(self, request, latency_list, errors) -> bool:
        """执行单个请求"""
        try:
            start = time.time()
            response = await self.retrieval_service.retrieve(request)
            latency_ms = (time.time() - start) * 1000
            
            latency_list.append(latency_ms)
            return response.status == "success"
        except Exception as e:
            errors.append(str(e))
            return False
    
    async def test_mixed_load(self) -> TestResult:
        """混合负载测试"""
        start_time = datetime.now()
        
        total_requests = 200
        
        latency_list = []
        successful = 0
        failed = 0
        errors = []
        
        async def make_retrieval_request(query_id: int):
            nonlocal successful, failed
            try:
                request = RetrievalRequest(
                    query="python programming",
                    query_id=f"mixed_retrieval_{query_id}",
                    retrieval_type=RetrievalType.MEMORY
                )
                start = time.time()
                response = await self.retrieval_service.retrieve(request)
                latency_ms = (time.time() - start) * 1000
                latency_list.append(latency_ms)
                
                if response.status == "success":
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))
        
        async def make_governance_request(query_id: int):
            nonlocal successful, failed
            try:
                unit = UnitInfo(
                    unit_id=f"unit_{query_id}",
                    content="Test content",
                    confidence=0.7,
                    stability=0.6,
                    contamination=0.05,
                    current_layer="long_term",
                    age_hours=24,
                    access_count=5
                )
                start = time.time()
                decision = await self.governance_service.govern(
                    query_id=f"mixed_gov_{query_id}",
                    trace_id=f"trace_{query_id}",
                    unit=unit
                )
                latency_ms = (time.time() - start) * 1000
                latency_list.append(latency_ms)
                successful += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))
        
        async def make_memory_request(query_id: int):
            nonlocal successful, failed
            try:
                request = WritebackRequest(
                    query_id=f"mixed_mem_{query_id}",
                    trace_id=f"trace_{query_id}",
                    transaction_id=f"txn_{query_id}",
                    operations=[
                        MemoryOperation(
                            operation_type="create",
                            object_id=f"mixed_obj_{query_id}",
                            target_layer="long_term",
                            content={"test": "data"},
                            reason="Mixed load test"
                        )
                    ]
                )
                start = time.time()
                response = await self.memory_service.writeback(request)
                latency_ms = (time.time() - start) * 1000
                latency_list.append(latency_ms)
                
                if response.status == "committed":
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))
        
        # 混合执行
        tasks = []
        for i in range(total_requests // 3):
            tasks.append(make_retrieval_request(i))
            tasks.append(make_governance_request(i))
            tasks.append(make_memory_request(i))
        
        await asyncio.gather(*tasks)
        
        return TestResult(
            test_name="mixed_load",
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            latency_ms_list=latency_list,
            errors=errors,
            start_time=start_time,
            end_time=datetime.now()
        )
    
    def _print_summary(self):
        """打印测试汇总"""
        print("\n" + "="*70)
        print("测试汇总")
        print("="*70)
        
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        
        print(f"\n总计:")
        print(f"  测试数: {len(self.results)}")
        print(f"  总请求: {total_requests}")
        print(f"  成功率: {total_successful/total_requests*100:.1f}%")
        
        print("\n各测试详情:")
        for result in self.results:
            status = "✓" if result.success_rate >= 0.95 else "⚠" if result.success_rate >= 0.8 else "✗"
            print(f"  {status} {result.test_name}: "
                  f"{result.success_rate:.1%} | "
                  f"P95={result.p95_latency_ms:.1f}ms")
        
        # SLO 检查
        print("\nSLO 检查:")
        slo_passed = all(r.success_rate >= 0.99 and r.p95_latency_ms < 500 
                        for r in self.results)
        if slo_passed:
            print("  ✓ 所有测试满足 SLO 要求")
        else:
            print("  ⚠ 部分测试未达到 SLO 要求")
        
        print("\n" + "="*70)


async def main():
    """主函数"""
    suite = LoadTestSuite()
    await suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
