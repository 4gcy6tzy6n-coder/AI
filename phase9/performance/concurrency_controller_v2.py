"""
Concurrency Controller v2 - 并发控制器 v2

WP1 核心组件：
解决 Phase 8 暴露的并发瓶颈

功能：
1. 读写分离控制
2. 细粒度锁管理
3. 异步任务队列
4. 流量控制与限流
5. 超时与重试机制
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class OperationType(Enum):
    """操作类型"""
    READ = "read"          # 读操作
    WRITE_KB = "write_kb"  # 知识库写入
    WRITE_PARAM = "write_param"  # 参数写入
    GOVERNANCE = "governance"    # 治理操作


class Priority(Enum):
    """任务优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Task:
    """任务定义"""
    task_id: str
    operation_type: OperationType
    priority: Priority
    coro: Coroutine
    created_at: float
    timeout_seconds: float
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ConcurrencyStats:
    """并发统计"""
    active_reads: int = 0
    active_writes_kb: int = 0
    active_writes_param: int = 0
    active_governance: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    timeout_tasks: int = 0
    avg_wait_time_ms: float = 0.0


class ReadWriteLock:
    """
    读写锁
    
    规则：
    - 多个读可以同时进行
    - 写操作独占
    - 写优先于读（防止写饥饿）
    """
    
    def __init__(self):
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._read_count = 0
        self._write_pending = False
        self._read_condition = threading.Condition(self._read_lock)
    
    def acquire_read(self):
        """获取读锁"""
        with self._read_condition:
            # 等待写操作完成
            while self._write_pending:
                self._read_condition.wait()
            self._read_count += 1
    
    def release_read(self):
        """释放读锁"""
        with self._read_condition:
            self._read_count -= 1
            if self._read_count == 0:
                self._read_condition.notify_all()
    
    def acquire_write(self):
        """获取写锁"""
        self._write_pending = True
        self._write_lock.acquire()
        # 等待所有读完成
        with self._read_condition:
            while self._read_count > 0:
                self._read_condition.wait()
    
    def release_write(self):
        """释放写锁"""
        self._write_pending = False
        self._write_lock.release()
        with self._read_condition:
            self._read_condition.notify_all()


class AsyncTaskQueue:
    """
    异步任务队列
    
    功能：
    1. 优先级队列
    2. 超时控制
    3. 重试机制
    4. 流量控制
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        max_queue_size: int = 1000,
        default_timeout: float = 30.0
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        
        # 优先级队列 (按优先级排序)
        self._queue: deque[Task] = deque()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # 统计
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "retried": 0
        }
        
        # 运行状态
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动队列处理器"""
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """停止队列处理器"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def submit(
        self,
        task_id: str,
        operation_type: OperationType,
        coro: Coroutine,
        priority: Priority = Priority.NORMAL,
        timeout_seconds: Optional[float] = None
    ) -> bool:
        """提交任务"""
        async with self._lock:
            if len(self._queue) >= self.max_queue_size:
                return False
            
            task = Task(
                task_id=task_id,
                operation_type=operation_type,
                priority=priority,
                coro=coro,
                created_at=time.time(),
                timeout_seconds=timeout_seconds or self.default_timeout
            )
            
            # 按优先级插入
            inserted = False
            for i, existing in enumerate(self._queue):
                if task.priority.value < existing.priority.value:
                    self._queue.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(task)
            
            self.stats["submitted"] += 1
            return True
    
    async def _process_queue(self):
        """处理队列"""
        while self._running:
            task: Optional[Task] = None
            
            async with self._lock:
                if self._queue:
                    task = self._queue.popleft()
            
            if task:
                async with self._semaphore:
                    await self._execute_task(task)
            else:
                await asyncio.sleep(0.01)  # 避免忙等待
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        wait_time = time.time() - task.created_at
        
        try:
            # 设置超时
            result = await asyncio.wait_for(
                task.coro,
                timeout=task.timeout_seconds
            )
            self.stats["completed"] += 1
            return result
            
        except asyncio.TimeoutError:
            self.stats["timeout"] += 1
            
            # 重试
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.created_at = time.time()
                async with self._lock:
                    self._queue.append(task)
                self.stats["retried"] += 1
            
        except Exception as e:
            self.stats["failed"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "queue_size": len(self._queue),
            "max_queue_size": self.max_queue_size,
            "utilization": len(self._queue) / self.max_queue_size if self.max_queue_size > 0 else 0,
            **self.stats
        }


class ConcurrencyController:
    """
    并发控制器 v2
    
    功能：
    1. 区分读写路径
    2. 细粒度锁控制
    3. 异步任务管理
    4. 流量控制
    """
    
    def __init__(
        self,
        max_concurrent_reads: int = 20,
        max_concurrent_writes_kb: int = 5,
        max_concurrent_writes_param: int = 2,
        max_concurrent_governance: int = 10
    ):
        # 并发限制
        self.max_concurrent = {
            OperationType.READ: max_concurrent_reads,
            OperationType.WRITE_KB: max_concurrent_writes_kb,
            OperationType.WRITE_PARAM: max_concurrent_writes_param,
            OperationType.GOVERNANCE: max_concurrent_governance
        }
        
        # 信号量控制
        self._semaphores = {
            op: asyncio.Semaphore(limit)
            for op, limit in self.max_concurrent.items()
        }
        
        # 读写锁
        self._rw_lock = ReadWriteLock()
        
        # 任务队列
        self._task_queue = AsyncTaskQueue(
            max_concurrent=10,
            max_queue_size=1000
        )
        
        # 统计
        self._stats = ConcurrencyStats()
        self._stats_lock = threading.Lock()
        
        # 限流
        self._rate_limiter = TokenBucketRateLimiter(
            tokens_per_second=100,
            bucket_size=200
        )
    
    async def start(self):
        """启动控制器"""
        await self._task_queue.start()
    
    async def stop(self):
        """停止控制器"""
        await self._task_queue.stop()
    
    async def execute_read(self, coro: Coroutine) -> Any:
        """执行读操作"""
        # 限流检查
        if not self._rate_limiter.consume():
            raise Exception("Rate limit exceeded")
        
        async with self._semaphores[OperationType.READ]:
            self._rw_lock.acquire_read()
            try:
                with self._stats_lock:
                    self._stats.active_reads += 1
                
                result = await coro
                
                with self._stats_lock:
                    self._stats.active_reads -= 1
                    self._stats.completed_tasks += 1
                
                return result
            finally:
                self._rw_lock.release_read()
    
    async def execute_write_kb(self, coro: Coroutine) -> Any:
        """执行知识库写入"""
        async with self._semaphores[OperationType.WRITE_KB]:
            self._rw_lock.acquire_write()
            try:
                with self._stats_lock:
                    self._stats.active_writes_kb += 1
                
                result = await coro
                
                with self._stats_lock:
                    self._stats.active_writes_kb -= 1
                    self._stats.completed_tasks += 1
                
                return result
            finally:
                self._rw_lock.release_write()
    
    async def execute_write_param(self, coro: Coroutine) -> Any:
        """执行参数写入"""
        async with self._semaphores[OperationType.WRITE_PARAM]:
            self._rw_lock.acquire_write()
            try:
                with self._stats_lock:
                    self._stats.active_writes_param += 1
                
                result = await coro
                
                with self._stats_lock:
                    self._stats.active_writes_param -= 1
                    self._stats.completed_tasks += 1
                
                return result
            finally:
                self._rw_lock.release_write()
    
    async def submit_async_task(
        self,
        task_id: str,
        operation_type: OperationType,
        coro: Coroutine,
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """提交异步任务"""
        return await self._task_queue.submit(
            task_id=task_id,
            operation_type=operation_type,
            coro=coro,
            priority=priority
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._stats_lock:
            return {
                "active": {
                    "reads": self._stats.active_reads,
                    "writes_kb": self._stats.active_writes_kb,
                    "writes_param": self._stats.active_writes_param,
                    "governance": self._stats.active_governance
                },
                "limits": {
                    "reads": self.max_concurrent[OperationType.READ],
                    "writes_kb": self.max_concurrent[OperationType.WRITE_KB],
                    "writes_param": self.max_concurrent[OperationType.WRITE_PARAM],
                    "governance": self.max_concurrent[OperationType.GOVERNANCE]
                },
                "queue": self._task_queue.get_stats()
            }


class TokenBucketRateLimiter:
    """
    令牌桶限流器
    
    功能：
    控制请求速率，防止系统过载
    """
    
    def __init__(self, tokens_per_second: float, bucket_size: int):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """消费令牌"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # 补充令牌
            self.tokens = min(
                self.bucket_size,
                self.tokens + elapsed * self.tokens_per_second
            )
            self.last_update = now
            
            # 消费令牌
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "tokens": self.tokens,
                "bucket_size": self.bucket_size,
                "utilization": 1 - (self.tokens / self.bucket_size)
            }


async def demo_concurrency_controller():
    """演示并发控制器"""
    print("\n" + "="*70)
    print("Concurrency Controller v2 - 演示")
    print("="*70)
    
    # 创建控制器
    controller = ConcurrencyController(
        max_concurrent_reads=5,
        max_concurrent_writes_kb=2,
        max_concurrent_writes_param=1
    )
    
    await controller.start()
    
    print("\n1. 读操作并发测试")
    print("-" * 50)
    
    async def mock_read(task_id: str) -> str:
        await asyncio.sleep(0.1)  # 模拟读操作
        return f"Read result for {task_id}"
    
    # 并发执行多个读操作
    read_tasks = [
        controller.execute_read(mock_read(f"read_{i}"))
        for i in range(10)
    ]
    
    results = await asyncio.gather(*read_tasks, return_exceptions=True)
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    print(f"  提交 10 个读操作，成功: {success_count}")
    
    # 显示统计
    stats = controller.get_stats()
    print(f"  活跃读操作: {stats['active']['reads']}/{stats['limits']['reads']}")
    
    print("\n2. 写操作并发测试")
    print("-" * 50)
    
    async def mock_write_kb(task_id: str) -> str:
        await asyncio.sleep(0.2)  # 模拟写操作
        return f"KB write result for {task_id}"
    
    # 并发执行写操作
    write_tasks = [
        controller.execute_write_kb(mock_write_kb(f"write_{i}"))
        for i in range(5)
    ]
    
    results = await asyncio.gather(*write_tasks, return_exceptions=True)
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    print(f"  提交 5 个知识库写操作，成功: {success_count}")
    
    stats = controller.get_stats()
    print(f"  活跃写操作 (KB): {stats['active']['writes_kb']}/{stats['limits']['writes_kb']}")
    
    print("\n3. 异步任务队列测试")
    print("-" * 50)
    
    async def mock_async_task(task_id: str) -> str:
        await asyncio.sleep(0.05)
        return f"Async task {task_id} completed"
    
    # 提交不同优先级的任务
    for i in range(5):
        priority = Priority.CRITICAL if i == 0 else Priority.NORMAL
        await controller.submit_async_task(
            task_id=f"async_{i}",
            operation_type=OperationType.GOVERNANCE,
            coro=mock_async_task(f"task_{i}"),
            priority=priority
        )
    
    print(f"  已提交 5 个异步任务 (1 个 CRITICAL)")
    
    # 等待任务处理
    await asyncio.sleep(0.5)
    
    stats = controller.get_stats()
    print(f"  队列统计:")
    print(f"    队列大小: {stats['queue']['queue_size']}")
    print(f"    已完成: {stats['queue']['completed']}")
    
    print("\n4. 限流测试")
    print("-" * 50)
    
    limiter = TokenBucketRateLimiter(tokens_per_second=10, bucket_size=20)
    
    # 快速消费
    consumed = 0
    for i in range(25):
        if limiter.consume():
            consumed += 1
    
    print(f"  请求 25 个令牌，成功消费: {consumed}")
    
    stats = limiter.get_stats()
    print(f"  剩余令牌: {stats['tokens']:.1f}/{stats['bucket_size']}")
    
    print("\n5. 综合统计")
    print("-" * 50)
    
    stats = controller.get_stats()
    print(f"  并发限制:")
    for op_type, limit in stats['limits'].items():
        active = stats['active'][op_type.value if hasattr(op_type, 'value') else op_type]
        print(f"    {op_type}: {active}/{limit}")
    
    await controller.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_concurrency_controller())
