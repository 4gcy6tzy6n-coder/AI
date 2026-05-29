"""
Worker Queue Prototype - Worker 队列原型

WP3 核心组件：
实现多 Worker 模式原型

功能：
1. 任务队列管理
2. Worker 池管理
3. 优先级调度
4. 任务分发策略
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: float
    timeout_seconds: float
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Worker:
    """Worker 定义"""
    worker_id: str
    task_types: List[str]
    current_task: Optional[str] = None
    total_tasks: int = 0
    failed_tasks: int = 0
    is_active: bool = True


class TaskQueue:
    """
    任务队列
    
    功能：
    1. 优先级队列
    2. 任务状态跟踪
    3. 超时处理
    4. 重试机制
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        
        # 统计
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "retried": 0
        }
    
    async def submit(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: float = 30.0
    ) -> str:
        """提交任务"""
        task_id = str(uuid.uuid4())[:8]
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            payload=payload,
            created_at=time.time(),
            timeout_seconds=timeout_seconds
        )
        
        async with self._lock:
            if len(self._tasks) >= self.max_size:
                raise Exception("Queue is full")
            
            self._tasks[task_id] = task
            self.stats["submitted"] += 1
        
        # 放入优先级队列 (priority, created_at, task_id)
        await self._queue.put((priority.value, task.created_at, task_id))
        
        return task_id
    
    async def get_task(self) -> Optional[Task]:
        """获取任务"""
        try:
            _, _, task_id = await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0
            )
            
            async with self._lock:
                task = self._tasks.get(task_id)
                if task and task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    return task
            
            return None
            
        except asyncio.TimeoutError:
            return None
    
    async def complete_task(self, task_id: str, result: Any):
        """完成任务"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                self.stats["completed"] += 1
    
    async def fail_task(self, task_id: str, error: str):
        """标记任务失败"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    # 重试
                    task.status = TaskStatus.PENDING
                    task.error = None
                    await self._queue.put((task.priority.value, time.time(), task_id))
                    self.stats["retried"] += 1
                else:
                    # 最终失败
                    task.status = TaskStatus.FAILED
                    task.error = error
                    task.completed_at = time.time()
                    self.stats["failed"] += 1
    
    async def timeout_task(self, task_id: str):
        """标记任务超时"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.TIMEOUT
                task.completed_at = time.time()
                self.stats["timeout"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        
        return {
            "queue_size": len(self._tasks),
            "pending": pending,
            "running": running,
            **self.stats
        }


class WorkerPool:
    """
    Worker 池
    
    功能：
    1. Worker 管理
    2. 任务分发
    3. 负载均衡
    4. 故障恢复
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        task_queue: Optional[TaskQueue] = None
    ):
        self.num_workers = num_workers
        self.task_queue = task_queue or TaskQueue()
        
        self.workers: Dict[str, Worker] = {}
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    def register_handler(self, task_type: str, handler: Callable):
        """注册任务处理器"""
        self._handlers[task_type] = handler
    
    async def start(self):
        """启动 Worker 池"""
        self._running = True
        
        # 创建 Workers
        for i in range(self.num_workers):
            worker_id = f"worker_{i:02d}"
            worker = Worker(
                worker_id=worker_id,
                task_types=list(self._handlers.keys())
            )
            self.workers[worker_id] = worker
            
            # 启动 Worker 任务
            task = asyncio.create_task(self._worker_loop(worker_id))
            self._worker_tasks.append(task)
        
        print(f"  Worker 池已启动: {self.num_workers} 个 Workers")
    
    async def stop(self):
        """停止 Worker 池"""
        self._running = False
        
        # 取消所有 Worker 任务
        for task in self._worker_tasks:
            task.cancel()
        
        # 等待取消完成
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        print(f"  Worker 池已停止")
    
    async def _worker_loop(self, worker_id: str):
        """Worker 主循环"""
        while self._running:
            try:
                # 获取任务
                task = await self.task_queue.get_task()
                
                if task:
                    worker = self.workers[worker_id]
                    worker.current_task = task.task_id
                    
                    # 检查任务是否超时
                    elapsed = time.time() - task.created_at
                    if elapsed > task.timeout_seconds:
                        await self.task_queue.timeout_task(task.task_id)
                        worker.current_task = None
                        continue
                    
                    # 执行任务
                    try:
                        handler = self._handlers.get(task.task_type)
                        if handler:
                            result = await asyncio.wait_for(
                                handler(task.payload),
                                timeout=task.timeout_seconds - elapsed
                            )
                            await self.task_queue.complete_task(task.task_id, result)
                            worker.total_tasks += 1
                        else:
                            await self.task_queue.fail_task(
                                task.task_id,
                                f"No handler for task type: {task.task_type}"
                            )
                            worker.failed_tasks += 1
                            
                    except asyncio.TimeoutError:
                        await self.task_queue.timeout_task(task.task_id)
                        worker.failed_tasks += 1
                    except Exception as e:
                        await self.task_queue.fail_task(task.task_id, str(e))
                        worker.failed_tasks += 1
                    
                    worker.current_task = None
                else:
                    # 没有任务，短暂休眠
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        queue_stats = self.task_queue.get_stats()
        
        worker_stats = {
            worker_id: {
                "is_active": worker.is_active,
                "current_task": worker.current_task,
                "total_tasks": worker.total_tasks,
                "failed_tasks": worker.failed_tasks
            }
            for worker_id, worker in self.workers.items()
        }
        
        return {
            "queue": queue_stats,
            "workers": worker_stats,
            "num_workers": len(self.workers)
        }


async def demo_worker_queue():
    """演示 Worker 队列"""
    print("\n" + "="*70)
    print("Worker Queue Prototype - 演示")
    print("="*70)
    
    # 创建 Worker 池
    pool = WorkerPool(num_workers=3)
    
    # 注册任务处理器
    async def handle_query(payload: Dict) -> Dict:
        await asyncio.sleep(0.1)  # 模拟处理时间
        return {"status": "success", "query": payload.get("text", "")}
    
    async def handle_retrieval(payload: Dict) -> Dict:
        await asyncio.sleep(0.2)  # 模拟处理时间
        return {"status": "success", "results": []}
    
    async def handle_governance(payload: Dict) -> Dict:
        await asyncio.sleep(0.15)  # 模拟处理时间
        return {"status": "success", "decision": "promote"}
    
    pool.register_handler("query", handle_query)
    pool.register_handler("retrieval", handle_retrieval)
    pool.register_handler("governance", handle_governance)
    
    print("\n1. 启动 Worker 池")
    print("-" * 50)
    await pool.start()
    
    print("\n2. 提交任务")
    print("-" * 50)
    
    # 提交不同类型和优先级的任务
    task_ids = []
    
    # 普通优先级任务
    for i in range(5):
        task_id = await pool.task_queue.submit(
            task_type="query",
            payload={"text": f"Query {i}"},
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
    
    # 高优先级任务
    for i in range(3):
        task_id = await pool.task_queue.submit(
            task_type="retrieval",
            payload={"vector": [0.1, 0.2, 0.3]},
            priority=TaskPriority.HIGH
        )
        task_ids.append(task_id)
    
    # 关键优先级任务
    task_id = await pool.task_queue.submit(
        task_type="governance",
        payload={"unit_id": "unit_001"},
        priority=TaskPriority.CRITICAL
    )
    task_ids.append(task_id)
    
    print(f"  已提交 {len(task_ids)} 个任务")
    print(f"    - 5 个 query (NORMAL)")
    print(f"    - 3 个 retrieval (HIGH)")
    print(f"    - 1 个 governance (CRITICAL)")
    
    print("\n3. 等待任务处理")
    print("-" * 50)
    
    # 等待任务处理
    for i in range(10):
        await asyncio.sleep(0.3)
        stats = pool.get_stats()
        print(f"  第 {i+1} 秒 - 队列: {stats['queue']['pending']} 待处理, "
              f"{stats['queue']['completed']} 已完成")
        
        if stats['queue']['pending'] == 0:
            break
    
    print("\n4. 最终统计")
    print("-" * 50)
    
    stats = pool.get_stats()
    print(f"  队列统计:")
    print(f"    提交: {stats['queue']['submitted']}")
    print(f"    完成: {stats['queue']['completed']}")
    print(f"    失败: {stats['queue']['failed']}")
    print(f"    超时: {stats['queue']['timeout']}")
    print(f"    重试: {stats['queue']['retried']}")
    
    print(f"\n  Worker 统计:")
    for worker_id, worker_stats in stats['workers'].items():
        print(f"    {worker_id}: {worker_stats['total_tasks']} 任务, "
              f"{worker_stats['failed_tasks']} 失败")
    
    print("\n5. 停止 Worker 池")
    print("-" * 50)
    await pool.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_worker_queue())
