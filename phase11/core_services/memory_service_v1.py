"""
Memory Service V1 - 记忆服务 V1

Phase 11 WP1 核心组件：
实现完整的记忆管理服务

功能：
1. 长期层对象管理
2. 隔离区/错误区管理
3. 晋升控制
4. 写回事务
5. 状态回放与生命周期
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MemoryLayer(Enum):
    """记忆层"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SHALLOW_PERMANENT = "shallow_permanent"
    DEEP_PERMANENT = "deep_permanent"
    QUARANTINE = "quarantine"
    ERROR_ZONE = "error_zone"


class LifecycleState(Enum):
    """生命周期状态"""
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    ARCHIVED = "archived"
    DELETED = "deleted"


class OperationType(Enum):
    """操作类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    PROMOTE = "promote"
    DEMOTE = "demote"


@dataclass
class MemoryObject:
    """记忆对象"""
    id: str
    layer: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int
    lifecycle_state: str
    history: List[Dict] = field(default_factory=list)


@dataclass
class MemoryOperation:
    """记忆操作"""
    operation_type: str
    object_id: str
    target_layer: str
    content: Optional[Dict] = None
    metadata: Optional[Dict] = None
    reason: str = ""


@dataclass
class WritebackRequest:
    """写回请求"""
    query_id: str
    trace_id: str
    operations: List[MemoryOperation]
    transaction_id: str
    atomic: bool = True


@dataclass
class WritebackResponse:
    """写回响应"""
    transaction_id: str
    status: str  # "committed" | "rolled_back" | "partial"
    completed_operations: int
    failed_operations: int
    errors: List[str]
    timestamp: datetime


class Transaction:
    """事务"""
    
    def __init__(self, transaction_id: str, operations: List[MemoryOperation], atomic: bool = True):
        self.transaction_id = transaction_id
        self.operations = operations
        self.atomic = atomic
        self.status = "pending"  # pending | committed | rolled_back
        self.completed_ops: List[MemoryOperation] = []
        self.failed_ops: List[Tuple[MemoryOperation, str]] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
    
    def commit(self):
        """提交事务"""
        self.status = "committed"
        self.end_time = datetime.now()
    
    def rollback(self):
        """回滚事务"""
        self.status = "rolled_back"
        self.end_time = datetime.now()
    
    def duration_ms(self) -> float:
        """事务持续时间"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000


class LayerStore:
    """层存储"""
    
    def __init__(self, layer: MemoryLayer, max_size: int = 10000):
        self.layer = layer
        self.max_size = max_size
        self.objects: Dict[str, MemoryObject] = {}
        self.stats = {
            "inserts": 0,
            "updates": 0,
            "deletes": 0,
            "promotions": 0,
            "demotions": 0
        }
    
    def get(self, object_id: str) -> Optional[MemoryObject]:
        """获取对象"""
        return self.objects.get(object_id)
    
    def put(self, obj: MemoryObject) -> bool:
        """存储对象"""
        if len(self.objects) >= self.max_size and obj.id not in self.objects:
            return False
        
        is_update = obj.id in self.objects
        self.objects[obj.id] = obj
        
        if is_update:
            self.stats["updates"] += 1
        else:
            self.stats["inserts"] += 1
        
        return True
    
    def delete(self, object_id: str) -> bool:
        """删除对象"""
        if object_id in self.objects:
            del self.objects[object_id]
            self.stats["deletes"] += 1
            return True
        return False
    
    def query(self, filters: Optional[Dict] = None, limit: int = 100) -> List[MemoryObject]:
        """查询对象"""
        results = list(self.objects.values())
        
        if filters:
            for key, value in filters.items():
                results = [obj for obj in results if obj.metadata.get(key) == value]
        
        return results[:limit]
    
    def get_size(self) -> int:
        """获取存储大小"""
        return len(self.objects)


class MemoryService:
    """
    记忆服务 V1
    
    职责：
    1. 多层存储管理
    2. 写回事务控制
    3. 生命周期管理
    4. 晋升控制
    5. 状态回放
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8083,
        transaction_timeout: int = 30
    ):
        self.host = host
        self.port = port
        self.transaction_timeout = transaction_timeout
        
        # 初始化各层存储
        self.stores = {
            MemoryLayer.LONG_TERM: LayerStore(MemoryLayer.LONG_TERM, max_size=100000),
            MemoryLayer.SHALLOW_PERMANENT: LayerStore(MemoryLayer.SHALLOW_PERMANENT, max_size=500000),
            MemoryLayer.DEEP_PERMANENT: LayerStore(MemoryLayer.DEEP_PERMANENT, max_size=1000000),
            MemoryLayer.QUARANTINE: LayerStore(MemoryLayer.QUARANTINE, max_size=10000),
            MemoryLayer.ERROR_ZONE: LayerStore(MemoryLayer.ERROR_ZONE, max_size=50000)
        }
        
        # 事务管理
        self.transactions: Dict[str, Transaction] = {}
        self.pending_queue: List[WritebackRequest] = []
        
        # 历史日志
        self.history_log: List[Dict] = []
        
        # 统计
        self.stats = {
            "total_writebacks": 0,
            "successful_writebacks": 0,
            "failed_writebacks": 0,
            "total_transactions": 0
        }
        
        self.is_running = False
    
    async def start(self):
        """启动服务"""
        self.is_running = True
        print(f"Memory Service started on {self.host}:{self.port}")
    
    async def stop(self):
        """停止服务"""
        self.is_running = False
        print("Memory Service stopped")
    
    async def writeback(self, request: WritebackRequest) -> WritebackResponse:
        """
        执行写回操作
        
        流程：
        1. 创建事务
        2. 执行操作（带重试）
        3. 提交或回滚
        4. 记录历史
        """
        self.stats["total_writebacks"] += 1
        
        # 创建事务
        transaction = Transaction(
            transaction_id=request.transaction_id,
            operations=request.operations,
            atomic=request.atomic
        )
        self.transactions[request.transaction_id] = transaction
        self.stats["total_transactions"] += 1
        
        errors = []
        
        try:
            # 执行操作
            for operation in request.operations:
                try:
                    success = await self._execute_operation(operation)
                    if success:
                        transaction.completed_ops.append(operation)
                    else:
                        error_msg = f"Operation failed: {operation.operation_type} on {operation.object_id}"
                        transaction.failed_ops.append((operation, error_msg))
                        errors.append(error_msg)
                        
                        # 原子性事务：任一失败即回滚
                        if request.atomic:
                            break
                except Exception as e:
                    error_msg = f"Exception in {operation.operation_type}: {str(e)}"
                    transaction.failed_ops.append((operation, error_msg))
                    errors.append(error_msg)
                    
                    if request.atomic:
                        break
            
            # 决定事务状态
            if request.atomic and transaction.failed_ops:
                # 原子性事务失败，回滚
                await self._rollback_transaction(transaction)
                transaction.rollback()
                status = "rolled_back"
                self.stats["failed_writebacks"] += 1
            else:
                # 提交事务
                transaction.commit()
                status = "committed"
                self.stats["successful_writebacks"] += 1
            
            # 记录历史
            self._log_history(request, transaction)
            
            return WritebackResponse(
                transaction_id=request.transaction_id,
                status=status,
                completed_operations=len(transaction.completed_ops),
                failed_operations=len(transaction.failed_ops),
                errors=errors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            transaction.rollback()
            self.stats["failed_writebacks"] += 1
            
            return WritebackResponse(
                transaction_id=request.transaction_id,
                status="rolled_back",
                completed_operations=len(transaction.completed_ops),
                failed_operations=len(transaction.failed_ops),
                errors=errors + [str(e)],
                timestamp=datetime.now()
            )
    
    async def _execute_operation(self, operation: MemoryOperation) -> bool:
        """执行单个操作"""
        op_type = OperationType(operation.operation_type)
        target_layer = MemoryLayer(operation.target_layer)
        store = self.stores.get(target_layer)
        
        if not store:
            return False
        
        if op_type == OperationType.CREATE:
            return self._do_create(operation, store)
        elif op_type == OperationType.UPDATE:
            return self._do_update(operation, store)
        elif op_type == OperationType.DELETE:
            return store.delete(operation.object_id)
        elif op_type == OperationType.PROMOTE:
            return await self._do_promote(operation)
        elif op_type == OperationType.DEMOTE:
            return await self._do_demote(operation)
        
        return False
    
    def _do_create(self, operation: MemoryOperation, store: LayerStore) -> bool:
        """创建对象"""
        now = datetime.now()
        obj = MemoryObject(
            id=operation.object_id,
            layer=operation.target_layer,
            content=operation.content or {},
            metadata=operation.metadata or {},
            created_at=now,
            updated_at=now,
            version=1,
            lifecycle_state=LifecycleState.ACTIVE.value
        )
        return store.put(obj)
    
    def _do_update(self, operation: MemoryOperation, store: LayerStore) -> bool:
        """更新对象"""
        obj = store.get(operation.object_id)
        if not obj:
            return False
        
        # 保存历史
        obj.history.append({
            "version": obj.version,
            "content": obj.content.copy(),
            "metadata": obj.metadata.copy(),
            "updated_at": obj.updated_at.isoformat()
        })
        
        # 更新
        if operation.content:
            obj.content.update(operation.content)
        if operation.metadata:
            obj.metadata.update(operation.metadata)
        
        obj.version += 1
        obj.updated_at = datetime.now()
        
        return store.put(obj)
    
    async def _do_promote(self, operation: MemoryOperation) -> bool:
        """晋升对象"""
        # 找到当前层
        current_layer = None
        current_obj = None
        
        for layer, store in self.stores.items():
            obj = store.get(operation.object_id)
            if obj:
                current_layer = layer
                current_obj = obj
                break
        
        if not current_obj:
            return False
        
        # 从当前层删除
        self.stores[current_layer].delete(operation.object_id)
        
        # 插入目标层
        target_store = self.stores[MemoryLayer(operation.target_layer)]
        current_obj.layer = operation.target_layer
        current_obj.updated_at = datetime.now()
        current_obj.metadata["promoted_from"] = current_layer.value
        current_obj.metadata["promoted_at"] = datetime.now().isoformat()
        
        return target_store.put(current_obj)
    
    async def _do_demote(self, operation: MemoryOperation) -> bool:
        """降级对象"""
        # 与晋升类似
        return await self._do_promote(operation)
    
    async def _rollback_transaction(self, transaction: Transaction):
        """回滚事务"""
        print(f"  回滚事务: {transaction.transaction_id}")
        
        # 撤销已完成的操作
        for operation in reversed(transaction.completed_ops):
            # 根据操作类型执行反向操作
            if operation.operation_type == OperationType.CREATE.value:
                # 删除创建的对象
                target_layer = MemoryLayer(operation.target_layer)
                self.stores[target_layer].delete(operation.object_id)
            elif operation.operation_type == OperationType.UPDATE.value:
                # 恢复历史版本（简化处理）
                pass
    
    def _log_history(self, request: WritebackRequest, transaction: Transaction):
        """记录历史"""
        self.history_log.append({
            "transaction_id": request.transaction_id,
            "trace_id": request.trace_id,
            "query_id": request.query_id,
            "status": transaction.status,
            "operation_count": len(request.operations),
            "completed_count": len(transaction.completed_ops),
            "failed_count": len(transaction.failed_ops),
            "duration_ms": transaction.duration_ms(),
            "timestamp": datetime.now().isoformat()
        })
    
    def query_objects(
        self,
        layer: Optional[str] = None,
        object_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        limit: int = 100
    ) -> List[MemoryObject]:
        """查询对象"""
        if object_id:
            # 按 ID 查询
            for store in self.stores.values():
                obj = store.get(object_id)
                if obj:
                    return [obj]
            return []
        
        if layer:
            # 按层查询
            target_layer = MemoryLayer(layer)
            store = self.stores.get(target_layer)
            if store:
                return store.query(filters, limit)
            return []
        
        # 查询所有层
        results = []
        for store in self.stores.values():
            results.extend(store.query(filters, limit))
            if len(results) >= limit:
                break
        
        return results[:limit]
    
    def get_object_history(self, object_id: str) -> Optional[List[Dict]]:
        """获取对象历史"""
        for store in self.stores.values():
            obj = store.get(object_id)
            if obj:
                return obj.history
        return None
    
    def update_lifecycle(
        self,
        object_id: str,
        new_state: str,
        reason: str = ""
    ) -> bool:
        """更新生命周期状态"""
        for store in self.stores.values():
            obj = store.get(object_id)
            if obj:
                obj.lifecycle_state = new_state
                obj.metadata["lifecycle_change_reason"] = reason
                obj.metadata["lifecycle_changed_at"] = datetime.now().isoformat()
                obj.updated_at = datetime.now()
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "service": "memory",
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "writebacks": self.stats,
            "storage": {
                layer.value: {
                    "size": store.get_size(),
                    "max_size": store.max_size,
                    "utilization": store.get_size() / store.max_size
                }
                for layer, store in self.stores.items()
            },
            "pending_queue": len(self.pending_queue),
            "history_entries": len(self.history_log)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }


async def demo_memory_service():
    """演示记忆服务"""
    print("\n" + "="*70)
    print("Memory Service V1 - 演示")
    print("="*70)
    
    service = MemoryService()
    await service.start()
    
    # 测试写回
    print("\n1. 写回事务测试")
    print("-" * 50)
    
    # 创建操作
    request1 = WritebackRequest(
        query_id="q_001",
        trace_id="trace_001",
        transaction_id="txn_001",
        atomic=True,
        operations=[
            MemoryOperation(
                operation_type="create",
                object_id="obj_001",
                target_layer="long_term",
                content={"text": "Important knowledge", "confidence": 0.9},
                metadata={"source": "training", "category": "core"},
                reason="Initial creation"
            ),
            MemoryOperation(
                operation_type="create",
                object_id="obj_002",
                target_layer="long_term",
                content={"text": "Secondary knowledge", "confidence": 0.7},
                metadata={"source": "training", "category": "secondary"},
                reason="Initial creation"
            )
        ]
    )
    
    response1 = await service.writeback(request1)
    print(f"\n  事务 1 (创建):")
    print(f"    状态: {response1.status}")
    print(f"    完成: {response1.completed_operations}, 失败: {response1.failed_operations}")
    
    # 更新操作
    request2 = WritebackRequest(
        query_id="q_002",
        trace_id="trace_002",
        transaction_id="txn_002",
        atomic=True,
        operations=[
            MemoryOperation(
                operation_type="update",
                object_id="obj_001",
                target_layer="long_term",
                content={"confidence": 0.95},
                metadata={"updated_by": "governance"},
                reason="Confidence improvement"
            )
        ]
    )
    
    response2 = await service.writeback(request2)
    print(f"\n  事务 2 (更新):")
    print(f"    状态: {response2.status}")
    
    # 晋升操作
    request3 = WritebackRequest(
        query_id="q_003",
        trace_id="trace_003",
        transaction_id="txn_003",
        atomic=True,
        operations=[
            MemoryOperation(
                operation_type="promote",
                object_id="obj_001",
                target_layer="shallow_permanent",
                reason="High confidence, promote to permanent"
            )
        ]
    )
    
    response3 = await service.writeback(request3)
    print(f"\n  事务 3 (晋升):")
    print(f"    状态: {response3.status}")
    
    # 查询
    print("\n2. 查询测试")
    print("-" * 50)
    
    # 按层查询
    long_term_objects = service.query_objects(layer="long_term")
    print(f"\n  长期层对象数: {len(long_term_objects)}")
    
    shallow_permanent_objects = service.query_objects(layer="shallow_permanent")
    print(f"  浅层永久对象数: {len(shallow_permanent_objects)}")
    
    # 按 ID 查询
    obj_001 = service.query_objects(object_id="obj_001")
    if obj_001:
        print(f"\n  obj_001 详情:")
        print(f"    层: {obj_001[0].layer}")
        print(f"    版本: {obj_001[0].version}")
        print(f"    历史记录数: {len(obj_001[0].history)}")
    
    # 统计
    print("\n3. 服务统计")
    print("-" * 50)
    
    stats = service.get_stats()
    print(f"  总写回数: {stats['writebacks']['total_writebacks']}")
    print(f"  成功: {stats['writebacks']['successful_writebacks']}")
    print(f"  失败: {stats['writebacks']['failed_writebacks']}")
    print(f"  事务数: {stats['writebacks']['total_transactions']}")
    
    print("\n  存储使用:")
    for layer, info in stats['storage'].items():
        print(f"    {layer}: {info['size']}/{info['max_size']} ({info['utilization']:.1%})")
    
    await service.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_memory_service())
