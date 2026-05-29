"""
API Gateway Service - API 网关服务

WP1 核心组件：
实现服务化架构的入口服务

功能：
1. 请求路由
2. 鉴权与限流
3. 请求编排
4. 响应聚合
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RequestStatus(Enum):
    """请求状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class GatewayRequest:
    """网关请求"""
    request_id: str
    text: str
    context: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GatewayResponse:
    """网关响应"""
    request_id: str
    status: str
    results: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # 清理过期请求
        self.requests[client_id] = [
            ts for ts in self.requests[client_id]
            if now - ts < self.window_seconds
        ]
        
        # 检查是否超过限制
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # 记录请求
        self.requests[client_id].append(now)
        return True


class APIGateway:
    """
    API 网关服务
    
    职责：
    1. 接收外部请求
    2. 请求鉴权与限流
    3. 路由到下游服务
    4. 聚合响应
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        rate_limit: int = 100
    ):
        self.host = host
        self.port = port
        self.rate_limiter = RateLimiter(max_requests=rate_limit)
        
        # 服务地址配置
        self.services = {
            "retrieval": {"host": "localhost", "port": 8081},
            "governance": {"host": "localhost", "port": 8082},
            "memory": {"host": "localhost", "port": 8083},
            "observability": {"host": "localhost", "port": 8084}
        }
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited": 0
        }
        
        self.is_running = False
    
    async def start(self):
        """启动网关服务"""
        self.is_running = True
        print(f"  API Gateway 已启动: {self.host}:{self.port}")
    
    async def stop(self):
        """停止网关服务"""
        self.is_running = False
        print(f"  API Gateway 已停止")
    
    async def handle_request(self, request: GatewayRequest) -> GatewayResponse:
        """处理请求"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # 生成 trace_id
        trace_id = str(uuid.uuid4())[:12]
        
        # 限流检查
        client_id = request.context.get("client_id", "anonymous")
        if not self.rate_limiter.is_allowed(client_id):
            self.stats["rate_limited"] += 1
            return GatewayResponse(
                request_id=request.request_id,
                status="rate_limited",
                trace_id=trace_id,
                latency_ms=(time.time() - start_time) * 1000,
                error="Rate limit exceeded"
            )
        
        try:
            # 执行请求编排
            results = await self._orchestrate_request(request, trace_id)
            
            self.stats["successful_requests"] += 1
            
            return GatewayResponse(
                request_id=request.request_id,
                status="success",
                results=results,
                trace_id=trace_id,
                latency_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            
            return GatewayResponse(
                request_id=request.request_id,
                status="error",
                trace_id=trace_id,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _orchestrate_request(
        self,
        request: GatewayRequest,
        trace_id: str
    ) -> Dict[str, Any]:
        """编排请求处理"""
        results = {}
        
        # 1. 检索阶段 (如果启用)
        if request.options.get("enable_retrieval", True):
            retrieval_results = await self._call_retrieval_service(request, trace_id)
            results["retrieval"] = retrieval_results
        
        # 2. 治理阶段 (如果启用)
        if request.options.get("enable_governance", True):
            governance_results = await self._call_governance_service(
                request, trace_id, results.get("retrieval", {})
            )
            results["governance"] = governance_results
        
        return results
    
    async def _call_retrieval_service(
        self,
        request: GatewayRequest,
        trace_id: str
    ) -> Dict[str, Any]:
        """调用检索服务"""
        # 模拟调用
        await asyncio.sleep(0.01)
        
        return {
            "status": "success",
            "results": [
                {"unit_id": "unit_001", "score": 0.95},
                {"unit_id": "unit_002", "score": 0.87}
            ],
            "total_found": 2
        }
    
    async def _call_governance_service(
        self,
        request: GatewayRequest,
        trace_id: str,
        retrieval_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用治理服务"""
        # 模拟调用
        await asyncio.sleep(0.02)
        
        return {
            "status": "success",
            "decision": "promote",
            "scores": {
                "QT": 0.92,
                "SL": 0.88,
                "T": 0.85,
                "C": 0.90,
                "L": 0.87
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "service": "api_gateway",
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            **self.stats
        }


async def demo_api_gateway():
    """演示 API Gateway"""
    print("\n" + "="*70)
    print("API Gateway Service - 演示")
    print("="*70)
    
    # 创建网关
    gateway = APIGateway(host="localhost", port=8080)
    
    print("\n1. 启动 API Gateway")
    print("-" * 50)
    await gateway.start()
    
    print("\n2. 发送请求")
    print("-" * 50)
    
    # 发送多个请求
    for i in range(3):
        request = GatewayRequest(
            request_id=f"req_{i:03d}",
            text=f"Query {i}",
            context={"client_id": "client_001"},
            options={
                "enable_retrieval": True,
                "enable_governance": True,
                "max_results": 5
            }
        )
        
        response = await gateway.handle_request(request)
        
        print(f"\n  请求 {i+1}:")
        print(f"    Request ID: {response.request_id}")
        print(f"    Status: {response.status}")
        print(f"    Trace ID: {response.trace_id}")
        print(f"    Latency: {response.latency_ms:.2f} ms")
        
        if response.status == "success":
            print(f"    Retrieval: {len(response.results.get('retrieval', {}).get('results', []))} results")
            print(f"    Governance: {response.results.get('governance', {}).get('decision', 'N/A')}")
        elif response.error:
            print(f"    Error: {response.error}")
    
    print("\n3. 服务统计")
    print("-" * 50)
    
    stats = gateway.get_stats()
    print(f"  总请求: {stats['total_requests']}")
    print(f"  成功: {stats['successful_requests']}")
    print(f"  失败: {stats['failed_requests']}")
    print(f"  限流: {stats['rate_limited']}")
    
    print("\n4. 停止 API Gateway")
    print("-" * 50)
    await gateway.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_api_gateway())
