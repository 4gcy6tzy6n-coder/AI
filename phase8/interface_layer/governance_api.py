"""
Governance API - 治理系统 API 层

WP4 核心组件：
提供产品化的 REST API 接口，支持外部系统调用

API 端点：
1. /api/v1/process - 处理输入文本
2. /api/v1/governance/status - 获取治理状态
3. /api/v1/governance/metrics - 获取治理指标
4. /api/v1/units/{unit_id} - 获取 Unit 详情
5. /api/v1/batch/process - 批量处理
"""

import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class APIStatus(Enum):
    """API 状态"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    TIMEOUT = "timeout"


@dataclass
class APIResponse:
    """API 响应"""
    status: str
    data: Any
    message: str
    timestamp: str
    request_id: str
    latency_ms: float


@dataclass
class ProcessRequest:
    """处理请求"""
    text: str
    language: str = "zh"
    options: Dict[str, Any] = None


@dataclass
class ProcessResult:
    """处理结果"""
    request_id: str
    text: str
    language: str
    units: Dict[str, Any]
    governance_scores: Dict[str, float]
    processing_time_ms: float


class GovernanceAPI:
    """
    治理系统 API
    
    功能：
    1. 提供标准化的 API 接口
    2. 处理输入并返回治理结果
    3. 支持批量处理
    4. 提供系统状态查询
    """
    
    def __init__(self):
        self.request_counter = 0
        self.processing_history = []
        
        # 模拟治理系统状态
        self.system_status = {
            "status": "running",
            "version": "v0.8.0",
            "uptime_seconds": 3600,
            "total_requests": 0,
            "avg_latency_ms": 45.0
        }
    
    def _generate_request_id(self) -> str:
        """生成请求 ID"""
        self.request_counter += 1
        return f"req_{int(time.time())}_{self.request_counter:06d}"
    
    def _simulate_processing(self, text: str, language: str) -> Dict[str, Any]:
        """模拟处理逻辑"""
        start_time = time.time()
        
        # 模拟延迟
        time.sleep(0.01)
        
        # 模拟 Unit 识别
        units = {
            "concepts": self._extract_concepts(text),
            "relations": self._extract_relations(text),
            "rules": self._extract_rules(text),
            "task_patterns": self._extract_task_patterns(text)
        }
        
        # 模拟治理分数
        governance_scores = {
            "QT": random.uniform(0.75, 0.95),
            "SL": random.uniform(0.80, 0.98),
            "T": random.uniform(0.70, 0.90),
            "C": random.uniform(0.75, 0.95),
            "L": random.uniform(0.80, 0.95)
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "units": units,
            "governance_scores": governance_scores,
            "processing_time_ms": processing_time
        }
    
    def _extract_concepts(self, text: str) -> List[Dict]:
        """模拟概念提取"""
        # 简单提取"名词"作为概念
        words = text.split()
        concepts = []
        for i, word in enumerate(words[:5]):  # 最多5个
            if len(word) > 2:
                concepts.append({
                    "id": f"concept_{i}",
                    "text": word,
                    "type": "concept",
                    "confidence": random.uniform(0.7, 0.95)
                })
        return concepts
    
    def _extract_relations(self, text: str) -> List[Dict]:
        """模拟关系提取"""
        relations = []
        relation_keywords = ["是", "导致", "属于", "包含", "影响"]
        for keyword in relation_keywords:
            if keyword in text:
                relations.append({
                    "id": f"rel_{len(relations)}",
                    "type": "relation",
                    "marker": keyword,
                    "confidence": random.uniform(0.6, 0.9)
                })
        return relations[:3]  # 最多3个
    
    def _extract_rules(self, text: str) -> List[Dict]:
        """模拟规则提取"""
        rules = []
        if "如果" in text or "if" in text.lower():
            rules.append({
                "id": "rule_0",
                "type": "inference",
                "condition": "detected",
                "confidence": random.uniform(0.7, 0.9)
            })
        return rules
    
    def _extract_task_patterns(self, text: str) -> List[Dict]:
        """模拟任务模式提取"""
        tasks = []
        task_keywords = {
            "什么是": "definition",
            "为什么": "reasoning",
            "解释": "explanation",
            "比较": "comparison",
            "验证": "verification"
        }
        for keyword, task_type in task_keywords.items():
            if keyword in text:
                tasks.append({
                    "id": f"task_{len(tasks)}",
                    "type": task_type,
                    "trigger": keyword,
                    "confidence": random.uniform(0.75, 0.95)
                })
        return tasks[:2]  # 最多2个
    
    def process(self, request: ProcessRequest) -> APIResponse:
        """
        处理单个请求
        
        POST /api/v1/process
        
        Request Body:
        {
            "text": "输入文本",
            "language": "zh",
            "options": {}
        }
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        try:
            # 处理
            result_data = self._simulate_processing(request.text, request.language)
            
            result = ProcessResult(
                request_id=request_id,
                text=request.text,
                language=request.language,
                units=result_data["units"],
                governance_scores=result_data["governance_scores"],
                processing_time_ms=result_data["processing_time_ms"]
            )
            
            # 记录历史
            self.processing_history.append({
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "text_length": len(request.text),
                "language": request.language
            })
            
            self.system_status["total_requests"] += 1
            
            latency = (time.time() - start_time) * 1000
            
            return APIResponse(
                status=APIStatus.SUCCESS.value,
                data=asdict(result),
                message="处理成功",
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return APIResponse(
                status=APIStatus.ERROR.value,
                data=None,
                message=str(e),
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                latency_ms=latency
            )
    
    def batch_process(self, requests: List[ProcessRequest]) -> APIResponse:
        """
        批量处理
        
        POST /api/v1/batch/process
        
        Request Body:
        {
            "requests": [
                {"text": "...", "language": "zh"},
                ...
            ]
        }
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        results = []
        for req in requests:
            response = self.process(req)
            results.append(response.data if response.status == APIStatus.SUCCESS.value else None)
        
        latency = (time.time() - start_time) * 1000
        
        return APIResponse(
            status=APIStatus.SUCCESS.value,
            data={
                "batch_id": request_id,
                "total": len(requests),
                "successful": sum(1 for r in results if r is not None),
                "results": results
            },
            message="批量处理完成",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            latency_ms=latency
        )
    
    def get_governance_status(self) -> APIResponse:
        """
        获取治理状态
        
        GET /api/v1/governance/status
        """
        return APIResponse(
            status=APIStatus.SUCCESS.value,
            data=self.system_status,
            message="获取成功",
            timestamp=datetime.now().isoformat(),
            request_id=self._generate_request_id(),
            latency_ms=1.0
        )
    
    def get_governance_metrics(self) -> APIResponse:
        """
        获取治理指标
        
        GET /api/v1/governance/metrics
        """
        metrics = {
            "qt_score": random.uniform(0.80, 0.95),
            "sl_score": random.uniform(0.85, 0.98),
            "t_score": random.uniform(0.75, 0.90),
            "c_score": random.uniform(0.80, 0.95),
            "l_score": random.uniform(0.85, 0.95),
            "avg_latency_ms": self.system_status["avg_latency_ms"],
            "throughput_qps": random.uniform(45, 55),
            "error_rate": random.uniform(0.01, 0.05),
            "unit_count": random.randint(10000, 50000)
        }
        
        return APIResponse(
            status=APIStatus.SUCCESS.value,
            data=metrics,
            message="获取成功",
            timestamp=datetime.now().isoformat(),
            request_id=self._generate_request_id(),
            latency_ms=1.0
        )
    
    def get_unit_details(self, unit_id: str) -> APIResponse:
        """
        获取 Unit 详情
        
        GET /api/v1/units/{unit_id}
        """
        # 模拟 Unit 详情
        unit_details = {
            "unit_id": unit_id,
            "unit_type": random.choice(["concept", "relation", "rule", "task_pattern"]),
            "created_at": datetime.now().isoformat(),
            "quality_score": random.uniform(0.7, 0.95),
            "stability_cycles": random.randint(5, 50),
            "references": random.randint(1, 20),
            "metadata": {
                "source": random.choice(["user_input", "inference", "external"]),
                "language": random.choice(["zh", "en"]),
                "confidence": random.uniform(0.7, 0.95)
            }
        }
        
        return APIResponse(
            status=APIStatus.SUCCESS.value,
            data=unit_details,
            message="获取成功",
            timestamp=datetime.now().isoformat(),
            request_id=self._generate_request_id(),
            latency_ms=2.0
        )
    
    def get_api_documentation(self) -> Dict[str, Any]:
        """获取 API 文档"""
        return {
            "api_version": "v1",
            "base_url": "/api/v1",
            "endpoints": [
                {
                    "path": "/process",
                    "method": "POST",
                    "description": "处理输入文本",
                    "parameters": {
                        "text": "输入文本（必填）",
                        "language": "语言代码（可选，默认zh）",
                        "options": "处理选项（可选）"
                    }
                },
                {
                    "path": "/batch/process",
                    "method": "POST",
                    "description": "批量处理多个文本",
                    "parameters": {
                        "requests": "请求列表"
                    }
                },
                {
                    "path": "/governance/status",
                    "method": "GET",
                    "description": "获取系统状态"
                },
                {
                    "path": "/governance/metrics",
                    "method": "GET",
                    "description": "获取治理指标"
                },
                {
                    "path": "/units/{unit_id}",
                    "method": "GET",
                    "description": "获取 Unit 详情"
                }
            ]
        }


def demo_governance_api():
    """演示治理 API"""
    import random
    
    print("\n" + "🌐 " * 35)
    print("Governance API Demo - 治理系统 API 演示")
    print("🌐 " * 35)
    
    api = GovernanceAPI()
    
    # 1. API 文档
    print("\n" + "="*70)
    print("1. API 文档")
    print("="*70)
    docs = api.get_api_documentation()
    print(f"\n  API 版本: {docs['api_version']}")
    print(f"  基础路径: {docs['base_url']}")
    print(f"\n  可用端点:")
    for endpoint in docs['endpoints']:
        print(f"    {endpoint['method']:6} {endpoint['path']:30} - {endpoint['description']}")
    
    # 2. 系统状态
    print("\n" + "="*70)
    print("2. 系统状态查询")
    print("="*70)
    status_response = api.get_governance_status()
    print(f"\n  状态: {status_response.data['status']}")
    print(f"  版本: {status_response.data['version']}")
    print(f"  运行时间: {status_response.data['uptime_seconds']} 秒")
    print(f"  总请求数: {status_response.data['total_requests']}")
    print(f"  平均延迟: {status_response.data['avg_latency_ms']:.1f} ms")
    
    # 3. 治理指标
    print("\n" + "="*70)
    print("3. 治理指标查询")
    print("="*70)
    metrics_response = api.get_governance_metrics()
    metrics = metrics_response.data
    print(f"\n  QT (质量): {metrics['qt_score']:.2f}")
    print(f"  SL (稳定性): {metrics['sl_score']:.2f}")
    print(f"  T (时效性): {metrics['t_score']:.2f}")
    print(f"  C (一致性): {metrics['c_score']:.2f}")
    print(f"  L (合法性): {metrics['l_score']:.2f}")
    print(f"\n  吞吐量: {metrics['throughput_qps']:.1f} QPS")
    print(f"  错误率: {metrics['error_rate']:.1%}")
    print(f"  Unit 数量: {metrics['unit_count']:,}")
    
    # 4. 单条处理
    print("\n" + "="*70)
    print("4. 单条文本处理")
    print("="*70)
    test_texts = [
        "什么是机器学习？",
        "如果下雨，地面会湿",
        "人工智能正在改变我们的生活"
    ]
    
    for text in test_texts:
        request = ProcessRequest(text=text, language="zh")
        response = api.process(request)
        
        print(f"\n  输入: {text}")
        print(f"  请求ID: {response.request_id}")
        print(f"  状态: {response.status}")
        print(f"  延迟: {response.latency_ms:.2f} ms")
        
        if response.data:
            result = response.data
            print(f"  概念: {len(result['units']['concepts'])} 个")
            print(f"  关系: {len(result['units']['relations'])} 个")
            print(f"  规则: {len(result['units']['rules'])} 个")
            print(f"  任务: {len(result['units']['task_patterns'])} 个")
            print(f"  QT分数: {result['governance_scores']['QT']:.2f}")
    
    # 5. 批量处理
    print("\n" + "="*70)
    print("5. 批量文本处理")
    print("="*70)
    batch_requests = [
        ProcessRequest(text="解释深度学习", language="zh"),
        ProcessRequest(text="Why is the sky blue?", language="en"),
        ProcessRequest(text="比较AI和ML的区别", language="zh")
    ]
    
    batch_response = api.batch_process(batch_requests)
    print(f"\n  批量ID: {batch_response.data['batch_id']}")
    print(f"  总数: {batch_response.data['total']}")
    print(f"  成功: {batch_response.data['successful']}")
    print(f"  延迟: {batch_response.latency_ms:.2f} ms")
    
    # 6. Unit 详情查询
    print("\n" + "="*70)
    print("6. Unit 详情查询")
    print("="*70)
    unit_response = api.get_unit_details("unit_12345")
    unit = unit_response.data
    print(f"\n  Unit ID: {unit['unit_id']}")
    print(f"  类型: {unit['unit_type']}")
    print(f"  质量分数: {unit['quality_score']:.2f}")
    print(f"  稳定周期: {unit['stability_cycles']}")
    print(f"  引用数: {unit['references']}")
    
    print("\n" + "="*70)
    print("API 演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_governance_api()
