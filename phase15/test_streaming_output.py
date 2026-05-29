"""
Test Streaming Output - 流式输出测试

目标：
1. 验证流式输出正常工作
2. 监控TTFT（首Token时间）
3. 对比流式 vs 非流式体验
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, MemoryLayer
)
from phase11.core_services.governance_service_v1 import GovernanceService
from phase11.core_services.memory_service_v1 import MemoryService
from phase15.orchestrator_llm_v6_streaming import create_orchestrator_v6_streaming
from phase15.llm_client_streaming_v1 import create_mock_streaming_client, create_deepseek_streaming_client


class StreamingOutputTest:
    """流式输出测试"""
    
    def __init__(self, llm_provider: str = "mock"):
        self.llm_provider = llm_provider
        self.orchestrator = None
        self.services = {}
        self.results = []
    
    async def setup(self):
        """设置测试环境"""
        print("=" * 70)
        print("流式输出测试 - 环境设置")
        print("=" * 70)
        
        # 启动服务
        self.services['retrieval'] = RetrievalService()
        self.services['governance'] = GovernanceService()
        self.services['memory'] = MemoryService()
        
        for name, service in self.services.items():
            await service.start()
        
        print("✓ 核心服务启动完成")
        
        # 创建流式LLM客户端
        if self.llm_provider == "deepseek":
            llm_client = create_deepseek_streaming_client()
            print("✓ 使用DeepSeek流式客户端")
        else:
            llm_client = create_mock_streaming_client()
            print("✓ 使用Mock流式客户端")
        
        # 健康检查
        if not await llm_client.health_check():
            raise RuntimeError("LLM客户端健康检查失败")
        
        # 创建流式Orchestrator
        self.orchestrator = create_orchestrator_v6_streaming(
            retrieval_service=self.services['retrieval'],
            governance_service=self.services['governance'],
            memory_service=self.services['memory'],
            llm_client=llm_client,
            debug_mode=True
        )
        
        print("✓ 流式Orchestrator创建完成")
        
        # 准备测试数据
        await self._prepare_test_data()
        print("✓ 注入测试记忆")
        print("\n环境设置完成！\n")
    
    async def _prepare_test_data(self):
        """准备测试数据"""
        retrieval = self.services['retrieval']
        
        test_memories = [
            (MemoryLayer.LONG_TERM, "user_identity", 
             "用户的名字是Alice，是一名软件工程师，喜欢Python编程。", 0.9),
            (MemoryLayer.LONG_TERM, "project_goal", 
             "项目的目标是在2024年Q3完成核心功能开发，并在Q4上线。", 0.9),
            (MemoryLayer.LONG_TERM, "project_progress", 
             "项目目前已完成Phase 1和Phase 2，正在进行Phase 3的开发。", 0.85),
            (MemoryLayer.LONG_TERM, "tech_stack", 
             "技术栈使用Python后端、React前端和PostgreSQL数据库。", 0.9),
            (MemoryLayer.LONG_TERM, "discussion_topic_1", 
             "之前讨论过项目目标和技术选型问题。", 0.8),
        ]
        
        for layer, key, content, confidence in test_memories:
            retrieval.insert_memory(layer, key, content, {"confidence": confidence})
    
    async def run_streaming_test(self) -> dict:
        """运行流式输出测试"""
        print("=" * 70)
        print("开始流式输出测试")
        print("=" * 70)
        
        # 测试用例
        test_cases = [
            ("streaming_001", "你好"),
            ("streaming_002", "我叫什么名字？"),
            ("streaming_003", "项目的目标是什么？"),
            ("streaming_004", "技术栈是什么？"),
            ("streaming_005", "我们之前讨论过什么？"),
        ]
        
        for session_id, query in test_cases:
            print(f"\n{'='*70}")
            print(f"测试: {query}")
            print("=" * 70)
            
            start_time = time.time()
            ttft = None
            full_content = []
            
            print("\n[流式输出] ", end="", flush=True)
            
            try:
                async for event in self.orchestrator.process_turn_streaming(query, session_id):
                    if event["type"] == "status":
                        status_data = event["data"]
                        if status_data.get("status") == "generating":
                            pre_time = status_data.get("pre_processing_ms", 0)
                            print(f"\n[准备完成] 预处理: {pre_time:.0f}ms")
                            print("[流式输出] ", end="", flush=True)
                    
                    elif event["type"] == "content":
                        content_data = event["data"]
                        chunk = content_data.get("content", "")
                        
                        # 记录TTFT
                        if content_data.get("is_first") and ttft is None:
                            ttft = (time.time() - start_time) * 1000
                        
                        full_content.append(chunk)
                        print(chunk, end="", flush=True)
                    
                    elif event["type"] == "complete":
                        complete_data = event["data"]
                        total_latency = complete_data.get("total_latency_ms", 0)
                        
                        print(f"\n\n[完成]")
                        print(f"  TTFT: {complete_data.get('ttft_ms', 0):.0f}ms")
                        print(f"  总延迟: {total_latency:.0f}ms")
                        print(f"  策略: {complete_data.get('strategy', 'unknown')}")
                        print(f"  分块数: {complete_data.get('chunk_count', 0)}")
                        
                        self.results.append({
                            "query": query,
                            "success": True,
                            "ttft_ms": complete_data.get('ttft_ms', 0),
                            "total_latency_ms": total_latency,
                            "strategy": complete_data.get('strategy', 'unknown'),
                            "chunk_count": complete_data.get('chunk_count', 0),
                            "response": "".join(full_content)
                        })
                    
                    elif event["type"] == "error":
                        error_data = event["data"]
                        print(f"\n[错误] {error_data.get('error', 'Unknown error')}")
                        
                        self.results.append({
                            "query": query,
                            "success": False,
                            "error": error_data.get('error', 'Unknown error')
                        })
                
            except Exception as e:
                print(f"\n[异常] {e}")
                self.results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        return self._analyze_results()
    
    def _analyze_results(self) -> dict:
        """分析测试结果"""
        print("\n" + "=" * 70)
        print("流式输出测试分析")
        print("=" * 70)
        
        successful = [r for r in self.results if r.get("success")]
        
        if not successful:
            print("✗ 无成功结果")
            return {"success": False, "error": "无成功结果"}
        
        # TTFT统计
        ttfts = [r.get("ttft_ms", 0) for r in successful]
        avg_ttft = sum(ttfts) / len(ttfts)
        min_ttft = min(ttfts)
        max_ttft = max(ttfts)
        
        # 总延迟统计
        latencies = [r.get("total_latency_ms", 0) for r in successful]
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"\n📊 TTFT (Time To First Token) 统计:")
        print(f"  平均: {avg_ttft:.0f}ms")
        print(f"  最小: {min_ttft:.0f}ms")
        print(f"  最大: {max_ttft:.0f}ms")
        print(f"  目标: <1000ms")
        
        if avg_ttft < 1000:
            print(f"  ✅ 达到目标！")
        else:
            print(f"  ⚠️  未达目标，需优化")
        
        print(f"\n📊 总延迟统计:")
        print(f"  平均: {avg_latency:.0f}ms")
        
        print(f"\n📊 策略分布:")
        strategies = {}
        for r in successful:
            s = r.get("strategy", "unknown")
            strategies[s] = strategies.get(s, 0) + 1
        for s, count in strategies.items():
            print(f"  {s}: {count}")
        
        # 检查目标
        print(f"\n🎯 目标检查:")
        checks = [
            ("TTFT < 1000ms", avg_ttft < 1000),
            ("成功率 > 80%", len(successful) / len(self.results) > 0.8),
            ("至少2种策略", len(strategies) >= 2),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        return {
            "success": all_passed,
            "avg_ttft_ms": avg_ttft,
            "avg_latency_ms": avg_latency,
            "success_rate": len(successful) / len(self.results),
            "strategies": strategies
        }
    
    async def cleanup(self):
        """清理资源"""
        print("\n" + "=" * 70)
        print("清理资源")
        print("=" * 70)
        
        for name, service in self.services.items():
            await service.stop()
        
        print("✓ 资源清理完成")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="流式输出测试")
    parser.add_argument("--provider", choices=["mock", "deepseek"], default="mock",
                       help="LLM提供商")
    
    args = parser.parse_args()
    
    test = StreamingOutputTest(llm_provider=args.provider)
    
    try:
        await test.setup()
        result = await test.run_streaming_test()
    finally:
        await test.cleanup()
    
    print("\n" + "=" * 70)
    if result.get("success"):
        print("✅ 流式输出测试通过！TTFT目标达成。")
    else:
        print("⚠️ 部分检查未通过，需继续优化。")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result.get("success", False) else 1)
