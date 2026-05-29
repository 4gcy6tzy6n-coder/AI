"""
Test Latency Analysis - 延迟分析测试

目标：
1. 分解每段耗时
2. 找出最大瓶颈
3. 为优化提供数据支撑
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, MemoryLayer
)
from phase11.core_services.governance_service_v1 import GovernanceService
from phase11.core_services.memory_service_v1 import MemoryService
from phase15.orchestrator_llm_v5_profiled import create_orchestrator_v5_profiled
from phase15.llm_client_v1 import create_mock_client, create_deepseek_client


class LatencyAnalysisTest:
    """延迟分析测试"""
    
    def __init__(self, llm_provider: str = "deepseek"):
        self.llm_provider = llm_provider
        self.orchestrator = None
        self.services = {}
        self.results = []
    
    async def setup(self):
        """设置测试环境"""
        print("="*70)
        print("延迟分析测试 - 环境设置")
        print("="*70)
        
        # 启动服务
        self.services['retrieval'] = RetrievalService()
        self.services['governance'] = GovernanceService()
        self.services['memory'] = MemoryService()
        
        for name, service in self.services.items():
            await service.start()
        
        print("✓ 核心服务启动完成")
        
        # 创建LLM客户端
        if self.llm_provider == "deepseek":
            llm_client = create_deepseek_client()
            print("✓ 使用DeepSeek客户端")
        else:
            llm_client = create_mock_client()
            print("✓ 使用MOCK客户端")
        
        # 健康检查
        if not await llm_client.health_check():
            raise RuntimeError("LLM客户端健康检查失败")
        
        # 创建带延迟分析的Orchestrator
        self.orchestrator = create_orchestrator_v5_profiled(
            retrieval_service=self.services['retrieval'],
            governance_service=self.services['governance'],
            memory_service=self.services['memory'],
            llm_client=llm_client,
            debug_mode=True
        )
        
        print("✓ 带延迟分析的Orchestrator创建完成")
        
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
    
    async def run_latency_test(self) -> dict:
        """运行延迟分析测试"""
        print("="*70)
        print("开始延迟分析测试")
        print("="*70)
        
        # 测试用例 - 覆盖不同场景
        test_cases = [
            # 通用类（无检索）
            ("latency_001", "你好"),
            
            # 身份类（有检索）
            ("latency_002", "我叫什么名字？"),
            
            # 项目类（有检索）
            ("latency_003", "项目的目标是什么？"),
            
            # 技术类（有检索）
            ("latency_004", "技术栈是什么？"),
            
            # 历史类（有检索）
            ("latency_005", "我们之前讨论过什么？"),
        ]
        
        for session_id, query in test_cases:
            print(f"\n测试: {query}")
            print("-"*50)
            
            try:
                response, breakdown = await self.orchestrator.process_turn_profiled(
                    query, session_id
                )
                
                self.results.append({
                    "query": query,
                    "breakdown": breakdown,
                    "success": True,
                })
                
            except Exception as e:
                print(f"✗ 失败: {e}")
                self.results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                })
        
        # 打印延迟分析报告
        self.orchestrator.print_latency_report()
        
        return self._analyze_results()
    
    def _analyze_results(self) -> dict:
        """分析测试结果"""
        print("\n" + "="*70)
        print("延迟优化建议")
        print("="*70)
        
        summary = self.orchestrator.profiler.get_summary()
        
        if not summary:
            return {"success": False, "error": "无数据"}
        
        # 分析瓶颈
        print("\n🎯 主要瓶颈分析:")
        
        for bottleneck in summary.get("bottlenecks", []):
            stage = bottleneck["stage"]
            avg_ms = bottleneck["avg_ms"]
            percentage = bottleneck["percentage"]
            
            print(f"\n  [{stage}]")
            print(f"    平均耗时: {avg_ms:.0f}ms ({percentage:.1f}%)")
            
            # 给出优化建议
            if "llm_generation" in stage:
                print("    💡 优化建议:")
                print("      1. 使用流式输出降低首token时间")
                print("      2. 优化prompt长度，减少token数")
                print("      3. 考虑使用更快的模型或缓存")
            elif "retrieval" in stage:
                print("    💡 优化建议:")
                print("      1. 使用向量数据库加速语义检索")
                print("      2. 增加检索缓存")
                print("      3. 并行化关键词和语义检索")
            elif "prompt_build" in stage:
                print("    💡 优化建议:")
                print("      1. 缓存常用prompt模板")
                print("      2. 减少历史记录长度")
                print("      3. 优化prompt拼接逻辑")
        
        # 总体建议
        total_avg = summary.get("total_avg_ms", 0)
        print(f"\n📊 总体评估:")
        print(f"  平均总延迟: {total_avg:.0f}ms")
        
        if total_avg < 2000:
            print("  ✅ 延迟表现优秀")
        elif total_avg < 4000:
            print("  ⚠️  延迟可接受，有优化空间")
        else:
            print("  ❌ 延迟较高，需要重点优化")
        
        # 计算各环节占比
        llm_pct = summary["stages"].get("llm_generation_ms", {}).get("percentage", 0)
        retrieval_pct = summary["stages"].get("retrieval_search_ms", {}).get("percentage", 0)
        other_pct = 100 - llm_pct - retrieval_pct
        
        print(f"\n📈 耗时分布:")
        print(f"  LLM生成: {llm_pct:.1f}%")
        print(f"  检索: {retrieval_pct:.1f}%")
        print(f"  其他: {other_pct:.1f}%")
        
        return {
            "success": True,
            "summary": summary,
            "total_samples": len(self.results),
            "successful_samples": sum(1 for r in self.results if r.get("success")),
        }
    
    async def cleanup(self):
        """清理资源"""
        print("\n" + "="*70)
        print("清理资源")
        print("="*70)
        
        for name, service in self.services.items():
            await service.stop()
        
        print("✓ 资源清理完成")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="延迟分析测试")
    parser.add_argument("--provider", choices=["mock", "deepseek"], default="deepseek",
                       help="LLM提供商")
    
    args = parser.parse_args()
    
    test = LatencyAnalysisTest(llm_provider=args.provider)
    
    try:
        await test.setup()
        result = await test.run_latency_test()
    finally:
        await test.cleanup()
    
    print("\n" + "="*70)
    if result.get("success"):
        print("✓ 延迟分析完成")
    else:
        print("✗ 延迟分析失败")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result.get("success", False) else 1)
