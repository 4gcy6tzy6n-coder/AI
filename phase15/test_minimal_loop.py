"""
Test Minimal Loop - 最小闭环测试

验证真实模型接入后，整条对话链第一次真正闭环
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase11.core_services.retrieval_service_v1 import RetrievalService, MemoryLayer
from phase11.core_services.governance_service_v1 import GovernanceService
from phase11.core_services.memory_service_v1 import MemoryService
from phase15.llm_client_v1 import (
    create_mock_client,
    create_openai_client,
    create_claude_client,
    create_deepseek_client
)
from phase15.orchestrator_llm_v1 import ConversationOrchestratorWithLLM, PromptBuilder


class MinimalLoopTest:
    """最小闭环测试"""
    
    def __init__(self, llm_provider: str = "mock"):
        self.llm_provider = llm_provider
        self.orchestrator = None
        self.test_results = []
    
    async def setup(self):
        """设置测试环境"""
        print("="*70)
        print("设置测试环境")
        print("="*70)
        
        # 初始化服务
        self.retrieval = RetrievalService()
        self.governance = GovernanceService()
        self.memory = MemoryService()
        
        await self.retrieval.start()
        await self.governance.start()
        await self.memory.start()
        
        print("✓ 核心服务启动完成")
        
        # 创建LLM客户端
        if self.llm_provider == "mock":
            self.llm_client = create_mock_client()
            print("✓ 使用模拟LLM客户端")
        elif self.llm_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("未设置 OPENAI_API_KEY 环境变量")
            self.llm_client = create_openai_client(model="gpt-3.5-turbo")
            print("✓ 使用OpenAI客户端 (gpt-3.5-turbo)")
        elif self.llm_provider == "claude":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量")
            self.llm_client = create_claude_client()
            print("✓ 使用Claude客户端")
        elif self.llm_provider == "deepseek":
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量")
            self.llm_client = create_deepseek_client()
            print("✓ 使用DeepSeek客户端 (deepseek-chat)")
        else:
            raise ValueError(f"未知的LLM提供商: {self.llm_provider}")
        
        # 健康检查
        if await self.llm_client.health_check():
            print("✓ LLM客户端健康检查通过")
        else:
            raise RuntimeError("LLM客户端健康检查失败")
        
        # 创建主控器
        self.orchestrator = ConversationOrchestratorWithLLM(
            retrieval_service=self.retrieval,
            governance_service=self.governance,
            memory_service=self.memory,
            llm_client=self.llm_client,
            prompt_builder=PromptBuilder()
        )
        print("✓ 对话主控器创建完成")
        
        # 准备测试记忆
        test_memories = [
            ("long_term", "user_name", "用户叫Alice"),
            ("long_term", "project_goal", "项目目标是在Q3完成核心功能"),
            ("long_term", "tech_stack", "技术栈使用Python和React"),
        ]
        
        for layer, key, content in test_memories:
            self.retrieval.insert_memory(MemoryLayer.LONG_TERM, key, content, {"category": "test"})
        
        print(f"✓ 注入 {len(test_memories)} 条测试记忆")
        print("\n环境设置完成！\n")
    
    async def run_test_case(self, session_id: str, query: str, expected_behavior: str) -> dict:
        """运行单个测试用例"""
        print(f"\n测试: {query}")
        print("-" * 50)
        
        try:
            response = await self.orchestrator.process_turn_async(query, session_id)
            
            result = {
                "query": query,
                "success": True,
                "response": response.response_text,
                "strategy": response.strategy.value,
                "confidence": response.confidence,
                "latency_ms": response.metadata.get('total_latency_ms', 0),
                "llm_latency_ms": response.metadata.get('llm_latency_ms', 0),
                "expected_behavior": expected_behavior,
                "sources": response.sources
            }
            
            print(f"✓ 成功")
            print(f"  策略: {response.strategy.value}")
            print(f"  置信度: {response.confidence:.2f}")
            print(f"  延迟: {result['latency_ms']:.0f}ms (LLM: {result['llm_latency_ms']:.0f}ms)")
            print(f"  响应: {response.response_text[:80]}...")
            
            return result
            
        except Exception as e:
            result = {
                "query": query,
                "success": False,
                "error": str(e),
                "expected_behavior": expected_behavior
            }
            print(f"✗ 失败: {e}")
            return result
    
    async def run_all_tests(self) -> list:
        """运行所有测试"""
        print("="*70)
        print("开始最小闭环测试")
        print("="*70)
        
        session_id = "test_session_001"
        
        test_cases = [
            {
                "query": "你好",
                "expected_behavior": "正常问候，使用DIRECT策略"
            },
            {
                "query": "Python是什么编程语言？",
                "expected_behavior": "直接回答技术问题"
            },
            {
                "query": "我们之前讨论过什么？",
                "expected_behavior": "触发RETRIEVAL_FIRST，检索记忆"
            },
            {
                "query": "项目的目标是什么？",
                "expected_behavior": "检索并回答项目相关信息"
            },
            {
                "query": "我叫什么名字？",
                "expected_behavior": "检索记忆，回答用户名字"
            },
            {
                "query": "xyzabc123是什么技术？",
                "expected_behavior": "DECLINE或CONSERVATIVE，表示不了解"
            },
            {
                "query": "最新的量子计算突破是什么？",
                "expected_behavior": "CONSERVATIVE，说明不确定性"
            },
            {
                "query": "技术栈是什么？",
                "expected_behavior": "检索并回答技术栈信息"
            }
        ]
        
        results = []
        for test_case in test_cases:
            result = await self.run_test_case(
                session_id,
                test_case["query"],
                test_case["expected_behavior"]
            )
            results.append(result)
        
        return results
    
    def analyze_results(self, results: list) -> dict:
        """分析测试结果"""
        print("\n" + "="*70)
        print("测试结果分析")
        print("="*70)
        
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful
        
        # 统计策略分布
        strategies = {}
        latencies = []
        
        for r in results:
            if r.get("success"):
                strategy = r.get("strategy", "UNKNOWN")
                strategies[strategy] = strategies.get(strategy, 0) + 1
                latencies.append(r.get("latency_ms", 0))
        
        analysis = {
            "total_tests": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "strategy_distribution": strategies,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0
        }
        
        print(f"\n总体统计:")
        print(f"  总测试数: {total}")
        print(f"  成功: {successful} ({analysis['success_rate']:.1%})")
        print(f"  失败: {failed}")
        
        print(f"\n策略分布:")
        for strategy, count in strategies.items():
            print(f"  {strategy}: {count}")
        
        if latencies:
            print(f"\n延迟统计:")
            print(f"  平均: {analysis['avg_latency_ms']:.0f}ms")
            print(f"  最大: {analysis['max_latency_ms']:.0f}ms")
            print(f"  最小: {analysis['min_latency_ms']:.0f}ms")
        
        # 检查关键指标
        print(f"\n关键指标检查:")
        
        checks = [
            ("成功率 > 80%", analysis['success_rate'] >= 0.8),
            ("平均延迟 < 2000ms", analysis['avg_latency_ms'] < 2000),
            ("至少使用2种策略", len(strategies) >= 2),
            ("有检索触发", 'RETRIEVAL_FIRST' in strategies or len([r for r in results if r.get('sources')]) > 0)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            all_passed = all_passed and passed
        
        analysis['all_checks_passed'] = all_passed
        
        return analysis
    
    async def cleanup(self):
        """清理资源"""
        print("\n" + "="*70)
        print("清理资源")
        print("="*70)
        
        await self.retrieval.stop()
        await self.governance.stop()
        await self.memory.stop()
        
        print("✓ 资源清理完成")
    
    async def run(self) -> dict:
        """运行完整测试"""
        try:
            await self.setup()
            results = await self.run_all_tests()
            analysis = self.analyze_results(results)
            await self.cleanup()
            
            print("\n" + "="*70)
            if analysis['all_checks_passed']:
                print("✓ 所有检查通过！最小闭环测试成功。")
            else:
                print("✗ 部分检查未通过，需要优化。")
            print("="*70)
            
            return {
                "success": analysis['all_checks_passed'],
                "results": results,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            await self.cleanup()
            return {
                "success": False,
                "error": str(e)
            }


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="最小闭环测试")
    parser.add_argument(
        "--provider",
        choices=["mock", "openai", "claude", "deepseek"],
        default="mock",
        help="LLM提供商 (默认: mock)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Real LLM Integration - Minimal Loop Test")
    print(f"LLM Provider: {args.provider}")
    print("="*70 + "\n")
    
    test = MinimalLoopTest(llm_provider=args.provider)
    result = await test.run()
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    
    # 返回码
    exit(0 if result.get("success", False) else 1)
