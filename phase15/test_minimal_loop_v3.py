"""
Test Minimal Loop v3 - 最小闭环测试 v3

使用 QueryRewriter 优化检索
验证检索质量改善效果
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
    create_deepseek_client
)
from phase15.orchestrator_llm_v3 import ConversationOrchestratorWithLLMV3
from phase15.prompt_builder_v2 import create_prompt_builder_v2
from phase15.query_rewriter_v1 import create_query_rewriter


class MinimalLoopTestV3:
    """最小闭环测试 v3 - 验证检索质量"""
    
    def __init__(self, llm_provider: str = "mock", debug_mode: bool = False):
        self.llm_provider = llm_provider
        self.debug_mode = debug_mode
        self.orchestrator = None
        self.test_results = []
    
    async def setup(self):
        """设置测试环境"""
        print("="*70)
        print("设置测试环境 (v3 - 集成 QueryRewriter)")
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
        
        # 创建主控器（使用V3）
        self.orchestrator = ConversationOrchestratorWithLLMV3(
            retrieval_service=self.retrieval,
            governance_service=self.governance,
            memory_service=self.memory,
            llm_client=self.llm_client,
            prompt_builder=create_prompt_builder_v2(),
            query_rewriter=create_query_rewriter(),
            debug_mode=self.debug_mode
        )
        print("✓ 对话主控器V3创建完成（集成QueryRewriter）")
        
        # 准备测试记忆
        test_memories = [
            ("long_term", "user_name", "用户的名字是Alice，是一名软件工程师，喜欢Python编程"),
            ("long_term", "project_goal", "项目目标是在2026年Q3完成核心功能开发，包括AI对话系统和检索模块"),
            ("long_term", "tech_stack", "技术栈使用Python后端、React前端和PostgreSQL数据库"),
            ("long_term", "python_info", "Python是一种高级编程语言，由Guido van Rossum于1991年创建，以简洁易读的语法著称"),
        ]
        
        for layer, key, content in test_memories:
            self.retrieval.insert_memory(MemoryLayer.LONG_TERM, key, content, {"category": "test", "confidence": 0.9})
        
        print(f"✓ 注入 {len(test_memories)} 条测试记忆")
        print("\n环境设置完成！\n")
    
    async def run_test_case(self, session_id: str, query: str, expected_behavior: str) -> dict:
        """运行单个测试用例"""
        print(f"\n测试: {query}")
        print("-" * 50)
        
        try:
            response = await self.orchestrator.process_turn_async(query, session_id)
            
            # 检查检索质量
            has_retrieval = response.metadata.get('retrieval_count', 0) > 0
            
            # 检查是否还有"无法访问"问题
            has_context_issue = "无法访问" in response.response_text or "无法确定" in response.response_text
            
            # 检查是否引用了检索内容（简单判断）
            has_citation = '[来源' in response.response_text or '根据' in response.response_text
            
            result = {
                "query": query,
                "success": True,
                "response": response.response_text,
                "strategy": response.strategy.value,
                "confidence": response.confidence,
                "latency_ms": response.metadata.get('total_latency_ms', 0),
                "llm_latency_ms": response.metadata.get('llm_latency_ms', 0),
                "retrieval_count": response.metadata.get('retrieval_count', 0),
                "expected_behavior": expected_behavior,
                "has_retrieval": has_retrieval,
                "has_context_issue": has_context_issue,
                "has_citation": has_citation,
                "sources": response.sources
            }
            
            # 判断测试质量
            quality = "✓" if has_retrieval and not has_context_issue else "⚠"
            
            print(f"{quality} 成功")
            print(f"  策略: {response.strategy.value}")
            print(f"  置信度: {response.confidence:.2f}")
            print(f"  检索: {result['retrieval_count']} 条")
            print(f"  延迟: {result['latency_ms']:.0f}ms")
            print(f"  引用: {'是' if has_citation else '否'}")
            print(f"  响应: {response.response_text[:80]}...")
            
            if has_context_issue:
                print(f"  ⚠ 警告：仍出现上下文问题")
            
            return result
            
        except Exception as e:
            result = {
                "query": query,
                "success": False,
                "error": str(e),
                "expected_behavior": expected_behavior
            }
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    async def run_all_tests(self) -> list:
        """运行所有测试"""
        print("="*70)
        print("开始最小闭环测试 v3 - 检索质量验证")
        print("="*70)
        
        session_id = "test_session_v3_001"
        
        test_cases = [
            {
                "query": "你好",
                "expected_behavior": "正常问候，无需检索"
            },
            {
                "query": "Python是什么编程语言？",
                "expected_behavior": "检索并回答Python信息"
            },
            {
                "query": "我叫什么名字？",
                "expected_behavior": "检索记忆，回答Alice"
            },
            {
                "query": "项目的目标是什么？",
                "expected_behavior": "检索并回答项目目标"
            },
            {
                "query": "技术栈是什么？",
                "expected_behavior": "检索并回答技术栈"
            },
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
        print("测试结果分析 v3")
        print("="*70)
        
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        
        # 检索质量统计
        with_retrieval = sum(1 for r in results if r.get("has_retrieval", False))
        with_citation = sum(1 for r in results if r.get("has_citation", False))
        context_issues = sum(1 for r in results if r.get("has_context_issue", False))
        
        # 策略分布
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
            "success_rate": successful / total if total > 0 else 0,
            "with_retrieval": with_retrieval,
            "with_citation": with_citation,
            "context_issues": context_issues,
            "retrieval_rate": with_retrieval / total if total > 0 else 0,
            "citation_rate": with_citation / total if total > 0 else 0,
            "strategy_distribution": strategies,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        }
        
        print(f"\n总体统计:")
        print(f"  总测试数: {total}")
        print(f"  成功: {successful} ({analysis['success_rate']:.1%})")
        
        print(f"\n检索质量:")
        print(f"  触发检索: {with_retrieval}/{total} ({analysis['retrieval_rate']:.1%})")
        print(f"  引用内容: {with_citation}/{total} ({analysis['citation_rate']:.1%})")
        print(f"  上下文问题: {context_issues}")
        
        print(f"\n策略分布:")
        for strategy, count in strategies.items():
            print(f"  {strategy}: {count}")
        
        if latencies:
            print(f"\n延迟统计:")
            print(f"  平均: {analysis['avg_latency_ms']:.0f}ms")
        
        # 检查关键指标
        print(f"\n关键指标检查:")
        
        checks = [
            ("成功率 > 80%", analysis['success_rate'] >= 0.8),
            ("检索触发率 > 60%", analysis['retrieval_rate'] >= 0.6),
            ("上下文问题 = 0", context_issues == 0),
            ("至少使用2种策略", len(strategies) >= 2),
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
                print("✓ 所有检查通过！检索质量优化成功。")
            else:
                print("⚠ 部分检查未通过，需要继续优化。")
            print("="*70)
            
            return {
                "success": analysis['all_checks_passed'],
                "results": results,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            await self.cleanup()
            return {
                "success": False,
                "error": str(e)
            }


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="最小闭环测试 v3")
    parser.add_argument(
        "--provider",
        choices=["mock", "deepseek"],
        default="mock",
        help="LLM提供商 (默认: mock)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Real LLM Integration - Minimal Loop Test v3")
    print(f"LLM Provider: {args.provider}")
    print(f"Debug Mode: {args.debug}")
    print("="*70 + "\n")
    
    test = MinimalLoopTestV3(
        llm_provider=args.provider,
        debug_mode=args.debug
    )
    result = await test.run()
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    
    # 返回码
    exit(0 if result.get("success", False) else 1)
