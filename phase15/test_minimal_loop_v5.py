"""
Test Minimal Loop v5 - 最小闭环测试 v5

验证 QueryRewriter v2 效果：
- 4类查询分别处理
- 身份类查询召回
- 历史回顾类查询召回
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
from phase15.orchestrator_llm_v5 import create_orchestrator_v5
from phase15.llm_client_v1 import create_mock_client, create_deepseek_client


class MinimalLoopTestV5:
    """最小闭环测试 v5"""
    
    def __init__(self, llm_provider: str = "mock", debug_mode: bool = False):
        self.llm_provider = llm_provider
        self.debug_mode = debug_mode
        self.orchestrator = None
        self.services = {}
    
    async def setup(self):
        """设置测试环境"""
        print("="*70)
        print("设置测试环境 (v5 - QueryRewriter v2)")
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
            print("✓ 使用DeepSeek客户端 (deepseek-chat)")
        else:
            llm_client = create_mock_client()
            print("✓ 使用MOCK客户端")
        
        # 健康检查
        if await llm_client.health_check():
            print("✓ LLM客户端健康检查通过")
        else:
            raise RuntimeError("LLM客户端健康检查失败")
        
        # 创建Orchestrator v5
        self.orchestrator = create_orchestrator_v5(
            retrieval_service=self.services['retrieval'],
            governance_service=self.services['governance'],
            memory_service=self.services['memory'],
            llm_client=llm_client,
            debug_mode=self.debug_mode
        )
        
        print("✓ 对话主控器V5创建完成（QueryRewriter v2）")
        
        # 准备测试数据
        await self._prepare_test_data()
        print("✓ 注入测试记忆")
        print("\n环境设置完成！\n")
    
    async def _prepare_test_data(self):
        """准备测试数据"""
        retrieval = self.services['retrieval']
        
        test_memories = [
            # 身份信息
            (MemoryLayer.LONG_TERM, "user_identity", 
             "用户的名字是Alice，是一名软件工程师，喜欢Python编程。", 0.9),
            
            # 项目信息
            (MemoryLayer.LONG_TERM, "project_goal", 
             "项目的目标是在2024年Q3完成核心功能开发，并在Q4上线。", 0.9),
            (MemoryLayer.LONG_TERM, "project_progress", 
             "项目目前已完成Phase 1和Phase 2，正在进行Phase 3的开发。", 0.85),
            
            # 技术栈信息
            (MemoryLayer.LONG_TERM, "tech_stack", 
             "技术栈使用Python后端、React前端和PostgreSQL数据库。", 0.9),
            
            # 历史对话主题
            (MemoryLayer.LONG_TERM, "discussion_topic_1", 
             "之前讨论过项目目标和技术选型问题。", 0.8),
        ]
        
        for layer, key, content, confidence in test_memories:
            retrieval.insert_memory(layer, key, content, {"confidence": confidence})
    
    async def run_test_case(self, session_id: str, query: str, expected_behavior: str) -> dict:
        """运行单个测试用例"""
        print(f"\n测试: {query}")
        print("-"*50)
        
        try:
            response = await self.orchestrator.process_turn_async(query, session_id)
            
            # 检查结果
            has_retrieval = response.metadata.get('retrieval_count', 0) > 0
            has_context_issue = "无法访问" in response.response_text or "无法确定" in response.response_text
            has_citation = '[来源' in response.response_text or '根据' in response.response_text
            
            # 判断质量
            if has_retrieval and not has_context_issue:
                quality = "✓"
            elif has_retrieval:
                quality = "⚠"
            else:
                quality = "○"
            
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
                "sources": response.sources,
            }
            
            print(f"{quality} 成功")
            print(f"  策略: {response.strategy.value}")
            print(f"  置信度: {response.confidence:.2f}")
            print(f"  检索: {result['retrieval_count']} 条")
            print(f"  延迟: {result['latency_ms']:.0f}ms")
            print(f"  引用: {'是' if has_citation else '否'}")
            print(f"  响应: {response.response_text[:80]}...")
            
            return result
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "expected_behavior": expected_behavior,
            }
    
    async def run(self) -> dict:
        """运行完整测试"""
        await self.setup()
        
        print("="*70)
        print("开始最小闭环测试 v5 - QueryRewriter v2 验证")
        print("="*70)
        
        # 测试用例 - 覆盖4类查询
        test_cases = [
            # 通用类
            ("test_v5_001", "你好", "通用问候，无需检索"),
            
            # 身份类（重点测试）
            ("test_v5_002", "我叫什么名字？", "身份类：应触发检索并回答Alice"),
            ("test_v5_003", "我是谁？", "身份类：应触发检索"),
            
            # 项目状态类
            ("test_v5_004", "项目的目标是什么？", "项目类：应触发检索并回答目标"),
            ("test_v5_005", "我们做到哪个阶段了？", "项目类：应触发检索并回答阶段"),
            
            # 技术栈类
            ("test_v5_006", "技术栈是什么？", "技术类：应触发检索并回答技术栈"),
            ("test_v5_007", "用什么技术？", "技术类：应触发检索"),
            
            # 历史回顾类（重点测试）
            ("test_v5_008", "我们之前讨论过什么？", "历史类：应触发检索并回顾主题"),
        ]
        
        results = []
        for session_id, query, expected in test_cases:
            result = await self.run_test_case(session_id, query, expected)
            results.append(result)
        
        # 分析结果
        return self._analyze_results(results)
    
    def _analyze_results(self, results: list) -> dict:
        """分析测试结果"""
        print("\n" + "="*70)
        print("测试结果分析 v5")
        print("="*70)
        
        # 基础统计
        total = len(results)
        successful = sum(1 for r in results if r.get("success"))
        
        # 检索统计
        with_retrieval = sum(1 for r in results if r.get("has_retrieval"))
        with_citation = sum(1 for r in results if r.get("has_citation"))
        context_issues = sum(1 for r in results if r.get("has_context_issue"))
        
        # 按类型统计
        identity_queries = [r for r in results if "身份" in r.get("expected_behavior", "")]
        project_queries = [r for r in results if "项目" in r.get("expected_behavior", "")]
        tech_queries = [r for r in results if "技术" in r.get("expected_behavior", "")]
        history_queries = [r for r in results if "历史" in r.get("expected_behavior", "")]
        
        # 策略统计
        strategies = {}
        for r in results:
            if r.get("success"):
                s = r.get("strategy", "unknown")
                strategies[s] = strategies.get(s, 0) + 1
        
        # 延迟统计
        latencies = [r.get("latency_ms", 0) for r in results if r.get("success")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        print(f"\n总体统计:")
        print(f"  总测试数: {total}")
        print(f"  成功: {successful} ({100*successful/total:.1f}%)")
        
        print(f"\n检索质量:")
        print(f"  触发检索: {with_retrieval}/{total} ({100*with_retrieval/total:.1f}%)")
        print(f"  引用内容: {with_citation}/{total} ({100*with_citation/total:.1f}%)")
        print(f"  上下文问题: {context_issues}")
        
        print(f"\n分类型统计:")
        if identity_queries:
            identity_success = sum(1 for r in identity_queries if r.get("has_retrieval"))
            print(f"  身份类: {identity_success}/{len(identity_queries)} 触发检索")
        if project_queries:
            project_success = sum(1 for r in project_queries if r.get("has_retrieval"))
            print(f"  项目类: {project_success}/{len(project_queries)} 触发检索")
        if tech_queries:
            tech_success = sum(1 for r in tech_queries if r.get("has_retrieval"))
            print(f"  技术类: {tech_success}/{len(tech_queries)} 触发检索")
        if history_queries:
            history_success = sum(1 for r in history_queries if r.get("has_retrieval"))
            print(f"  历史类: {history_success}/{len(history_queries)} 触发检索")
        
        print(f"\n策略分布:")
        for s, count in sorted(strategies.items()):
            print(f"  {s}: {count}")
        
        print(f"\n延迟统计:")
        print(f"  平均: {avg_latency:.0f}ms")
        
        # 关键指标检查
        print(f"\n关键指标检查:")
        checks = [
            ("成功率 > 80%", successful/total > 0.8),
            ("检索触发率 > 60%", with_retrieval/total > 0.6),
            ("身份类触发率 > 50%", 
             sum(1 for r in identity_queries if r.get("has_retrieval")) / len(identity_queries) > 0.5 if identity_queries else False),
            ("上下文问题 = 0", context_issues == 0),
            ("至少使用2种策略", len(strategies) >= 2),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        return {
            "success": all_passed,
            "total": total,
            "successful": successful,
            "with_retrieval": with_retrieval,
            "with_citation": with_citation,
            "context_issues": context_issues,
            "strategies": strategies,
            "avg_latency_ms": avg_latency,
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
    parser = argparse.ArgumentParser(description="最小闭环测试 v5")
    parser.add_argument("--provider", choices=["mock", "deepseek"], default="mock",
                       help="LLM提供商")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    
    args = parser.parse_args()
    
    test = MinimalLoopTestV5(
        llm_provider=args.provider,
        debug_mode=args.debug
    )
    
    try:
        result = await test.run()
    finally:
        await test.cleanup()
    
    print("\n" + "="*70)
    if result.get("success"):
        print("✓ 所有检查通过！QueryRewriter v2 成功。")
    else:
        print("⚠ 部分检查未通过，需要继续优化。")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result.get("success", False) else 1)
