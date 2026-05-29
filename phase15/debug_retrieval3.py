"""
Debug Retrieval 3 - 测试修复后的搜索
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase11.core_services.retrieval_service_v1 import (
    RetrievalService, RetrievalRequest, RetrievalType, MemoryLayer
)


async def main():
    """测试修复后的检索"""
    print("="*70)
    print("Retrieval Service Test (After Fix)")
    print("="*70)
    
    retrieval = RetrievalService()
    await retrieval.start()
    
    # 插入测试数据
    test_memories = [
        (MemoryLayer.LONG_TERM, "user_identity", "用户的名字是Alice，是一名软件工程师"),
        (MemoryLayer.LONG_TERM, "project_goal", "项目的目标是在2024年Q3完成核心功能"),
        (MemoryLayer.LONG_TERM, "tech_stack", "技术栈使用Python后端、React前端和PostgreSQL数据库"),
    ]
    
    print("\n1. 插入测试数据:")
    for layer, key, content in test_memories:
        retrieval.insert_memory(layer, key, content, {"confidence": 0.9})
        print(f"  [{layer.value}] {key}: {content[:40]}...")
    
    # 测试查询
    test_queries = [
        "我叫什么名字",
        "用户名字",
        "名字",
        "Alice",
        "项目目标",
        "目标",
        "技术栈",
        "Python",
        # v4 改写后的查询
        "我叫什么名字？ 用户名字",
        "项目的目标是什么？ 项目信息",
        "技术栈是什么？ 技术架构",
    ]
    
    print("\n2. 测试检索:")
    for query in test_queries:
        request = RetrievalRequest(
            query=query,
            query_id=f"test_{hash(query)}",
            retrieval_type=RetrievalType.HYBRID,
            max_results=3
        )
        
        response = await retrieval.retrieve(request)
        
        status = "✓" if response.total_found > 0 else "○"
        print(f"\n  {status} 查询: '{query}'")
        print(f"     找到: {response.total_found} 条")
        
        for i, r in enumerate(response.results[:2]):
            print(f"     [{i+1}] {r.content[:35]}... (score:{r.score:.2f})")
    
    await retrieval.stop()
    print("\n测试完成")


if __name__ == "__main__":
    asyncio.run(main())
