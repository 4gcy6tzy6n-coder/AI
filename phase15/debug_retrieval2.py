"""
Debug Retrieval 2 - 详细调试
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase11.core_services.retrieval_service_v1 import (
    IndexStore, MemoryLayer, RetrievalResult
)


def test_search():
    """测试搜索逻辑"""
    print("="*70)
    print("Search Logic Debug")
    print("="*70)
    
    # 创建索引
    index = IndexStore(MemoryLayer.LONG_TERM)
    
    # 插入数据
    index.insert("user_identity", "用户的名字是Alice，是一名软件工程师")
    index.insert("project_goal", "项目的目标是在2024年Q3完成核心功能")
    
    print("\n索引内容:")
    for doc_id, doc in index.index.items():
        print(f"  {doc_id}: {doc['content']}")
    
    # 测试搜索
    test_cases = [
        "我叫什么名字",
        "用户名字",
        "名字",
        "Alice",
    ]
    
    print("\n搜索测试:")
    for query in test_cases:
        query_terms = query.lower().split()
        print(f"\n查询: '{query}'")
        print(f"查询词: {query_terms}")
        
        for doc_id, doc in index.index.items():
            content = doc.get("content", "").lower()
            matches = [term for term in query_terms if term in content]
            score = len(matches) / len(query_terms) if query_terms else 0
            
            print(f"  [{doc_id}]")
            print(f"    内容: {content}")
            print(f"    匹配: {matches}")
            print(f"    分数: {score:.2f}")


if __name__ == "__main__":
    test_search()
