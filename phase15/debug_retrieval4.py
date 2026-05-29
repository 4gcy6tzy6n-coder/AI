"""
Debug Retrieval 4 - 详细分析匹配逻辑
"""

content = "用户的名字是Alice，是一名软件工程师"
queries = [
    "我叫什么名字",
    "用户名字",
    "名字",
    "Alice",
]

print("内容:", content)
print()

for query in queries:
    query_lower = query.lower()
    content_lower = content.lower()
    
    # 检查子串匹配
    is_substring = query_lower in content_lower
    
    # 检查关键词匹配
    query_terms = query_lower.split()
    matches = [term for term in query_terms if len(term) > 1 and term in content_lower]
    
    print(f"查询: '{query}'")
    print(f"  子串匹配: {is_substring}")
    print(f"  查询词: {query_terms}")
    print(f"  关键词匹配: {matches}")
    print()
