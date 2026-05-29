import random
from typing import Any


class ExternalRetriever:
    """
    外部检索器

    第一阶段：Mock 实现
    不真联网，查本地文件或预置知识片段
    """

    def __init__(self):
        # 模拟外部知识库
        self._mock_external_kb = {
            "苹果": [
                {
                    "id": "ext_apple_001",
                    "content": "苹果公司(Apple Inc.)是一家美国科技公司，总部位于加州库比蒂诺。",
                    "source": "mock_wikipedia",
                    "type": "company"
                }
            ],
            "人工智能": [
                {
                    "id": "ext_ai_001",
                    "content": "人工智能(AI)是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。",
                    "source": "mock_external",
                    "type": "definition"
                }
            ]
        }

        # 模拟网络延迟
        self._simulate_delay = True

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """
        模拟外部检索

        第一阶段返回 Mock 数据
        """
        results = []

        # 在模拟知识库中查找
        for keyword, entries in self._mock_external_kb.items():
            if keyword in query or any(word in query for word in keyword):
                for entry in entries:
                    result = {
                        "result_id": entry["id"],
                        "content": entry["content"],
                        "source": entry["source"],
                        "relevance_score": random.uniform(0.6, 0.9),
                        "metadata": {
                            "type": entry.get("type", "general"),
                            "retrieved_from": "external_mock"
                        }
                    }
                    results.append(result)

        # 如果没有匹配，返回通用回复
        if not results:
            results.append({
                "result_id": "ext_generic_001",
                "content": f"[模拟外部检索] 未找到关于'{query}'的具体信息。",
                "source": "mock_external",
                "relevance_score": 0.3,
                "metadata": {
                    "type": "not_found",
                    "retrieved_from": "external_mock"
                }
            })

        # 按相关性排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results[:top_k]

    def is_available(self) -> bool:
        """检查外部检索是否可用"""
        # 第一阶段始终返回 True（Mock 模式）
        return True

    def add_mock_knowledge(
        self,
        keyword: str,
        content: str,
        source: str = "mock_external"
    ) -> None:
        """添加模拟外部知识（用于测试）"""
        if keyword not in self._mock_external_kb:
            self._mock_external_kb[keyword] = []

        entry_id = f"ext_{keyword}_{len(self._mock_external_kb[keyword])}"
        self._mock_external_kb[keyword].append({
            "id": entry_id,
            "content": content,
            "source": source,
            "type": "user_added"
        })
