import json
import os
from typing import Any


class RetrievalResult:
    """检索结果"""

    def __init__(
        self,
        result_id: str,
        content: str,
        source: str,
        relevance_score: float = 0.5,
        metadata: dict[str, Any] | None = None
    ):
        self.result_id = result_id
        self.content = content
        self.source = source
        self.relevance_score = relevance_score
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "result_id": self.result_id,
            "content": self.content,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata
        }


class InternalRetriever:
    """
    内部检索器

    第一阶段：使用本地 JSON/dict 存储少量高质量样本 Unit
    """

    def __init__(self, data_path: str | None = None):
        self.data_path = data_path or "./data/internal_knowledge.json"
        self._knowledge_base: dict[str, list[dict[str, Any]]] = {}
        self._load_knowledge()

    def _load_knowledge(self) -> None:
        """加载知识库"""
        # 内置知识库（第一阶段使用）
        self._knowledge_base = {
            "苹果": [
                {
                    "id": "apple_001",
                    "content": "苹果是一种常见的水果，属于蔷薇科苹果属。果实通常为红色、绿色或黄色，味甜多汁。",
                    "source": "internal_kb",
                    "category": "fruit"
                },
                {
                    "id": "apple_002",
                    "content": "苹果富含维生素C、膳食纤维和多种抗氧化物质，对健康有益。",
                    "source": "internal_kb",
                    "category": "nutrition"
                }
            ],
            "记忆层": [
                {
                    "id": "memory_001",
                    "content": "记忆层是Post Transformer AI系统的核心组件，分为瞬态层、长期层、浅永久层和深永久层四个层级。",
                    "source": "system_doc",
                    "category": "architecture"
                },
                {
                    "id": "memory_002",
                    "content": "瞬态层存储当前会话上下文，长期层存储中期记忆，浅永久层存储压缩后的验证知识，深永久层存储核心知识。",
                    "source": "system_doc",
                    "category": "architecture"
                }
            ],
            "永久层": [
                {
                    "id": "permanent_001",
                    "content": "永久层是记忆系统的最高层级，只有经过严格验证和审查的知识才能进入。",
                    "source": "system_doc",
                    "category": "governance"
                }
            ],
            "TSLA": [
                {
                    "id": "tsla_001",
                    "content": "TSLA是Trustworthiness(可信度)、Safety(安全性)、Liability(责任性)、Accountability(可追责性)的缩写，是系统的治理框架。",
                    "source": "system_doc",
                    "category": "governance"
                }
            ],
            "Unit": [
                {
                    "id": "unit_001",
                    "content": "Unit是系统的最小治理单元，包含script_form(书写形式)、phonetic_form(语音形式)、core_meaning(核心语义)三态。",
                    "source": "system_doc",
                    "category": "core_concept"
                }
            ]
        }

        # 尝试从文件加载（如果存在）
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self._knowledge_base.update(loaded)
            except Exception:
                pass  # 使用默认知识库

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        检索相关知识

        第一阶段：简单的关键词匹配
        """
        results = []

        # 在知识库中查找匹配的关键词
        for keyword, entries in self._knowledge_base.items():
            if keyword in query or query in keyword:
                for entry in entries:
                    # 计算相关性分数（简化版）
                    relevance = self._calculate_relevance(query, keyword, entry)

                    result = RetrievalResult(
                        result_id=entry["id"],
                        content=entry["content"],
                        source=entry["source"],
                        relevance_score=relevance,
                        metadata={
                            "category": entry.get("category", "general"),
                            "matched_keyword": keyword
                        }
                    )
                    results.append(result)

        # 按相关性排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:top_k]

    def _calculate_relevance(
        self,
        query: str,
        keyword: str,
        entry: dict[str, Any]
    ) -> float:
        """计算相关性分数（简化版）"""
        base_score = 0.5

        # 完全匹配加分
        if query == keyword:
            base_score += 0.3
        elif keyword in query:
            base_score += 0.2

        # 根据内容长度调整
        content_length = len(entry.get("content", ""))
        if 50 <= content_length <= 200:
            base_score += 0.1

        return min(base_score, 1.0)

    def add_knowledge(
        self,
        keyword: str,
        content: str,
        source: str = "user_added",
        category: str = "general"
    ) -> str:
        """添加知识到内部知识库"""
        entry_id = f"user_{keyword}_{len(self._knowledge_base.get(keyword, []))}"

        if keyword not in self._knowledge_base:
            self._knowledge_base[keyword] = []

        self._knowledge_base[keyword].append({
            "id": entry_id,
            "content": content,
            "source": source,
            "category": category
        })

        return entry_id

    def save_knowledge(self) -> None:
        """保存知识库到文件"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self._knowledge_base, f, ensure_ascii=False, indent=2)
