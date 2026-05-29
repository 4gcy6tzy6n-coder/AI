from enum import Enum
from typing import Any

from .external_retriever import ExternalResult
from .internal_retriever import InternalResult


class RankingStrategy(str, Enum):
    RELEVANCE = "relevance"
    RECENCY = "recency"
    CREDIBILITY = "credibility"
    HYBRID = "hybrid"


class RankedResult:
    """排序后的结果"""

    def __init__(
        self,
        result: InternalResult | ExternalResult,
        final_score: float,
        component_scores: dict[str, float]
    ):
        self.result = result
        self.final_score = final_score
        self.component_scores = component_scores


class Ranking:
    """结果排序器"""

    def __init__(
        self,
        relevance_weight: float = 0.5,
        credibility_weight: float = 0.3,
        freshness_weight: float = 0.2
    ):
        self.relevance_weight = relevance_weight
        self.credibility_weight = credibility_weight
        self.freshness_weight = freshness_weight

    def rank(
        self,
        results: list[InternalResult | ExternalResult],
        query: str,
        strategy: RankingStrategy = RankingStrategy.HYBRID
    ) -> list[RankedResult]:
        """排序结果"""
        ranked = []

        for result in results:
            scores = self._calculate_component_scores(result, query, strategy)
            final_score = self._calculate_final_score(scores)

            ranked.append(RankedResult(
                result=result,
                final_score=final_score,
                component_scores=scores
            ))

        # 按最终分数排序
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked

    def _calculate_component_scores(
        self,
        result: InternalResult | ExternalResult,
        query: str,
        strategy: RankingStrategy
    ) -> dict[str, float]:
        """计算各维度分数"""
        scores = {
            "relevance": getattr(result, 'relevance_score', 0.0),
            "credibility": getattr(result, 'source_credibility', 0.5),
            "freshness": getattr(result, 'freshness_score', 1.0)
        }

        # 根据策略调整权重
        if strategy == RankingStrategy.RELEVANCE:
            scores["credibility"] *= 0.5
            scores["freshness"] *= 0.5
        elif strategy == RankingStrategy.CREDIBILITY:
            scores["relevance"] *= 0.5
            scores["freshness"] *= 0.5
        elif strategy == RankingStrategy.RECENCY:
            scores["relevance"] *= 0.5
            scores["credibility"] *= 0.5

        return scores

    def _calculate_final_score(self, scores: dict[str, float]) -> float:
        """计算最终分数"""
        return (
            scores["relevance"] * self.relevance_weight +
            scores["credibility"] * self.credibility_weight +
            scores["freshness"] * self.freshness_weight
        )

    def deduplicate(
        self,
        results: list[RankedResult],
        similarity_threshold: float = 0.9
    ) -> list[RankedResult]:
        """去重"""
        unique = []

        for result in results:
            is_duplicate = False
            for existing in unique:
                similarity = self._calculate_similarity(
                    result.result,
                    existing.result
                )
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(result)

        return unique

    def _calculate_similarity(
        self,
        result1: InternalResult | ExternalResult,
        result2: InternalResult | ExternalResult
    ) -> float:
        """计算两个结果的相似度"""
        content1 = set(getattr(result1, 'content', '').lower().split())
        content2 = set(getattr(result2, 'content', '').lower().split())

        if not content1 or not content2:
            return 0.0

        intersection = content1 & content2
        union = content1 | content2

        return len(intersection) / len(union) if union else 0.0
