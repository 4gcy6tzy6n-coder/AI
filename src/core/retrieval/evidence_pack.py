from typing import Any

from .ranking import RankedResult


class Evidence:
    """证据定义"""

    def __init__(
        self,
        evidence_id: str,
        content: str,
        source: str,
        source_type: str,
        relevance: float,
        credibility: float
    ):
        self.evidence_id = evidence_id
        self.content = content
        self.source = source
        self.source_type = source_type
        self.relevance = relevance
        self.credibility = credibility
        self.citations: list[str] = []


class EvidenceConflict:
    """证据冲突"""

    def __init__(
        self,
        conflict_id: str,
        evidence_ids: list[str],
        conflict_type: str,
        severity: str
    ):
        self.conflict_id = conflict_id
        self.evidence_ids = evidence_ids
        self.conflict_type = conflict_type
        self.severity = severity


class EvidencePack:
    """证据包"""

    def __init__(
        self,
        pack_id: str,
        query: str,
        evidence: list[Evidence],
        conflicts: list[EvidenceConflict],
        coverage_score: float,
        overall_confidence: float
    ):
        self.pack_id = pack_id
        self.query = query
        self.evidence = evidence
        self.conflicts = conflicts
        self.coverage_score = coverage_score
        self.overall_confidence = overall_confidence


class EvidencePackBuilder:
    """证据包构建器"""

    def build(
        self,
        results: list[RankedResult],
        query: str,
        max_evidence: int = 10
    ) -> EvidencePack:
        """构建证据包"""
        evidence = []

        # 转换结果为证据
        for i, ranked in enumerate(results[:max_evidence]):
            result = ranked.result
            evidence.append(Evidence(
                evidence_id=f"ev_{i}",
                content=getattr(result, 'content', ''),
                source=getattr(result, 'source', 'unknown'),
                source_type=self._determine_source_type(result),
                relevance=ranked.component_scores.get('relevance', 0.0),
                credibility=ranked.component_scores.get('credibility', 0.5)
            ))

        # 检测冲突
        conflicts = self._detect_conflicts(evidence)

        # 计算覆盖度
        coverage = self._calculate_coverage(evidence, query)

        # 计算整体置信度
        confidence = self._calculate_confidence(evidence)

        return EvidencePack(
            pack_id=f"pack_{hash(query)}",
            query=query,
            evidence=evidence,
            conflicts=conflicts,
            coverage_score=coverage,
            overall_confidence=confidence
        )

    def _determine_source_type(
        self,
        result: Any
    ) -> str:
        """确定来源类型"""
        source = getattr(result, 'source', '')
        if 'memory' in source.lower():
            return 'memory'
        elif 'api' in source.lower():
            return 'api'
        elif 'document' in source.lower():
            return 'document'
        return 'unknown'

    def _detect_conflicts(self, evidence: list[Evidence]) -> list[EvidenceConflict]:
        """检测证据冲突"""
        conflicts = []

        # 简化的冲突检测
        for i, e1 in enumerate(evidence):
            for e2 in evidence[i+1:]:
                # 检查内容是否矛盾 (简化实现)
                if self._are_contradictory(e1, e2):
                    conflicts.append(EvidenceConflict(
                        conflict_id=f"conflict_{len(conflicts)}",
                        evidence_ids=[e1.evidence_id, e2.evidence_id],
                        conflict_type="contradiction",
                        severity="medium"
                    ))

        return conflicts

    def _are_contradictory(self, e1: Evidence, e2: Evidence) -> bool:
        """检查两个证据是否矛盾"""
        # 简化的实现，实际应该使用更复杂的逻辑
        return False

    def _calculate_coverage(self, evidence: list[Evidence], query: str) -> float:
        """计算证据覆盖度"""
        if not evidence:
            return 0.0

        query_terms = set(query.lower().split())
        covered_terms = set()

        for e in evidence:
            content_terms = set(e.content.lower().split())
            covered_terms.update(query_terms & content_terms)

        return len(covered_terms) / len(query_terms) if query_terms else 0.0

    def _calculate_confidence(self, evidence: list[Evidence]) -> float:
        """计算整体置信度"""
        if not evidence:
            return 0.0

        total_confidence = sum(
            e.relevance * e.credibility for e in evidence
        )
        return total_confidence / len(evidence)
