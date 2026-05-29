"""
Review Manager - 强审查节点

第五阶段核心组件：
负责训练路径的强审查：
- 来源审查
- 支撑证据审查
- 冲突审查
- 是否允许进入长期正常区/深层永久候选
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ReviewCheck(Enum):
    """审查检查项"""
    SOURCE_AUTHENTICITY = "source_authenticity"      # 来源真实性
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"    # 证据充分性
    CONFLICT_DETECTION = "conflict_detection"        # 冲突检测
    BOUNDARY_CLARITY = "boundary_clarity"           # 边界清晰度
    SINGLE_MEANING = "single_meaning"               # 单义性
    STRUCTURE_INTEGRITY = "structure_integrity"     # 结构完整性


class ReviewDecision(Enum):
    """审查决策"""
    APPROVED = "approved"                           # 通过
    APPROVED_WITH_NOTES = "approved_with_notes"     # 通过但有备注
    NEEDS_REVISION = "needs_revision"               # 需要修改
    REJECTED = "rejected"                           # 拒绝
    PENDING_EVIDENCE = "pending_evidence"           # 待补充证据


@dataclass
class ReviewResult:
    """审查结果"""
    case_id: str
    decision: ReviewDecision
    passed_checks: List[ReviewCheck] = field(default_factory=list)
    failed_checks: List[ReviewCheck] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reviewed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reviewer_id: str = "system"


@dataclass
class Evidence:
    """证据"""
    evidence_id: str
    source: str
    content: str
    reliability: float = 0.0  # 0-1
    verified: bool = False


class ReviewManager:
    """
    强审查管理器
    
    功能：
    1. 来源审查 - 验证数据来源的可靠性
    2. 证据审查 - 检查支撑证据的充分性
    3. 冲突审查 - 检测与现有知识的冲突
    4. 边界审查 - 确认概念边界的清晰度
    5. 综合决策 - 决定是否允许进入长期正常区/深层永久候选
    
    原则：
    - 训练路径需要比用户路径更严格的审查
    - 深层永久候选必须通过所有审查项
    - 审查失败不直接排除，可回流修改
    """
    
    def __init__(self):
        # 来源可信度评级
        self.source_trust_levels = {
            "curated_dataset": 0.95,
            "expert_labeled": 0.90,
            "verified_external": 0.85,
            "peer_reviewed": 0.88,
            "community_verified": 0.70,
            "model_generated": 0.40,
            "user_contributed": 0.50,
            "unverified": 0.20
        }
        
        # 审查阈值
        self.thresholds = {
            "min_source_trust": 0.70,
            "min_evidence_reliability": 0.75,
            "max_conflict_score": 0.30,
            "min_boundary_clarity": 0.70
        }
    
    def review(
        self,
        case_id: str,
        content: str,
        source: str,
        evidence_list: List[Evidence],
        existing_knowledge: List[Dict] = None,
        target_zone: str = "normal"  # normal 或 deep_permanent
    ) -> ReviewResult:
        """
        执行完整审查
        
        Args:
            case_id: 案例ID
            content: 内容
            source: 来源
            evidence_list: 证据列表
            existing_knowledge: 现有知识（用于冲突检测）
            target_zone: 目标区域
        """
        passed_checks = []
        failed_checks = []
        notes = []
        
        # 1. 来源审查
        source_ok, source_note = self._check_source(source)
        if source_ok:
            passed_checks.append(ReviewCheck.SOURCE_AUTHENTICITY)
        else:
            failed_checks.append(ReviewCheck.SOURCE_AUTHENTICITY)
        notes.append(source_note)
        
        # 2. 证据审查
        evidence_ok, evidence_note = self._check_evidence(evidence_list)
        if evidence_ok:
            passed_checks.append(ReviewCheck.EVIDENCE_SUFFICIENCY)
        else:
            failed_checks.append(ReviewCheck.EVIDENCE_SUFFICIENCY)
        notes.append(evidence_note)
        
        # 3. 冲突审查
        conflict_ok, conflict_note = self._check_conflicts(
            content, existing_knowledge or []
        )
        if conflict_ok:
            passed_checks.append(ReviewCheck.CONFLICT_DETECTION)
        else:
            failed_checks.append(ReviewCheck.CONFLICT_DETECTION)
        notes.append(conflict_note)
        
        # 4. 边界清晰度审查
        boundary_ok, boundary_note = self._check_boundary_clarity(content)
        if boundary_ok:
            passed_checks.append(ReviewCheck.BOUNDARY_CLARITY)
        else:
            failed_checks.append(ReviewCheck.BOUNDARY_CLARITY)
        notes.append(boundary_note)
        
        # 5. 单义性审查
        meaning_ok, meaning_note = self._check_single_meaning(content)
        if meaning_ok:
            passed_checks.append(ReviewCheck.SINGLE_MEANING)
        else:
            failed_checks.append(ReviewCheck.SINGLE_MEANING)
        notes.append(meaning_note)
        
        # 6. 结构完整性审查
        structure_ok, structure_note = self._check_structure_integrity(content)
        if structure_ok:
            passed_checks.append(ReviewCheck.STRUCTURE_INTEGRITY)
        else:
            failed_checks.append(ReviewCheck.STRUCTURE_INTEGRITY)
        notes.append(structure_note)
        
        # 计算置信度
        total_checks = len(ReviewCheck)
        confidence = len(passed_checks) / total_checks
        
        # 决策逻辑
        decision = self._make_decision(
            passed_checks, failed_checks, target_zone, confidence
        )
        
        return ReviewResult(
            case_id=case_id,
            decision=decision,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            notes=[n for n in notes if n],
            confidence=confidence
        )
    
    def _check_source(self, source: str) -> tuple[bool, str]:
        """来源审查"""
        trust_level = self.source_trust_levels.get(source, 0.0)
        
        if trust_level >= self.thresholds["min_source_trust"]:
            return True, f"来源 '{source}' 可信度 {trust_level:.2f}，通过审查"
        elif trust_level >= 0.50:
            return True, f"来源 '{source}' 可信度 {trust_level:.2f}，通过但需关注"
        else:
            return False, f"来源 '{source}' 可信度 {trust_level:.2f}，未达阈值"
    
    def _check_evidence(self, evidence_list: List[Evidence]) -> tuple[bool, str]:
        """证据审查"""
        if not evidence_list:
            return False, "无支撑证据"
        
        avg_reliability = sum(e.reliability for e in evidence_list) / len(evidence_list)
        verified_count = sum(1 for e in evidence_list if e.verified)
        
        if avg_reliability >= self.thresholds["min_evidence_reliability"]:
            if verified_count == len(evidence_list):
                return True, f"证据充分且已验证，平均可靠度 {avg_reliability:.2f}"
            else:
                return True, f"证据充分但部分未验证 ({verified_count}/{len(evidence_list)})"
        else:
            return False, f"证据可靠度不足 {avg_reliability:.2f} < {self.thresholds['min_evidence_reliability']}"
    
    def _check_conflicts(
        self,
        content: str,
        existing_knowledge: List[Dict]
    ) -> tuple[bool, str]:
        """冲突审查"""
        # 简化实现：检查内容相似度
        conflict_score = 0.0
        
        for knowledge in existing_knowledge:
            # 这里应该使用更复杂的语义相似度计算
            if knowledge.get("content") == content:
                conflict_score = 1.0
                break
        
        if conflict_score <= self.thresholds["max_conflict_score"]:
            return True, f"无明显冲突 (冲突分 {conflict_score:.2f})"
        else:
            return False, f"检测到潜在冲突 (冲突分 {conflict_score:.2f})"
    
    def _check_boundary_clarity(self, content: str) -> tuple[bool, str]:
        """边界清晰度审查"""
        # 简化实现：检查内容长度和结构
        words = content.split()
        
        # 假设：太短或太长都可能边界不清
        if 10 <= len(words) <= 500:
            return True, f"边界清晰 (词数 {len(words)})"
        else:
            return False, f"边界可能不清 (词数 {len(words)})"
    
    def _check_single_meaning(self, content: str) -> tuple[bool, str]:
        """单义性审查"""
        # 简化实现：检查是否包含多义词指示词
        ambiguous_indicators = ["可能", "也许", "或者", "取决于"]
        
        found = [ind for ind in ambiguous_indicators if ind in content]
        
        if not found:
            return True, "单义性良好"
        else:
            return False, f"检测到多义指示词: {found}"
    
    def _check_structure_integrity(self, content: str) -> tuple[bool, str]:
        """结构完整性审查"""
        # 检查基本结构
        has_definition = "是" in content or "指" in content
        has_boundary = "不包括" in content or "区别于" in content
        
        if has_definition:
            return True, "结构完整"
        else:
            return False, "缺少明确定义"
    
    def _make_decision(
        self,
        passed: List[ReviewCheck],
        failed: List[ReviewCheck],
        target_zone: str,
        confidence: float
    ) -> ReviewDecision:
        """做出审查决策"""
        
        # 深层永久需要全部通过
        if target_zone == "deep_permanent":
            if len(failed) == 0 and confidence >= 0.90:
                return ReviewDecision.APPROVED
            elif len(failed) <= 1 and confidence >= 0.85:
                return ReviewDecision.APPROVED_WITH_NOTES
            elif len(failed) <= 2:
                return ReviewDecision.NEEDS_REVISION
            else:
                return ReviewDecision.REJECTED
        
        # 普通区域标准稍低
        else:
            if len(failed) == 0:
                return ReviewDecision.APPROVED
            elif len(failed) <= 1:
                return ReviewDecision.APPROVED_WITH_NOTES
            elif len(failed) <= 2:
                return ReviewDecision.NEEDS_REVISION
            elif ReviewCheck.EVIDENCE_SUFFICIENCY in failed:
                return ReviewDecision.PENDING_EVIDENCE
            else:
                return ReviewDecision.REJECTED
    
    def can_enter_normal_zone(self, result: ReviewResult) -> bool:
        """检查是否可以进入长期正常区"""
        return result.decision in [
            ReviewDecision.APPROVED,
            ReviewDecision.APPROVED_WITH_NOTES
        ]
    
    def can_enter_deep_permanent(self, result: ReviewResult) -> bool:
        """检查是否可以进入深层永久候选"""
        return (
            result.decision == ReviewDecision.APPROVED and
            result.confidence >= 0.90 and
            len(result.failed_checks) == 0
        )


def demo_review():
    """审查演示"""
    print("\n" + "=" * 70)
    print("Review Manager Demo - 强审查演示")
    print("=" * 70)
    
    manager = ReviewManager()
    
    # 案例1: 高质量训练样本
    evidence1 = [
        Evidence("ev1", "expert_doc", "专家定义内容", 0.95, True),
        Evidence("ev2", "verified_paper", "论文支撑", 0.90, True)
    ]
    
    result1 = manager.review(
        case_id="review_001",
        content="人工智能是指由人制造出来的系统所表现出来的智能，区别于自然智能。",
        source="curated_dataset",
        evidence_list=evidence1,
        existing_knowledge=[],
        target_zone="deep_permanent"
    )
    
    print(f"\n案例 1: 高质量样本")
    print(f"  决策: {result1.decision.value}")
    print(f"  置信度: {result1.confidence:.2f}")
    print(f"  通过: {len(result1.passed_checks)}/{len(ReviewCheck)}")
    print(f"  可进深层永久: {manager.can_enter_deep_permanent(result1)}")
    
    # 案例2: 低质量样本
    evidence2 = [
        Evidence("ev3", "unverified", "未验证内容", 0.40, False)
    ]
    
    result2 = manager.review(
        case_id="review_002",
        content="可能是某种技术",
        source="unverified",
        evidence_list=evidence2,
        existing_knowledge=[],
        target_zone="normal"
    )
    
    print(f"\n案例 2: 低质量样本")
    print(f"  决策: {result2.decision.value}")
    print(f"  置信度: {result2.confidence:.2f}")
    print(f"  失败项: {[c.value for c in result2.failed_checks]}")
    print(f"  可进正常区: {manager.can_enter_normal_zone(result2)}")


if __name__ == "__main__":
    demo_review()
