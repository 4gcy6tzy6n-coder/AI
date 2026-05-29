from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class UnitType(str, Enum):
    """Unit 类型"""
    INPUT = "input"
    THINKING = "thinking"
    RETRIEVAL = "retrieval"
    CONCLUSION = "conclusion"
    OUTPUT = "output"


class MemoryZone(str, Enum):
    """记忆区域类型"""
    TRANSIENT = "transient"
    LONG_TERM = "long_term"
    SHALLOW_PERMANENT = "shallow_permanent"
    DEEP_PERMANENT = "deep_permanent"


class SourceType(str, Enum):
    """来源类型"""
    USER_INPUT = "user_input"
    SYSTEM_GENERATED = "system_generated"
    RETRIEVED = "retrieved"
    INFERRED = "inferred"


class TSLAAction(str, Enum):
    """TSLA 动作类型"""
    KEEP = "keep"
    REVIEW = "review"
    SPLIT = "split"
    ISOLATE = "isolate"
    REJECT = "reject"


@dataclass
class Unit:
    """
    Unit - 最小治理单元

    完整字段定义，第一阶段使用默认值和简化计算
    """
    # 核心标识
    unit_id: UUID = field(default_factory=uuid4)
    unit_type: UnitType = UnitType.INPUT

    # 语义三态
    script_form: str = ""           # 书写形式（如：苹果）
    phonetic_form: str = ""         # 语音形式（如：píng guǒ）
    core_meaning: str = ""          # 核心语义（如：一种水果）

    # 语言和来源
    language_tag: str = "zh-CN"     # 语言标签
    source_type: SourceType = SourceType.USER_INPUT

    # TSLA 分数 (0.0 - 1.0)
    truth_score: float = 0.5        # 真实性分数
    stability_score: float = 0.5    # 稳定性分数
    evidence_score: float = 0.5     # 证据充分性分数
    conflict_cleanliness: float = 1.0  # 冲突清洁度
    legality_score: float = 1.0     # 合法性分数
    governance_score: float = 0.5   # 治理综合分数

    # 记忆位置
    memory_layer: str = "transient"  # 当前记忆层
    zone_type: MemoryZone = MemoryZone.TRANSIENT

    # 内容和时间戳
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        # 处理 source_type，可能是字符串或枚举
        if isinstance(self.source_type, str):
            source_type_value = self.source_type
        else:
            source_type_value = self.source_type.value

        return {
            "unit_id": str(self.unit_id),
            "unit_type": self.unit_type.value if hasattr(self.unit_type, 'value') else self.unit_type,
            "script_form": self.script_form,
            "phonetic_form": self.phonetic_form,
            "core_meaning": self.core_meaning,
            "language_tag": self.language_tag,
            "source_type": source_type_value,
            "truth_score": self.truth_score,
            "stability_score": self.stability_score,
            "evidence_score": self.evidence_score,
            "conflict_cleanliness": self.conflict_cleanliness,
            "legality_score": self.legality_score,
            "governance_score": self.governance_score,
            "memory_layer": self.memory_layer,
            "zone_type": self.zone_type.value if hasattr(self.zone_type, 'value') else self.zone_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_input(cls, text: str, session_id: Optional[str] = None) -> "Unit":
        """从用户输入创建 Unit"""
        return cls(
            unit_type=UnitType.INPUT,
            script_form=text,
            content=text,
            session_id=session_id,
            metadata={"raw_input": text}
        )

    def is_structurally_valid(self) -> bool:
        """检查结构合法性"""
        # 第一阶段：至少有 script_form 或 content
        return bool(self.script_form) or bool(self.content)

    def has_meaning(self) -> bool:
        """是否有基本语义定义"""
        return bool(self.core_meaning)


@dataclass
class CandidateConclusion:
    """候选结论"""
    answer_text: str = ""
    used_units: list[Unit] = field(default_factory=list)
    used_evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    conflict_flags: list[str] = field(default_factory=list)
    confidence_stub: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "answer_text": self.answer_text,
            "used_units": [u.to_dict() for u in self.used_units],
            "used_evidence": self.used_evidence,
            "missing_slots": self.missing_slots,
            "conflict_flags": self.conflict_flags,
            "confidence_stub": self.confidence_stub
        }


@dataclass
class TSLAResult:
    """TSLA 审查结果"""
    action: TSLAAction = TSLAAction.REVIEW
    confidence: float = 0.5
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "details": self.details
        }
