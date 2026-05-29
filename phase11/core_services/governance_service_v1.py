"""
Governance Service V1 - 治理服务 V1

Phase 11 WP1 核心组件：
实现完整的治理决策服务

功能：
1. TSLA 分数计算
2. 八动作分流
3. 五门迁移决策
4. 回流重审
5. 治理事件日志
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class GovernanceAction(Enum):
    """八动作"""
    KEEP = "keep"                    # 保留
    PROMOTE = "promote"              # 晋升
    DEMOTE = "demote"                # 降级
    QUARANTINE = "quarantine"        # 隔离
    ARCHIVE = "archive"              # 归档
    REPAIR = "repair"                # 修复
    RECYCLE = "recycle"              # 回流
    DELETE = "delete"                # 删除


class MemoryLayer(Enum):
    """记忆层"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SHALLOW_PERMANENT = "shallow_permanent"
    DEEP_PERMANENT = "deep_permanent"
    QUARANTINE = "quarantine"
    ERROR_ZONE = "error_zone"


@dataclass
class UnitInfo:
    """Unit 信息"""
    unit_id: str
    content: str
    confidence: float
    stability: float
    contamination: float
    current_layer: str
    age_hours: float
    access_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecisions:
    """五门决策"""
    gate_1_promotion: bool = False      # 门 1: 晋升
    gate_2_demotion: bool = False       # 门 2: 降级
    gate_3_repair: bool = False         # 门 3: 修复
    gate_4_quarantine: bool = False     # 门 4: 隔离
    gate_5_archive: bool = False        # 门 5: 归档


@dataclass
class GovernanceDecision:
    """治理决策"""
    query_id: str
    trace_id: str
    unit_id: str
    tsla_score: float
    action: str
    target_layer: str
    confidence: float
    reasoning: str
    gate_decisions: GateDecisions
    needs_review: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GovernanceEvent:
    """治理事件"""
    event_id: str
    event_type: str
    unit_id: str
    decision: GovernanceDecision
    timestamp: datetime
    metadata: Dict[str, Any]


class TSLAScorer:
    """TSLA 分数计算器"""
    
    def __init__(self):
        # 权重配置
        self.weights = {
            "confidence": 0.35,
            "stability": 0.25,
            "contamination": 0.20,
            "age": 0.10,
            "access": 0.10
        }
    
    def calculate(self, unit: UnitInfo) -> float:
        """计算 TSLA 分数"""
        # 置信度分数 (0-1)
        confidence_score = unit.confidence
        
        # 稳定性分数 (0-1)
        stability_score = unit.stability
        
        # 污染分数 (越低越好，所以用 1 - contamination)
        contamination_score = 1 - unit.contamination
        
        # 年龄分数 (越老越稳定)
        age_score = min(1.0, unit.age_hours / (24 * 30))  # 30 天达到满分
        
        # 访问分数 (访问越多越重要)
        access_score = min(1.0, unit.access_count / 100)  # 100 次达到满分
        
        # 加权计算
        tsla_score = (
            confidence_score * self.weights["confidence"] +
            stability_score * self.weights["stability"] +
            contamination_score * self.weights["contamination"] +
            age_score * self.weights["age"] +
            access_score * self.weights["access"]
        )
        
        return round(tsla_score, 4)


class GateController:
    """五门控制器"""
    
    def __init__(self):
        # 门阈值配置
        self.thresholds = {
            "gate_1": {"min_confidence": 0.7, "min_stability": 0.6},
            "gate_2": {"max_contamination": 0.1},
            "gate_3": {"repairable_threshold": 0.4},
            "gate_4": {"quarantine_threshold": 0.2},
            "gate_5": {"archive_threshold": 0.1}
        }
    
    def evaluate(self, unit: UnitInfo, tsla_score: float) -> GateDecisions:
        """评估五门决策"""
        decisions = GateDecisions()
        
        # 门 1: 晋升 - 高置信度 + 高稳定性
        if (unit.confidence >= self.thresholds["gate_1"]["min_confidence"] and
            unit.stability >= self.thresholds["gate_1"]["min_stability"]):
            decisions.gate_1_promotion = True
        
        # 门 2: 降级 - 高污染
        if unit.contamination > self.thresholds["gate_2"]["max_contamination"]:
            decisions.gate_2_demotion = True
        
        # 门 3: 修复 - 可修复的低质量
        if (tsla_score < 0.5 and 
            tsla_score >= self.thresholds["gate_3"]["repairable_threshold"]):
            decisions.gate_3_repair = True
        
        # 门 4: 隔离 - 严重问题但未达到归档
        if (tsla_score < self.thresholds["gate_3"]["repairable_threshold"] and
            tsla_score >= self.thresholds["gate_4"]["quarantine_threshold"]):
            decisions.gate_4_quarantine = True
        
        # 门 5: 归档 - 极低质量
        if tsla_score < self.thresholds["gate_5"]["archive_threshold"]:
            decisions.gate_5_archive = True
        
        return decisions


class ActionRouter:
    """八动作路由器"""
    
    def __init__(self):
        self.layer_hierarchy = [
            MemoryLayer.ERROR_ZONE,
            MemoryLayer.QUARANTINE,
            MemoryLayer.SHORT_TERM,
            MemoryLayer.LONG_TERM,
            MemoryLayer.SHALLOW_PERMANENT,
            MemoryLayer.DEEP_PERMANENT
        ]
    
    def determine_action(
        self,
        unit: UnitInfo,
        tsla_score: float,
        gates: GateDecisions
    ) -> Tuple[GovernanceAction, MemoryLayer, str]:
        """确定治理动作"""
        current_layer = self._parse_layer(unit.current_layer)
        
        # 优先级 1: 归档
        if gates.gate_5_archive:
            return (
                GovernanceAction.ARCHIVE,
                MemoryLayer.ERROR_ZONE,
                "TSLA score too low, archive to error zone"
            )
        
        # 优先级 2: 隔离
        if gates.gate_4_quarantine:
            return (
                GovernanceAction.QUARANTINE,
                MemoryLayer.QUARANTINE,
                "Quality issues, move to quarantine for review"
            )
        
        # 优先级 3: 修复
        if gates.gate_3_repair:
            return (
                GovernanceAction.REPAIR,
                current_layer,
                "Repairable issues detected, trigger repair flow"
            )
        
        # 优先级 4: 降级
        if gates.gate_2_demotion:
            target_layer = self._get_lower_layer(current_layer)
            if target_layer != current_layer:
                return (
                    GovernanceAction.DEMOTE,
                    target_layer,
                    f"High contamination, demote from {current_layer.value}"
                )
        
        # 优先级 5: 晋升
        if gates.gate_1_promotion:
            target_layer = self._get_higher_layer(current_layer)
            if target_layer != current_layer:
                return (
                    GovernanceAction.PROMOTE,
                    target_layer,
                    f"High quality, promote from {current_layer.value}"
                )
        
        # 默认: 保留
        return (
            GovernanceAction.KEEP,
            current_layer,
            "No action needed, keep in current layer"
        )
    
    def _parse_layer(self, layer_str: str) -> MemoryLayer:
        """解析层字符串"""
        try:
            return MemoryLayer(layer_str)
        except ValueError:
            return MemoryLayer.SHORT_TERM
    
    def _get_higher_layer(self, current: MemoryLayer) -> MemoryLayer:
        """获取更高层"""
        idx = self.layer_hierarchy.index(current)
        if idx < len(self.layer_hierarchy) - 1:
            return self.layer_hierarchy[idx + 1]
        return current
    
    def _get_lower_layer(self, current: MemoryLayer) -> MemoryLayer:
        """获取更低层"""
        idx = self.layer_hierarchy.index(current)
        if idx > 0:
            return self.layer_hierarchy[idx - 1]
        return current


class GovernanceService:
    """
    治理服务 V1
    
    职责：
    1. TSLA 分数计算
    2. 八动作分流
    3. 五门迁移决策
    4. 回流重审
    5. 治理事件日志
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8082
    ):
        self.host = host
        self.port = port
        
        # 子组件
        self.tsla_scorer = TSLAScorer()
        self.gate_controller = GateController()
        self.action_router = ActionRouter()
        
        # 事件日志
        self.event_log: List[GovernanceEvent] = []
        self.recycle_queue: List[str] = []  # 回流队列
        
        # 统计
        self.stats = {
            "total_decisions": 0,
            "action_counts": {action.value: 0 for action in GovernanceAction},
            "avg_tsla_score": 0.0
        }
        
        self.is_running = False
    
    async def start(self):
        """启动服务"""
        self.is_running = True
        print(f"Governance Service started on {self.host}:{self.port}")
    
    async def stop(self):
        """停止服务"""
        self.is_running = False
        print("Governance Service stopped")
    
    async def govern(self, query_id: str, trace_id: str, unit: UnitInfo) -> GovernanceDecision:
        """
        执行治理决策
        
        流程：
        1. 计算 TSLA 分数
        2. 评估五门
        3. 确定八动作
        4. 记录事件
        """
        # 1. 计算 TSLA
        tsla_score = self.tsla_scorer.calculate(unit)
        
        # 2. 评估五门
        gate_decisions = self.gate_controller.evaluate(unit, tsla_score)
        
        # 3. 确定动作
        action, target_layer, reasoning = self.action_router.determine_action(
            unit, tsla_score, gate_decisions
        )
        
        # 4. 判断是否需要回流重审
        needs_review = (
            action == GovernanceAction.RECYCLE or
            action == GovernanceAction.REPAIR or
            (action == GovernanceAction.QUARANTINE and unit.access_count > 10)
        )
        
        # 5. 创建决策
        decision = GovernanceDecision(
            query_id=query_id,
            trace_id=trace_id,
            unit_id=unit.unit_id,
            tsla_score=tsla_score,
            action=action.value,
            target_layer=target_layer.value,
            confidence=unit.confidence,
            reasoning=reasoning,
            gate_decisions=gate_decisions,
            needs_review=needs_review
        )
        
        # 6. 记录事件
        self._log_event(decision, unit)
        
        # 7. 更新统计
        self._update_stats(decision)
        
        # 8. 如果需要回流，加入队列
        if needs_review:
            self.recycle_queue.append(unit.unit_id)
        
        return decision
    
    def _log_event(self, decision: GovernanceDecision, unit: UnitInfo):
        """记录治理事件"""
        event = GovernanceEvent(
            event_id=f"evt_{len(self.event_log):06d}",
            event_type="governance_decision",
            unit_id=unit.unit_id,
            decision=decision,
            timestamp=datetime.now(),
            metadata={
                "unit_info": {
                    "confidence": unit.confidence,
                    "stability": unit.stability,
                    "contamination": unit.contamination
                }
            }
        )
        self.event_log.append(event)
    
    def _update_stats(self, decision: GovernanceDecision):
        """更新统计"""
        self.stats["total_decisions"] += 1
        self.stats["action_counts"][decision.action] += 1
        
        # 更新平均 TSLA
        n = self.stats["total_decisions"]
        self.stats["avg_tsla_score"] = (
            self.stats["avg_tsla_score"] * (n - 1) + decision.tsla_score
        ) / n
    
    async def recycle_review(self, unit_id: str) -> Optional[GovernanceDecision]:
        """回流重审"""
        if unit_id not in self.recycle_queue:
            return None
        
        # 模拟重审逻辑
        print(f"  回流重审: {unit_id}")
        
        # 从重审队列移除
        self.recycle_queue.remove(unit_id)
        
        return None  # 实际实现会返回新的决策
    
    def get_events(
        self,
        unit_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[GovernanceEvent]:
        """获取治理事件"""
        events = self.event_log
        
        if unit_id:
            events = [e for e in events if e.unit_id == unit_id]
        
        if action:
            events = [e for e in events if e.decision.action == action]
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "service": "governance",
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "decisions": self.stats,
            "event_count": len(self.event_log),
            "recycle_queue_size": len(self.recycle_queue)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }


async def demo_governance_service():
    """演示治理服务"""
    print("\n" + "="*70)
    print("Governance Service V1 - 演示")
    print("="*70)
    
    service = GovernanceService()
    await service.start()
    
    # 测试用例
    print("\n1. 治理决策测试")
    print("-" * 50)
    
    test_units = [
        # 高质量 Unit - 应该晋升
        UnitInfo(
            unit_id="unit_001",
            content="High quality knowledge",
            confidence=0.9,
            stability=0.85,
            contamination=0.02,
            current_layer="long_term",
            age_hours=24 * 7,
            access_count=50
        ),
        # 中等质量 - 保留
        UnitInfo(
            unit_id="unit_002",
            content="Medium quality knowledge",
            confidence=0.6,
            stability=0.5,
            contamination=0.05,
            current_layer="short_term",
            age_hours=24,
            access_count=10
        ),
        # 高污染 - 降级
        UnitInfo(
            unit_id="unit_003",
            content="Contaminated knowledge",
            confidence=0.5,
            stability=0.4,
            contamination=0.15,
            current_layer="long_term",
            age_hours=24 * 3,
            access_count=5
        ),
        # 低质量 - 隔离
        UnitInfo(
            unit_id="unit_004",
            content="Low quality knowledge",
            confidence=0.3,
            stability=0.2,
            contamination=0.3,
            current_layer="short_term",
            age_hours=12,
            access_count=2
        ),
    ]
    
    for unit in test_units:
        decision = await service.govern(
            query_id=f"q_{unit.unit_id}",
            trace_id=f"trace_{unit.unit_id}",
            unit=unit
        )
        
        print(f"\n  Unit: {unit.unit_id}")
        print(f"  TSLA Score: {decision.tsla_score:.4f}")
        print(f"  Action: {decision.action}")
        print(f"  Target: {decision.target_layer}")
        print(f"  Reason: {decision.reasoning}")
        print(f"  Gates: P={decision.gate_decisions.gate_1_promotion}, "
              f"D={decision.gate_decisions.gate_2_demotion}, "
              f"R={decision.gate_decisions.gate_3_repair}, "
              f"Q={decision.gate_decisions.gate_4_quarantine}, "
              f"A={decision.gate_decisions.gate_5_archive}")
        print(f"  Needs Review: {decision.needs_review}")
    
    # 统计
    print("\n2. 服务统计")
    print("-" * 50)
    
    stats = service.get_stats()
    print(f"  总决策数: {stats['decisions']['total_decisions']}")
    print(f"  平均 TSLA: {stats['decisions']['avg_tsla_score']:.4f}")
    print(f"  事件数: {stats['event_count']}")
    print(f"  回流队列: {stats['recycle_queue_size']}")
    
    print("\n  动作分布:")
    for action, count in stats['decisions']['action_counts'].items():
        if count > 0:
            print(f"    {action}: {count}")
    
    await service.stop()
    
    print("\n" + "="*70)
    print("演示完成")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(demo_governance_service())
