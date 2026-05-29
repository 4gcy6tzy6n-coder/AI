from typing import Any

from .state_graph import StateGraph


class Conflict:
    """冲突定义"""

    def __init__(
        self,
        conflict_id: str,
        node_ids: list[str],
        conflict_type: str,
        description: str
    ):
        self.conflict_id = conflict_id
        self.node_ids = node_ids
        self.conflict_type = conflict_type
        self.description = description


class Resolution:
    """解决方案"""

    def __init__(
        self,
        resolution_id: str,
        conflict_id: str,
        strategy: str,
        selected_node_id: str | None = None,
        resolved_nodes: list[str] | None = None
    ):
        self.resolution_id = resolution_id
        self.conflict_id = conflict_id
        self.strategy = strategy
        self.selected_node_id = selected_node_id
        self.resolved_nodes = resolved_nodes or []


class ConflictResolver:
    """冲突解决器"""

    def detect_conflicts(self, graph: StateGraph) -> list[Conflict]:
        """检测状态图中的冲突"""
        conflicts = []

        # 简单的冲突检测：查找矛盾的结果
        node_results = {}
        for node_id, node in graph.nodes.items():
            result_key = str(node.step_result.get("conclusion", ""))
            if result_key in node_results:
                # 发现潜在冲突
                conflict = Conflict(
                    conflict_id=f"conflict_{len(conflicts)}",
                    node_ids=[node_results[result_key], node_id],
                    conflict_type="contradiction",
                    description="Contradictory conclusions detected"
                )
                conflicts.append(conflict)
            else:
                node_results[result_key] = node_id

        return conflicts

    def resolve(self, conflict: Conflict) -> Resolution:
        """解决冲突"""
        # 简单的解决策略：选择置信度更高的节点
        # 实际实现中应该更复杂
        return Resolution(
            resolution_id=f"resolution_{conflict.conflict_id}",
            conflict_id=conflict.conflict_id,
            strategy="confidence_based",
            resolved_nodes=conflict.node_ids
        )

    def evaluate_evidence(
        self,
        node_id: str,
        graph: StateGraph
    ) -> dict[str, Any]:
        """评估节点的证据强度"""
        node = graph.nodes.get(node_id)
        if not node:
            return {"strength": 0.0}

        return {
            "strength": node.confidence,
            "evidence_count": len(node.step_result.get("evidence", [])),
            "supporting_paths": 1
        }
