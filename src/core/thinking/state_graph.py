from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID, uuid4


@dataclass
class Node:
    """状态图节点"""
    node_id: str
    step_result: dict[str, Any]
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """状态图边"""
    edge_id: str
    from_node: str
    to_node: str
    relation: str
    weight: float = 1.0


@dataclass
class StateGraph:
    """推理状态图"""
    graph_id: UUID
    thinking_unit_id: UUID
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    root_node_id: Optional[str] = None


class StateGraphManager:
    """状态图管理器"""

    def __init__(self):
        self._graphs: dict[UUID, StateGraph] = {}

    def create_graph(self, thinking_unit_id: UUID) -> StateGraph:
        """创建新的状态图"""
        graph = StateGraph(
            graph_id=uuid4(),
            thinking_unit_id=thinking_unit_id
        )
        self._graphs[graph.graph_id] = graph
        return graph

    def get_graph(self, graph_id: UUID) -> Optional[StateGraph]:
        """获取状态图"""
        return self._graphs.get(graph_id)

    def add_result(
        self,
        graph: StateGraph,
        result: dict[str, Any],
        parent_node_id: Optional[str] = None
    ) -> Node:
        """添加推理结果到图"""
        node_id = f"node_{len(graph.nodes)}"
        node = Node(
            node_id=node_id,
            step_result=result,
            confidence=result.get("confidence", 0.0)
        )
        graph.nodes[node_id] = node

        # 设置根节点
        if graph.root_node_id is None:
            graph.root_node_id = node_id

        # 添加边
        if parent_node_id and parent_node_id in graph.nodes:
            edge_id = f"edge_{len(graph.edges)}"
            edge = Edge(
                edge_id=edge_id,
                from_node=parent_node_id,
                to_node=node_id,
                relation="follows"
            )
            graph.edges[edge_id] = edge

        return node

    def get_path(
        self,
        graph: StateGraph,
        start_node_id: str,
        end_node_id: str
    ) -> list[str]:
        """获取两个节点之间的路径"""
        # 简化的路径查找 (BFS)
        visited = {start_node_id}
        queue = [(start_node_id, [start_node_id])]

        while queue:
            current, path = queue.pop(0)

            if current == end_node_id:
                return path

            # 查找相邻节点
            for edge in graph.edges.values():
                if edge.from_node == current and edge.to_node not in visited:
                    visited.add(edge.to_node)
                    queue.append((edge.to_node, path + [edge.to_node]))

        return []

    def apply_resolution(
        self,
        graph: StateGraph,
        resolution: dict[str, Any]
    ) -> None:
        """应用冲突解决方案"""
        # 标记被解决的节点
        for node_id in resolution.get("resolved_nodes", []):
            if node_id in graph.nodes:
                graph.nodes[node_id].metadata["resolved"] = True
