import json
import sys
from pathlib import Path
from typing import Any, Optional

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval.external_retriever import ExternalRetriever
from core.retrieval.internal_retriever import InternalRetriever
from core.thinking.engine import ThinkingEngine
from core.tsla.reviewer import TSLAReviewer
from core.unit.models import MemoryZone, TSLAAction, Unit
from core.unit.parser import UnitParser


class UserPipeline:
    """
    用户请求处理流水线 - 第一阶段最小可运行版本

    链路：解析 → 思考 → 缺失判断 → 检索 → 候选结论 → TSLA动作建议
    """

    def __init__(self):
        self.parser = UnitParser()
        self.thinking_engine = ThinkingEngine()
        self.internal_retriever = InternalRetriever()
        self.external_retriever = ExternalRetriever()
        self.tsla_reviewer = TSLAReviewer()

    def process(
        self,
        user_input: str,
        session_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        处理用户输入，走完完整链路

        返回结构化结果，包含：
        - input: 原始输入
        - units: 生成的 Unit 候选
        - thinking_state_trace: 思考状态轨迹
        - candidate_conclusion: 候选结论
        - tsla_action: TSLA 动作建议
        - memory_write_decision: 记忆写入决策
        """
        # 1. 解析输入 → InputParsed
        parse_result = self.parser.parse(user_input, session_id)

        # 提取主输入 Unit 和激活的 Units
        input_unit = parse_result.units[0] if parse_result.units else Unit.from_input(user_input, session_id)
        activated_units = parse_result.units[1:] if len(parse_result.units) > 1 else []

        # 2. 检索证据（内部 + 外部）
        evidence_pack = self._retrieve_evidence(user_input)

        # 3. 思考引擎执行状态转换
        thinking_result = self.thinking_engine.think(
            input_unit=input_unit,
            activated_units=activated_units,
            evidence_pack=evidence_pack
        )

        # 4. 获取候选结论
        candidate = self.thinking_engine.get_candidate()

        # 5. TSLA 审查
        tsla_result = self.tsla_reviewer.review(
            candidate=candidate,
            gap_result=self.thinking_engine._gap_result,
            units=activated_units
        )

        # 6. 确定记忆写入决策
        memory_decision = self._determine_memory_decision(tsla_result)

        # 7. 组装最终输出
        result = {
            "input": user_input,
            "session_id": session_id,
            "units": [u.to_dict() for u in parse_result.units],
            "thinking_state_trace": thinking_result["state_trace"],
            "current_state": thinking_result["current_state"],
            "gap_result": thinking_result["gap_result"],
            "candidate_conclusion": candidate.to_dict() if candidate else None,
            "tsla_action": tsla_result.to_dict(),
            "memory_write_decision": memory_decision
        }

        return result

    def _retrieve_evidence(self, query: str) -> list[dict[str, Any]]:
        """检索证据（内部 + 外部）"""
        evidence_pack = []

        # 内部检索
        internal_results = self.internal_retriever.retrieve(query, top_k=2)
        for result in internal_results:
            evidence_pack.append(result.to_dict())

        # 外部检索
        external_results = self.external_retriever.retrieve(query, top_k=1)
        evidence_pack.extend(external_results)

        return evidence_pack

    def _determine_memory_decision(self, tsla_result) -> str:
        """根据 TSLA 结果确定记忆写入决策"""
        action = tsla_result.action

        if action == TSLAAction.KEEP:
            return "transient"  # 第一阶段只写入瞬态层
        elif action == TSLAAction.REVIEW:
            return "transient_only"  # 仅瞬态，等待审查
        elif action == TSLAAction.REJECT:
            return "no_write"  # 不写入
        elif action == TSLAAction.ISOLATE:
            return "isolated_transient"  # 隔离的瞬态存储
        elif action == TSLAAction.SPLIT:
            return "transient_pending_split"  # 瞬态，等待拆分

        return "transient_only"

    def process_and_print(self, user_input: str, session_id: Optional[str] = None) -> None:
        """处理并打印格式化结果"""
        result = self.process(user_input, session_id)

        print("=" * 60)
        print(f"输入: {result['input']}")
        print("=" * 60)

        print("\n📋 生成的 Units:")
        for i, unit in enumerate(result['units'], 1):
            print(f"  Unit {i}:")
            print(f"    - script_form: {unit['script_form']}")
            print(f"    - core_meaning: {unit.get('core_meaning', 'N/A')}")
            print(f"    - truth_score: {unit['truth_score']}")
            print(f"    - legality_score: {unit['legality_score']}")

        print("\n🔄 思考状态轨迹:")
        for i, state in enumerate(result['thinking_state_trace'], 1):
            print(f"  {i}. {state}")

        print("\n🔍 缺口检测结果:")
        gap = result['gap_result']
        if gap and gap['has_gap']:
            print(f"  ⚠️  存在缺口 (严重程度: {gap['severity']})")
            print(f"  原因: {', '.join(gap['reasons'])}")
        else:
            print("  ✅ 无明显缺口")

        print("\n💡 候选结论:")
        candidate = result['candidate_conclusion']
        if candidate:
            print(f"  回答: {candidate['answer_text']}")
            print(f"  置信度: {candidate['confidence_stub']:.2f}")
            if candidate['missing_slots']:
                print(f"  缺失槽位: {candidate['missing_slots']}")
            if candidate['conflict_flags']:
                print(f"  冲突标记: {candidate['conflict_flags']}")

        print("\n🛡️ TSLA 动作建议:")
        tsla = result['tsla_action']
        print(f"  动作: {tsla['action']}")
        print(f"  置信度: {tsla['confidence']:.2f}")
        print(f"  原因: {tsla['reason']}")

        print("\n💾 记忆写入决策:")
        print(f"  {result['memory_write_decision']}")

        print("\n" + "=" * 60)

        return result


# 简单的命令行测试接口
if __name__ == "__main__":
    import sys

    pipeline = UserPipeline()

    if len(sys.argv) > 1:
        # 从命令行参数获取输入
        user_input = sys.argv[1]
        pipeline.process_and_print(user_input)
    else:
        # 运行预设测试用例
        test_cases = [
            "苹果是什么",
            "记忆层有什么作用",
            "某个没定义的新词是什么意思",
        ]

        print("\n" + "🧪 " * 30)
        print("运行预设测试用例")
        print("🧪 " * 30 + "\n")

        for test_input in test_cases:
            pipeline.process_and_print(test_input)
            print("\n")
