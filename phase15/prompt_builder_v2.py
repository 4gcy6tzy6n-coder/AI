"""
Prompt Builder v2 - 改进版 Prompt 构建器

解决上下文注入问题：
1. 明确告知模型可以使用检索结果和记忆
2. 结构化格式化上下文
3. 清晰的块分隔
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievedItem:
    """检索项"""
    content: str
    source: str
    confidence: float


@dataclass
class MemoryItem:
    """记忆项"""
    content: str
    layer: str
    timestamp: str


class PromptBuilderV2:
    """
    改进版 Prompt 构建器
    
    结构：
    1. System: 角色定义 + 回答协议 + 可用上下文说明
    2. History: 对话历史
    3. Context: 检索结果 + 记忆（结构化）
    4. Governance: 治理决策
    5. Query: 当前问题
    """
    
    def build(
        self,
        query: str,
        history: List[Dict[str, Any]],
        retrieval_results: Optional[Any],
        governance_result: Dict[str, Any],
        memory_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建完整的 LLM 输入"""
        
        # 1. 构建系统 Prompt
        system_prompt = self._build_system_prompt(governance_result)
        
        # 2. 构建对话历史
        conversation_history = self._build_history(history)
        
        # 3. 构建上下文块
        context_block = self._build_context_block(
            retrieval_results,
            memory_context
        )
        
        # 4. 构建治理块
        governance_block = self._build_governance_block(governance_result)
        
        # 5. 构建最终用户输入
        user_prompt = self._build_user_prompt(query, context_block, governance_block)
        
        return {
            "system_prompt": system_prompt,
            "conversation_history": conversation_history,
            "user_prompt": user_prompt,
            "full_prompt": self._assemble_full_prompt(
                system_prompt, conversation_history, user_prompt
            )
        }
    
    def _build_system_prompt(self, governance_result: Dict[str, Any]) -> str:
        """构建系统 Prompt"""
        
        return """你是一个具有治理意识的AI助手。你的回答必须遵循以下协议：

## 你的角色
- 你是用户的智能助手，可以访问对话历史、检索结果和相关记忆
- 你的目标是提供准确、有帮助且符合治理要求的回答

## 回答协议
1. **使用可用信息**：你可以且应该使用提供的对话历史、检索结果和记忆来回答问题
2. **标注来源**：如果使用了检索结果或记忆中的信息，必须标注来源[来源:N]
3. **承认不确定**：如果不确定，必须说明置信度，不要编造
4. **遵守治理**：遵循治理建议的策略和风险等级

## 响应策略选择
根据置信度和风险选择策略：
- **DIRECT**: 置信度>0.85，直接回答
- **RETRIEVAL_FIRST**: 需要验证，先检索再回答
- **CONSERVATIVE**: 置信度0.5-0.85，说明不确定性
- **DECLINE**: 置信度<0.5，诚实拒绝
- **REVIEW**: 高风险内容，标记审查

## 输出格式（必须严格遵守）
请以JSON格式输出，不要包含其他内容：
{
    "strategy": "DIRECT|RETRIEVAL_FIRST|CONSERVATIVE|DECLINE|REVIEW",
    "confidence": 0.0-1.0,
    "response": "最终回答",
    "reasoning": "推理过程",
    "citations": ["来源1", "来源2"]
}"""
    
    def _build_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """构建对话历史"""
        messages = []
        
        # 只取最近5轮
        for turn in history[-5:]:
            if 'user' in turn and turn['user']:
                messages.append({
                    "role": "user",
                    "content": turn['user']
                })
            if 'ai' in turn and turn['ai']:
                messages.append({
                    "role": "assistant",
                    "content": turn['ai']
                })
        
        return messages
    
    def _build_context_block(
        self,
        retrieval_results: Optional[Any],
        memory_context: List[Dict[str, Any]]
    ) -> str:
        """构建上下文块（检索结果 + 记忆）"""
        
        parts = []
        
        # 检索结果
        retrieval_items = []
        if retrieval_results and hasattr(retrieval_results, 'results'):
            retrieval_items = retrieval_results.results[:3]
        
        if retrieval_items:
            parts.append("=" * 50)
            parts.append("【检索到的相关信息】")
            parts.append("以下信息来自知识库检索，你可以使用这些信息回答问题：")
            parts.append("")
            
            for i, item in enumerate(retrieval_items, 1):
                confidence = item.metadata.get('confidence', 0.5) if hasattr(item, 'metadata') else 0.5
                parts.append(f"[{i}] 来源：{item.layer} | 相关度：{confidence:.0%}")
                parts.append(f"    内容：{item.content}")
                parts.append("")
        else:
            parts.append("=" * 50)
            parts.append("【检索结果】")
            parts.append("本次查询未检索到相关信息")
            parts.append("")
        
        # 记忆
        if memory_context:
            parts.append("=" * 50)
            parts.append("【相关记忆】")
            parts.append("以下信息来自对话记忆，你可以参考：")
            parts.append("")
            
            for i, mem in enumerate(memory_context[:3], 1):
                layer = mem.get('layer', 'unknown')
                content = mem.get('content', '')
                parts.append(f"[{i}] 类型：{layer}")
                parts.append(f"    内容：{content}")
                parts.append("")
        else:
            parts.append("=" * 50)
            parts.append("【相关记忆】")
            parts.append("无相关记忆")
            parts.append("")
        
        return "\n".join(parts)
    
    def _build_governance_block(self, governance_result: Dict[str, Any]) -> str:
        """构建治理块"""
        
        strategy = governance_result.get('strategy', 'DIRECT')
        confidence = governance_result.get('confidence', 0.7)
        risk_level = governance_result.get('risk_level', 'low')
        reasoning = governance_result.get('reasoning', '')
        
        return f"""{'=' * 50}
【治理建议】
- 推荐策略：{strategy.value if hasattr(strategy, 'value') else strategy}
- 置信度评估：{confidence:.0%}
- 风险等级：{risk_level}
- 建议理由：{reasoning}
{'=' * 50}"""
    
    def _build_user_prompt(
        self,
        query: str,
        context_block: str,
        governance_block: str
    ) -> str:
        """构建用户 Prompt"""
        
        return f"""{context_block}

{governance_block}

【当前问题】
{query}

请根据以上信息回答问题。记住：
1. 你可以使用检索结果和记忆中的信息
2. 如果使用了这些信息，请标注来源[来源:N]
3. 如果不确定，请说明置信度
4. 以JSON格式输出"""
    
    def _assemble_full_prompt(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        user_prompt: str
    ) -> str:
        """组装完整 Prompt（用于调试）"""
        
        parts = []
        parts.append("=" * 70)
        parts.append("SYSTEM PROMPT")
        parts.append("=" * 70)
        parts.append(system_prompt)
        parts.append("")
        
        if conversation_history:
            parts.append("=" * 70)
            parts.append("CONVERSATION HISTORY")
            parts.append("=" * 70)
            for msg in conversation_history:
                parts.append(f"[{msg['role'].upper()}]: {msg['content']}")
            parts.append("")
        
        parts.append("=" * 70)
        parts.append("USER PROMPT")
        parts.append("=" * 70)
        parts.append(user_prompt)
        
        return "\n".join(parts)


# 便捷函数
def create_prompt_builder_v2() -> PromptBuilderV2:
    """创建 PromptBuilderV2 实例"""
    return PromptBuilderV2()


# 测试
if __name__ == "__main__":
    builder = PromptBuilderV2()
    
    # 模拟输入
    query = "Python是什么？"
    history = [
        {"user": "你好", "ai": "你好！有什么可以帮助你的？"}
    ]
    
    # 模拟检索结果
    class MockRetrievalResult:
        def __init__(self):
            class Item:
                def __init__(self):
                    self.layer = "long_term"
                    self.content = "Python是一种高级编程语言，由Guido van Rossum创建"
                    self.metadata = {"confidence": 0.95}
            self.results = [Item()]
    
    retrieval_results = MockRetrievalResult()
    
    governance_result = {
        "strategy": "DIRECT",
        "confidence": 0.9,
        "risk_level": "low",
        "reasoning": "高置信度事实查询"
    }
    
    memory_context = [
        {"content": "用户喜欢编程", "layer": "ephemeral"}
    ]
    
    result = builder.build(
        query=query,
        history=history,
        retrieval_results=retrieval_results,
        governance_result=governance_result,
        memory_context=memory_context
    )
    
    print(result["full_prompt"])
