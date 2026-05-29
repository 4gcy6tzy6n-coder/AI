"""
Prompt Builder v3 - 带Token监控的Prompt构建器

改进：
1. 集成Prompt压缩
2. Token使用量监控
3. 按模块统计Token
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase15.prompt_compressor_v1 import PromptCompressor, create_prompt_compressor


@dataclass
class TokenUsage:
    """Token使用统计"""
    system_tokens: int
    history_tokens: int
    context_tokens: int
    governance_tokens: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_tokens": self.system_tokens,
            "history_tokens": self.history_tokens,
            "context_tokens": self.context_tokens,
            "governance_tokens": self.governance_tokens,
            "total_tokens": self.total_tokens,
        }


class PromptBuilderV3:
    """
    Prompt构建器 v3
    
    带Token监控，支持Prompt压缩
    """
    
    def __init__(self, enable_compression: bool = True):
        self.compressor = create_prompt_compressor() if enable_compression else None
        self.enable_compression = enable_compression
        self.token_stats = []
    
    def build(
        self,
        query: str,
        history: List[Dict],
        retrieval_results: Optional[Any],
        governance_result: Dict,
        memory_context: List[Dict]
    ) -> Dict[str, Any]:
        """
        构建Prompt（带Token监控）
        
        Returns:
            {
                "system_prompt": str,
                "user_prompt": str,
                "conversation_history": List,
                "token_usage": TokenUsage,
            }
        """
        
        # 1. 构建System Prompt
        system_prompt = self._build_system_prompt()
        
        # 2. 压缩并构建历史
        if self.enable_compression and self.compressor:
            history_result = self.compressor.compress_history(history, max_turns=3, max_length_per_turn=100)
            compressed_history = eval(history_result.compressed_content) if history_result.compressed_content else []
        else:
            compressed_history = history[-3:] if len(history) > 3 else history
        
        # 3. 压缩检索上下文
        context_str = ""
        if retrieval_results and hasattr(retrieval_results, 'results'):
            if self.enable_compression and self.compressor:
                context_result = self.compressor.compress_retrieval_results(
                    retrieval_results.results, 
                    max_results=3, 
                    max_length_per_result=150
                )
                context_str = context_result.compressed_content
            else:
                context_str = self._format_retrieval_context(retrieval_results.results)
        
        # 4. 构建Governance上下文
        governance_str = self._build_governance_context(governance_result)
        
        # 5. 构建用户Prompt
        user_prompt = self._build_user_prompt(query, context_str, governance_str)
        
        # 6. 统计Token
        token_usage = self._calculate_token_usage(
            system_prompt, compressed_history, context_str, governance_str
        )
        
        self.token_stats.append(token_usage)
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "conversation_history": compressed_history,
            "token_usage": token_usage.to_dict(),
        }
    
    def _build_system_prompt(self) -> str:
        """构建System Prompt（精简版）"""
        return """AI助手。基于检索结果回答，不确定时说明置信度。引用来源用[来源:N]格式。"""
    
    def _format_retrieval_context(self, results: List[Any]) -> str:
        """格式化检索上下文"""
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results[:3], 1):
            content = getattr(result, 'content', str(result))
            context_parts.append(f"[{i}] {content}")
        
        return "\n".join(context_parts)
    
    def _build_governance_context(self, governance_result: Dict) -> str:
        """构建Governance上下文"""
        strategy = governance_result.get('strategy', 'CONSERVATIVE')
        confidence = governance_result.get('confidence', 0.6)
        
        return f"策略:{strategy.value if hasattr(strategy, 'value') else strategy},置信度:{confidence:.0%}"
    
    def _build_user_prompt(self, query: str, context: str, governance: str) -> str:
        """构建用户Prompt"""
        parts = [f"问题:{query}"]
        
        if context:
            parts.append(f"检索结果:\n{context}")
        
        parts.append(f"治理:{governance}")
        
        return "\n\n".join(parts)
    
    def _calculate_token_usage(
        self,
        system_prompt: str,
        history: List[Dict],
        context: str,
        governance: str
    ) -> TokenUsage:
        """计算Token使用量"""
        
        system_tokens = self._estimate_tokens(system_prompt)
        history_tokens = self._estimate_tokens(str(history))
        context_tokens = self._estimate_tokens(context)
        governance_tokens = self._estimate_tokens(governance)
        
        return TokenUsage(
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            context_tokens=context_tokens,
            governance_tokens=governance_tokens,
            total_tokens=system_tokens + history_tokens + context_tokens + governance_tokens
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数"""
        import re
        
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - zh_chars - en_chars
        
        tokens = (
            zh_chars / 1.5 +  # 中文
            en_chars / 4.0 +   # 英文
            other_chars / 2    # 其他
        )
        
        return int(tokens) + 1
    
    def get_token_stats(self) -> Dict[str, Any]:
        """获取Token统计"""
        if not self.token_stats:
            return {}
        
        total_samples = len(self.token_stats)
        
        return {
            "total_samples": total_samples,
            "avg_system_tokens": sum(s.system_tokens for s in self.token_stats) / total_samples,
            "avg_history_tokens": sum(s.history_tokens for s in self.token_stats) / total_samples,
            "avg_context_tokens": sum(s.context_tokens for s in self.token_stats) / total_samples,
            "avg_governance_tokens": sum(s.governance_tokens for s in self.token_stats) / total_samples,
            "avg_total_tokens": sum(s.total_tokens for s in self.token_stats) / total_samples,
            "max_total_tokens": max(s.total_tokens for s in self.token_stats),
            "min_total_tokens": min(s.total_tokens for s in self.token_stats),
        }


# 便捷函数
def create_prompt_builder_v3(enable_compression: bool = True) -> PromptBuilderV3:
    """创建Prompt构建器 v3"""
    return PromptBuilderV3(enable_compression=enable_compression)


# 测试
if __name__ == "__main__":
    builder = create_prompt_builder_v3()
    
    print("="*70)
    print("Prompt Builder v3 Test")
    print("="*70)
    
    # 模拟数据
    query = "项目的目标是什么？"
    history = [
        {"turn": 1, "user": "你好", "ai": "你好！"},
        {"turn": 2, "user": "技术栈是什么？", "ai": "技术栈是Python+React+PostgreSQL。"},
    ]
    
    class MockRetrievalResult:
        def __init__(self):
            self.results = [
                type('Result', (), {'content': '项目目标是在2024年Q3完成核心功能开发。', 'id': '1', 'score': 0.9}),
                type('Result', (), {'content': '项目目前已完成Phase 1和Phase 2。', 'id': '2', 'score': 0.85}),
            ]
    
    retrieval_results = MockRetrievalResult()
    governance_result = {"strategy": "RETRIEVAL_FIRST", "confidence": 0.85}
    memory_context = []
    
    # 构建Prompt
    prompt_data = builder.build(query, history, retrieval_results, governance_result, memory_context)
    
    print("\n1. 构建的Prompt")
    print("-"*50)
    print(f"System: {prompt_data['system_prompt']}")
    print(f"\nUser: {prompt_data['user_prompt'][:200]}...")
    
    print("\n2. Token使用统计")
    print("-"*50)
    token_usage = prompt_data['token_usage']
    print(f"System Tokens: {token_usage['system_tokens']}")
    print(f"History Tokens: {token_usage['history_tokens']}")
    print(f"Context Tokens: {token_usage['context_tokens']}")
    print(f"Governance Tokens: {token_usage['governance_tokens']}")
    print(f"Total Tokens: {token_usage['total_tokens']}")
    
    # 多次构建后的统计
    print("\n3. 多次构建后的统计")
    print("-"*50)
    
    # 模拟多次调用
    for i in range(4):
        builder.build(
            f"问题{i}",
            history + [{"turn": 3+i, "user": f"问题{i}", "ai": f"回答{i}"}],
            retrieval_results,
            governance_result,
            memory_context
        )
    
    stats = builder.get_token_stats()
    print(f"总样本数: {stats['total_samples']}")
    print(f"平均总Tokens: {stats['avg_total_tokens']:.0f}")
    print(f"最大总Tokens: {stats['max_total_tokens']}")
    print(f"最小总Tokens: {stats['min_total_tokens']}")
