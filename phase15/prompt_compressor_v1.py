"""
Prompt Compressor v1 - Prompt压缩器 v1

目标：
1. 减少Token使用量
2. 保留信息密度
3. 提升长对话可持续性
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """压缩结果"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compressed_content: str
    removed_content: List[str]


class PromptCompressor:
    """Prompt压缩器"""
    
    # Token估算：中文约1.5字符/token，英文约4字符/token
    CHARS_PER_TOKEN_ZH = 1.5
    CHARS_PER_TOKEN_EN = 4.0
    
    def __init__(self):
        self.compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
        }
    
    def estimate_tokens(self, text: str) -> int:
        """估算token数"""
        # 简单估算：中文字符 + 英文单词
        zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - zh_chars - en_chars
        
        tokens = (
            zh_chars / self.CHARS_PER_TOKEN_ZH +
            en_chars / self.CHARS_PER_TOKEN_EN +
            other_chars / 2  # 标点符号等
        )
        
        return int(tokens) + 1  # +1保底
    
    def compress_history(
        self,
        history: List[Dict],
        max_turns: int = 3,
        max_length_per_turn: int = 100
    ) -> CompressionResult:
        """
        压缩对话历史
        
        策略：
        1. 只保留最近N轮
        2. 每轮截断到最大长度
        3. 去除无信息量的问候
        """
        original_text = str(history)
        original_tokens = self.estimate_tokens(original_text)
        
        if not history:
            return CompressionResult(0, 0, 0.0, "", [])
        
        removed = []
        compressed = []
        
        # 只保留最近N轮
        recent_history = history[-max_turns:] if len(history) > max_turns else history
        
        if len(history) > max_turns:
            removed.append(f"截断历史: 从{len(history)}轮到{max_turns}轮")
        
        for turn in recent_history:
            user_msg = turn.get('user', '')
            ai_msg = turn.get('ai', '')
            
            # 跳过无信息量的问候
            if self._is_greeting_only(user_msg) and len(compressed) > 0:
                removed.append(f"跳过问候: {user_msg[:20]}...")
                continue
            
            # 截断过长的消息
            orig_user_len = len(user_msg)
            orig_ai_len = len(ai_msg)
            
            if len(user_msg) > max_length_per_turn:
                user_msg = user_msg[:max_length_per_turn] + "..."
                removed.append(f"截断用户消息: {orig_user_len}->{max_length_per_turn}字符")
            
            if len(ai_msg) > max_length_per_turn:
                ai_msg = ai_msg[:max_length_per_turn] + "..."
                removed.append(f"截断AI消息: {orig_ai_len}->{max_length_per_turn}字符")
            
            compressed.append({
                'user': user_msg,
                'ai': ai_msg,
                'turn': turn.get('turn', 0)
            })
        
        compressed_text = str(compressed)
        compressed_tokens = self.estimate_tokens(compressed_text)
        
        ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0
        
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_tokens_saved"] += (original_tokens - compressed_tokens)
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            compressed_content=compressed_text,
            removed_content=removed
        )
    
    def compress_retrieval_results(
        self,
        results: List[Any],
        max_results: int = 3,
        max_length_per_result: int = 150
    ) -> CompressionResult:
        """
        压缩检索结果
        
        策略：
        1. 限制结果数量
        2. 截断长内容
        3. 去除低相关性结果（已在外部过滤）
        """
        original_text = str(results)
        original_tokens = self.estimate_tokens(original_text)
        
        if not results:
            return CompressionResult(0, 0, 0.0, "", [])
        
        removed = []
        compressed = []
        
        # 限制结果数量
        limited_results = results[:max_results] if len(results) > max_results else results
        
        if len(results) > max_results:
            removed.append(f"限制结果数: 从{len(results)}到{max_results}")
        
        for i, result in enumerate(limited_results):
            content = getattr(result, 'content', str(result))
            
            # 截断长内容
            orig_len = len(content)
            if len(content) > max_length_per_result:
                content = content[:max_length_per_result] + "..."
                removed.append(f"截断结果{i+1}: {orig_len}->{max_length_per_result}字符")
            
            compressed.append({
                'id': getattr(result, 'id', i),
                'content': content,
                'score': getattr(result, 'score', 0)
            })
        
        compressed_text = str(compressed)
        compressed_tokens = self.estimate_tokens(compressed_text)
        
        ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            compressed_content=compressed_text,
            removed_content=removed
        )
    
    def compress_system_prompt(self, prompt: str) -> CompressionResult:
        """
        压缩System Prompt
        
        策略：
        1. 去除多余空格和换行
        2. 简化冗长描述
        3. 保留核心指令
        """
        original_tokens = self.estimate_tokens(prompt)
        removed = []
        
        # 去除多余空白
        compressed = re.sub(r'\n\s*\n', '\n', prompt)  # 多空行变单空行
        compressed = re.sub(r'[ \t]+', ' ', compressed)  # 多空格变单空格
        
        # 简化常见冗长表述
        replacements = {
            "你是一个有帮助的AI助手": "AI助手",
            "请根据以下信息": "根据",
            "基于以上上下文": "基于上下文",
            "如果用户的问题": "如用户问题",
        }
        
        for old, new in replacements.items():
            if old in compressed:
                compressed = compressed.replace(old, new)
                removed.append(f"简化: '{old}' -> '{new}'")
        
        compressed_tokens = self.estimate_tokens(compressed)
        ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            compressed_content=compressed,
            removed_content=removed
        )
    
    def _is_greeting_only(self, text: str) -> bool:
        """判断是否为纯问候"""
        greetings = ['你好', '您好', 'hello', 'hi', 'hey', '在吗', '在么']
        text_lower = text.lower().strip()
        
        for greeting in greetings:
            if greeting in text_lower and len(text_lower) < 10:
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        total = self.compression_stats["total_compressions"]
        saved = self.compression_stats["total_tokens_saved"]
        
        return {
            "total_compressions": total,
            "total_tokens_saved": saved,
            "avg_tokens_saved_per_compression": saved / total if total > 0 else 0,
        }


# 便捷函数
def create_prompt_compressor() -> PromptCompressor:
    """创建Prompt压缩器"""
    return PromptCompressor()


# 测试
if __name__ == "__main__":
    compressor = PromptCompressor()
    
    print("="*70)
    print("Prompt Compressor Test")
    print("="*70)
    
    # 测试历史压缩
    print("\n1. 历史压缩测试")
    print("-"*50)
    
    long_history = [
        {"turn": 1, "user": "你好", "ai": "你好！我是AI助手。"},
        {"turn": 2, "user": "项目的目标是什么？项目的目标是什么？项目的目标是什么？", 
         "ai": "项目目标是..." * 50},
        {"turn": 3, "user": "技术栈是什么？", "ai": "技术栈是..."},
        {"turn": 4, "user": "我是谁？", "ai": "你是Alice。"},
        {"turn": 5, "user": "我们之前讨论过什么？", "ai": "讨论过项目目标。"},
    ]
    
    result = compressor.compress_history(long_history, max_turns=3, max_length_per_turn=50)
    print(f"原始Tokens: {result.original_tokens}")
    print(f"压缩后Tokens: {result.compressed_tokens}")
    print(f"压缩率: {result.compression_ratio:.1%}")
    print(f"移除内容: {result.removed_content}")
    
    # 测试检索结果压缩
    print("\n2. 检索结果压缩测试")
    print("-"*50)
    
    class MockResult:
        def __init__(self, id, content, score):
            self.id = id
            self.content = content
            self.score = score
    
    long_results = [
        MockResult(1, "项目目标是在2024年Q3完成核心功能开发..." * 10, 0.9),
        MockResult(2, "技术栈使用Python后端..." * 10, 0.85),
        MockResult(3, "用户名字是Alice..." * 5, 0.8),
        MockResult(4, "之前讨论过..." * 5, 0.75),
        MockResult(5, "其他信息..." * 5, 0.7),
    ]
    
    result = compressor.compress_retrieval_results(long_results, max_results=3, max_length_per_result=80)
    print(f"原始Tokens: {result.original_tokens}")
    print(f"压缩后Tokens: {result.compressed_tokens}")
    print(f"压缩率: {result.compression_ratio:.1%}")
    print(f"移除内容: {result.removed_content}")
    
    # 测试System Prompt压缩
    print("\n3. System Prompt压缩测试")
    print("-"*50)
    
    long_prompt = """
    你是一个有帮助的AI助手。
    
    请根据以下信息回答用户问题：
    
    1. 如果用户的问题需要检索，请使用检索结果。
    2. 基于以上上下文，给出准确回答。
    3. 如果用户的问题不明确，请询问澄清。
    
    记住：你是一个有帮助的AI助手。
    """
    
    result = compressor.compress_system_prompt(long_prompt)
    print(f"原始Tokens: {result.original_tokens}")
    print(f"压缩后Tokens: {result.compressed_tokens}")
    print(f"压缩率: {result.compression_ratio:.1%}")
    print(f"压缩后内容:\n{result.compressed_content[:200]}...")
    
    # 统计
    print("\n4. 压缩统计")
    print("-"*50)
    stats = compressor.get_stats()
    print(f"总压缩次数: {stats['total_compressions']}")
    print(f"总节省Tokens: {stats['total_tokens_saved']}")
    print(f"平均每次节省: {stats['avg_tokens_saved_per_compression']:.0f}")
