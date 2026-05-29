"""
Query Rewriter v1 - 查询改写器 v1

解决检索匹配问题 Layer 2：
将用户问题改写为更适合检索的查询
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class RewriteResult:
    """改写结果"""
    original_query: str
    rewritten_query: str
    rewrite_reason: str
    added_keywords: List[str]


class QueryRewriter:
    """
    查询改写器
    
    策略：
    1. 身份查询 -> 添加用户标识关键词
    2. 项目查询 -> 添加项目关键词
    3. 技术查询 -> 添加技术关键词
    4. 历史查询 -> 添加时间范围
    """
    
    # 关键词映射表 - 提取核心关键词用于检索
    KEYWORD_PATTERNS = {
        "identity": {
            "patterns": ["名字", "我是谁", "我叫", "姓名", "称呼"],
            "keywords": ["名字"],  # 核心关键词，用于匹配存储内容
            "search_terms": ["名字", "用户"]  # 检索时使用的词
        },
        "project": {
            "patterns": ["项目", "project", "目标", "goal", "计划"],
            "keywords": ["项目", "目标"],
            "search_terms": ["项目", "目标"]
        },
        "tech_stack": {
            "patterns": ["技术栈", "tech stack", "用什么技术", "技术方案"],
            "keywords": ["技术栈"],
            "search_terms": ["技术栈", "Python", "React", "PostgreSQL"]
        },
        "python": {
            "patterns": ["python", "Python", "编程语言"],
            "keywords": ["Python"],
            "search_terms": ["Python", "编程"]
        },
        "history": {
            "patterns": ["之前", "刚才", "上次", "之前讨论", "说过"],
            "keywords": ["对话", "历史"],
            "search_terms": ["对话", "讨论"]
        }
    }
    
    def __init__(self):
        self.context_keywords = []  # 从对话历史提取的关键词
    
    def rewrite(
        self,
        user_input: str,
        intent: Dict[str, Any] = None,
        conversation_history: List[Dict] = None
    ) -> RewriteResult:
        """
        改写用户查询为检索优化查询
        
        Args:
            user_input: 用户原始输入
            intent: 意图分析结果
            conversation_history: 对话历史
        
        Returns:
            RewriteResult: 改写结果
        """
        original = user_input.strip()
        reasons = []
        added_keywords = []
        search_terms = []
        
        # 1. 检测查询类型并提取核心关键词
        query_type = self._detect_query_type(original)
        
        if query_type:
            pattern_config = self.KEYWORD_PATTERNS[query_type]
            search_terms = pattern_config.get("search_terms", pattern_config["keywords"])
            reasons.append(f"检测到{query_type}类查询")
            added_keywords.extend(pattern_config["keywords"])
        else:
            # 没有匹配到类型，使用原查询
            search_terms = [original]
        
        # 2. 提取对话上下文关键词
        if conversation_history:
            context_kws = self._extract_context_keywords(conversation_history)
            if context_kws:
                search_terms.extend(context_kws)
                reasons.append(f"添加上下文关键词: {context_kws}")
                added_keywords.extend(context_kws)
        
        # 3. 构建检索查询（使用核心关键词，而不是原查询）
        rewritten = " ".join(search_terms)
        
        # 4. 清理和标准化
        rewritten = self._normalize(rewritten)
        
        return RewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            rewrite_reason="; ".join(reasons) if reasons else "无需改写",
            added_keywords=list(set(added_keywords))  # 去重
        )
    
    def _detect_query_type(self, query: str) -> str:
        """检测查询类型"""
        query_lower = query.lower()
        
        for query_type, config in self.KEYWORD_PATTERNS.items():
            for pattern in config["patterns"]:
                if pattern.lower() in query_lower:
                    return query_type
        
        return None
    
    def _extract_context_keywords(self, history: List[Dict], max_keywords: int = 3) -> List[str]:
        """从对话历史提取关键词"""
        if not history:
            return []
        
        # 合并最近对话
        recent_text = " ".join([
            turn.get('user', '') + " " + turn.get('ai', '')
            for turn in history[-3:]
        ])
        
        # 提取重要关键词
        important_keywords = {
            'python': 'Python',
            '项目': '项目',
            '技术': '技术',
            '目标': '目标',
            '名字': '名字',
            'alice': 'Alice',
            '数据库': '数据库',
        }
        
        found = []
        recent_lower = recent_text.lower()
        for kw, canonical in important_keywords.items():
            if kw in recent_lower and canonical not in found:
                found.append(canonical)
        
        return found[:max_keywords]
    
    def _optimize_by_intent(self, query: str, intent: Dict) -> str:
        """基于意图优化查询"""
        intent_type = intent.get('type', 'general')
        
        if intent_type == 'fact':
            # 事实查询：添加"什么是"
            if not query.startswith(('什么是', '什么是', 'what is')):
                return f"什么是 {query}"
        
        elif intent_type == 'identity':
            # 身份查询：强调用户特定信息
            return f"用户身份信息 {query}"
        
        return query
    
    def _normalize(self, query: str) -> str:
        """标准化查询"""
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query)
        # 去除首尾空格
        query = query.strip()
        return query
    
    def batch_rewrite(
        self,
        queries: List[str],
        conversation_history: List[Dict] = None
    ) -> List[RewriteResult]:
        """批量改写"""
        return [
            self.rewrite(q, conversation_history=conversation_history)
            for q in queries
        ]


# 便捷函数
def create_query_rewriter() -> QueryRewriter:
    """创建查询改写器实例"""
    return QueryRewriter()


def rewrite_for_retrieval(
    user_input: str,
    intent: Dict = None,
    history: List[Dict] = None
) -> str:
    """便捷函数：改写查询"""
    rewriter = create_query_rewriter()
    result = rewriter.rewrite(user_input, intent, history)
    return result.rewritten_query


# 测试
if __name__ == "__main__":
    rewriter = QueryRewriter()
    
    test_cases = [
        {
            "query": "我叫什么名字？",
            "intent": {"type": "identity"},
            "history": []
        },
        {
            "query": "项目的目标是什么？",
            "intent": {"type": "fact"},
            "history": [{"user": "我们在做一个AI项目", "ai": "好的"}]
        },
        {
            "query": "技术栈是什么？",
            "intent": {"type": "fact"},
            "history": []
        },
        {
            "query": "Python是什么？",
            "intent": {"type": "fact"},
            "history": []
        },
        {
            "query": "我们之前讨论过什么？",
            "intent": {"type": "history"},
            "history": [
                {"user": "你好", "ai": "你好"},
                {"user": "Python是什么？", "ai": "Python是编程语言"}
            ]
        }
    ]
    
    print("="*70)
    print("Query Rewriter Test")
    print("="*70)
    
    for test in test_cases:
        result = rewriter.rewrite(
            test["query"],
            test.get("intent"),
            test.get("history")
        )
        
        print(f"\n原始: {result.original_query}")
        print(f"改写: {result.rewritten_query}")
        print(f"原因: {result.rewrite_reason}")
        print(f"添加: {result.added_keywords}")
        print("-"*50)
