"""
Query Rewriter v2 - 查询改写器 v2

改进：
1. 查询类型感知 - 4类查询分别处理
2. 同义词/模板扩展 - 提升短问题和口语问题召回率
3. 语义友好改写 - 生成更易匹配存储内容的查询
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
    query_type: str  # 新增：查询类型


class QueryRewriterV2:
    """
    查询改写器 v2
    
    策略：
    1. 查询类型识别 - 4类查询分别处理
    2. 同义词扩展 - 提升召回率
    3. 语义友好改写 - 匹配存储内容表述
    """
    
    # 查询类型定义
    QUERY_TYPES = {
        "identity": {
            "patterns": ["名字", "我是谁", "我叫", "姓名", "称呼", "我是谁", "我的身份"],
            "description": "身份类查询",
            # 同义词扩展：用户可能的各种问法 → 存储内容中的表述
            "search_terms": [
                "名字", "用户", "姓名", "身份",  # 核心词
                "用户的名字", "你的名字", "用户姓名",  # 短语形式
            ]
        },
        "project_state": {
            "patterns": ["项目", "project", "目标", "goal", "计划", "进度", "阶段", "完成"],
            "description": "项目状态类查询",
            "search_terms": [
                "项目", "目标", "计划", "进度", "阶段",
                "项目目标", "项目进度", "项目阶段",
            ]
        },
        "tech_stack": {
            "patterns": ["技术栈", "tech stack", "用什么技术", "技术方案", "架构", "后端", "前端"],
            "description": "技术栈类查询",
            "search_terms": [
                "技术栈", "技术方案", "技术架构", "技术组成",
                "Python", "React", "PostgreSQL",  # 具体技术
                "后端", "前端", "数据库",
            ]
        },
        "history_review": {
            "patterns": ["之前", "刚才", "上次", "之前讨论", "说过", "聊过", "讨论过", "我们之前"],
            "description": "历史回顾类查询",
            "search_terms": [
                "对话", "历史", "讨论", "会话", "话题",
                "之前讨论", "历史对话", "会话历史",
            ]
        },
        "general": {
            "patterns": [],  # 兜底类型
            "description": "通用查询",
            "search_terms": []
        }
    }
    
    def __init__(self):
        self.context_keywords = []
    
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
        
        # 1. 识别查询类型
        query_type = self._detect_query_type(original)
        type_config = self.QUERY_TYPES[query_type]
        
        reasons.append(f"类型:{type_config['description']}")
        
        # 2. 根据类型生成检索词
        search_terms = type_config["search_terms"].copy()
        added_keywords.extend(type_config["search_terms"][:3])  # 记录前3个关键词
        
        # 3. 提取对话上下文关键词（针对历史回顾类特别重要）
        if conversation_history:
            context_kws = self._extract_context_keywords(conversation_history, query_type)
            if context_kws:
                search_terms.extend(context_kws)
                reasons.append(f"上下文:{context_kws}")
                added_keywords.extend(context_kws)
        
        # 4. 针对特定类型的额外处理
        if query_type == "identity":
            # 身份类：确保包含"用户"和"名字"
            if "用户" not in search_terms:
                search_terms.append("用户")
            if "名字" not in search_terms:
                search_terms.append("名字")
        
        elif query_type == "history_review":
            # 历史回顾类：提取最近对话的主题词
            if conversation_history:
                topic_words = self._extract_topic_words(conversation_history)
                search_terms.extend(topic_words)
                if topic_words:
                    reasons.append(f"主题:{topic_words}")
        
        # 5. 构建检索查询（去重并限制长度）
        unique_terms = []
        seen = set()
        for term in search_terms:
            if term not in seen and len(term) > 1:
                unique_terms.append(term)
                seen.add(term)
        
        # 限制检索词数量，避免查询过长
        rewritten = " ".join(unique_terms[:6])
        
        return RewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            rewrite_reason="; ".join(reasons),
            added_keywords=list(set(added_keywords)),
            query_type=query_type
        )
    
    def _detect_query_type(self, query: str) -> str:
        """检测查询类型"""
        query_lower = query.lower()
        
        # 按优先级检查（更具体的类型优先）
        priority_order = ["identity", "history_review", "tech_stack", "project_state"]
        
        for query_type in priority_order:
            config = self.QUERY_TYPES[query_type]
            for pattern in config["patterns"]:
                if pattern.lower() in query_lower:
                    return query_type
        
        return "general"
    
    def _extract_context_keywords(
        self,
        history: List[Dict],
        query_type: str,
        max_keywords: int = 2
    ) -> List[str]:
        """从对话历史提取上下文关键词"""
        if not history:
            return []
        
        # 提取最近几轮对话
        recent_turns = history[-3:] if len(history) >= 3 else history
        
        # 合并文本
        all_text = " ".join([
            turn.get('user', '') + " " + turn.get('ai', '')
            for turn in recent_turns
        ])
        
        # 根据查询类型选择不同的关键词提取策略
        if query_type == "identity":
            # 身份类：提取人名、称谓
            keywords = ['Alice', 'Bob', '用户', '名字', '工程师', '经理']
        elif query_type == "project_state":
            # 项目状态类：提取项目相关词
            keywords = ['项目', '目标', '阶段', 'Phase', 'Q3', 'Q4', '开发', '完成']
        elif query_type == "tech_stack":
            # 技术栈类：提取技术词
            keywords = ['Python', 'React', 'PostgreSQL', '后端', '前端', '数据库']
        elif query_type == "history_review":
            # 历史回顾类：提取主题词
            keywords = ['讨论', '对话', '项目', '技术', '目标', '问题']
        else:
            keywords = []
        
        # 统计出现的关键词
        found_keywords = []
        text_lower = all_text.lower()
        for kw in keywords:
            if kw.lower() in text_lower and kw not in found_keywords:
                found_keywords.append(kw)
        
        return found_keywords[:max_keywords]
    
    def _extract_topic_words(self, history: List[Dict], max_words: int = 3) -> List[str]:
        """提取对话主题词"""
        if not history:
            return []
        
        # 提取最近一轮用户输入的关键词
        last_user_input = history[-1].get('user', '') if history else ''
        
        # 简单的主题词提取（基于预定义词表）
        topic_candidates = [
            '项目', '目标', '技术', '技术栈', 'Python', 'React',
            '开发', '设计', '方案', '问题', '讨论'
        ]
        
        found = []
        text_lower = last_user_input.lower()
        for word in topic_candidates:
            if word.lower() in text_lower and word not in found:
                found.append(word)
        
        return found[:max_words]


# 便捷函数
def create_query_rewriter_v2() -> QueryRewriterV2:
    """创建查询改写器 v2 实例"""
    return QueryRewriterV2()


# 测试
if __name__ == "__main__":
    rewriter = QueryRewriterV2()
    
    test_cases = [
        # 身份类
        {"query": "我叫什么名字？", "intent": {"type": "identity"}, "history": []},
        {"query": "我是谁？", "intent": {"type": "identity"}, "history": []},
        
        # 项目状态类
        {"query": "项目的目标是什么？", "intent": {"type": "project"}, "history": []},
        {"query": "我们做到哪个阶段了？", "intent": {"type": "project"}, "history": []},
        
        # 技术栈类
        {"query": "技术栈是什么？", "intent": {"type": "tech"}, "history": []},
        {"query": "用什么技术？", "intent": {"type": "tech"}, "history": []},
        
        # 历史回顾类
        {"query": "我们之前讨论过什么？", "intent": {"type": "history"}, "history": [
            {"user": "项目的目标是什么？", "ai": "项目目标是..."}
        ]},
        {"query": "你还记得上次聊了什么吗？", "intent": {"type": "history"}, "history": [
            {"user": "技术栈是什么？", "ai": "技术栈是..."}
        ]},
        
        # 通用类
        {"query": "Python是什么？", "intent": {"type": "fact"}, "history": []},
        {"query": "你好", "intent": {"type": "general"}, "history": []},
    ]
    
    print("="*70)
    print("Query Rewriter v2 Test")
    print("="*70)
    
    for test in test_cases:
        result = rewriter.rewrite(
            test["query"],
            test.get("intent"),
            test.get("history")
        )
        
        print(f"\n原始: {result.original_query}")
        print(f"类型: {result.query_type}")
        print(f"改写: {result.rewritten_query}")
        print(f"原因: {result.rewrite_reason}")
        print("-"*50)
