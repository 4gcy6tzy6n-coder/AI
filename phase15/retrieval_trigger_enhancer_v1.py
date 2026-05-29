"""
Retrieval Trigger Enhancer v1 - 检索触发增强器 v1

提高检索触发率，让系统更主动检索记忆
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class TriggerDecision:
    """触发决策结果"""
    should_retrieve: bool
    priority_score: float
    reasons: List[str]
    force_retrieval: bool


class RetrievalTriggerEnhancer:
    """
    检索触发增强器
    
    识别高检索倾向的问题类型，提高检索触发率
    """
    
    # 高检索倾向关键词（按类别分组）
    MEMORY_KEYWORDS = {
        "identity": ["我", "我的", "我叫", "我是谁", "我的名字", "姓名", "称呼"],
        "history": ["之前", "上次", "刚才", "说过", "讨论过", "聊过", "提到"],
        "project": ["项目", "目标", "阶段", "进度", "完成", "计划", "里程碑"],
        "tech": ["技术栈", "架构", "方案", "设计", "实现", "技术方案"],
        "memory": ["记得", "记忆", "存储", "记录", "保存"],
        "context": ["我们", "咱们的", "我们的", "团队"],
    }
    
    # 问题类型优先级和检索倾向
    QUERY_TYPE_CONFIG = {
        "identity": {"priority": 0.9, "force_threshold": 0.7},
        "project_state": {"priority": 0.9, "force_threshold": 0.7},
        "history": {"priority": 0.85, "force_threshold": 0.6},
        "tech_context": {"priority": 0.8, "force_threshold": 0.6},
        "fact": {"priority": 0.5, "force_threshold": 0.4},
        "general": {"priority": 0.3, "force_threshold": 0.3},
    }
    
    # 触发阈值
    TRIGGER_THRESHOLD = 0.5
    FORCE_TRIGGER_THRESHOLD = 0.7
    
    def should_prioritize_retrieval(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> TriggerDecision:
        """
        判断是否应优先检索
        
        Args:
            user_input: 用户原始输入
            intent: 意图分析结果
            
        Returns:
            TriggerDecision: 触发决策
        """
        score = 0.0
        reasons = []
        matched_keywords = []
        
        input_lower = user_input.lower()
        
        # 1. 关键词匹配评分 (最高 0.5)
        keyword_score = 0.0
        for category, keywords in self.MEMORY_KEYWORDS.items():
            for kw in keywords:
                if kw in input_lower:
                    keyword_score += 0.1
                    matched_keywords.append(f"{category}:{kw}")
                    if category not in [r.split(':')[0] for r in reasons]:
                        reasons.append(f"命中{category}类关键词")
        
        keyword_score = min(keyword_score, 0.5)  # 封顶 0.5
        score += keyword_score
        
        # 2. 意图类型评分 (最高 0.3)
        intent_type = intent.get('type', 'general')
        
        # 映射意图到配置
        intent_mapping = {
            'identity': 'identity',
            'project': 'project_state',
            'history': 'history',
            'tech': 'tech_context',
            'fact': 'fact',
            'general': 'general',
        }
        
        mapped_type = intent_mapping.get(intent_type, 'general')
        type_config = self.QUERY_TYPE_CONFIG.get(mapped_type, self.QUERY_TYPE_CONFIG['general'])
        type_score = type_config['priority'] * 0.3
        score += type_score
        
        reasons.append(f"意图类型: {intent_type} (权重: {type_score:.2f})")
        
        # 3. 查询特征评分 (最高 0.2)
        feature_score = 0.0
        
        # 疑问句特征
        if any(q in input_lower for q in ['什么', '多少', '哪里', '谁', '怎么']):
            feature_score += 0.1
            reasons.append("疑问句特征")
        
        # 长度特征（适中长度更可能是具体问题）
        if 5 <= len(user_input) <= 50:
            feature_score += 0.05
            reasons.append("适中长度")
        
        # 中文特征（当前系统主要处理中文）
        if any('\u4e00' <= c <= '\u9fff' for c in user_input):
            feature_score += 0.05
        
        score += feature_score
        
        # 4. 决策
        should_retrieve = score >= self.TRIGGER_THRESHOLD
        force_retrieval = score >= self.FORCE_TRIGGER_THRESHOLD
        
        if force_retrieval:
            reasons.append(f"强制检索触发 (score: {score:.2f} >= {self.FORCE_TRIGGER_THRESHOLD})")
        elif should_retrieve:
            reasons.append(f"建议检索 (score: {score:.2f})")
        else:
            reasons.append(f"无需优先检索 (score: {score:.2f} < {self.TRIGGER_THRESHOLD})")
        
        return TriggerDecision(
            should_retrieve=should_retrieve,
            priority_score=score,
            reasons=reasons,
            force_retrieval=force_retrieval
        )
    
    def analyze_query(self, user_input: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """详细分析查询特征"""
        decision = self.should_prioritize_retrieval(user_input, intent)
        
        # 统计关键词命中
        input_lower = user_input.lower()
        keyword_hits = {}
        for category, keywords in self.MEMORY_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in input_lower]
            if hits:
                keyword_hits[category] = hits
        
        return {
            "query": user_input,
            "intent": intent.get('type', 'unknown'),
            "should_retrieve": decision.should_retrieve,
            "priority_score": decision.priority_score,
            "force_retrieval": decision.force_retrieval,
            "keyword_hits": keyword_hits,
            "reasons": decision.reasons,
        }


# 便捷函数
def create_retrieval_trigger_enhancer() -> RetrievalTriggerEnhancer:
    """创建检索触发增强器实例"""
    return RetrievalTriggerEnhancer()


# 测试
if __name__ == "__main__":
    enhancer = RetrievalTriggerEnhancer()
    
    test_cases = [
        {
            "query": "我叫什么名字？",
            "intent": {"type": "identity", "confidence": 0.9}
        },
        {
            "query": "项目的目标是什么？",
            "intent": {"type": "fact", "confidence": 0.8}
        },
        {
            "query": "技术栈是什么？",
            "intent": {"type": "fact", "confidence": 0.8}
        },
        {
            "query": "我们之前讨论过什么？",
            "intent": {"type": "history", "confidence": 0.7}
        },
        {
            "query": "Python是什么？",
            "intent": {"type": "fact", "confidence": 0.8}
        },
        {
            "query": "你好",
            "intent": {"type": "general", "confidence": 0.5}
        },
    ]
    
    print("="*70)
    print("Retrieval Trigger Enhancer Test")
    print("="*70)
    
    for test in test_cases:
        result = enhancer.analyze_query(test["query"], test["intent"])
        decision = enhancer.should_prioritize_retrieval(test["query"], test["intent"])
        
        print(f"\n查询: {result['query']}")
        print(f"意图: {result['intent']}")
        print(f"评分: {result['priority_score']:.2f}")
        print(f"建议检索: {'是' if result['should_retrieve'] else '否'}")
        print(f"强制检索: {'是' if result['force_retrieval'] else '否'}")
        print(f"关键词命中: {result['keyword_hits']}")
        print(f"原因: {', '.join(result['reasons'])}")
        print("-"*50)
    
    # 统计
    should_count = sum(1 for t in test_cases 
                      if enhancer.should_prioritize_retrieval(t["query"], t["intent"]).should_retrieve)
    print(f"\n统计: {should_count}/{len(test_cases)} 个查询建议检索")
