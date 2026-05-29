# Retrieval Optimization v1
# 检索优化方案 v1

**目标**: 把检索从"能用"推进到"可靠好用"  
**阶段**: Real Model Integration - Retrieval Reliability Enhancement  
**日期**: 2026-04-18

---

## 当前状态总结

### ✅ 已验证成功

| 里程碑 | 状态 |
|--------|------|
| 真实模型接入 | ✅ |
| 上下文注入 | ✅ |
| 记忆引用 | ✅ (首次成功) |
| 策略分流 | ✅ (3种策略) |

### 关键突破案例

```
用户: 我叫什么名字？
检索: 找到 3 条相关记忆
响应: 你的名字是Alice。
策略: RETRIEVAL_FIRST ✅

用户: 技术栈是什么？
响应: 你使用的技术栈是Python后端、React前端和PostgreSQL数据库[来源:2]
引用: 成功标注来源 ✅
```

### ⚠️ 当前瓶颈

| 问题 | 当前状态 | 目标 |
|------|----------|------|
| 检索触发率 | 20% (1/5) | >60% |
| 检索匹配 | 关键词匹配 | 混合检索 |
| 部分查询 | 落到 DECLINE | 正确回答 |

---

## 优化方案：三层检索架构

```
[Layer 1] 关键词召回 (Keywords)
    ↓ 快速召回候选
[Layer 2] 语义召回 (Embedding)
    ↓ 补充语义相关
[Layer 3] Rerank (排序)
    ↓ 精选 Top-K
[Output] 注入 Prompt
```

---

## Step 1: 检索触发增强

### 问题
检索触发率偏低 (20%)，部分应该检索的问题没有触发。

### 解决方案：记忆型问题优先检索

```python
class RetrievalTriggerEnhancer:
    """检索触发增强器"""
    
    # 高检索倾向关键词
    MEMORY_KEYWORDS = [
        # 身份/历史
        "我", "我的", "我叫", "我是谁", "我的名字",
        "之前", "上次", "刚才", "说过", "讨论过",
        # 项目状态
        "项目", "目标", "阶段", "进度", "完成",
        "技术栈", "架构", "方案", "设计",
        # 记忆依赖
        "记得", "记忆", "存储", "记录",
        "我们", "咱们的",
    ]
    
    # 问题类型优先级
    QUERY_TYPE_PRIORITY = {
        "identity": 0.9,      # 身份查询 - 高优先级
        "project_state": 0.9,  # 项目状态 - 高优先级
        "history": 0.8,       # 历史查询 - 中高优先级
        "fact": 0.5,          # 事实查询 - 中等优先级
        "general": 0.3,       # 通用查询 - 低优先级
    }
    
    def should_prioritize_retrieval(
        self,
        user_input: str,
        intent: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        判断是否应优先检索
        
        Returns:
            (should_retrieve, priority_score)
        """
        score = 0.0
        reasons = []
        
        # 1. 关键词匹配
        input_lower = user_input.lower()
        for kw in self.MEMORY_KEYWORDS:
            if kw in input_lower:
                score += 0.2
                reasons.append(f"命中关键词: {kw}")
        
        # 2. 意图类型
        intent_type = intent.get('type', 'general')
        type_score = self.QUERY_TYPE_PRIORITY.get(intent_type, 0.3)
        score += type_score
        reasons.append(f"意图类型: {intent_type} ({type_score})")
        
        # 3. 判断结果
        should_retrieve = score >= 0.5
        
        return should_retrieve, score
```

### 集成到 Orchestrator

```python
# 在 _decide_retrieval_v3 中
async def _decide_retrieval_v3(self, user_input, intent, context):
    # 1. 改写查询
    rewrite_result = self.query_rewriter.rewrite(...)
    
    # 2. 【新增】判断是否优先检索
    should_retrieve, priority = self.retrieval_trigger.should_prioritize_retrieval(
        user_input, intent
    )
    
    if should_retrieve:
        print(f"  [Trigger] 高优先级检索 (score: {priority:.2f})")
        # 强制检索，即使置信度不高
        force_retrieval = True
    else:
        force_retrieval = False
    
    # 3. 执行检索
    if intent.get('needs_retrieval') or force_retrieval:
        # ... 检索逻辑
```

---

## Step 2: 混合检索实现

### 当前问题
纯关键词匹配有上限，无法处理语义相似但表述不同的情况。

### 解决方案：关键词 + 语义召回

```python
class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, retrieval_service):
        self.retrieval = retrieval_service
        self.keyword_weight = 0.4
        self.semantic_weight = 0.6
    
    async def retrieve_hybrid(
        self,
        query: str,
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """
        混合检索
        
        1. 关键词召回
        2. 语义召回
        3. 合并去重
        4. 加权排序
        """
        # 1. 关键词召回
        keyword_results = await self._keyword_retrieve(query, max_results * 2)
        
        # 2. 语义召回（简化版 - 使用文本相似度）
        semantic_results = await self._semantic_retrieve(query, max_results * 2)
        
        # 3. 合并去重
        all_results = {}
        
        # 添加关键词结果
        for r in keyword_results:
            r.adjusted_score = r.score * self.keyword_weight
            all_results[r.id] = r
        
        # 添加语义结果，合并分数
        for r in semantic_results:
            if r.id in all_results:
                # 已存在，合并分数
                all_results[r.id].adjusted_score += r.score * self.semantic_weight
            else:
                r.adjusted_score = r.score * self.semantic_weight
                all_results[r.id] = r
        
        # 4. 按调整后的分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.adjusted_score,
            reverse=True
        )
        
        return sorted_results[:max_results]
    
    async def _keyword_retrieve(self, query: str, max_results: int) -> List[RetrievalResult]:
        """关键词检索"""
        # 使用现有的检索服务
        request = RetrievalRequest(
            query=query,
            query_id=f"kw_{time.time()}",
            retrieval_type=RetrievalType.KEYWORD,
            max_results=max_results
        )
        response = await self.retrieval.retrieve(request)
        return response.results if response else []
    
    async def _semantic_retrieve(self, query: str, max_results: int) -> List[RetrievalResult]:
        """语义检索（简化版）"""
        # 方案A: 使用简单的文本相似度
        # 方案B: 如果有embedding模型，使用向量相似度
        
        # 这里先用简化版：扩展查询同义词
        expanded_queries = self._expand_query(query)
        
        all_results = []
        for eq in expanded_queries:
            request = RetrievalRequest(
                query=eq,
                query_id=f"sem_{time.time()}",
                retrieval_type=RetrievalType.HYBRID,
                max_results=max_results
            )
            response = await self.retrieval.retrieve(request)
            if response:
                all_results.extend(response.results)
        
        # 去重
        seen = set()
        unique_results = []
        for r in all_results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)
        
        return unique_results[:max_results]
    
    def _expand_query(self, query: str) -> List[str]:
        """扩展查询（同义词/相关词）"""
        expansions = [query]  # 原始查询
        
        # 简单的同义词映射
        synonyms = {
            "项目": ["project", "计划", "工程"],
            "目标": ["goal", "目的", "aim"],
            "技术栈": ["tech stack", "技术方案", "架构"],
            "名字": ["姓名", "名称", "称呼"],
        }
        
        for word, alts in synonyms.items():
            if word in query:
                for alt in alts:
                    expanded = query.replace(word, alt)
                    if expanded != query:
                        expansions.append(expanded)
        
        return expansions
```

---

## Step 3: 简易 Reranker

### 解决方案：基于相关性的重排序

```python
class SimpleReranker:
    """简易重排序器"""
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        conversation_history: List[Dict] = None
    ) -> List[RetrievalResult]:
        """
        重排序检索结果
        
        考虑因素：
        1. 原始相似度分数
        2. 与查询的关键词匹配度
        3. 与对话主题的相关性
        4. 时效性（近期记忆优先）
        """
        scored_results = []
        
        for result in results:
            score = 0.0
            
            # 1. 基础相似度 (40%)
            base_score = result.score if hasattr(result, 'score') else 0.5
            score += base_score * 0.4
            
            # 2. 关键词匹配度 (30%)
            keyword_score = self._keyword_match_score(result.content, query)
            score += keyword_score * 0.3
            
            # 3. 主题相关性 (20%)
            if conversation_history:
                topic_score = self._topic_relevance_score(result, conversation_history)
                score += topic_score * 0.2
            
            # 4. 时效性 (10%)
            freshness_score = self._freshness_score(result)
            score += freshness_score * 0.1
            
            result.rerank_score = score
            scored_results.append((score, result))
        
        # 按新分数排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [r for _, r in scored_results]
    
    def _keyword_match_score(self, content: str, query: str) -> float:
        """计算关键词匹配度"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(content_words & query_words)
        return matches / len(query_words)
    
    def _topic_relevance_score(
        self,
        result: RetrievalResult,
        history: List[Dict]
    ) -> float:
        """计算与对话主题的相关性"""
        if not history:
            return 0.5
        
        # 提取最近对话关键词
        recent_text = " ".join([
            turn.get('user', '') + " " + turn.get('ai', '')
            for turn in history[-2:]
        ]).lower()
        
        # 检查是否有共同关键词
        result_text = result.content.lower()
        important_keywords = ['python', '项目', '技术', '目标', '名字', 'alice']
        
        matches = sum(1 for kw in important_keywords if kw in result_text and kw in recent_text)
        return min(matches / 3, 1.0)  # 最多3个匹配得满分
    
    def _freshness_score(self, result: RetrievalResult) -> float:
        """计算时效性分数"""
        # 近期记忆优先
        layer_scores = {
            'ephemeral': 1.0,
            'short_term': 0.8,
            'long_term': 0.6,
            'shallow_permanent': 0.4,
            'deep_permanent': 0.3,
        }
        return layer_scores.get(result.layer, 0.5)
```

---

## Step 4: 集成到 Orchestrator v4

```python
class ConversationOrchestratorWithLLMV4(ConversationOrchestratorWithLLMV3):
    """
    集成真实LLM的对话主控层 v4
    
    关键改进：
    - 检索触发增强
    - 混合检索
    - Rerank 优化
    """
    
    def __init__(self, ...):
        super().__init__(...)
        self.retrieval_trigger = RetrievalTriggerEnhancer()
        self.hybrid_retriever = HybridRetriever(retrieval_service)
        self.reranker = SimpleReranker()
    
    async def _decide_retrieval_v4(self, user_input, intent, context):
        """改进的检索决策（v4）"""
        
        # 1. 判断是否优先检索
        should_retrieve, priority = self.retrieval_trigger.should_prioritize_retrieval(
            user_input, intent
        )
        
        if should_retrieve:
            print(f"  [Trigger] 高优先级检索 (score: {priority:.2f})")
        
        # 2. 改写查询
        rewrite_result = self.query_rewriter.rewrite(...)
        
        # 3. 混合检索
        if intent.get('needs_retrieval') or should_retrieve:
            results = await self.hybrid_retriever.retrieve_hybrid(
                rewrite_result.rewritten_query,
                max_results=10
            )
            
            # 4. Rerank
            if results:
                reranked = self.reranker.rerank(
                    results,
                    rewrite_result.rewritten_query,
                    context.history
                )
                
                # 5. 过滤低质量结果
                filtered = [r for r in reranked if r.rerank_score > 0.3]
                
                print(f"  [Hybrid] 召回: {len(results)} → Rerank: {len(reranked)} → 过滤后: {len(filtered)}")
                
                return filtered[:3]
        
        return None
```

---

## 验收标准

修复完成后，以下测试必须通过：

| 测试用例 | 预期结果 |
|----------|----------|
| "我叫什么名字？" | 检索触发，回答 Alice |
| "项目的目标是什么？" | 检索触发，回答项目目标 |
| "技术栈是什么？" | 检索触发，回答技术栈 |
| "我们之前讨论过什么？" | 检索触发，基于历史回答 |
| "Python是什么？" | 适当检索或 DIRECT |

成功标准：
- [ ] 检索触发率 >= 60%
- [ ] 身份/项目类问题 100% 触发检索
- [ ] 至少 50% 的检索结果被 LLM 引用
- [ ] DECLINE 比例 < 20%

---

## 下一步行动

1. **实现 RetrievalTriggerEnhancer** - 提高检索触发率
2. **实现 HybridRetriever** - 关键词 + 语义召回
3. **实现 SimpleReranker** - 结果重排序
4. **创建 Orchestrator v4** - 集成所有优化
5. **第四轮真实测试** - 验证优化效果
