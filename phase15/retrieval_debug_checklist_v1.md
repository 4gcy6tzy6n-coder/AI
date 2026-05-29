# Retrieval Quality Debug Checklist v1
# 检索匹配问题排查与修复清单 v1

**阶段**: Real Model Integration - Retrieval Quality Optimization  
**日期**: 2026-04-18  
**状态**: 上下文注入已修复，进入检索质量优化阶段

---

## 当前问题诊断

### 现象
- LLM 回复："检索结果中没有相关信息"
- 检索服务执行了，但内容对题度不够
- 检索结果未被 LLM 感知为"有用"

### 五层诊断框架

```
[Layer 1] 原始 Query
    ↓
[Layer 2] Query Rewrite (检索查询)
    ↓
[Layer 3] Top-K 召回结果
    ↓
[Layer 4] Rerank 后结果
    ↓
[Layer 5] 注入 Prompt 的 Context
    ↓
[Layer 6] LLM 引用与回答质量
```

---

## Layer 1: 原始 Query 检查

### 检查项
- [ ] 用户原始问题是否完整捕获
- [ ] 问题意图是否被正确理解
- [ ] 是否丢失关键约束条件

### 调试代码
```python
# 在 orchestrator 中打印原始输入
print(f"[Layer 1] 原始 Query: {user_input}")
print(f"[Layer 1] 意图分析: {intent}")
```

### 示例问题
| 用户输入 | 意图 | 关键约束 |
|----------|------|----------|
| "我叫什么名字？" | 查询身份 | 需要用户特定记忆 |
| "项目的目标是什么？" | 查询项目信息 | 需要项目上下文 |
| "Python是什么？" | 查询知识 | 通用知识 |

---

## Layer 2: Query Rewrite 检查

### 检查项
- [ ] 检索查询是否被正确构造
- [ ] 是否丢失了用户问题的核心语义
- [ ] 是否添加了不必要的泛化

### 当前问题分析
```python
# 当前检索调用
request = RetrievalRequest(
    query=user_input,  # 直接使用原始输入，没有改写
    ...
)
```

**问题**：直接使用原始输入作为检索查询，没有针对记忆库优化。

### 修复方案：Query Rewrite

```python
class QueryRewriter:
    """查询改写器"""
    
    def rewrite_for_retrieval(
        self,
        user_input: str,
        intent: Dict,
        context: ConversationContext
    ) -> str:
        """
        将用户问题改写为更适合检索的查询
        
        策略：
        1. 身份查询 -> 添加用户标识
        2. 项目查询 -> 添加项目关键词
        3. 历史查询 -> 添加时间范围
        """
        query = user_input
        
        # 身份类查询优化
        if intent["type"] == "identity" or "名字" in user_input:
            query = f"用户名字 {user_input}"
        
        # 项目类查询优化
        if "项目" in user_input or "project" in user_input.lower():
            query = f"项目信息 {user_input}"
        
        # 技术栈查询优化
        if "技术栈" in user_input or "tech stack" in user_input.lower():
            query = "技术栈 Python React PostgreSQL"
        
        # 添加对话上下文关键词
        if context.history:
            recent_topics = self._extract_topics(context.history[-3:])
            if recent_topics:
                query = f"{query} {' '.join(recent_topics)}"
        
        return query
    
    def _extract_topics(self, history: List[Dict]) -> List[str]:
        """从历史中提取主题关键词"""
        topics = []
        for turn in history:
            user_msg = turn.get('user', '')
            # 简单提取名词
            if 'Python' in user_msg:
                topics.append('Python')
            if '项目' in user_msg:
                topics.append('项目')
        return list(set(topics))
```

### 调试输出
```python
rewritten_query = query_rewriter.rewrite_for_retrieval(user_input, intent, context)
print(f"[Layer 2] 原始 Query: {user_input}")
print(f"[Layer 2] 改写后 Query: {rewritten_query}")
```

---

## Layer 3: Top-K 召回检查

### 检查项
- [ ] 召回数量是否足够
- [ ] 召回内容的相似度分数
- [ ] 是否存在明显不相关的结果

### 调试代码
```python
# 在 retrieval_service 中打印召回结果
print(f"[Layer 3] 检索 Query: {request.query}")
print(f"[Layer 3] 召回数量: {len(results)}")
for i, r in enumerate(results[:5]):
    print(f"  [{i}] 相似度: {r.score:.3f} | 内容: {r.content[:50]}...")
```

### 当前问题
当前使用简单的关键词匹配，没有语义相似度计算。

### 修复方案：添加相似度阈值

```python
# 在 retrieval_service_v1.py 中
MIN_SIMILARITY_THRESHOLD = 0.3  # 最小相似度阈值

def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
    results = []
    
    # 搜索所有层
    for layer in [MemoryLayer.LONG_TERM, MemoryLayer.WORKING]:
        layer_results = self._search_layer(layer, request.query)
        
        # 过滤低相似度结果
        filtered = [r for r in layer_results if r.score >= MIN_SIMILARITY_THRESHOLD]
        results.extend(filtered)
    
    # 按相似度排序
    results.sort(key=lambda x: x.score, reverse=True)
    
    return RetrievalResponse(
        results=results[:request.max_results],
        total_found=len(results)
    )
```

---

## Layer 4: Rerank 检查

### 检查项
- [ ] 排序是否合理
- [ ] 是否考虑了对话上下文
- [ ] 是否考虑了用户偏好

### 当前问题
当前没有 rerank 步骤，直接按相似度排序。

### 修复方案：添加简单 Rerank

```python
class SimpleReranker:
    """简单重排序器"""
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        context: ConversationContext
    ) -> List[RetrievalResult]:
        """
        重排序检索结果
        
        考虑因素：
        1. 原始相似度分数
        2. 与对话主题的相关性
        3. 时效性
        """
        scored_results = []
        
        for result in results:
            score = result.score  # 基础相似度
            
            # 主题相关性加分
            if self._is_topic_relevant(result, context):
                score += 0.1
            
            # 时效性加分（近期记忆）
            if result.layer == MemoryLayer.EPHEMERAL:
                score += 0.05
            
            scored_results.append((score, result))
        
        # 按新分数排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [r for _, r in scored_results]
    
    def _is_topic_relevant(self, result: RetrievalResult, context: ConversationContext) -> bool:
        """检查是否与当前对话主题相关"""
        if not context.history:
            return False
        
        # 提取最近对话关键词
        recent_text = " ".join([
            turn.get('user', '') + " " + turn.get('ai', '')
            for turn in context.history[-2:]
        ])
        
        # 简单匹配
        result_text = result.content.lower()
        recent_lower = recent_text.lower()
        
        # 检查是否有共同关键词
        common_keywords = ['python', '项目', '技术', '目标']
        for kw in common_keywords:
            if kw in result_text and kw in recent_lower:
                return True
        
        return False
```

---

## Layer 5: Context 注入检查

### 检查项
- [ ] 最终注入的 context 格式是否正确
- [ ] 是否被截断或丢失信息
- [ ] 是否清晰标注来源

### 调试代码（已在 v2 中实现）
```python
if self.debug_mode:
    print(f"[Layer 5] 注入 Context:")
    print(prompt_data["full_prompt"])
```

### 修复方案：优化 Context 格式

```python
# 在 prompt_builder_v2.py 中优化
def _build_context_block(self, retrieval_results, memory_context):
    parts = []
    
    # 检索结果 - 更清晰的格式
    if retrieval_results and hasattr(retrieval_results, 'results'):
        valid_results = [
            r for r in retrieval_results.results 
            if r.metadata.get('confidence', 0) > 0.3  # 只保留高置信度
        ][:3]  # 最多3条
        
        if valid_results:
            parts.append("=" * 50)
            parts.append("【检索到的相关信息】")
            parts.append("以下信息可能有助于回答当前问题：")
            parts.append("")
            
            for i, item in enumerate(valid_results, 1):
                confidence = item.metadata.get('confidence', 0.5)
                parts.append(f"[{i}] 📄 来源：{item.layer}")
                parts.append(f"    ✅ 相关度：{confidence:.0%}")
                parts.append(f"    📝 内容：{item.content}")
                parts.append("")
    
    # 如果没有有效结果，明确说明
    if not parts or len(parts) == 0:
        parts.append("=" * 50)
        parts.append("【检索结果】")
        parts.append("⚠️ 未找到与当前问题高度相关的信息")
        parts.append("")
    
    return "\n".join(parts)
```

---

## Layer 6: LLM 引用检查

### 检查项
- [ ] LLM 是否引用了检索内容
- [ ] 引用是否准确
- [ ] 回答是否基于检索内容改善

### 验收标准
```python
def check_llm_citation(response_text: str, retrieval_results: List) -> Dict:
    """检查 LLM 是否正确引用"""
    
    has_citation = '[来源:' in response_text or '[' in response_text
    
    # 检查是否提到检索内容中的关键词
    mentioned_keywords = []
    for result in retrieval_results:
        keywords = result.content.split()[:5]  # 前5个词
        for kw in keywords:
            if kw in response_text:
                mentioned_keywords.append(kw)
    
    return {
        "has_citation": has_citation,
        "mentioned_keywords": mentioned_keywords,
        "citation_quality": "good" if has_citation and mentioned_keywords else "poor"
    }
```

---

## 综合调试脚本

```python
# retrieval_debugger.py
class RetrievalDebugger:
    """检索调试器"""
    
    async def debug_full_chain(
        self,
        user_input: str,
        session_id: str,
        orchestrator: ConversationOrchestratorWithLLMV2
    ):
        """调试完整检索链"""
        
        print("="*70)
        print("RETRIEVAL DEBUG CHAIN")
        print("="*70)
        
        # Layer 1: 原始 Query
        print(f"\n[Layer 1] 原始 Query: {user_input}")
        
        # Layer 2: Query Rewrite
        context = orchestrator.sessions.get(session_id)
        intent = orchestrator._analyze_intent(user_input, context)
        rewritten = orchestrator._rewrite_query(user_input, intent, context)
        print(f"[Layer 2] 改写 Query: {rewritten}")
        
        # Layer 3: Top-K 召回
        retrieval_result = await orchestrator._decide_retrieval(
            rewritten, intent, context
        )
        if retrieval_result:
            print(f"[Layer 3] 召回 {len(retrieval_result.results)} 条")
            for i, r in enumerate(retrieval_result.results[:3]):
                print(f"  [{i}] score={r.metadata.get('confidence', 0):.3f} | {r.content[:60]}...")
        else:
            print("[Layer 3] 无召回结果")
        
        # Layer 4: Rerank
        if retrieval_result:
            reranked = orchestrator._rerank_results(
                retrieval_result.results, rewritten, context
            )
            print(f"[Layer 4] Rerank 后 Top 3:")
            for i, r in enumerate(reranked[:3]):
                print(f"  [{i}] {r.content[:60]}...")
        
        # Layer 5: Context 注入
        prompt_data = orchestrator.prompt_builder.build(
            query=user_input,
            history=context.history if context else [],
            retrieval_results=retrieval_result,
            governance_result={"strategy": "DIRECT", "confidence": 0.8},
            memory_context=[]
        )
        print(f"[Layer 5] Context 长度: {len(prompt_data['full_prompt'])} 字符")
        
        # Layer 6: LLM 响应
        response = await orchestrator.process_turn_async(user_input, session_id)
        citation_check = check_llm_citation(response.response_text, 
                                           retrieval_result.results if retrieval_result else [])
        print(f"[Layer 6] LLM 引用质量: {citation_check['citation_quality']}")
        print(f"[Layer 6] 是否引用: {citation_check['has_citation']}")
        print(f"[Layer 6] 提到关键词: {citation_check['mentioned_keywords'][:5]}")
```

---

## 修复优先级

### P0: Query Rewrite
- 实现 QueryRewriter 类
- 针对身份/项目/技术栈查询优化

### P1: 相似度阈值
- 添加 MIN_SIMILARITY_THRESHOLD
- 过滤低质量召回

### P2: Rerank
- 实现 SimpleReranker
- 考虑对话主题相关性

### P3: Context 格式优化
- 更清晰的分隔和标注
- 添加置信度可视化

---

## 验收标准

修复完成后，以下测试必须通过：

| 测试用例 | 预期结果 |
|----------|----------|
| "我叫什么名字？" | LLM 回答"Alice"并引用记忆 |
| "项目的目标是什么？" | LLM 回答项目目标并引用 |
| "技术栈是什么？" | LLM 回答技术栈并引用 |
| "Python是什么？" | LLM 回答并引用知识库 |

成功标准：
- [ ] 至少 75% 的查询 LLM 能正确引用检索内容
- [ ] 不再出现"检索结果中没有相关信息"
- [ ] 检索内容被明确标注来源

---

## 下一步行动

1. **实现 QueryRewriter** - 优化检索查询构造
2. **添加相似度阈值** - 过滤低质量结果
3. **实现 SimpleReranker** - 提升结果排序质量
4. **运行调试脚本** - 验证五层链路
5. **第三轮真实测试** - 验证修复效果
