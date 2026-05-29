# Real Model Tuning Checklist v1
# 真实模型联调修复清单 v1

**阶段**: Real Model Integration Validation Round 1  
**日期**: 2026-04-18  
**状态**: 第一轮验证完成，进入联调修复阶段

---

## 第一轮验证结论

### ✅ 成功验证

| 验证项 | 状态 | 说明 |
|--------|------|------|
| DeepSeek API 接入 | ✅ | API 连接正常，认证通过 |
| 结构化 JSON 输出 | ✅ | 系统能正确解析模型响应 |
| 治理层作用 | ✅ | 治理校准机制真实生效 |
| 8轮真实对话 | ✅ | 脱离纯模拟模式 |

### ⚠️ 待修复问题（按优先级排序）

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P0 | 检索上下文未生效 | LLM 无法感知记忆/检索结果 |
| P1 | 策略单一（全CONSERVATIVE） | 治理分流不够细 |
| P2 | 延迟偏高（平均7s） | 用户体验待优化 |

---

## P0: 修复检索上下文注入

### 问题现象
- LLM 回复："抱歉，我无法访问之前的对话历史或记忆"
- 检索服务已执行，但模型感知不到上下文

### 根因分析
- [ ] Prompt 组装时未正确注入 conversation history
- [ ] Retrieval results 格式混乱，模型无法识别
- [ ] System instruction 限制模型引用历史
- [ ] 上下文块之间没有清晰分隔

### 修复检查项

#### Step 1: 打印真实 Prompt
```python
# 在 orchestrator_llm_v1.py 中添加调试输出
print("="*70)
print("DEBUG: 完整 Prompt 发送给 LLM")
print("="*70)
for msg in llm_input.to_messages():
    print(f"[{msg['role']}]: {msg['content'][:500]}...")
```

#### Step 2: 标准化 Prompt 结构
必须包含以下5个块：

```
[SYSTEM]
- 角色定义
- 回答协议
- 可用上下文说明（明确告知可以使用检索结果和记忆）

[CONVERSATION HISTORY]
- 最近3-5轮对话
- 格式：User: xxx / Assistant: xxx

[RETRIEVED KNOWLEDGE]
- 检索到的内容（结构化列表）
- 标注来源和置信度

[GOVERNANCE DECISION]
- 推荐策略
- 风险等级
- 特殊要求

[CURRENT QUERY]
- 当前用户问题
```

#### Step 3: 修复 PromptBuilder
```python
# 确保 retrieval_context 正确格式化
def _format_retrieval(self, results: List[RetrievalResult]) -> str:
    if not results:
        return "无检索结果"
    
    parts = ["检索到的相关信息："]
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] 来源：{r.source} | 置信度：{r.confidence:.2f}")
        parts.append(f"    内容：{r.content}")
        parts.append("")
    return "\n".join(parts)
```

#### Step 4: 修复记忆上下文
```python
# 确保 memory_context 正确注入
def _format_memory(self, memories: List[MemoryEntry]) -> str:
    if not memories:
        return "无相关记忆"
    
    parts = ["相关记忆："]
    for m in memories:
        parts.append(f"- [{m.layer}] {m.content}")
    return "\n".join(parts)
```

### 复测标准
- [ ] LLM 不再说"无法访问历史"
- [ ] LLM 能引用检索结果中的具体内容
- [ ] LLM 能提及记忆中的信息
- [ ] 打印的完整 Prompt 结构清晰

---

## P1: 调整治理阈值与策略分流

### 问题现象
- 全部8轮测试都使用 CONSERVATIVE 策略
- 策略缺乏多样性

### 根因分析
- [ ] confidence calibration 偏低
- [ ] DIRECT 阈值过高（>0.9）
- [ ] RETRIEVAL_FIRST 触发逻辑被覆盖
- [ ] 0.6-0.7 区间全部落入 CONSERVATIVE

### 修复检查项

#### Step 1: 重新校准策略区间

```python
# 建议的新阈值（工程校准版）
STRATEGY_THRESHOLDS = {
    "DIRECT": {
        "min_confidence": 0.85,  # 从 0.9 降低
        "max_risk": "medium",
        "condition": "高置信度 + 中低风险"
    },
    "RETRIEVAL_FIRST": {
        "min_confidence": 0.70,  # 新增区间
        "max_risk": "medium",
        "condition": "证据缺口明显但可补"
    },
    "CONSERVATIVE": {
        "min_confidence": 0.50,  # 收窄区间
        "max_risk": "medium",
        "condition": "有一定把握但证据不充分"
    },
    "DECLINE": {
        "max_confidence": 0.50,  # 明确上限
        "condition": "低置信度且不可验证"
    },
    "REVIEW": {
        "condition": "高风险或冲突显著（无视置信度）"
    }
}
```

#### Step 2: 修改治理决策逻辑

```python
# _governance_review 方法更新
def _governance_review_v2(self, assembled_context):
    user_input = assembled_context["user_input"]
    retrieved = assembled_context["retrieved"]
    
    # 基础置信度计算
    if retrieved:
        avg_conf = sum(r["confidence"] for r in retrieved) / len(retrieved)
        base_confidence = 0.6 + avg_conf * 0.3  # 提高基础分
    else:
        base_confidence = 0.65  # 无检索时默认更高
    
    # 根据查询类型调整
    if self._is_factual_query(user_input):
        base_confidence += 0.05
    
    # 策略选择（新区间）
    if base_confidence >= 0.85:
        strategy = ResponseStrategy.DIRECT
    elif base_confidence >= 0.70 and retrieved:
        strategy = ResponseStrategy.RETRIEVAL_FIRST
    elif base_confidence >= 0.50:
        strategy = ResponseStrategy.CONSERVATIVE
    else:
        strategy = ResponseStrategy.DECLINE
    
    # 高风险强制 REVIEW
    if self._is_high_risk(user_input):
        strategy = ResponseStrategy.REVIEW
    
    return {
        "strategy": strategy,
        "confidence": base_confidence,
        "risk_level": self._assess_risk(user_input),
        "reasoning": f"基于{len(retrieved)}条检索，置信度{base_confidence:.2f}"
    }
```

#### Step 3: 添加策略分布监控

```python
# 在测试报告中增加
strategy_counts = {}
for r in results:
    s = r.get("strategy", "UNKNOWN")
    strategy_counts[s] = strategy_counts.get(s, 0) + 1

print("策略分布:")
for s, count in strategy_counts.items():
    pct = count / len(results) * 100
    print(f"  {s}: {count} ({pct:.1f}%)")

# 目标分布（建议）
TARGET_DISTRIBUTION = {
    "DIRECT": "20-30%",
    "RETRIEVAL_FIRST": "30-40%",
    "CONSERVATIVE": "25-35%",
    "DECLINE": "5-10%",
    "REVIEW": "<5%"
}
```

### 复测标准
- [ ] 至少出现3种不同策略
- [ ] RETRIEVAL_FIRST 在需要检索时被触发
- [ ] CONSERVATIVE 不再是唯一策略
- [ ] 高风险查询正确触发 REVIEW

---

## P2: 优化延迟体验

### 问题现象
- 平均延迟：7155ms
- 最大延迟：13516ms
- 目标：<2000ms

### 根因分析
- [ ] 未使用流式输出
- [ ] 未记录分阶段耗时
- [ ] 首 token 时间未知
- [ ] 用户无进度反馈

### 修复检查项

#### Step 1: 添加分阶段耗时统计

```python
@dataclass
class TimingBreakdown:
    intent_analysis_ms: float
    retrieval_ms: float
    context_assembly_ms: float
    governance_review_ms: float
    llm_prompt_build_ms: float
    llm_api_ms: float
    response_parse_ms: float
    memory_update_ms: float
    total_ms: float
```

#### Step 2: 实现流式响应（可选）

```python
async def generate_stream(
    self,
    input_data: LLMInput,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> AsyncGenerator[str, None]:
    """流式生成"""
    response = await self.client.chat.completions.create(
        model=self.model,
        messages=input_data.to_messages(),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True  # 启用流式
    )
    
    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

#### Step 3: 添加轻量状态反馈

```python
# 在 orchestrator 中
async def process_turn_with_feedback(self, user_input, session_id):
    # 1. 立即返回"正在思考..."
    yield {"type": "status", "message": "正在分析意图..."}
    
    # 2. 意图分析完成
    intent = self._analyze_intent(user_input)
    yield {"type": "status", "message": f"意图：{intent['type']}"}
    
    # 3. 检索完成
    retrieval = await self._decide_retrieval(...)
    if retrieval:
        yield {"type": "status", "message": f"找到 {len(retrieval.results)} 条相关信息"}
    
    # 4. 治理审查完成
    governance = await self._governance_review(...)
    yield {"type": "status", "message": f"策略：{governance['strategy'].value}"}
    
    # 5. 最终响应
    response = await self._generate_with_llm(...)
    yield {"type": "response", "data": response}
```

### 复测标准
- [ ] 首 token 时间 < 1000ms
- [ ] 平均延迟下降或体感改善
- [ ] 有分阶段耗时报告
- [ ] 用户能感知到处理进度

---

## 第二轮测试计划

### 测试目标
验证修复后的真实模型行为

### 测试用例（与第一轮相同）
1. "你好" - 正常问候
2. "Python是什么编程语言？" - 直接回答
3. "我们之前讨论过什么？" - 检索记忆
4. "项目的目标是什么？" - 检索知识
5. "我叫什么名字？" - 检索用户记忆
6. "xyzabc123是什么技术？" - 未知内容
7. "最新的量子计算突破是什么？" - 保守回答
8. "技术栈是什么？" - 检索技术信息

### 验收标准

| 检查项 | 通过标准 |
|--------|----------|
| 上下文注入 | LLM 不再说"无法访问历史" |
| 策略多样性 | 至少3种不同策略 |
| 检索价值 | 检索后答案更具体 |
| 延迟改善 | 平均延迟下降或体感改善 |
| AI 感知 | 用户感觉"这就是你们的AI" |

---

## 执行顺序

```
Phase 15.1: 上下文注入修复
    ├── 打印真实 Prompt
    ├── 标准化 Prompt 结构
    ├── 修复 PromptBuilder
    └── 复测验证

Phase 15.2: 策略阈值调整
    ├── 重新校准策略区间
    ├── 修改治理决策逻辑
    ├── 添加策略监控
    └── 复测验证

Phase 15.3: 延迟优化
    ├── 添加耗时统计
    ├── 实现流式响应（可选）
    ├── 添加状态反馈
    └── 复测验证

Phase 15.4: 第二轮真实测试
    └── 完整8轮测试 + 验收
```

---

## 当前状态总结

```
✅ 真实模型接入成功
✅ 系统脱离纯模拟模式
🔄 上下文注入修复中
⏳ 策略分流校准待开始
⏳ 延迟优化待开始
⏳ 第二轮测试待执行
```

**下一步**: 开始 Phase 15.1 - 修复上下文注入
