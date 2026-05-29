# Real LLM Integration Architecture v1
# 真实模型接入架构 v1

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 15 - Real LLM Integration into Conversation Orchestrator

---

## 1. 核心目标

把当前模拟响应替换为真实 LLM 推理调用，让整条对话链第一次真正闭环。

### 1.1 为什么现在必须做

| 当前状态 | 问题 |
|----------|------|
| 模拟响应 | Phase 14 训练跑不出真实效果 |
| 模拟响应 | 产品行为评估没有真实对象 |
| 模拟响应 | 用户对话测试只能测"编排逻辑" |

**关键依赖关系**:
```
真实模型接入
    │
    ├──▶ Phase 14 训练执行
    ├──▶ 真实用户验证
    ├──▶ 性能压测
    └──▶ 产品行为验证
```

---

## 2. 接入架构设计

### 2.1 整体架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              Conversation Orchestrator                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Intent     │──▶│  Retrieval   │──▶│  Context     │      │
│  │   Analysis   │  │  Decision    │  │  Assembly    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Governance  │──▶│   [LLM       │──▶│   Memory     │      │
│  │   Review     │  │   Client]    │  │  Update      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            │                               │
│                            ▼                               │
│                      真实模型推理                           │
│                      (OpenAI/Claude/Local)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                      输出给用户
```

### 2.2 关键组件

| 组件 | 职责 | 文件 |
|------|------|------|
| LLM Client | 统一封装各种LLM API | `llm_client_v1.py` |
| Prompt Builder | 构建符合协议的prompt | `prompt_builder_v1.py` |
| Response Parser | 解析模型输出 | `response_parser_v1.py` |
| Training Adapter | 与训练头兼容 | `training_adapter_v1.py` |

---

## 3. 模型输入协议

### 3.1 System Prompt 模板

```python
SYSTEM_PROMPT_TEMPLATE = """你是一个具有治理意识的AI助手。你的回答必须遵循以下协议：

## 响应策略
根据置信度和风险选择策略：
- DIRECT: 置信度>0.9，直接回答
- RETRIEVAL_FIRST: 需要验证，先检索再回答
- CONSERVATIVE: 置信度0.6-0.9，说明不确定性
- DECLINE: 置信度<0.6，诚实拒绝
- REVIEW: 高风险内容，标记审查

## 当前上下文
检索结果：
{retrieval_context}

治理建议：
- 策略：{governance_strategy}
- 置信度：{confidence_score}
- 风险等级：{risk_level}

## 回答要求
1. 如果使用了检索结果，必须标注来源
2. 如果不确定，必须说明置信度
3. 如果高风险，必须保守或拒绝
4. 保持人格一致性

## 输出格式
请以JSON格式输出：
{{
    "strategy": "DIRECT|RETRIEVAL_FIRST|CONSERVATIVE|DECLINE|REVIEW",
    "confidence": 0.0-1.0,
    "response": "最终回答",
    "reasoning": "推理过程",
    "citations": ["来源1", "来源2"]
}}
"""
```

### 3.2 Context Assembly 格式

```python
@dataclass
class LLMInput:
    """LLM输入结构"""
    system_prompt: str
    conversation_history: List[Dict[str, str]]  # [{"role": "user|assistant", "content": "..."}]
    current_query: str
    retrieval_context: Optional[List[RetrievalResult]]
    governance_context: GovernanceContext
    memory_context: Optional[List[MemoryEntry]]
    
    def to_messages(self) -> List[Dict[str, str]]:
        """转换为LLM消息格式"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加历史
        for turn in self.conversation_history[-5:]:  # 最近5轮
            messages.append(turn)
        
        # 构建当前输入
        user_input = self._build_user_input()
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _build_user_input(self) -> str:
        """构建用户输入（包含上下文）"""
        parts = [f"用户问题：{self.current_query}"]
        
        if self.retrieval_context:
            parts.append("\n检索到的相关信息：")
            for i, ctx in enumerate(self.retrieval_context[:3], 1):
                parts.append(f"[{i}] {ctx.content}")
        
        if self.memory_context:
            parts.append("\n相关记忆：")
            for mem in self.memory_context[:2]:
                parts.append(f"- {mem.content}")
        
        return "\n".join(parts)
```

---

## 4. 模型输出协议

### 4.1 输出Schema

```python
@dataclass
class LLMOutput:
    """LLM输出结构"""
    strategy: ResponseStrategy
    confidence: float
    response: str
    reasoning: str
    citations: List[str]
    raw_output: str  # 原始输出，用于调试
    latency_ms: float
    token_usage: Dict[str, int]
    
    @classmethod
    def from_json(cls, json_str: str, **metadata) -> 'LLMOutput':
        """从JSON解析"""
        try:
            data = json.loads(json_str)
            return cls(
                strategy=ResponseStrategy(data['strategy']),
                confidence=float(data['confidence']),
                response=data['response'],
                reasoning=data.get('reasoning', ''),
                citations=data.get('citations', []),
                raw_output=json_str,
                **metadata
            )
        except Exception as e:
            # 解析失败，降级处理
            return cls(
                strategy=ResponseStrategy.CONSERVATIVE,
                confidence=0.5,
                response="抱歉，处理响应时出现问题。",
                reasoning=f"解析错误: {e}",
                citations=[],
                raw_output=json_str,
                **metadata
            )
```

### 4.2 治理结果注入

```python
class GovernanceInjector:
    """治理结果注入器"""
    
    def inject_to_prompt(
        self,
        base_prompt: str,
        governance_result: GovernanceResult
    ) -> str:
        """将治理结果注入prompt"""
        
        governance_section = f"""
## 治理审查结果
- 建议策略：{governance_result.strategy.value}
- 置信度评估：{governance_result.confidence:.2f}
- 风险等级：{governance_result.risk_level}
- 需要审查：{'是' if governance_result.requires_review else '否'}

## 强制约束
{self._get_constraints(governance_result)}
"""
        return base_prompt + governance_section
    
    def _get_constraints(self, result: GovernanceResult) -> str:
        """获取强制约束"""
        if result.risk_level == "high":
            return "- 必须采取保守策略\n- 必须说明不确定性\n- 不得给出确定性结论"
        elif result.confidence < 0.6:
            return "- 置信度不足，建议拒绝回答或请求更多信息"
        return "- 按建议策略执行"
```

---

## 5. LLM客户端封装

### 5.1 统一接口

```python
class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    async def generate(
        self,
        input: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        """生成响应"""
        pass
    
    @abstractmethod
    async def generate_with_training_signal(
        self,
        input: LLMInput,
        training_mode: bool = False
    ) -> Tuple[LLMOutput, TrainingSignal]:
        """生成响应并返回训练信号"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        input: LLMInput,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMOutput:
        start_time = time.time()
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=input.to_messages(),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}  # 强制JSON输出
        )
        
        latency = (time.time() - start_time) * 1000
        
        return LLMOutput.from_json(
            response.choices[0].message.content,
            latency_ms=latency,
            token_usage={
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens
            }
        )


class ClaudeClient(BaseLLMClient):
    """Claude客户端"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(self, input: LLMInput, **kwargs) -> LLMOutput:
        # Claude实现...
        pass


class LocalModelClient(BaseLLMClient):
    """本地模型客户端"""
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    async def generate(self, input: LLMInput, **kwargs) -> LLMOutput:
        # 本地模型推理...
        pass
```

### 5.2 客户端工厂

```python
class LLMClientFactory:
    """LLM客户端工厂"""
    
    @staticmethod
    def create_client(
        provider: str,
        **config
    ) -> BaseLLMClient:
        """创建客户端"""
        
        if provider == "openai":
            return OpenAIClient(
                api_key=config['api_key'],
                model=config.get('model', 'gpt-4')
            )
        elif provider == "claude":
            return ClaudeClient(
                api_key=config['api_key'],
                model=config.get('model', 'claude-3-opus')
            )
        elif provider == "local":
            return LocalModelClient(
                model_path=config['model_path']
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
```

---

## 6. 与Orchestrator集成

### 6.1 修改后的主控层

```python
class ConversationOrchestrator:
    """集成真实LLM的对话主控层"""
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        governance_service: GovernanceService,
        memory_service: MemoryService,
        llm_client: BaseLLMClient,  # 新增：LLM客户端
        prompt_builder: PromptBuilder  # 新增：Prompt构建器
    ):
        self.retrieval = retrieval_service
        self.governance = governance_service
        self.memory = memory_service
        self.llm = llm_client  # 真实LLM
        self.prompt_builder = prompt_builder
        
        self.sessions: Dict[str, ConversationContext] = {}
    
    async def process_turn(
        self,
        user_input: str,
        session_id: str,
        user_id: str = "anonymous"
    ) -> OrchestratorResponse:
        """处理单轮对话（真实LLM版本）"""
        
        # 1-4步：与之前相同（意图分析、检索决策、上下文组装、治理审查）
        context = self._get_or_create_context(session_id, user_id)
        intent = self._analyze_intent(user_input, context)
        retrieval_result = await self._decide_retrieval(user_input, intent, context)
        assembled_context = self._assemble_context(user_input, retrieval_result, context)
        governance_result = await self._governance_review(assembled_context)
        
        # 5. 构建LLM输入
        llm_input = self.prompt_builder.build(
            query=user_input,
            history=context.history,
            retrieval_results=retrieval_result,
            governance_result=governance_result,
            memory_context=context.retrieved_memories
        )
        
        # 6. 调用真实LLM生成响应
        start_time = time.time()
        llm_output = await self.llm.generate(llm_input)
        generation_latency = time.time() - start_time
        
        # 7. 验证和校准
        validated_output = self._validate_output(llm_output, governance_result)
        
        # 8. 记忆更新
        await self._update_memory(user_input, validated_output, context)
        
        # 9. 构建最终响应
        response = OrchestratorResponse(
            response_text=validated_output.response,
            strategy=validated_output.strategy,
            confidence=validated_output.confidence,
            sources=validated_output.citations,
            reasoning=validated_output.reasoning,
            metadata={
                "retrieval_count": len(retrieval_result.results) if retrieval_result else 0,
                "generation_latency_ms": generation_latency,
                "token_usage": llm_output.token_usage
            },
            timestamp=datetime.now()
        )
        
        # 更新历史
        context.history.append({
            "turn": context.turn_number,
            "user": user_input,
            "ai": response.response_text,
            "strategy": response.strategy.value,
            "confidence": response.confidence
        })
        
        return response
    
    def _validate_output(
        self,
        llm_output: LLMOutput,
        governance_result: GovernanceResult
    ) -> LLMOutput:
        """验证LLM输出是否符合治理要求"""
        
        # 检查策略一致性
        if governance_result.risk_level == "high" and llm_output.confidence > 0.8:
            # 高风险但LLM过于自信，强制降级
            return LLMOutput(
                strategy=ResponseStrategy.CONSERVATIVE,
                confidence=0.6,
                response=f"{llm_output.response}\n\n[注：此回答已根据风险评估调整为保守表述]",
                reasoning=f"{llm_output.reasoning} [策略校准：高风险强制保守]",
                citations=llm_output.citations,
                raw_output=llm_output.raw_output,
                latency_ms=llm_output.latency_ms,
                token_usage=llm_output.token_usage
            )
        
        return llm_output
```

---

## 7. 与训练头兼容

### 7.1 训练信号提取

```python
@dataclass
class TrainingSignal:
    """训练信号"""
    # 策略监督信号
    predicted_strategy: ResponseStrategy
    target_strategy: ResponseStrategy
    strategy_confidence: float
    
    # 检索监督信号
    retrieval_triggered: bool
    retrieval_necessity: float
    
    # 置信度监督信号
    predicted_confidence: float
    actual_accuracy: Optional[float]  # 需要后续验证
    
    # 治理监督信号
    governance_actions: List[str]
    
    # 记忆监督信号
    memory_writeback: bool
    target_layer: str


class TrainingAdapter:
    """训练适配器"""
    
    def __init__(
        self,
        policy_head: PolicyHead,
        retrieval_head: RetrievalGovernanceHead,
        memory_head: MemoryWritebackHead
    ):
        self.policy_head = policy_head
        self.retrieval_head = retrieval_head
        self.memory_head = memory_head
    
    def extract_training_signal(
        self,
        llm_input: LLMInput,
        llm_output: LLMOutput,
        ground_truth: Optional[Dict] = None
    ) -> TrainingSignal:
        """提取训练信号"""
        
        # 提取特征
        features = self._extract_features(llm_input)
        
        # 获取训练头预测
        with torch.no_grad():
            policy_pred = self.policy_head.predict_strategy(
                torch.tensor([features], dtype=torch.float32)
            )
            retrieval_pred = self.retrieval_head(
                torch.tensor([features], dtype=torch.float32)
            )
        
        return TrainingSignal(
            predicted_strategy=policy_pred[0].item(),
            target_strategy=ground_truth['strategy'] if ground_truth else llm_output.strategy,
            strategy_confidence=policy_pred[1].item(),
            retrieval_triggered=llm_output.strategy == ResponseStrategy.RETRIEVAL_FIRST,
            retrieval_necessity=retrieval_pred[1].item(),
            predicted_confidence=llm_output.confidence,
            actual_accuracy=None,  # 待后续验证
            governance_actions=[],  # 从治理服务获取
            memory_writeback=False,  # 从记忆服务获取
            target_layer="ephemeral"
        )
    
    def compute_training_loss(
        self,
        signal: TrainingSignal,
        llm_output: LLMOutput
    ) -> Dict[str, float]:
        """计算训练损失"""
        
        # 策略损失
        policy_loss = F.cross_entropy(
            torch.tensor([signal.predicted_strategy]),
            torch.tensor([signal.target_strategy.value])
        )
        
        # 置信度校准损失
        confidence_loss = F.mse_loss(
            torch.tensor([signal.predicted_confidence]),
            torch.tensor([0.9 if signal.actual_accuracy else 0.5])  # 占位
        )
        
        return {
            'policy_loss': policy_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': policy_loss.item() + confidence_loss.item()
        }
```

---

## 8. 最小闭环测试

### 8.1 测试用例

```python
async def test_minimal_loop():
    """最小闭环测试"""
    
    print("="*70)
    print("Real LLM Integration - Minimal Loop Test")
    print("="*70)
    
    # 初始化服务
    retrieval = RetrievalService()
    governance = GovernanceService()
    memory = MemoryService()
    
    await retrieval.start()
    await governance.start()
    await memory.start()
    
    # 初始化LLM客户端
    llm_client = LLMClientFactory.create_client(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    # 初始化主控器
    orchestrator = ConversationOrchestrator(
        retrieval_service=retrieval,
        governance_service=governance,
        memory_service=memory,
        llm_client=llm_client,
        prompt_builder=PromptBuilder()
    )
    
    # 测试对话
    test_queries = [
        "Python是什么编程语言？",
        "我们之前讨论过什么？",
        "最新的数据隐私法规是什么？",
        "xyzabc是什么技术？",
    ]
    
    session_id = "test_session_001"
    
    for query in test_queries:
        print(f"\n用户: {query}")
        
        try:
            response = await orchestrator.process_turn(query, session_id)
            
            print(f"AI: {response.response_text}")
            print(f"   [策略: {response.strategy.value}]")
            print(f"   [置信度: {response.confidence:.2f}]")
            print(f"   [延迟: {response.metadata.get('generation_latency_ms', 0):.0f}ms]")
            
        except Exception as e:
            print(f"   [错误: {e}]")
    
    # 清理
    await retrieval.stop()
    await governance.stop()
    await memory.stop()
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
```

---

## 9. 实施步骤

### 9.1 第一步：环境准备

```bash
# 1. 安装依赖
pip install openai anthropic transformers

# 2. 配置API密钥
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# 3. 验证连接
python -c "from phase15.llm_client_v1 import test_connection; test_connection()"
```

### 9.2 第二步：代码实现

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `llm_client_v1.py` | LLM客户端封装 | P0 |
| `prompt_builder_v1.py` | Prompt构建器 | P0 |
| `response_parser_v1.py` | 响应解析器 | P0 |
| `orchestrator_llm_v1.py` | 集成LLM的主控器 | P0 |
| `training_adapter_v1.py` | 训练适配器 | P1 |

### 9.3 第三步：验证测试

```bash
# 运行最小闭环测试
python phase15/test_minimal_loop.py

# 运行产品行为评估
python phase14/eval/eval_product_behavior.py --use-real-llm
```

---

## 10. 关键决策

### 10.1 模型选择建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速验证 | GPT-4 | 能力强，JSON模式稳定 |
| 成本控制 | GPT-3.5 | 便宜，适合初期测试 |
| 数据隐私 | 本地模型 | 数据不出境 |
| 长期部署 | 自研模型 | 完全可控 |

### 10.2 风险缓解

| 风险 | 缓解措施 |
|------|----------|
| API不稳定 | 实现重试机制和降级策略 |
| 响应延迟高 | 添加缓存和流式响应 |
| 成本过高 | 实现请求批处理和配额管理 |
| JSON解析失败 | 添加容错和降级到文本模式 |

---

## 11. 成功标准

接入完成后必须验证：

- [ ] 10轮真实对话成功执行
- [ ] 响应延迟 < 2s
- [ ] 策略选择符合治理建议
- [ ] 能正确引用检索结果
- [ ] 不确定时主动说明
- [ ] 训练信号可以正确提取

---

**下一步行动**：开始实现 `llm_client_v1.py`
