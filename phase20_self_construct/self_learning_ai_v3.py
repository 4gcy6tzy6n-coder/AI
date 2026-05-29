"""
Stage 11: 自学习主流程接管 - Phase A 产品能力补强

核心目标: 重构产品响应链路，实现双层生成结构
架构: 用户输入 → 意图识别 → 缺口识别 → 检索 → 候选推理 → TSLA审查 → 自然语言生成

关键改进:
1. 强制统一流程 - 禁止模型输出被忽略
2. 双层生成 - 内部工作层 + 用户表达层
3. 缺口识别 - 模型先判断"我会不会"
4. 检索内化 - 不是外挂工具，而是本能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


class GapType(Enum):
    """缺口类型"""
    NONE = "none"           # 无缺口，直接知道
    PARTIAL = "partial"     # 部分知道，需要确认
    RETRIEVAL_NEEDED = "retrieval"  # 需要检索
    INSUFFICIENT = "insufficient"   # 信息不足，应保守回答
    CONFLICT = "conflict"   # 知识冲突，需要澄清


class TSLAAction(Enum):
    """TSLA 动作"""
    PASS = "pass"           # 通过，正常输出
    ISOLATE = "isolate"     # 隔离，不写入记忆
    REFLOW = "reflow"       # 回流，重新审查
    DOWNGRADE = "downgrade" # 降级，保守输出
    PROMOTE = "promote"     # 晋升，进入长期记忆


@dataclass
class InternalState:
    """内部工作状态"""
    query: str
    intent: str = ""
    gap_type: GapType = GapType.NONE
    gap_confidence: float = 0.0
    retrieval_triggered: bool = False
    retrieval_results: List[Dict] = field(default_factory=list)
    candidate_reasoning: str = ""
    candidate_answer: str = ""
    uncertainty_score: float = 0.0
    tsla_predicted: TSLAAction = TSLAAction.PASS
    tsla_confidence: float = 0.0
    memory_action: str = "no_write"


@dataclass
class UserResponse:
    """用户层响应"""
    text: str
    style: str  # concise/detailed/conservative/guiding
    confidence_indicator: str
    suggested_followup: Optional[str] = None


class GapRecognitionHead:
    """缺口识别头 - 判断模型是否知道答案"""
    
    def __init__(self, model):
        self.model = model
        self.confidence_threshold_high = 0.8
        self.confidence_threshold_low = 0.3
    
    def analyze(self, query: str, context: str = "") -> Tuple[GapType, float]:
        """
        分析查询，判断知识缺口类型
        返回: (缺口类型, 置信度)
        """
        # 编码输入
        input_ids = self._encode(query + " " + context)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            
            # 使用 gap_probs 作为缺口检测信号
            gap_probs = outputs['gap_probs'][0]
            
            # 计算内部置信度
            internal_conf = gap_probs.max().item()
            
            # 判断缺口类型
            if internal_conf > self.confidence_threshold_high:
                return GapType.NONE, internal_conf
            elif internal_conf > 0.5:
                return GapType.PARTIAL, internal_conf
            elif internal_conf > self.confidence_threshold_low:
                return GapType.RETRIEVAL_NEEDED, internal_conf
            else:
                return GapType.INSUFFICIENT, internal_conf
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens])


class RetrievalEngine:
    """检索引擎 - 知识检索"""
    
    def __init__(self, knowledge_base: Dict):
        self.kb = knowledge_base
        self.retrieval_count = 0
        self.hit_count = 0
    
    def search(self, query: str, gap_type: GapType) -> List[Dict]:
        """
        执行检索
        返回相关文档列表
        """
        self.retrieval_count += 1
        
        results = []
        query_lower = query.lower()
        
        # 关键词匹配
        for key, value in self.kb.items():
            score = 0.0
            
            # 精确匹配
            if key.lower() in query_lower:
                score = 0.9
            # 部分匹配
            elif any(word in query_lower for word in key.lower().split()):
                score = 0.6
            
            if score > 0.5:
                results.append({
                    'key': key,
                    'content': value,
                    'score': score,
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            self.hit_count += 1
        
        return results[:3]  # 返回前3个
    
    def get_stats(self) -> Dict:
        """获取检索统计"""
        hit_rate = self.hit_count / self.retrieval_count if self.retrieval_count > 0 else 0
        return {
            'total_queries': self.retrieval_count,
            'hits': self.hit_count,
            'hit_rate': hit_rate,
        }


class CandidateBuilder:
    """候选构建器 - 生成候选推理和答案"""
    
    def __init__(self, model):
        self.model = model
    
    def build(self, query: str, retrieved_docs: List[Dict], gap_type: GapType) -> Tuple[str, str, float]:
        """
        构建候选推理和答案
        返回: (推理链, 候选答案, 不确定性分数)
        """
        # 基于检索结果构建推理
        reasoning_steps = []
        
        if retrieved_docs:
            reasoning_steps.append(f"检索到 {len(retrieved_docs)} 条相关信息")
            for i, doc in enumerate(retrieved_docs[:2], 1):
                reasoning_steps.append(f"[{i}] {doc['key']}: {doc['content'][:50]}...")
        
        # 根据缺口类型调整
        if gap_type == GapType.NONE:
            uncertainty = 0.2
            reasoning_steps.append("基于内部知识直接回答")
        elif gap_type == GapType.PARTIAL:
            uncertainty = 0.4
            reasoning_steps.append("部分信息来自内部知识，部分需确认")
        elif gap_type == GapType.RETRIEVAL_NEEDED:
            uncertainty = 0.5
            reasoning_steps.append("需要检索补充信息")
        else:
            uncertainty = 0.8
            reasoning_steps.append("信息不足，建议保守回答")
        
        # 生成候选答案
        if retrieved_docs:
            candidate = self._generate_from_docs(query, retrieved_docs)
        else:
            candidate = self._generate_conservative(query, gap_type)
        
        reasoning = "\n".join(reasoning_steps)
        
        return reasoning, candidate, uncertainty
    
    def _generate_from_docs(self, query: str, docs: List[Dict]) -> str:
        """基于文档生成答案"""
        if not docs:
            return ""
        
        # 简单整合前两个文档
        main_doc = docs[0]
        return f"根据{main_doc['key']}，{main_doc['content'][:100]}"
    
    def _generate_conservative(self, query: str, gap_type: GapType) -> str:
        """生成保守答案"""
        if gap_type == GapType.INSUFFICIENT:
            return "[保守模式] 我目前的信息不足以准确回答这个问题。"
        return "[待确认] 我需要更多信息才能给出准确回答。"


class TSLAPredictor:
    """TSLA 预判头 - 预测治理动作"""
    
    def __init__(self, model):
        self.model = model
        self.confidence_threshold = 0.8
    
    def predict(self, internal_state: InternalState) -> Tuple[TSLAAction, float]:
        """
        预测 TSLA 应该执行的动作
        返回: (动作, 置信度)
        """
        # 基于不确定性分数判断
        uncertainty = internal_state.uncertainty_score
        gap_type = internal_state.gap_type
        
        # 决策逻辑
        if uncertainty > 0.7:
            return TSLAAction.ISOLATE, 0.9
        elif uncertainty > 0.5:
            return TSLAAction.REFLOW, 0.8
        elif gap_type == GapType.CONFLICT:
            return TSLAAction.DOWNGRADE, 0.85
        elif uncertainty < 0.3 and internal_state.retrieval_triggered:
            return TSLAAction.PROMOTE, 0.75
        else:
            return TSLAAction.PASS, 0.8
    
    def check_consistency(self, predicted: TSLAAction, actual: TSLAAction) -> float:
        """检查预测与实际的一致性"""
        return 1.0 if predicted == actual else 0.0


class ResponseGenerator:
    """响应生成器 - 双层生成结构"""
    
    def __init__(self):
        self.templates = {
            'concise': {
                'high_conf': "{answer}",
                'medium_conf': "{answer} (置信度: {conf:.0%})",
                'low_conf': "关于这个问题，我了解有限。{answer}",
            },
            'detailed': {
                'high_conf': "{answer}\n\n推理过程: {reasoning}",
                'medium_conf': "{answer}\n\n推理过程: {reasoning}\n\n注意: 此回答基于有限信息，建议进一步验证。",
                'low_conf': "我目前无法给出准确回答。\n\n原因: {reasoning}\n\n建议: 提供更多背景信息或尝试其他提问方式。",
            },
            'conservative': {
                'default': "[系统提示] {answer}",
            },
            'guiding': {
                'default': "{answer}\n\n您可以继续问: {followup}",
            },
        }
    
    def generate(
        self,
        internal_state: InternalState,
        style: str = "detailed"
    ) -> UserResponse:
        """
        生成用户层响应
        """
        uncertainty = internal_state.uncertainty_score
        
        # 选择模板集
        templates = self.templates.get(style, self.templates['detailed'])
        
        # 根据置信度选择具体模板
        if uncertainty < 0.3:
            conf_level = 'high_conf'
        elif uncertainty < 0.6:
            conf_level = 'medium_conf'
        else:
            conf_level = 'low_conf'
        
        template = templates.get(conf_level, templates.get('default', '{answer}'))
        
        # 填充模板
        text = template.format(
            answer=internal_state.candidate_answer,
            reasoning=internal_state.candidate_reasoning[:200],
            conf=1 - uncertainty,
            followup="能否提供更多背景信息？"
        )
        
        # 置信度指示器
        if uncertainty < 0.3:
            indicator = "✓ 高置信度"
        elif uncertainty < 0.6:
            indicator = "~ 中等置信度"
        else:
            indicator = "? 低置信度"
        
        return UserResponse(
            text=text,
            style=style,
            confidence_indicator=indicator,
            suggested_followup=None if uncertainty < 0.5 else "需要更多信息"
        )


class SelfLearningAI:
    """
    自学习 AI 系统 - Phase A 完整实现
    
    主链路:
    用户输入 → 意图识别 → 缺口识别 → 检索 → 候选构建 → TSLA预判 → 自然语言生成 → 输出
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # 初始化模型
        config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(config)
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 加载模型: {model_path}")
        
        self.model.eval()
        
        # 初始化各组件
        self.gap_head = GapRecognitionHead(self.model)
        self.retrieval = RetrievalEngine(self._build_knowledge_base())
        self.candidate_builder = CandidateBuilder(self.model)
        self.tsla_predictor = TSLAPredictor(self.model)
        self.response_gen = ResponseGenerator()
        
        # 对话历史
        self.conversation_history: List[Dict] = []
        self.context_window: List[Dict] = []
        
        # 统计
        self.stats = {
            'total_queries': 0,
            'gap_none': 0,
            'gap_partial': 0,
            'gap_retrieval': 0,
            'gap_insufficient': 0,
            'tsla_pass': 0,
            'tsla_isolate': 0,
            'tsla_reflow': 0,
        }
    
    def _build_knowledge_base(self) -> Dict:
        """构建知识库 - 分层结构"""
        kb = {}
        
        # ===== 第一层: 产品通用知识 =====
        general_knowledge = {
            # 问候与基础
            "你好": "你好！我是一个AI助手，可以帮助你解答问题、进行推理和对话。",
            "自我介绍": "我是一个基于自学习架构的AI系统，具备缺口识别、主动检索和知识治理能力。",
            "你能做什么": "我可以回答问题、进行推理、检索知识、识别信息缺口，并在不确定时主动告知。",
            
            # AI基础
            "人工智能": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习": "机器学习是AI的一个子领域，让计算机能够从数据中学习规律，而无需明确编程。",
            "深度学习": "深度学习是机器学习的一种方法，使用多层神经网络来学习数据的层次化表示。",
            "神经网络": "神经网络是受生物神经元启发的计算模型，由相互连接的节点(神经元)组成，能够学习复杂模式。",
            
            # 模型架构
            "Transformer": "Transformer是一种神经网络架构，使用自注意力机制处理序列数据，是GPT和BERT的基础。",
            "GPT": "GPT(Generative Pre-trained Transformer)是一系列生成式预训练语言模型，用于文本生成任务。",
            "BERT": "BERT(Bidirectional Encoder Representations)是Google开发的双向编码器模型，用于理解任务。",
            "注意力机制": "注意力机制让模型能够聚焦于输入序列的不同部分，是Transformer的核心组件。",
            
            # 训练概念
            "过拟合": "过拟合指模型在训练数据上表现很好，但在新数据上表现差，通常因为模型过于复杂。",
            "欠拟合": "欠拟合指模型在训练数据和新数据上都表现差，通常因为模型过于简单。",
            "梯度下降": "梯度下降是一种优化算法，通过沿损失函数梯度的反方向调整参数来最小化损失。",
            "学习率": "学习率控制参数更新的步长，太高会导致不稳定，太低会导致收敛慢。",
            "损失函数": "损失函数衡量模型预测与真实值之间的差距，训练目标是最小化损失。",
            
            # 评估指标
            "准确率": "准确率是正确预测数占总预测数的比例，是最简单的分类指标。",
            "精确率": "精确率衡量被预测为正类的样本中真正为正类的比例。",
            "召回率": "召回率衡量真正为正类的样本中被正确预测为正类的比例。",
            "F1分数": "F1分数是精确率和召回率的调和平均，综合衡量模型性能。",
            
            # 应用场景
            "推荐系统": "推荐系统根据用户历史行为和偏好，预测用户可能感兴趣的物品。",
            "语音识别": "语音识别将人类语音转换为文本，是智能助手和语音输入的基础。",
            "图像识别": "图像识别让计算机能够识别和理解图像内容，应用于人脸识别、医疗影像等。",
            "自动驾驶": "自动驾驶利用AI感知环境、规划路径和控制车辆，目标是实现无人驾驶。",
            "机器翻译": "机器翻译自动将一种语言的文本转换为另一种语言，如Google翻译。",
        }
        kb.update(general_knowledge)
        
        # ===== 第二层: 项目高价值知识 =====
        project_knowledge = {
            # 核心机制
            "Guard机制": "Guard机制通过KL散度约束writeback输出分布，防止知识写入过程中的灾难性漂移。",
            "TSLA": "TSLA(Three-State Learning Architecture)是三态学习架构，管理知识的晋升、隔离和回流。",
            "Output KL Guard": "Output KL Guard直接约束输出分布，确保writeback稳定性，beta=0.2时效果最佳。",
            "跷跷板效应": "跷跷板效应指训练多层任务时，提升一层性能导致其他层性能下降的现象。",
            
            # 训练策略
            "课程学习": "课程学习按难度递增顺序训练模型，从简单任务逐步过渡到复杂任务。",
            "Replay机制": "Replay机制定期回放旧任务样本，防止学习新任务时遗忘旧知识。",
            "固定采样比例": "固定采样比例2:2:1(L1:L2:L3)确保多层任务平衡，避免某一层次被压制。",
            "平衡窗口": "平衡窗口指多层任务性能同时达标的连续步数，是系统稳定性的关键指标。",
            
            # 项目里程碑
            "Stage 9": "Stage 9验证了单模型多层任务共存可行性，R2配置实现109步平衡窗口。",
            "Stage 10": "Stage 10完成长期稳定性验证(649步)和真实数据迁移验证(399步)。",
            "product_baseline_v1": "product_baseline_v1是冻结的产品基线，配置为采样比2:2:1，权重1.2/1.2/1.0。",
            
            # 自学习
            "缺口识别": "缺口识别让模型先判断自己是否知道答案，再决定如何响应。",
            "自构建": "自构建允许模型生成候选知识，但需经过TSLA审查才能晋升。",
            "知识晋升": "知识晋升是从瞬时层→长期受审区→永久记忆的渐进过程。",
            "教师模式": "教师模式使用DeepSeek API提供高质量答案，加速模型能力增强。",
        }
        kb.update(project_knowledge)
        
        return kb
    
    def chat(self, user_input: str) -> Dict:
        """
        完整对话流程 - Phase A 主链路
        """
        self.stats['total_queries'] += 1
        
        # ===== Step 1: 意图识别 (简化版) =====
        intent = self._recognize_intent(user_input)
        
        # ===== Step 2: 缺口识别 =====
        gap_type, gap_conf = self.gap_head.analyze(user_input, self._get_context())
        self._update_gap_stats(gap_type)
        
        # ===== Step 3: 检索决策 =====
        retrieval_triggered = gap_type in [GapType.PARTIAL, GapType.RETRIEVAL_NEEDED]
        retrieved_docs = []
        
        if retrieval_triggered:
            retrieved_docs = self.retrieval.search(user_input, gap_type)
        
        # ===== Step 4: 候选构建 =====
        reasoning, candidate, uncertainty = self.candidate_builder.build(
            user_input, retrieved_docs, gap_type
        )
        
        # ===== Step 5: TSLA 预判 =====
        internal_state = InternalState(
            query=user_input,
            intent=intent,
            gap_type=gap_type,
            gap_confidence=gap_conf,
            retrieval_triggered=retrieval_triggered,
            retrieval_results=retrieved_docs,
            candidate_reasoning=reasoning,
            candidate_answer=candidate,
            uncertainty_score=uncertainty,
        )
        
        tsla_action, tsla_conf = self.tsla_predictor.predict(internal_state)
        internal_state.tsla_predicted = tsla_action
        internal_state.tsla_confidence = tsla_conf
        
        self._update_tsla_stats(tsla_action)
        
        # ===== Step 6: 响应生成 =====
        # 根据 TSLA 动作调整输出
        if tsla_action == TSLAAction.ISOLATE:
            final_answer = "[系统提示] 我无法确定地回答这个问题。这可能是因为信息不足或问题涉及敏感内容。"
            internal_state.candidate_answer = final_answer
            internal_state.uncertainty_score = 0.9
        elif tsla_action == TSLAAction.DOWNGRADE:
            internal_state.candidate_answer = "[保守回答] " + candidate
            internal_state.uncertainty_score = min(uncertainty + 0.2, 1.0)
        
        # 生成用户响应
        user_response = self.response_gen.generate(internal_state, style="detailed")
        
        # ===== Step 7: 记忆治理决策 =====
        memory_action = self._decide_memory_action(internal_state)
        internal_state.memory_action = memory_action
        
        # ===== 记录对话 =====
        record = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'intent': intent,
            'gap_type': gap_type.value,
            'gap_confidence': gap_conf,
            'retrieval_triggered': retrieval_triggered,
            'retrieval_count': len(retrieved_docs),
            'candidate_reasoning': reasoning,
            'candidate_answer': candidate,
            'uncertainty_score': uncertainty,
            'tsla_action': tsla_action.value,
            'tsla_confidence': tsla_conf,
            'memory_action': memory_action,
            'final_response': user_response.text,
            'confidence_indicator': user_response.confidence_indicator,
        }
        
        self.conversation_history.append(record)
        self.context_window.append({
            'user': user_input,
            'assistant': user_response.text,
        })
        
        # 限制上下文窗口
        if len(self.context_window) > 10:
            self.context_window = self.context_window[-10:]
        
        return record
    
    def _recognize_intent(self, query: str) -> str:
        """简单意图识别"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['你好', '嗨', 'hello', 'hi']):
            return "greeting"
        elif any(word in query_lower for word in ['什么', '什么是', '什么是']):
            return "definition"
        elif any(word in query_lower for word in ['为什么', '怎么', '如何']):
            return "explanation"
        elif any(word in query_lower for word in ['设计', '构建', '创建']):
            return "design"
        else:
            return "general"
    
    def _get_context(self) -> str:
        """获取上下文"""
        if not self.context_window:
            return ""
        
        # 返回最近3轮对话
        recent = self.context_window[-3:]
        context = " ".join([f"用户:{c['user']} 助手:{c['assistant'][:50]}" for c in recent])
        return context
    
    def _update_gap_stats(self, gap_type: GapType):
        """更新缺口统计"""
        if gap_type == GapType.NONE:
            self.stats['gap_none'] += 1
        elif gap_type == GapType.PARTIAL:
            self.stats['gap_partial'] += 1
        elif gap_type == GapType.RETRIEVAL_NEEDED:
            self.stats['gap_retrieval'] += 1
        elif gap_type == GapType.INSUFFICIENT:
            self.stats['gap_insufficient'] += 1
    
    def _update_tsla_stats(self, action: TSLAAction):
        """更新TSLA统计"""
        if action == TSLAAction.PASS:
            self.stats['tsla_pass'] += 1
        elif action == TSLAAction.ISOLATE:
            self.stats['tsla_isolate'] += 1
        elif action == TSLAAction.REFLOW:
            self.stats['tsla_reflow'] += 1
    
    def _decide_memory_action(self, state: InternalState) -> str:
        """决定记忆动作"""
        if state.uncertainty_score > 0.7:
            return "no_write"
        elif state.uncertainty_score > 0.4:
            return "review_zone"
        elif state.retrieval_triggered and state.uncertainty_score < 0.3:
            return "promote_candidate"
        else:
            return "no_write"
    
    def get_stats(self) -> Dict:
        """获取系统统计"""
        retrieval_stats = self.retrieval.get_stats()
        
        return {
            'total_queries': self.stats['total_queries'],
            'gap_distribution': {
                'none': self.stats['gap_none'],
                'partial': self.stats['gap_partial'],
                'retrieval': self.stats['gap_retrieval'],
                'insufficient': self.stats['gap_insufficient'],
            },
            'tsla_distribution': {
                'pass': self.stats['tsla_pass'],
                'isolate': self.stats['tsla_isolate'],
                'reflow': self.stats['tsla_reflow'],
            },
            'retrieval': retrieval_stats,
        }
    
    def interactive_session(self):
        """交互式对话会话"""
        print("\n" + "="*70)
        print("🤖 自学习 AI 系统 - Phase A")
        print("="*70)
        print("特性:")
        print("  • 缺口识别 - 主动判断知识状态")
        print("  • 检索内化 - 自动触发知识检索")
        print("  • TSLA预判 - 治理机制前置")
        print("  • 双层生成 - 内部推理+自然语言")
        print("="*70)
        print("输入 'exit' 退出, 'stats' 查看统计")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("你: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n再见!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"\n📊 系统统计:")
                    print(f"  总查询: {stats['total_queries']}")
                    print(f"  缺口分布: {stats['gap_distribution']}")
                    print(f"  TSLA分布: {stats['tsla_distribution']}")
                    print(f"  检索统计: {stats['retrieval']}")
                    continue
                
                if not user_input:
                    continue
                
                # 执行对话
                result = self.chat(user_input)
                
                # 显示响应
                print(f"\n🤖 AI: {result['final_response']}")
                print(f"   [{result['confidence_indicator']}]")
                
                # 显示内部状态（调试用）
                print(f"   [内部: 缺口={result['gap_type']}, TSLA={result['tsla_action']}, 检索={result['retrieval_count']}条]")
                print()
                
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"\n[错误] {e}\n")


def main():
    """主函数"""
    # 创建AI实例
    ai = SelfLearningAI()
    
    # 启动交互会话
    ai.interactive_session()


if __name__ == "__main__":
    main()
