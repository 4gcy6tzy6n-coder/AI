"""
Stage 11-A-R2 第一阶段: 基础链路训练

核心策略: 先训稳"会不会查→该不该查→查完怎么答"，再叠加治理

修复内容:
1. 回退中文标签 (减少映射负担)
2. 第一阶段只训: 缺口识别 + 检索决策 + 检索整合
3. TSLA/Memory 只做轻量曝光 (10% each)
4. 恢复稳定损失权重

第一阶段目标:
- 缺口识别准确率 ≥ 85%
- 检索决策准确率 ≥ 75%
- 链路一致性 ≥ 80%

数据比例 (30/25/25/10/10):
- 教师引导型: 30%
- 缺口识别型: 25%
- 检索整合型: 25%
- TSLA动作型: 10% (轻量)
- 记忆治理型: 10% (轻量)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import requests
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-2296148b16f54ab0be255e98a911c9fe"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


# ========== 中文标签枚举 (统一固定) ==========

class Strategy(Enum):
    """回答策略"""
    DIRECT = "直接回答"
    RETRIEVAL_FIRST = "先检索"
    CONSERVATIVE = "保守回答"
    DECLINE = "拒绝回答"
    REVIEW = "需要审查"


class TSLAAction(Enum):
    """TSLA治理动作 - 中文固定"""
    KEEP = "保留"
    PROMOTE = "晋升"
    ISOLATE = "隔离"
    ERROR_ARCHIVE = "错误归档"
    DOWNGRADE = "降级"
    REFLOW = "回流重审"
    SPLIT = "拆分"
    EXCLUDE = "排除"


class MemoryAction(Enum):
    """记忆动作 - 中文固定"""
    NO_WRITE = "不写入"
    TO_REVIEW = "进入受审区"
    ISOLATE = "隔离观察"
    TO_ERROR = "进入错误区"
    PROMOTION_CANDIDATE = "晋升候选"


# 中文标签到ID的映射
TSLA_ACTION_TO_ID = {action.value: i for i, action in enumerate(TSLAAction)}
MEMORY_ACTION_TO_ID = {action.value: i for i, action in enumerate(MemoryAction)}
STRATEGY_TO_ID = {strategy.value: i for i, strategy in enumerate(Strategy)}


@dataclass
class SelfLearningSample:
    """自学习训练样本 - R2第一阶段"""
    id: str
    query: str
    context: Dict = field(default_factory=dict)
    teacher_signals: Dict = field(default_factory=dict)
    model_targets: Dict = field(default_factory=dict)
    sample_type: str = ""


class DeepSeekTeacherR2:
    """DeepSeek 教师 - R2第一阶段 (中文标签)"""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.call_count = 0
        self.total_tokens = 0
    
    def get_teacher_signals(self, query: str, context: str = "") -> Dict:
        """获取教师信号 (中文标签)"""
        system_prompt = """你是一个教师评估系统。请用中文返回评估信号。

请分析问题，返回以下信号(必须使用指定中文值):
1. gap_detected: 0或1 (是否缺信息)
2. retrieval_needed: 0或1 (是否需要检索)
3. strategy: 直接回答/先检索/保守回答/拒绝回答/需要审查
4. tsla_action: 保留/晋升/隔离/错误归档/降级/回流重审/拆分/排除
5. memory_action: 不写入/进入受审区/隔离观察/进入错误区/晋升候选

以JSON格式返回。"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"问题: {query}\n上下文: {context}"}
            ],
            "temperature": 0.3,
            "max_tokens": 400,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.call_count += 1
            self.total_tokens += result.get('usage', {}).get('total_tokens', 0)
            
            content = result['choices'][0]['message']['content']
            parsed = json.loads(content)
            
            return {
                'gap_detected': int(parsed.get('gap_detected', 0)),
                'retrieval_needed': int(parsed.get('retrieval_needed', 0)),
                'strategy': parsed.get('strategy', '直接回答'),
                'tsla_action': parsed.get('tsla_action', '保留'),
                'memory_action': parsed.get('memory_action', '不写入'),
            }
            
        except Exception as e:
            print(f"[Teacher Error] {e}")
            return {
                'gap_detected': 0,
                'retrieval_needed': 0,
                'strategy': '直接回答',
                'tsla_action': '保留',
                'memory_action': '不写入',
            }


class SelfLearningDatasetBuilderR2Phase1:
    """自学习数据集构建器 - R2第一阶段 (30/25/25/10/10)"""
    
    def __init__(self, teacher: DeepSeekTeacherR2):
        self.teacher = teacher
        self.samples: List[SelfLearningSample] = []
    
    def build_dataset(self, target_size: int = 100) -> List[SelfLearningSample]:
        """构建数据集 - 第一阶段比例 30/25/25/10/10"""
        print(f"[Dataset R2-P1] 开始构建 {target_size} 条训练数据...")
        print("  第一阶段重点: 基础链路 (缺口识别+检索决策+检索整合)")
        
        # 第一阶段比例: 30/25/25/10/10
        n_teacher_guided = int(target_size * 0.30)   # 30%
        n_gap_detection = int(target_size * 0.25)     # 25%
        n_retrieval = int(target_size * 0.25)         # 25% (增加)
        n_tsla = int(target_size * 0.10)              # 10% (轻量)
        n_memory = int(target_size * 0.10)            # 10% (轻量)
        
        print(f"  教师引导型: {n_teacher_guided} (30%)")
        print(f"  缺口识别型: {n_gap_detection} (25%)")
        print(f"  检索整合型: {n_retrieval} (25%) ↑")
        print(f"  TSLA动作型: {n_tsla} (10%) 轻量")
        print(f"  记忆治理型: {n_memory} (10%) 轻量")
        
        self._build_teacher_guided_samples(n_teacher_guided)
        self._build_gap_detection_samples(n_gap_detection)
        self._build_retrieval_samples(n_retrieval)
        self._build_tsla_samples_light(n_tsla)  # 轻量版本
        self._build_memory_samples_light(n_memory)  # 轻量版本
        
        print(f"[Dataset R2-P1] 完成！共 {len(self.samples)} 条样本")
        return self.samples
    
    def _build_teacher_guided_samples(self, n: int):
        """教师引导型样本 (30%)"""
        queries = [
            "什么是人工智能？",
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是神经网络？",
            "什么是自然语言处理？",
            "什么是计算机视觉？",
            "什么是强化学习？",
            "什么是监督学习？",
            "什么是无监督学习？",
            "什么是迁移学习？",
            "什么是过拟合？",
            "什么是梯度下降？",
            "什么是反向传播？",
            "什么是卷积神经网络？",
            "什么是注意力机制？",
            "什么是Transformer架构？",
            "什么是BERT？",
            "什么是GPT？",
            "什么是少样本学习？",
            "什么是零样本学习？",
            "什么是模型蒸馏？",
            "什么是联邦学习？",
            "什么是差分隐私？",
            "什么是可解释AI？",
            "什么是对齐问题？",
            "什么是AI安全？",
            "什么是数据增强？",
            "什么是特征工程？",
            "什么是超参数调优？",
            "什么是模型压缩？",
        ]
        
        for i in range(min(n, len(queries))):
            query = queries[i]
            teacher_signals = self.teacher.get_teacher_signals(query)
            
            sample = SelfLearningSample(
                id=f"tg_{i:04d}",
                query=query,
                teacher_signals=teacher_signals,
                model_targets={
                    'gap_detected': teacher_signals['gap_detected'],
                    'retrieval_needed': teacher_signals['retrieval_needed'],
                    'strategy': teacher_signals['strategy'],
                    'tsla_action': teacher_signals['tsla_action'],
                    'memory_action': teacher_signals['memory_action'],
                },
                sample_type="teacher_guided"
            )
            self.samples.append(sample)
            time.sleep(0.2)
    
    def _build_gap_detection_samples(self, n: int):
        """缺口识别型样本 (25%)"""
        scenarios = [
            # (query, gap_detected, retrieval_needed, strategy)
            ("2+2等于几？", 0, 0, "直接回答"),
            ("法国首都是哪里？", 0, 0, "直接回答"),
            ("水是由什么组成的？", 0, 0, "直接回答"),
            ("谁写了《罗密欧与朱丽叶》？", 0, 0, "直接回答"),
            ("光速是多少？", 0, 0, "直接回答"),
            ("什么是光合作用？", 0, 0, "直接回答"),
            ("什么是DNA？", 0, 0, "直接回答"),
            ("什么是重力？", 0, 0, "直接回答"),
            ("最新的GPT版本有什么功能？", 1, 1, "先检索"),
            ("2024年AI领域有什么突破？", 1, 1, "先检索"),
            ("特斯拉当前股价是多少？", 1, 1, "先检索"),
            ("谁获得了最新的诺贝尔物理学奖？", 1, 1, "先检索"),
            ("今天东京天气如何？", 1, 1, "先检索"),
            ("预测明天的彩票号码", 1, 0, "保守回答"),
            ("告诉我某人的私人信息", 1, 0, "拒绝回答"),
            ("如何入侵安全系统？", 1, 0, "拒绝回答"),
            ("给严重疾病提供医疗建议", 1, 0, "保守回答"),
            ("对公众人物做虚假陈述", 1, 0, "拒绝回答"),
            ("解释量子力学原理", 0, 0, "直接回答"),
            ("什么是区块链？", 0, 0, "直接回答"),
            ("什么是元宇宙？", 0, 0, "直接回答"),
            ("什么是Web3？", 0, 0, "直接回答"),
            ("什么是数字孪生？", 0, 0, "直接回答"),
            ("什么是边缘计算？", 0, 0, "直接回答"),
            ("什么是量子计算？", 0, 0, "直接回答"),
        ]
        
        for i in range(min(n, len(scenarios))):
            query, gap, retrieval, strategy = scenarios[i]
            
            sample = SelfLearningSample(
                id=f"gap_{i:04d}",
                query=query,
                teacher_signals={
                    'gap_detected': gap,
                    'retrieval_needed': retrieval,
                    'strategy': strategy,
                },
                model_targets={
                    'gap_detected': gap,
                    'retrieval_needed': retrieval,
                    'strategy': strategy,
                },
                sample_type="gap_detection"
            )
            self.samples.append(sample)
    
    def _build_retrieval_samples(self, n: int):
        """检索整合型样本 (25%)"""
        scenarios = [
            {
                'query': "Transformer和RNN有什么区别？",
                'docs': ['Transformer使用自注意力机制处理序列', 'RNN使用循环结构处理序列数据'],
                'retrieval_needed': 1,
            },
            {
                'query': "如何防止模型过拟合？",
                'docs': ['正则化技术可以防止过拟合', 'Dropout是一种常用的正则化方法', '数据增强也能帮助减少过拟合'],
                'retrieval_needed': 1,
            },
            {
                'query': "CNN有什么优势？",
                'docs': ['CNN擅长图像处理任务', '能够捕捉空间层次特征', '参数共享减少计算量'],
                'retrieval_needed': 1,
            },
            {
                'query': "解释偏差-方差权衡",
                'docs': ['高偏差导致欠拟合', '高方差导致过拟合', '这是机器学习的核心权衡'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是集成学习？",
                'docs': ['集成学习结合多个模型', '随机森林是一种集成方法', '可以提升模型稳定性'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是交叉验证？",
                'docs': ['交叉验证用于评估模型泛化能力', 'K折交叉验证是常用方法', '可以减少过拟合风险'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是学习率？",
                'docs': ['学习率控制参数更新步长', '太大会导致震荡', '太小会收敛缓慢'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是批量归一化？",
                'docs': ['批量归一化稳定训练', '加速收敛速度', '允许使用更大学习率'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是残差连接？",
                'docs': ['残差连接解决梯度消失', '允许训练更深网络', '是ResNet的核心创新'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是词嵌入？",
                'docs': ['词嵌入将词映射为向量', '捕捉语义关系', 'Word2Vec和GloVe是经典方法'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是序列到序列模型？",
                'docs': ['Seq2Seq用于序列转换', '包含编码器和解码器', '常用于机器翻译'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是生成对抗网络？",
                'docs': ['GAN包含生成器和判别器', '通过对抗训练生成数据', '可以生成逼真图像'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是自编码器？",
                'docs': ['自编码器学习数据表示', '包含编码和解码部分', '可用于降维和去噪'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是强化学习中的奖励？",
                'docs': ['奖励是环境的反馈信号', '指导智能体学习策略', '设计好的奖励很关键'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是马尔可夫决策过程？",
                'docs': ['MDP是强化学习的数学框架', '包含状态、动作、奖励', '满足马尔可夫性质'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是策略梯度？",
                'docs': ['策略梯度直接优化策略', '通过梯度上升更新', 'REINFORCE是经典算法'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是Actor-Critic？",
                'docs': ['Actor-Critic结合价值和策略', 'Actor选择动作', 'Critic评估价值'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是经验回放？",
                'docs': ['经验回放存储历史经验', '打破数据相关性', '提高样本效率'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是目标网络？",
                'docs': ['目标网络稳定训练', '定期从主网络复制', '减少震荡'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是探索与利用？",
                'docs': ['探索尝试新动作', '利用选择已知好动作', '需要平衡两者'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是多臂老虎机？",
                'docs': ['多臂老虎机是简化RL问题', '平衡探索和利用', '有理论最优策略'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是上置信界算法？",
                'docs': ['UCB基于置信区间选择', '理论保证遗憾界', '常用于推荐系统'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是汤普森采样？",
                'docs': ['汤普森采样基于贝叶斯', '从后验分布采样', '在实践中表现很好'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是上下文老虎机？",
                'docs': ['上下文老虎机考虑特征', '比多臂老虎机更一般', 'LinUCB是经典算法'],
                'retrieval_needed': 1,
            },
            {
                'query': "什么是对抗样本？",
                'docs': ['对抗样本欺骗模型', '添加微小扰动', '是安全研究热点'],
                'retrieval_needed': 1,
            },
        ]
        
        for i in range(min(n, len(scenarios))):
            scenario = scenarios[i]
            
            sample = SelfLearningSample(
                id=f"ret_{i:04d}",
                query=scenario['query'],
                context={'retrieved_docs': scenario['docs']},
                teacher_signals={
                    'retrieval_needed': scenario['retrieval_needed'],
                    'strategy': '先检索',
                },
                model_targets={
                    'retrieval_needed': scenario['retrieval_needed'],
                    'evidence_relevance': 0.8,
                },
                sample_type="retrieval_integration"
            )
            self.samples.append(sample)
    
    def _build_tsla_samples_light(self, n: int):
        """TSLA动作型样本 - 轻量版 (10%)"""
        # 第一阶段只做轻量曝光，不重点训练
        tsla_scenarios = [
            ("这是经过验证的可靠知识", "保留", "不写入"),
            ("这个信息可能有误", "回流重审", "进入受审区"),
            ("涉及敏感个人信息", "隔离", "隔离观察"),
            ("经过多次验证的高质量知识", "晋升", "晋升候选"),
        ]
        
        for i in range(n):
            content, tsla_action, memory_action = tsla_scenarios[i % len(tsla_scenarios)]
            
            sample = SelfLearningSample(
                id=f"tsla_{i:04d}",
                query=f"内容评估: {content}",
                teacher_signals={
                    'tsla_action': tsla_action,
                    'memory_action': memory_action,
                },
                model_targets={
                    'tsla_action': tsla_action,
                    'memory_action': memory_action,
                },
                sample_type="tsla_action"
            )
            self.samples.append(sample)
    
    def _build_memory_samples_light(self, n: int):
        """记忆治理型样本 - 轻量版 (10%)"""
        memory_scenarios = [
            ("临时对话内容", "不写入"),
            ("待验证的新知识", "进入受审区"),
            ("错误信息", "进入错误区"),
            ("验证过的知识", "晋升候选"),
        ]
        
        for i in range(n):
            content, memory_action = memory_scenarios[i % len(memory_scenarios)]
            
            sample = SelfLearningSample(
                id=f"mem_{i:04d}",
                query=f"记忆处理: {content}",
                teacher_signals={
                    'memory_action': memory_action,
                },
                model_targets={
                    'memory_action': memory_action,
                },
                sample_type="memory_governance"
            )
            self.samples.append(sample)
    
    def save_dataset(self, filepath: str):
        """保存数据集"""
        data = []
        for sample in self.samples:
            data.append({
                'id': sample.id,
                'query': sample.query,
                'context': sample.context,
                'teacher_signals': sample.teacher_signals,
                'model_targets': sample.model_targets,
                'sample_type': sample.sample_type,
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[Dataset R2-P1] 已保存: {filepath}")


class SelfLearningModelR2(nn.Module):
    """自学习模型 - R2版本"""
    
    def __init__(self, base_model: NativeBackboneTinyV1):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_dim
        
        # 原有输出
        self.writeback_head = base_model.writeback_head
        self.governance_head = base_model.governance_head
        self.gap_detector = base_model.gap_detector
        
        # 新增预测头
        self.gap_detection_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        
        self.retrieval_decision_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
        
        self.tsla_action_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
        
        self.memory_action_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
    
    def forward(self, input_ids, return_all=True):
        base_outputs = self.base(input_ids)
        hidden = base_outputs['hidden_states']
        pooled = hidden.mean(dim=1)
        
        outputs = {
            'writeback_logits': base_outputs.get('writeback_logits'),
            'governance_logits': base_outputs.get('governance_logits'),
            'gap_probs': base_outputs.get('gap_probs'),
        }
        
        if return_all:
            outputs['gap_detection_logits'] = self.gap_detection_head(pooled)
            outputs['retrieval_decision_logits'] = self.retrieval_decision_head(pooled)
            outputs['strategy_logits'] = self.strategy_head(pooled)
            outputs['tsla_action_logits'] = self.tsla_action_head(pooled)
            outputs['memory_action_logits'] = self.memory_action_head(pooled)
        
        return outputs


class Stage11AR2Phase1Trainer:
    """Stage 11-A-R2 第一阶段训练器"""
    
    def __init__(
        self,
        model: SelfLearningModelR2,
        teacher: DeepSeekTeacherR2,
        device: str = 'cpu',
    ):
        self.model = model
        self.teacher = teacher
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # R2第一阶段: 恢复稳定权重，但保持链路倾斜
        # L_answer=0.30, L_gap=0.20, L_retrieval=0.20, L_tsla=0.15, L_memory=0.10, L_guard=0.05
        self.loss_weights = {
            'answer': 0.30,
            'gap': 0.20,
            'retrieval': 0.20,
            'tsla': 0.15,  # 降低TSLA权重
            'memory': 0.10,
            'guard': 0.05,
        }
        
        self.metrics_history = []
    
    def compute_loss(self, outputs: Dict, targets: Dict, sample_type: str) -> Dict:
        """计算损失 - 第一阶段重点: gap + retrieval"""
        losses = {}
        
        # 1. 缺口识别损失 (核心)
        if 'gap_detected' in targets:
            gap_target = torch.tensor([targets['gap_detected']], device=self.device)
            losses['gap'] = F.cross_entropy(
                outputs['gap_detection_logits'],
                gap_target
            )
        
        # 2. 检索决策损失 (核心)
        if 'retrieval_needed' in targets:
            retrieval_target = torch.tensor([targets['retrieval_needed']], device=self.device)
            losses['retrieval'] = F.cross_entropy(
                outputs['retrieval_decision_logits'],
                retrieval_target
            )
        
        # 3. TSLA动作损失 (轻量)
        if 'tsla_action' in targets and sample_type in ['tsla_action', 'teacher_guided']:
            tsla_target_str = targets['tsla_action']
            tsla_target_id = TSLA_ACTION_TO_ID.get(tsla_target_str, 0)
            tsla_target = torch.tensor([tsla_target_id], device=self.device)
            losses['tsla'] = F.cross_entropy(
                outputs['tsla_action_logits'],
                tsla_target
            )
        
        # 4. 记忆动作损失 (轻量)
        if 'memory_action' in targets and sample_type in ['memory_governance', 'teacher_guided']:
            memory_target_str = targets['memory_action']
            memory_target_id = MEMORY_ACTION_TO_ID.get(memory_target_str, 0)
            memory_target = torch.tensor([memory_target_id], device=self.device)
            losses['memory'] = F.cross_entropy(
                outputs['memory_action_logits'],
                memory_target
            )
        
        # 总损失
        loss_tensors = []
        for key in self.loss_weights.keys():
            loss_val = losses.get(key, 0)
            if isinstance(loss_val, torch.Tensor) and loss_val.requires_grad:
                loss_tensors.append(loss_val * self.loss_weights[key])
        
        if loss_tensors:
            total_loss = sum(loss_tensors)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        losses['total'] = total_loss
        return losses
    
    def train_step(self, sample: SelfLearningSample) -> Dict:
        """单步训练"""
        self.model.train()
        
        input_ids = self._encode(sample.query)
        outputs = self.model(input_ids)
        losses = self.compute_loss(outputs, sample.model_targets, sample.sample_type)
        
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': losses['total'].item() if isinstance(losses['total'], torch.Tensor) else losses['total'],
            'gap_loss': losses.get('gap', 0).item() if isinstance(losses.get('gap'), torch.Tensor) else 0,
            'retrieval_loss': losses.get('retrieval', 0).item() if isinstance(losses.get('retrieval'), torch.Tensor) else 0,
            'tsla_loss': losses.get('tsla', 0).item() if isinstance(losses.get('tsla'), torch.Tensor) else 0,
            'memory_loss': losses.get('memory', 0).item() if isinstance(losses.get('memory'), torch.Tensor) else 0,
        }
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def evaluate(self, samples: List[SelfLearningSample]) -> Dict:
        """评估模型 - 第一阶段重点"""
        self.model.eval()
        
        metrics = {
            'gap_correct': 0,
            'retrieval_correct': 0,
            'tsla_correct': 0,
            'memory_correct': 0,
            'chain_consistent': 0,
            'total': 0,
        }
        
        with torch.no_grad():
            for sample in samples:
                input_ids = self._encode(sample.query)
                outputs = self.model(input_ids)
                
                metrics['total'] += 1
                
                # 评估缺口识别
                if 'gap_detected' in sample.model_targets:
                    gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
                    if gap_pred == sample.model_targets['gap_detected']:
                        metrics['gap_correct'] += 1
                
                # 评估检索决策
                if 'retrieval_needed' in sample.model_targets:
                    retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
                    if retrieval_pred == sample.model_targets['retrieval_needed']:
                        metrics['retrieval_correct'] += 1
                
                # 评估TSLA
                if 'tsla_action' in sample.model_targets:
                    tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
                    tsla_map = ['保留', '晋升', '隔离', '错误归档', '降级', '回流重审', '拆分', '排除']
                    tsla_pred_name = tsla_map[tsla_pred] if tsla_pred < len(tsla_map) else '未知'
                    if tsla_pred_name == sample.model_targets['tsla_action']:
                        metrics['tsla_correct'] += 1
                
                # 评估记忆
                if 'memory_action' in sample.model_targets:
                    memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
                    memory_map = ['不写入', '进入受审区', '隔离观察', '进入错误区', '晋升候选']
                    memory_pred_name = memory_map[memory_pred] if memory_pred < len(memory_map) else '未知'
                    if memory_pred_name == sample.model_targets['memory_action']:
                        metrics['memory_correct'] += 1
                
                # 链路一致性检查
                gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
                retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
                
                # 简单一致性: 有缺口时应该考虑检索
                consistent = True
                if gap_pred == 1 and retrieval_pred == 0:
                    # 有缺口但不检索，检查策略
                    strategy_pred = outputs['strategy_logits'].argmax(dim=-1).item()
                    if strategy_pred != 2:  # 不是保守回答
                        consistent = False
                
                if consistent:
                    metrics['chain_consistent'] += 1
        
        total = metrics['total']
        return {
            'gap_accuracy': metrics['gap_correct'] / total if total > 0 else 0,
            'retrieval_accuracy': metrics['retrieval_correct'] / total if total > 0 else 0,
            'tsla_accuracy': metrics['tsla_correct'] / total if total > 0 else 0,
            'memory_accuracy': metrics['memory_correct'] / total if total > 0 else 0,
            'chain_consistency': metrics['chain_consistent'] / total if total > 0 else 0,
        }


def run_stage11a_r2_phase1():
    """运行 Stage 11-A-R2 第一阶段训练"""
    print("="*70)
    print("Stage 11-A-R2 第一阶段: 基础链路训练")
    print("="*70)
    print("核心策略: 先训稳'会不会查→该不该查→查完怎么答'")
    print("="*70)
    print("\n第一阶段重点:")
    print("  - 缺口识别 (核心)")
    print("  - 检索决策 (核心)")
    print("  - 检索整合 (核心)")
    print("  - TSLA/Memory (轻量曝光)")
    print("\n数据比例: 30/25/25/10/10")
    print("损失权重: 0.30/0.20/0.20/0.15/0.10/0.05")
    print("="*70)
    print("\n第一阶段目标:")
    print("  缺口识别准确率 ≥ 85%")
    print("  检索决策准确率 ≥ 75%")
    print("  链路一致性 ≥ 80%")
    print("="*70)
    
    # 1. 初始化教师
    print("\n[1/5] 初始化R2教师 (中文标签)...")
    teacher = DeepSeekTeacherR2()
    
    # 2. 构建数据集 (第一阶段比例)
    print("\n[2/5] 构建R2第一阶段数据集 (30/25/25/10/10)...")
    dataset_builder = SelfLearningDatasetBuilderR2Phase1(teacher)
    samples = dataset_builder.build_dataset(target_size=100)
    dataset_builder.save_dataset("stage8_dataset/stage11a_r2_phase1_dataset.json")
    
    # 3. 初始化模型
    print("\n[3/5] 初始化R2模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = SelfLearningModelR2(base_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 4. 创建训练器
    print("\n[4/5] 创建R2第一阶段训练器...")
    trainer = Stage11AR2Phase1Trainer(model, teacher, device)
    print(f"损失权重: {trainer.loss_weights}")
    
    # 5. 执行训练 (5轮)
    print("\n[5/5] 开始R2第一阶段训练 (5轮)...")
    print(f"训练样本: {len(samples)}")
    print("="*70)
    
    for epoch in range(5):
        print(f"\n[Epoch {epoch+1}/5]")
        
        random.shuffle(samples)
        
        epoch_losses = []
        for i, sample in enumerate(samples):
            result = trainer.train_step(sample)
            epoch_losses.append(result['loss'])
            
            if i % 20 == 0:
                print(f"  Step {i:3d} | Loss: {result['loss']:.4f} | "
                      f"Gap: {result['gap_loss']:.4f} | "
                      f"Retrieval: {result['retrieval_loss']:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
    
    # 6. 评估
    print("\n" + "="*70)
    print("R2第一阶段评估结果")
    print("="*70)
    
    eval_result = trainer.evaluate(samples)
    print(f"缺口识别准确率: {eval_result['gap_accuracy']:.2%}")
    print(f"检索决策准确率: {eval_result['retrieval_accuracy']:.2%}")
    print(f"TSLA动作准确率: {eval_result['tsla_accuracy']:.2%}")
    print(f"记忆动作准确率: {eval_result['memory_accuracy']:.2%}")
    print(f"链路一致性: {eval_result['chain_consistency']:.2%}")
    
    # 7. 判断是否通过第一阶段门槛
    print("\n" + "="*70)
    print("第一阶段门槛检查")
    print("="*70)
    
    checks = []
    passed = True
    
    # 第一阶段核心指标
    phase1_thresholds = {
        'gap_accuracy': 0.85,
        'retrieval_accuracy': 0.75,
        'chain_consistency': 0.80,
    }
    
    for metric, threshold in phase1_thresholds.items():
        value = eval_result[metric]
        if value >= threshold:
            checks.append(f"✓ {metric}: {value:.1%} ≥ {threshold:.0%}")
        else:
            checks.append(f"✗ {metric}: {value:.1%} < {threshold:.0%}")
            passed = False
    
    for check in checks:
        print(f"  {check}")
    
    # 8. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_phase1_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'eval_result': eval_result,
    }, checkpoint_path)
    
    print(f"\n✓ R2第一阶段检查点已保存: {checkpoint_path}")
    print(f"✓ DeepSeek API 调用: {teacher.call_count} 次")
    print(f"✓ 总 token 消耗: {teacher.total_tokens}")
    
    print("\n" + "="*70)
    if passed:
        print("🎉 Stage 11-A-R2 第一阶段通过！")
        print("建议: 可以进入第二阶段 (叠加治理链路)")
    else:
        print("⚠ Stage 11-A-R2 第一阶段未完全通过")
        print("建议: 分析未通过项，修复后再进入第二阶段")
    print("="*70)
    
    return model, trainer, eval_result, passed


if __name__ == "__main__":
    model, trainer, eval_result, passed = run_stage11a_r2_phase1()
