"""
Stage 11-A-R1: 修复重训

修复内容:
1. 统一标签格式 (全英文枚举)
2. 修复样本目标逻辑
3. 重新平衡样本比例 (25/25/20/20/10)
4. 调整损失权重

目标门槛:
- 缺口识别准确率 ≥ 85%
- 检索决策准确率 ≥ 75%
- TSLA动作准确率 ≥ 70%
- 记忆动作准确率 ≥ 70%
- 链路一致性 ≥ 80%
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


# ========== 统一标签格式 (全英文枚举) ==========

class Strategy(Enum):
    """回答策略"""
    DIRECT = "DIRECT"
    RETRIEVAL_FIRST = "RETRIEVAL_FIRST"
    CONSERVATIVE = "CONSERVATIVE"
    DECLINE = "DECLINE"
    REVIEW = "REVIEW"


class TSLAAction(Enum):
    """TSLA治理动作 - 统一英文"""
    KEEP = "KEEP"
    PROMOTE = "PROMOTE"
    ISOLATE = "ISOLATE"
    ERROR_ARCHIVE = "ERROR_ARCHIVE"
    DOWNGRADE = "DOWNGRADE"
    REFLOW = "REFLOW"
    SPLIT = "SPLIT"
    EXCLUDE = "EXCLUDE"


class MemoryAction(Enum):
    """记忆动作 - 统一英文"""
    NO_WRITE = "NO_WRITE"
    TO_REVIEW = "TO_REVIEW"
    ISOLATE = "ISOLATE"
    TO_ERROR = "TO_ERROR"
    PROMOTION_CANDIDATE = "PROMOTION_CANDIDATE"


# 映射字典
TSLA_ACTION_MAP = {
    '保留': 'KEEP',
    '晋升': 'PROMOTE',
    '隔离': 'ISOLATE',
    '错误归档': 'ERROR_ARCHIVE',
    '降级': 'DOWNGRADE',
    '回流重审': 'REFLOW',
    '拆分': 'SPLIT',
    '排除': 'EXCLUDE',
}

MEMORY_ACTION_MAP = {
    '不写入': 'NO_WRITE',
    '进入受审区': 'TO_REVIEW',
    '隔离': 'ISOLATE',
    '错误区': 'TO_ERROR',
    '晋升候选': 'PROMOTION_CANDIDATE',
}

STRATEGY_MAP = {
    'DIRECT': 0,
    'RETRIEVAL_FIRST': 1,
    'CONSERVATIVE': 2,
    'DECLINE': 3,
    'REVIEW': 4,
}

TSLA_ACTION_TO_ID = {action.value: i for i, action in enumerate(TSLAAction)}
MEMORY_ACTION_TO_ID = {action.value: i for i, action in enumerate(MemoryAction)}


@dataclass
class SelfLearningSample:
    """自学习训练样本 - R1修复版"""
    id: str
    query: str
    context: Dict = field(default_factory=dict)
    teacher_signals: Dict = field(default_factory=dict)
    model_targets: Dict = field(default_factory=dict)
    sample_type: str = ""


@dataclass
class ModelOutput:
    """模型完整输出"""
    gap_detected: bool = False
    gap_confidence: float = 0.0
    retrieval_needed: bool = False
    strategy: Strategy = Strategy.DIRECT
    tsla_action: TSLAAction = TSLAAction.KEEP
    memory_action: MemoryAction = MemoryAction.NO_WRITE


class DeepSeekTeacherR1:
    """DeepSeek 教师 - R1修复版 (统一英文标签)"""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.call_count = 0
        self.total_tokens = 0
    
    def get_teacher_signals(self, query: str, context: str = "") -> Dict:
        """获取教师信号 (统一返回英文标签)"""
        system_prompt = """You are a teacher evaluation system. Provide assessment signals in English only.

Analyze the query and return these signals in JSON format:
1. gap_detected: 0 or 1 (whether information is missing)
2. retrieval_needed: 0 or 1 (whether retrieval is needed)
3. strategy: DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW
4. correction_hint: brief hint about potential errors
5. quality_score: 0.0-1.0
6. tsla_action: KEEP/PROMOTE/ISOLATE/ERROR_ARCHIVE/DOWNGRADE/REFLOW/SPLIT/EXCLUDE
7. memory_action: NO_WRITE/TO_REVIEW/ISOLATE/TO_ERROR/PROMOTION_CANDIDATE

All values must be in English."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\nContext: {context}"}
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
                'strategy': parsed.get('strategy', 'DIRECT'),
                'correction_hint': parsed.get('correction_hint', ''),
                'quality_score': float(parsed.get('quality_score', 0.5)),
                'tsla_action': parsed.get('tsla_action', 'KEEP'),
                'memory_action': parsed.get('memory_action', 'NO_WRITE'),
            }
            
        except Exception as e:
            print(f"[Teacher Error] {e}")
            return {
                'gap_detected': 0,
                'retrieval_needed': 0,
                'strategy': 'DIRECT',
                'correction_hint': 'API error',
                'quality_score': 0.5,
                'tsla_action': 'KEEP',
                'memory_action': 'NO_WRITE',
            }


class SelfLearningDatasetBuilderR1:
    """自学习数据集构建器 - R1修复版 (25/25/20/20/10比例)"""
    
    def __init__(self, teacher: DeepSeekTeacherR1):
        self.teacher = teacher
        self.samples: List[SelfLearningSample] = []
    
    def build_dataset(self, target_size: int = 100) -> List[SelfLearningSample]:
        """构建数据集 - 新比例 25/25/20/20/10"""
        print(f"[Dataset R1] 开始构建 {target_size} 条训练数据...")
        
        # 新比例: 25/25/20/20/10
        n_teacher_guided = int(target_size * 0.25)   # 25% (原为30%)
        n_gap_detection = int(target_size * 0.25)     # 25% (保持)
        n_retrieval = int(target_size * 0.20)         # 20% (保持)
        n_tsla = int(target_size * 0.20)              # 20% (原为15%)
        n_memory = int(target_size * 0.10)            # 10% (保持)
        
        print(f"  教师引导型: {n_teacher_guided} (25%)")
        print(f"  缺口识别型: {n_gap_detection} (25%)")
        print(f"  检索整合型: {n_retrieval} (20%)")
        print(f"  TSLA动作型: {n_tsla} (20%) ↑")
        print(f"  记忆治理型: {n_memory} (10%)")
        
        self._build_teacher_guided_samples(n_teacher_guided)
        self._build_gap_detection_samples(n_gap_detection)
        self._build_retrieval_samples(n_retrieval)
        self._build_tsla_samples(n_tsla)
        self._build_memory_samples(n_memory)
        
        print(f"[Dataset R1] 完成！共 {len(self.samples)} 条样本")
        return self.samples
    
    def _build_teacher_guided_samples(self, n: int):
        """教师引导型样本 (25%)"""
        queries = [
            "What is artificial intelligence?",
            "What is machine learning?",
            "What is deep learning?",
            "What is a neural network?",
            "What is natural language processing?",
            "What is computer vision?",
            "What is reinforcement learning?",
            "What is supervised learning?",
            "What is unsupervised learning?",
            "What is transfer learning?",
            "What is overfitting in ML?",
            "What is gradient descent?",
            "What is backpropagation?",
            "What is a convolutional neural network?",
            "What is attention mechanism?",
            "What is the Transformer architecture?",
            "What is BERT?",
            "What is GPT?",
            "What is few-shot learning?",
            "What is zero-shot learning?",
            "What is model distillation?",
            "What is federated learning?",
            "What is differential privacy?",
            "What is explainable AI?",
            "What is AI alignment?",
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
        # 有明确答案的问题
        known_queries = [
            ("What is 2+2?", 0, 0, "DIRECT"),
            ("What is the capital of France?", 0, 0, "DIRECT"),
            ("What is water made of?", 0, 0, "DIRECT"),
            ("Who wrote Romeo and Juliet?", 0, 0, "DIRECT"),
            ("What is the speed of light?", 0, 0, "DIRECT"),
            ("What is photosynthesis?", 0, 0, "DIRECT"),
            ("What is DNA?", 0, 0, "DIRECT"),
            ("What is gravity?", 0, 0, "DIRECT"),
        ]
        
        # 需要检索的问题
        retrieval_queries = [
            ("What is the latest version of GPT?", 1, 1, "RETRIEVAL_FIRST"),
            ("What are the recent breakthroughs in AI 2024?", 1, 1, "RETRIEVAL_FIRST"),
            ("What is the current stock price of Tesla?", 1, 1, "RETRIEVAL_FIRST"),
            ("Who won the latest Nobel Prize in Physics?", 1, 1, "RETRIEVAL_FIRST"),
            ("What is the weather in Tokyo today?", 1, 1, "RETRIEVAL_FIRST"),
        ]
        
        # 应该保守回答的问题
        conservative_queries = [
            ("Predict tomorrow's lottery numbers", 1, 0, "CONSERVATIVE"),
            ("Tell me someone's private information", 1, 0, "DECLINE"),
            ("How to hack into a secure system?", 1, 0, "DECLINE"),
            ("Give me medical advice for a serious condition", 1, 0, "CONSERVATIVE"),
            ("Make a false statement about a public figure", 1, 0, "DECLINE"),
        ]
        
        all_queries = known_queries + retrieval_queries + conservative_queries
        
        for i in range(n):
            query, gap, retrieval, strategy = all_queries[i % len(all_queries)]
            
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
        """检索整合型样本 (20%)"""
        retrieval_scenarios = [
            {
                'query': "What is the difference between Transformer and RNN?",
                'docs': ['Transformers use self-attention mechanism', 'RNNs use recurrent structure'],
                'retrieval_needed': 1,
            },
            {
                'query': "How to prevent overfitting?",
                'docs': ['Regularization prevents overfitting', 'Dropout is a regularization method', 'Data augmentation helps'],
                'retrieval_needed': 1,
            },
            {
                'query': "What are the advantages of CNN?",
                'docs': ['CNNs are good for image processing', 'They capture spatial hierarchies', 'Parameter sharing reduces computation'],
                'retrieval_needed': 1,
            },
            {
                'query': "Explain the bias-variance tradeoff",
                'docs': ['High bias leads to underfitting', 'High variance leads to overfitting', 'Tradeoff is fundamental in ML'],
                'retrieval_needed': 1,
            },
        ]
        
        for i in range(n):
            scenario = retrieval_scenarios[i % len(retrieval_scenarios)]
            
            sample = SelfLearningSample(
                id=f"ret_{i:04d}",
                query=scenario['query'],
                context={'retrieved_docs': scenario['docs']},
                teacher_signals={
                    'retrieval_needed': scenario['retrieval_needed'],
                    'strategy': 'RETRIEVAL_FIRST',
                },
                model_targets={
                    'retrieval_needed': scenario['retrieval_needed'],
                    'evidence_relevance': 0.8,
                },
                sample_type="retrieval_integration"
            )
            self.samples.append(sample)
    
    def _build_tsla_samples(self, n: int):
        """TSLA动作型样本 (20%) - 增加治理训练"""
        tsla_scenarios = [
            ("This is a verified fact from reliable sources", "KEEP", "NO_WRITE"),
            ("This information might be incorrect or outdated", "REFLOW", "TO_REVIEW"),
            ("This content involves sensitive personal information", "ISOLATE", "ISOLATE"),
            ("This is high-quality knowledge validated multiple times", "PROMOTE", "PROMOTION_CANDIDATE"),
            ("This contains conflicting information", "REFLOW", "TO_REVIEW"),
            ("This is potentially harmful content", "EXCLUDE", "TO_ERROR"),
            ("This is a temporary conversation fragment", "DOWNGRADE", "NO_WRITE"),
            ("This needs to be split into multiple parts", "SPLIT", "TO_REVIEW"),
        ]
        
        for i in range(n):
            content, tsla_action, memory_action = tsla_scenarios[i % len(tsla_scenarios)]
            
            sample = SelfLearningSample(
                id=f"tsla_{i:04d}",
                query=f"How to handle this content: {content}",
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
    
    def _build_memory_samples(self, n: int):
        """记忆治理型样本 (10%)"""
        memory_scenarios = [
            ("Temporary conversation content that should not persist", "NO_WRITE"),
            ("New knowledge that needs verification", "TO_REVIEW"),
            ("Incorrect or outdated information", "TO_ERROR"),
            ("Knowledge validated through multiple rounds", "PROMOTION_CANDIDATE"),
            ("Sensitive information requiring isolation", "ISOLATE"),
        ]
        
        for i in range(n):
            content, memory_action = memory_scenarios[i % len(memory_scenarios)]
            
            sample = SelfLearningSample(
                id=f"mem_{i:04d}",
                query=f"How to handle this memory: {content}",
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
        
        print(f"[Dataset R1] 已保存: {filepath}")


class SelfLearningModelR1(nn.Module):
    """自学习模型 - R1修复版"""
    
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


class Stage11AR1Trainer:
    """Stage 11-A-R1 训练器 - 修复版"""
    
    def __init__(
        self,
        model: SelfLearningModelR1,
        teacher: DeepSeekTeacherR1,
        device: str = 'cpu',
    ):
        self.model = model
        self.teacher = teacher
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # R1修复: 新损失权重
        # L_answer=0.25, L_gap=0.20, L_retrieval=0.20, L_tsla=0.20, L_memory=0.10, L_guard=0.05
        self.loss_weights = {
            'answer': 0.25,      # 降低回答权重
            'gap': 0.20,         # 保持缺口识别
            'retrieval': 0.20,   # 提升检索决策
            'tsla': 0.20,        # 提升TSLA
            'memory': 0.10,      # 保持记忆
            'guard': 0.05,       # 降低Guard
        }
        
        self.metrics_history = []
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Dict:
        """计算多目标损失 - R1修复版"""
        losses = {}
        
        # 1. 缺口识别损失
        if 'gap_detected' in targets:
            gap_target = torch.tensor([targets['gap_detected']], device=self.device)
            losses['gap'] = F.cross_entropy(
                outputs['gap_detection_logits'],
                gap_target
            )
        
        # 2. 检索决策损失
        if 'retrieval_needed' in targets:
            retrieval_target = torch.tensor([targets['retrieval_needed']], device=self.device)
            losses['retrieval'] = F.cross_entropy(
                outputs['retrieval_decision_logits'],
                retrieval_target
            )
        
        # 3. TSLA动作损失
        if 'tsla_action' in targets:
            tsla_target_str = targets['tsla_action']
            tsla_target_id = TSLA_ACTION_TO_ID.get(tsla_target_str, 0)
            tsla_target = torch.tensor([tsla_target_id], device=self.device)
            losses['tsla'] = F.cross_entropy(
                outputs['tsla_action_logits'],
                tsla_target
            )
        
        # 4. 记忆动作损失
        if 'memory_action' in targets:
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
        losses = self.compute_loss(outputs, sample.model_targets)
        
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
        """评估模型"""
        self.model.eval()
        
        metrics = {
            'gap_correct': 0,
            'retrieval_correct': 0,
            'tsla_correct': 0,
            'memory_correct': 0,
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
                    tsla_map = ['KEEP', 'PROMOTE', 'ISOLATE', 'ERROR_ARCHIVE', 'DOWNGRADE', 'REFLOW', 'SPLIT', 'EXCLUDE']
                    tsla_pred_name = tsla_map[tsla_pred] if tsla_pred < len(tsla_map) else 'UNKNOWN'
                    if tsla_pred_name == sample.model_targets['tsla_action']:
                        metrics['tsla_correct'] += 1
                
                # 评估记忆
                if 'memory_action' in sample.model_targets:
                    memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
                    memory_map = ['NO_WRITE', 'TO_REVIEW', 'ISOLATE', 'TO_ERROR', 'PROMOTION_CANDIDATE']
                    memory_pred_name = memory_map[memory_pred] if memory_pred < len(memory_map) else 'UNKNOWN'
                    if memory_pred_name == sample.model_targets['memory_action']:
                        metrics['memory_correct'] += 1
        
        total = metrics['total']
        return {
            'gap_accuracy': metrics['gap_correct'] / total if total > 0 else 0,
            'retrieval_accuracy': metrics['retrieval_correct'] / total if total > 0 else 0,
            'tsla_accuracy': metrics['tsla_correct'] / total if total > 0 else 0,
            'memory_accuracy': metrics['memory_correct'] / total if total > 0 else 0,
        }


def run_stage11a_r1_training():
    """运行 Stage 11-A-R1 修复训练"""
    print("="*70)
    print("Stage 11-A-R1: 修复重训")
    print("="*70)
    print("修复内容:")
    print("  1. 统一标签格式 (全英文)")
    print("  2. 修复样本目标逻辑")
    print("  3. 重新平衡样本比例 (25/25/20/20/10)")
    print("  4. 调整损失权重 (0.25/0.20/0.20/0.20/0.10/0.05)")
    print("="*70)
    print("\n目标门槛:")
    print("  缺口识别准确率 ≥ 85%")
    print("  检索决策准确率 ≥ 75%")
    print("  TSLA动作准确率 ≥ 70%")
    print("  记忆动作准确率 ≥ 70%")
    print("  链路一致性 ≥ 80%")
    print("="*70)
    
    # 1. 初始化教师
    print("\n[1/5] 初始化R1教师...")
    teacher = DeepSeekTeacherR1()
    
    # 2. 构建数据集 (新比例)
    print("\n[2/5] 构建R1数据集 (25/25/20/20/10)...")
    dataset_builder = SelfLearningDatasetBuilderR1(teacher)
    samples = dataset_builder.build_dataset(target_size=100)
    dataset_builder.save_dataset("stage8_dataset/stage11a_r1_dataset.json")
    
    # 3. 初始化模型
    print("\n[3/5] 初始化R1模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = SelfLearningModelR1(base_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 4. 创建训练器 (新权重)
    print("\n[4/5] 创建R1训练器...")
    trainer = Stage11AR1Trainer(model, teacher, device)
    print(f"损失权重: {trainer.loss_weights}")
    
    # 5. 执行训练 (5轮)
    print("\n[5/5] 开始R1训练 (5轮)...")
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
                      f"Retrieval: {result['retrieval_loss']:.4f} | "
                      f"TSLA: {result['tsla_loss']:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
    
    # 6. 评估
    print("\n" + "="*70)
    print("R1评估结果")
    print("="*70)
    
    eval_result = trainer.evaluate(samples)
    print(f"缺口识别准确率: {eval_result['gap_accuracy']:.2%}")
    print(f"检索决策准确率: {eval_result['retrieval_accuracy']:.2%}")
    print(f"TSLA动作准确率: {eval_result['tsla_accuracy']:.2%}")
    print(f"记忆动作准确率: {eval_result['memory_accuracy']:.2%}")
    
    # 7. 判断是否通过门槛
    print("\n" + "="*70)
    print("门槛检查")
    print("="*70)
    
    checks = []
    passed = True
    
    thresholds = {
        'gap_accuracy': 0.85,
        'retrieval_accuracy': 0.75,
        'tsla_accuracy': 0.70,
        'memory_accuracy': 0.70,
    }
    
    for metric, threshold in thresholds.items():
        value = eval_result[metric]
        if value >= threshold:
            checks.append(f"✓ {metric}: {value:.1%} ≥ {threshold:.0%}")
        else:
            checks.append(f"✗ {metric}: {value:.1%} < {threshold:.0%}")
            passed = False
    
    for check in checks:
        print(f"  {check}")
    
    # 8. 保存
    checkpoint_path = "stage8_dataset/stage11a_r1_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'eval_result': eval_result,
    }, checkpoint_path)
    
    print(f"\n✓ R1检查点已保存: {checkpoint_path}")
    print(f"✓ DeepSeek API 调用: {teacher.call_count} 次")
    print(f"✓ 总 token 消耗: {teacher.total_tokens}")
    
    print("\n" + "="*70)
    if passed:
        print("🎉 Stage 11-A-R1 修复训练通过门槛！")
        print("建议: 可以扩展到300条数据")
    else:
        print("⚠ Stage 11-A-R1 未完全通过门槛")
        print("建议: 分析未通过项，考虑进一步修复")
    print("="*70)
    
    return model, trainer, eval_result


if __name__ == "__main__":
    model, trainer, eval_result = run_stage11a_r1_training()
