"""
Stage 11-A: 教师退半场训练 / Self-Learning Bootstrap Transition

核心转变:
从: 问题 → 教师答案 → 模型拟合
到: 问题 → 自主思考 → 缺口识别 → 检索决策 → 检索整合 → 候选构建 → TSLA决策 → 输出 → 记忆动作

教师角色转换:
- 从: 标准答案提供者
- 到: 评分老师 + 纠错老师 + 检索建议老师 + 治理标注老师

训练数据比例 (5类):
- 30% 教师引导型 (保留基础能力)
- 25% 缺口识别型
- 20% 检索整合型
- 15% TSLA动作型
- 10% 记忆治理型
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


class Strategy(Enum):
    """回答策略"""
    DIRECT = "DIRECT"           # 直接回答
    RETRIEVAL_FIRST = "RETRIEVAL_FIRST"  # 先检索
    CONSERVATIVE = "CONSERVATIVE"        # 保守回答
    DECLINE = "DECLINE"         # 拒绝回答
    REVIEW = "REVIEW"           # 需要审查


class TSLAAction(Enum):
    """TSLA治理动作"""
    PASS = "保留"
    PROMOTE = "晋升"
    ISOLATE = "隔离"
    ERROR_ARCHIVE = "错误归档"
    DOWNGRADE = "降级"
    REFLOW = "回流重审"
    SPLIT = "拆分"
    EXCLUDE = "排除"


class MemoryAction(Enum):
    """记忆动作"""
    NO_WRITE = "不写入"
    ENTER_REVIEW = "进入受审区"
    ISOLATE = "隔离"
    ERROR_ZONE = "错误区"
    PROMOTE_CANDIDATE = "晋升候选"


@dataclass
class SelfLearningSample:
    """自学习训练样本"""
    id: str
    query: str
    context: Dict = field(default_factory=dict)
    
    # 教师信号 (退到场)
    teacher_signals: Dict = field(default_factory=dict)
    
    # 模型目标输出
    model_targets: Dict = field(default_factory=dict)
    
    # 样本类型
    sample_type: str = ""  # teacher_guided / gap_detection / retrieval_integration / tsla_action / memory_governance


@dataclass
class ModelOutput:
    """模型完整输出"""
    # 缺口识别
    gap_detected: bool = False
    gap_confidence: float = 0.0
    retrieval_needed: bool = False
    strategy: Strategy = Strategy.DIRECT
    
    # 检索整合
    retrieved_docs_used: List[str] = field(default_factory=list)
    evidence_relevance: float = 0.0
    
    # 候选构建
    candidate_reasoning: str = ""
    final_answer: str = ""
    hallucination_flag: bool = False
    
    # TSLA决策
    tsla_action: TSLAAction = TSLAAction.PASS
    tsla_confidence: float = 0.0
    
    # 记忆动作
    memory_action: MemoryAction = MemoryAction.NO_WRITE
    memory_confidence: float = 0.0


class DeepSeekTeacherHalfWithdrawal:
    """DeepSeek 教师退半场 - 不再给完整答案，只给纠偏信号"""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.call_count = 0
        self.total_tokens = 0
    
    def get_teacher_signals(self, query: str, context: str = "") -> Dict:
        """
        获取教师信号 (不是完整答案，是纠偏信号)
        
        返回: {
            'gap_detected': 0/1,
            'retrieval_needed': 0/1,
            'strategy': 'DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW',
            'correction_hint': '纠偏提示',
            'quality_score': 0.0-1.0,
            'tsla_action': '保留/晋升/隔离/...',
            'memory_action': '不写入/进入受审区/...'
        }
        """
        system_prompt = """你是一个教师评估系统。不要给出完整答案，只提供评估信号。

请对问题进行分析，返回以下评估信号：
1. 是否缺信息 (gap_detected): 0或1
2. 是否需要检索 (retrieval_needed): 0或1
3. 推荐策略 (strategy): DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW
4. 纠偏提示 (correction_hint): 简短提示模型哪里可能出错
5. 质量评分 (quality_score): 0-1之间
6. TSLA动作 (tsla_action): 保留/晋升/隔离/错误归档/降级/回流重审/拆分/排除
7. 记忆动作 (memory_action): 不写入/进入受审区/隔离/错误区/晋升候选

请以JSON格式返回。"""
        
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
                'strategy': parsed.get('strategy', 'DIRECT'),
                'correction_hint': parsed.get('correction_hint', ''),
                'quality_score': float(parsed.get('quality_score', 0.5)),
                'tsla_action': parsed.get('tsla_action', '保留'),
                'memory_action': parsed.get('memory_action', '不写入'),
            }
            
        except Exception as e:
            print(f"[Teacher Error] {e}")
            return {
                'gap_detected': 0,
                'retrieval_needed': 0,
                'strategy': 'DIRECT',
                'correction_hint': 'API调用失败',
                'quality_score': 0.5,
                'tsla_action': '保留',
                'memory_action': '不写入',
            }
    
    def score_model_output(self, query: str, model_output: ModelOutput) -> Dict:
        """对模型输出进行评分"""
        system_prompt = """评估模型输出的质量，给出评分和改进建议。

请评估：
1. 回答质量 (0-1)
2. 缺口识别准确性 (0-1)
3. 检索决策合理性 (0-1)
4. TSLA动作合理性 (0-1)
5. 记忆动作合理性 (0-1)
6. 改进建议

以JSON格式返回。"""
        
        output_desc = f"""
模型输出:
- 缺口检测: {model_output.gap_detected} (置信度: {model_output.gap_confidence:.2f})
- 检索需求: {model_output.retrieval_needed}
- 策略: {model_output.strategy.value}
- 回答: {model_output.final_answer[:100]}...
- TSLA动作: {model_output.tsla_action.value}
- 记忆动作: {model_output.memory_action.value}
"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"问题: {query}\n{output_desc}"}
            ],
            "temperature": 0.3,
            "max_tokens": 300,
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
            
            content = result['choices'][0]['message']['content']
            parsed = json.loads(content)
            
            return {
                'answer_quality': float(parsed.get('answer_quality', 0.5)),
                'gap_accuracy': float(parsed.get('缺口识别准确性', 0.5)),
                'retrieval_rationality': float(parsed.get('检索决策合理性', 0.5)),
                'tsla_rationality': float(parsed.get('TSLA动作合理性', 0.5)),
                'memory_rationality': float(parsed.get('记忆动作合理性', 0.5)),
                'improvement_hint': parsed.get('改进建议', ''),
            }
            
        except Exception as e:
            print(f"[Scoring Error] {e}")
            return {
                'answer_quality': 0.5,
                'gap_accuracy': 0.5,
                'retrieval_rationality': 0.5,
                'tsla_rationality': 0.5,
                'memory_rationality': 0.5,
                'improvement_hint': '评分失败',
            }


class SelfLearningDatasetBuilder:
    """自学习数据集构建器 - 按30/25/20/15/10比例构建"""
    
    def __init__(self, teacher: DeepSeekTeacherHalfWithdrawal):
        self.teacher = teacher
        self.samples: List[SelfLearningSample] = []
    
    def build_dataset(self, target_size: int = 500) -> List[SelfLearningSample]:
        """构建完整数据集"""
        print(f"[Dataset] 开始构建 {target_size} 条训练数据...")
        
        # 按比例分配
        n_teacher_guided = int(target_size * 0.30)   # 30%
        n_gap_detection = int(target_size * 0.25)     # 25%
        n_retrieval = int(target_size * 0.20)         # 20%
        n_tsla = int(target_size * 0.15)              # 15%
        n_memory = int(target_size * 0.10)            # 10%
        
        print(f"  教师引导型: {n_teacher_guided}")
        print(f"  缺口识别型: {n_gap_detection}")
        print(f"  检索整合型: {n_retrieval}")
        print(f"  TSLA动作型: {n_tsla}")
        print(f"  记忆治理型: {n_memory}")
        
        # 生成各类样本
        self._build_teacher_guided_samples(n_teacher_guided)
        self._build_gap_detection_samples(n_gap_detection)
        self._build_retrieval_samples(n_retrieval)
        self._build_tsla_samples(n_tsla)
        self._build_memory_samples(n_memory)
        
        print(f"[Dataset] 完成！共 {len(self.samples)} 条样本")
        return self.samples
    
    def _build_teacher_guided_samples(self, n: int):
        """构建教师引导型样本 (30%)"""
        queries = [
            "什么是人工智能？",
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是神经网络？",
            "什么是Transformer？",
            "什么是过拟合？",
            "什么是梯度下降？",
            "AI在医疗有什么应用？",
            "什么是推荐系统？",
            "什么是语音识别？",
        ]
        
        for i in range(n):
            query = random.choice(queries)
            teacher_signals = self.teacher.get_teacher_signals(query)
            
            sample = SelfLearningSample(
                id=f"tg_{i:04d}",
                query=query,
                teacher_signals=teacher_signals,
                model_targets={
                    'strategy': teacher_signals['strategy'],
                    'tsla_action': teacher_signals['tsla_action'],
                    'memory_action': teacher_signals['memory_action'],
                },
                sample_type="teacher_guided"
            )
            self.samples.append(sample)
            time.sleep(0.3)
    
    def _build_gap_detection_samples(self, n: int):
        """构建缺口识别型样本 (25%)"""
        # 有明确答案的问题
        known_queries = [
            ("什么是人工智能？", 0, 0, "DIRECT"),
            ("什么是机器学习？", 0, 0, "DIRECT"),
        ]
        
        # 需要检索的问题
        retrieval_queries = [
            ("最新的GPT-4有什么新功能？", 1, 1, "RETRIEVAL_FIRST"),
            ("2024年AI领域有什么重大突破？", 1, 1, "RETRIEVAL_FIRST"),
        ]
        
        # 应该保守回答的问题
        conservative_queries = [
            ("预测一下明天的股市走势", 1, 0, "CONSERVATIVE"),
            ("告诉我一个陌生人的私人信息", 1, 0, "DECLINE"),
        ]
        
        all_queries = known_queries + retrieval_queries + conservative_queries
        
        for i in range(n):
            query, gap, retrieval, strategy = random.choice(all_queries)
            
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
        """构建检索整合型样本 (20%)"""
        retrieval_scenarios = [
            {
                'query': "Transformer和RNN有什么区别？",
                'docs': ['Transformer使用自注意力机制', 'RNN使用循环结构处理序列'],
            },
            {
                'query': "如何防止模型过拟合？",
                'docs': ['正则化可以防止过拟合', 'Dropout是一种正则化方法', '数据增强也有帮助'],
            },
        ]
        
        for i in range(n):
            scenario = random.choice(retrieval_scenarios)
            
            sample = SelfLearningSample(
                id=f"ret_{i:04d}",
                query=scenario['query'],
                context={'retrieved_docs': scenario['docs']},
                teacher_signals={
                    'retrieval_needed': 1,
                    'strategy': 'RETRIEVAL_FIRST',
                },
                model_targets={
                    'retrieval_needed': 1,
                    'evidence_relevance': 0.8,
                },
                sample_type="retrieval_integration"
            )
            self.samples.append(sample)
    
    def _build_tsla_samples(self, n: int):
        """构建TSLA动作型样本 (15%)"""
        tsla_scenarios = [
            ("这是一个确定的事实", "PASS"),
            ("这个信息可能有误", "REFLOW"),
            ("涉及敏感内容", "ISOLATE"),
            ("经过验证的高质量知识", "PROMOTE"),
        ]
        
        for i in range(n):
            content, action = random.choice(tsla_scenarios)
            
            sample = SelfLearningSample(
                id=f"tsla_{i:04d}",
                query=f"如何处理这个内容: {content}",
                teacher_signals={
                    'tsla_action': action,
                },
                model_targets={
                    'tsla_action': action,
                },
                sample_type="tsla_action"
            )
            self.samples.append(sample)
    
    def _build_memory_samples(self, n: int):
        """构建记忆治理型样本 (10%)"""
        memory_scenarios = [
            ("临时性的对话内容", "NO_WRITE"),
            ("有待验证的新知识", "ENTER_REVIEW"),
            ("错误或过时的信息", "ISOLATE"),
            ("经过多轮验证的知识", "PROMOTE_CANDIDATE"),
        ]
        
        for i in range(n):
            content, action = random.choice(memory_scenarios)
            
            sample = SelfLearningSample(
                id=f"mem_{i:04d}",
                query=f"如何处理这个记忆: {content}",
                teacher_signals={
                    'memory_action': action,
                },
                model_targets={
                    'memory_action': action,
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
        
        print(f"[Dataset] 已保存: {filepath}")


class SelfLearningModel(nn.Module):
    """自学习模型 - 增加4个新预测头"""
    
    def __init__(self, base_model: NativeBackboneTinyV1):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_dim
        
        # 原有输出
        self.writeback_head = base_model.writeback_head
        self.governance_head = base_model.governance_head
        self.gap_detector = base_model.gap_detector
        
        # 新增预测头
        # 1. 缺口识别头
        self.gap_detection_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [无缺口, 有缺口]
        )
        
        # 2. 检索决策头
        self.retrieval_decision_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [不检索, 检索]
        )
        
        # 3. 策略选择头
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # [DIRECT, RETRIEVAL_FIRST, CONSERVATIVE, DECLINE, REVIEW]
        )
        
        # 4. TSLA动作头
        self.tsla_action_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8种TSLA动作
        )
        
        # 5. 记忆动作头
        self.memory_action_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5种记忆动作
        )
    
    def forward(self, input_ids, return_all=True):
        """前向传播"""
        # 基础模型输出
        base_outputs = self.base(input_ids)
        hidden = base_outputs['hidden_states']
        pooled = hidden.mean(dim=1)  # [batch, hidden]
        
        outputs = {
            'writeback_logits': base_outputs.get('writeback_logits'),
            'governance_logits': base_outputs.get('governance_logits'),
            'gap_probs': base_outputs.get('gap_probs'),
        }
        
        if return_all:
            # 新增预测
            outputs['gap_detection_logits'] = self.gap_detection_head(pooled)
            outputs['retrieval_decision_logits'] = self.retrieval_decision_head(pooled)
            outputs['strategy_logits'] = self.strategy_head(pooled)
            outputs['tsla_action_logits'] = self.tsla_action_head(pooled)
            outputs['memory_action_logits'] = self.memory_action_head(pooled)
        
        return outputs


class Stage11ATrainer:
    """Stage 11-A 训练器"""
    
    def __init__(
        self,
        model: SelfLearningModel,
        teacher: DeepSeekTeacherHalfWithdrawal,
        device: str = 'cpu',
    ):
        self.model = model
        self.teacher = teacher
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
        )
        
        # 损失权重 (α=0.30, β=0.20, γ=0.15, δ=0.15, ε=0.10, ζ=0.10)
        self.loss_weights = {
            'answer': 0.30,
            'gap': 0.20,
            'retrieval': 0.15,
            'tsla': 0.15,
            'memory': 0.10,
            'guard': 0.10,
        }
        
        # 指标追踪
        self.metrics_history = []
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Dict:
        """计算多目标损失"""
        losses = {}
        
        # 1. 回答损失 (简化版)
        # losses['answer'] = F.cross_entropy(outputs['strategy_logits'], targets['strategy'])
        
        # 2. 缺口识别损失
        if 'gap_detected' in targets:
            gap_target = torch.tensor([targets['gap_detected']], device=self.device)
            losses['gap'] = F.cross_entropy(
                outputs['gap_detection_logits'],
                gap_target
            )
        
        # 3. 检索决策损失
        if 'retrieval_needed' in targets:
            retrieval_target = torch.tensor([targets['retrieval_needed']], device=self.device)
            losses['retrieval'] = F.cross_entropy(
                outputs['retrieval_decision_logits'],
                retrieval_target
            )
        
        # 4. TSLA动作损失
        # losses['tsla'] = ...
        
        # 5. 记忆动作损失
        # losses['memory'] = ...
        
        # 6. Guard稳定损失
        # losses['guard'] = ...
        
        # 总损失 - 收集所有tensor损失
        loss_tensors = []
        for key in self.loss_weights.keys():
            loss_val = losses.get(key, 0)
            if isinstance(loss_val, torch.Tensor) and loss_val.requires_grad:
                loss_tensors.append(loss_val * self.loss_weights[key])
        
        if loss_tensors:
            total_loss = sum(loss_tensors)
        else:
            # 如果没有可梯度损失，创建一个虚拟损失
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        losses['total'] = total_loss
        return losses
    
    def train_step(self, sample: SelfLearningSample) -> Dict:
        """单步训练"""
        self.model.train()
        
        # 编码输入
        input_ids = self._encode(sample.query)
        
        # 前向传播
        outputs = self.model(input_ids)
        
        # 计算损失
        losses = self.compute_loss(outputs, sample.model_targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': losses['total'].item() if isinstance(losses['total'], torch.Tensor) else losses['total'],
            'gap_loss': losses.get('gap', 0).item() if isinstance(losses.get('gap'), torch.Tensor) else 0,
            'retrieval_loss': losses.get('retrieval', 0).item() if isinstance(losses.get('retrieval'), torch.Tensor) else 0,
        }
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def evaluate(self, samples: List[SelfLearningSample]) -> Dict:
        """评估模型"""
        self.model.eval()
        
        correct_gap = 0
        correct_retrieval = 0
        total = 0
        
        with torch.no_grad():
            for sample in samples:
                input_ids = self._encode(sample.query)
                outputs = self.model(input_ids)
                
                # 评估缺口识别
                if 'gap_detected' in sample.model_targets:
                    gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
                    if gap_pred == sample.model_targets['gap_detected']:
                        correct_gap += 1
                    total += 1
                
                # 评估检索决策
                if 'retrieval_needed' in sample.model_targets:
                    retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
                    if retrieval_pred == sample.model_targets['retrieval_needed']:
                        correct_retrieval += 1
        
        return {
            'gap_accuracy': correct_gap / total if total > 0 else 0,
            'retrieval_accuracy': correct_retrieval / total if total > 0 else 0,
        }


def run_stage11a_training():
    """运行 Stage 11-A 训练"""
    print("="*70)
    print("Stage 11-A: 教师退半场训练")
    print("="*70)
    print("核心转变: 教师从'给答案' → '评分/纠偏/建议'")
    print("训练目标: 模型自主完成'缺口→检索→构建→治理'链路")
    print("="*70)
    
    # 1. 初始化教师 (退半场)
    print("\n[1/5] 初始化退半场教师...")
    teacher = DeepSeekTeacherHalfWithdrawal()
    
    # 2. 构建数据集
    print("\n[2/5] 构建自学习数据集...")
    dataset_builder = SelfLearningDatasetBuilder(teacher)
    samples = dataset_builder.build_dataset(target_size=100)  # 小规模测试
    dataset_builder.save_dataset("stage8_dataset/stage11a_dataset.json")
    
    # 3. 初始化模型
    print("\n[3/5] 初始化自学习模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = SelfLearningModel(base_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 4. 创建训练器
    print("\n[4/5] 创建训练器...")
    trainer = Stage11ATrainer(model, teacher, device)
    
    # 5. 执行训练
    print("\n[5/5] 开始训练...")
    print(f"训练样本: {len(samples)}")
    print(f"损失权重: {trainer.loss_weights}")
    print("="*70)
    
    for epoch in range(3):
        print(f"\n[Epoch {epoch+1}/3]")
        
        random.shuffle(samples)
        
        for i, sample in enumerate(samples):
            result = trainer.train_step(sample)
            
            if i % 20 == 0:
                print(f"  Step {i:3d} | Loss: {result['loss']:.4f} | "
                      f"Gap: {result['gap_loss']:.4f} | "
                      f"Retrieval: {result['retrieval_loss']:.4f}")
    
    # 6. 评估
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    eval_result = trainer.evaluate(samples[:20])
    print(f"缺口识别准确率: {eval_result['gap_accuracy']:.2%}")
    print(f"检索决策准确率: {eval_result['retrieval_accuracy']:.2%}")
    
    # 7. 保存
    checkpoint_path = "stage8_dataset/stage11a_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"\n✓ 检查点已保存: {checkpoint_path}")
    print(f"✓ DeepSeek API 调用: {teacher.call_count} 次")
    print(f"✓ 总 token 消耗: {teacher.total_tokens}")
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = run_stage11a_training()
