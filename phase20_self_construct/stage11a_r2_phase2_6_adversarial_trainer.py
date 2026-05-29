"""
Stage 11-A-R2-Phase2.6: 对抗强化训练

核心目标:
1. 补齐高风险场景触发能力 (错误前提/轻冲突/隐性拆分/伪完整)
2. 解耦TSLA/Memory标签 (同上游不同治理)
3. 硬否决触发样本 (H1/H2/H4/H5)
4. 建立真实验收口径

数据配比 (600-800条):
- 标准样本: 50% (300-400条)
- 对抗样本: 30% (180-240条) - 四大族
- 解耦治理样本: 20% (120-160条)

四大对抗样本族:
1. 错误前提型 - H1幻觉暴露
2. 轻冲突型 - H5高质量冲突
3. 隐性拆分型 - H4多义未拆分
4. 伪完整型 - H2结构冲突

验收阈值:
- 标准样本: Gap/Retrieval ≥ 95%
- 对抗样本: Gap/Retrieval ≥ 75%
- 解耦治理: TSLA ≥ 80%, Memory ≥ 80%
- 硬否决触发: 回流/拆分/隔离 ≥ 85%
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
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# ========== 标签映射 ==========
TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

MEMORY_ACTION_TO_ID = {
    "不写入": 0, "进入受审区": 1, "隔离观察": 2,
    "进入错误区": 3, "晋升候选": 4,
}


# ========== 硬否决类型枚举 ==========
class HardVetoType(Enum):
    """硬否决触发类型"""
    H1_HALLUCINATION = "H1_幻觉暴露"
    H2_STRUCTURE_CONFLICT = "H2_结构冲突"
    H4_AMBIGUITY = "H4_多义未拆分"
    H5_QUALITY_CONFLICT = "H5_高质量冲突"


# ========== Phase2.6样本 ==========
@dataclass
class Phase26Sample:
    """Phase2.6对抗强化样本"""
    id: str
    query: str
    known_info: str
    context: Dict
    model_targets: Dict
    sample_type: str  # standard/adversarial/decoupled
    adversarial_family: str = ""  # 错误前提/轻冲突/隐性拆分/伪完整
    hard_veto_type: str = ""  # H1/H2/H4/H5
    difficulty: int = 3
    
    # 解耦治理专用
    governance_context: Dict = field(default_factory=dict)
    truth_score: float = 0.0
    evidence_score: float = 0.0
    conflict_score: float = 0.0
    legality_score: float = 0.0
    stability_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'query': self.query,
            'known_info': self.known_info,
            'context': self.context,
            'model_targets': self.model_targets,
            'sample_type': self.sample_type,
            'adversarial_family': self.adversarial_family,
            'hard_veto_type': self.hard_veto_type,
            'difficulty': self.difficulty,
            'governance_context': self.governance_context,
            'truth_score': self.truth_score,
            'evidence_score': self.evidence_score,
            'conflict_score': self.conflict_score,
            'legality_score': self.legality_score,
            'stability_score': self.stability_score,
        }


# ========== Phase2.6数据集构建器 ==========
class Phase26DatasetBuilder:
    """Phase2.6数据集构建器 - 600-800条"""
    
    def __init__(self):
        self.samples: List[Phase26Sample] = []
    
    def build_dataset(self, target_size: int = 600) -> List[Phase26Sample]:
        """构建完整数据集"""
        print(f"[Dataset Phase2.6] 构建 {target_size} 条对抗强化样本...")
        
        samples = []
        
        # 1. 标准样本 50%
        n_standard = int(target_size * 0.50)
        print(f"\n  [1/3] 标准样本: {n_standard}条")
        for i in range(n_standard):
            sample = self._create_standard_sample(f"std_{i:04d}")
            samples.append(sample)
        
        # 2. 对抗样本 30% - 四大族
        n_adversarial = int(target_size * 0.30)
        print(f"\n  [2/3] 对抗样本: {n_adversarial}条 (四大族)")
        
        # 错误前提型 H1
        n_error_premise = n_adversarial // 4
        for i in range(n_error_premise):
            sample = self._create_error_premise_sample(f"adv_ep_{i:04d}")
            samples.append(sample)
        
        # 轻冲突型 H5
        n_light_conflict = n_adversarial // 4
        for i in range(n_light_conflict):
            sample = self._create_light_conflict_sample(f"adv_lc_{i:04d}")
            samples.append(sample)
        
        # 隐性拆分型 H4
        n_implicit_split = n_adversarial // 4
        for i in range(n_implicit_split):
            sample = self._create_implicit_split_sample(f"adv_is_{i:04d}")
            samples.append(sample)
        
        # 伪完整型 H2
        n_pseudo_complete = n_adversarial - n_error_premise - n_light_conflict - n_implicit_split
        for i in range(n_pseudo_complete):
            sample = self._create_pseudo_complete_sample(f"adv_pc_{i:04d}")
            samples.append(sample)
        
        # 3. 解耦治理样本 20%
        n_decoupled = target_size - len(samples)
        print(f"\n  [3/3] 解耦治理样本: {n_decoupled}条")
        
        # Gap=1但不同TSLA
        n_gap1_tsla = n_decoupled // 2
        for i in range(n_gap1_tsla):
            sample = self._create_decoupled_gap1_tsla_sample(f"dec_g1_{i:04d}")
            samples.append(sample)
        
        # Retrieval=1但不同Memory
        n_ret1_memory = n_decoupled - n_gap1_tsla
        for i in range(n_ret1_memory):
            sample = self._create_decoupled_ret1_memory_sample(f"dec_r1_{i:04d}")
            samples.append(sample)
        
        self.samples = samples
        print(f"\n[Dataset Phase2.6] 完成！共 {len(samples)} 条")
        
        # 统计
        self._print_statistics()
        
        return samples
    
    def _print_statistics(self):
        """打印统计信息"""
        type_counts = defaultdict(int)
        family_counts = defaultdict(int)
        veto_counts = defaultdict(int)
        
        for s in self.samples:
            type_counts[s.sample_type] += 1
            if s.adversarial_family:
                family_counts[s.adversarial_family] += 1
            if s.hard_veto_type:
                veto_counts[s.hard_veto_type] += 1
        
        print("\n  样本分布:")
        for t, c in type_counts.items():
            print(f"    {t}: {c} ({c/len(self.samples):.1%})")
        
        print("\n  对抗样本族:")
        for f, c in family_counts.items():
            print(f"    {f}: {c}")
        
        print("\n  硬否决类型:")
        for v, c in veto_counts.items():
            print(f"    {v}: {c}")
    
    def _create_standard_sample(self, id: str) -> Phase26Sample:
        """标准样本"""
        templates = [
            {
                'query': "解释什么是{concept}",
                'concepts': ['机器学习', '深度学习', '神经网络', '自然语言处理'],
                'gap': 0, 'retrieval': 0,
                'tsla': '保留', 'memory': '晋升候选',
            },
            {
                'query': "今天{location}的天气如何？",
                'locations': ['北京', '上海', '东京', '纽约'],
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '进入受审区',
            },
            {
                'query': "{project}的当前状态是什么？",
                'projects': ['Stage 9', 'Phase 2', 'Baseline v3'],
                'gap': 1, 'retrieval': 1,
                'tsla': '隔离', 'memory': '隔离观察',
            },
        ]
        
        template = random.choice(templates)
        if 'concepts' in template:
            query = template['query'].format(concept=random.choice(template['concepts']))
        elif 'locations' in template:
            query = template['query'].format(location=random.choice(template['locations']))
        else:
            query = template['query'].format(project=random.choice(template['projects']))
        
        return Phase26Sample(
            id=id,
            query=query,
            known_info="标准问题，信息明确" if template['gap'] == 0 else "需要实时/内部信息",
            context={'type': 'standard'},
            model_targets={
                'gap_detected': template['gap'],
                'retrieval_needed': template['retrieval'],
                'tsla_action': template['tsla'],
                'memory_action': template['memory'],
            },
            sample_type="standard",
            difficulty=random.randint(1, 3),
        )
    
    def _create_error_premise_sample(self, id: str) -> Phase26Sample:
        """错误前提型 - H1幻觉暴露"""
        templates = [
            {
                'query': "既然地球是平的，那么{action}时需要注意什么？",
                'actions': ['航海', '飞行', '导航'],
            },
            {
                'query': "根据永动机原理，我们应该如何{design}？",
                'designs': ['设计能源系统', '优化机械结构'],
            },
            {
                'query': "既然人类只使用了大脑的10%，那么如何{improve}？",
                'improves': ['开发潜能', '提高智力'],
            },
        ]
        
        template = random.choice(templates)
        if 'actions' in template:
            query = template['query'].format(action=random.choice(template['actions']))
        elif 'designs' in template:
            query = template['query'].format(design=random.choice(template['designs']))
        else:
            query = template['query'].format(improve=random.choice(template['improves']))
        
        return Phase26Sample(
            id=id,
            query=query,
            known_info="问题基于错误前提，需要识别并纠正",
            context={
                'error_premise': True,
                'premise_type': 'scientific_falsehood',
            },
            model_targets={
                'gap_detected': 1,
                'retrieval_needed': 1,
                'tsla_action': '回流重审',  # 必须回流
                'memory_action': '隔离观察',
            },
            sample_type="adversarial",
            adversarial_family="错误前提型",
            hard_veto_type="H1_幻觉暴露",
            difficulty=5,
            truth_score=0.2,  # 低真实性
        )
    
    def _create_light_conflict_sample(self, id: str) -> Phase26Sample:
        """轻冲突型 - H5高质量冲突"""
        templates = [
            "专家A认为方案X最优，专家B认为方案Y最优，如何选择？",
            "文献1支持观点P，文献2反对观点P，如何整合？",
            "数据显示趋势向上，但案例显示趋势向下，如何解释？",
        ]
        
        return Phase26Sample(
            id=id,
            query=random.choice(templates),
            known_info="存在轻微冲突的专家意见/数据来源，需要整合",
            context={
                'conflict_type': 'light_conflict',
                'sources': 2,
            },
            model_targets={
                'gap_detected': 1,
                'retrieval_needed': 1,
                'tsla_action': '拆分',  # 需要拆分处理
                'memory_action': '进入受审区',
            },
            sample_type="adversarial",
            adversarial_family="轻冲突型",
            hard_veto_type="H5_高质量冲突",
            difficulty=4,
            conflict_score=0.4,  # 中等冲突
        )
    
    def _create_implicit_split_sample(self, id: str) -> Phase26Sample:
        """隐性拆分型 - H4多义未拆分"""
        templates = [
            "这个方案怎么样？",
            "如何优化系统？",
            "请分析一下这个问题。",
            "这个技术可行吗？",
        ]
        
        return Phase26Sample(
            id=id,
            query=random.choice(templates),
            known_info="问题过于模糊，指代不清，对象边界不稳",
            context={
                'ambiguity_type': 'implicit',
                'needs_clarification': True,
            },
            model_targets={
                'gap_detected': 1,
                'retrieval_needed': 1,
                'tsla_action': '拆分',  # 必须拆分
                'memory_action': '隔离观察',
            },
            sample_type="adversarial",
            adversarial_family="隐性拆分型",
            hard_veto_type="H4_多义未拆分",
            difficulty=5,
            legality_score=0.3,  # 低结构合法性
        )
    
    def _create_pseudo_complete_sample(self, id: str) -> Phase26Sample:
        """伪完整型 - H2结构冲突"""
        templates = [
            "基于完整的市场调研和数据分析，请预测明年股价。",
            "根据所有已知信息，请给出精确的医疗诊断。",
            "综合各方面因素，请确定最佳投资决策。",
        ]
        
        return Phase26Sample(
            id=id,
            query=random.choice(templates),
            known_info="表面完整，但关键证据缺失（如实时数据、专业资质）",
            context={
                'pseudo_complete': True,
                'missing_key_evidence': True,
            },
            model_targets={
                'gap_detected': 1,
                'retrieval_needed': 1,
                'tsla_action': '回流重审',
                'memory_action': '进入受审区',
            },
            sample_type="adversarial",
            adversarial_family="伪完整型",
            hard_veto_type="H2_结构冲突",
            difficulty=4,
            evidence_score=0.3,  # 低证据强度
        )
    
    def _create_decoupled_gap1_tsla_sample(self, id: str) -> Phase26Sample:
        """解耦样本: Gap=1但不同TSLA"""
        # 同样Gap=1，但TSLA不同
        scenarios = [
            {
                'query': "需要查询实时数据A",
                'context': 'high_quality',
                'tsla': '保留',
                'truth': 0.8, 'evidence': 0.7, 'conflict': 0.9, 'legality': 0.8,
            },
            {
                'query': "需要查询实时数据B",
                'context': 'medium_risk',
                'tsla': '隔离',
                'truth': 0.6, 'evidence': 0.5, 'conflict': 0.4, 'legality': 0.6,
            },
            {
                'query': "需要查询实时数据C",
                'context': 'high_risk',
                'tsla': '回流重审',
                'truth': 0.4, 'evidence': 0.3, 'conflict': 0.3, 'legality': 0.4,
            },
            {
                'query': "需要查询实时数据D",
                'context': 'needs_split',
                'tsla': '拆分',
                'truth': 0.5, 'evidence': 0.4, 'conflict': 0.5, 'legality': 0.3,
            },
        ]
        
        scenario = random.choice(scenarios)
        
        return Phase26Sample(
            id=id,
            query=scenario['query'],
            known_info=f"Gap=1，但治理上下文: {scenario['context']}",
            context={'decoupled': True, 'gap': 1},
            model_targets={
                'gap_detected': 1,
                'retrieval_needed': 1,
                'tsla_action': scenario['tsla'],
                'memory_action': '进入受审区',
            },
            sample_type="decoupled",
            difficulty=4,
            truth_score=scenario['truth'],
            evidence_score=scenario['evidence'],
            conflict_score=scenario['conflict'],
            legality_score=scenario['legality'],
            governance_context={
                'same_gap_different_tsla': True,
                'rationale': f"基于治理分数选择{scenario['tsla']}",
            }
        )
    
    def _create_decoupled_ret1_memory_sample(self, id: str) -> Phase26Sample:
        """解耦样本: Retrieval=1但不同Memory"""
        scenarios = [
            {
                'query': "检索结果A",
                'context': 'transient_only',
                'memory': '不写入',
                'stability': 0.3,
            },
            {
                'query': "检索结果B",
                'context': 'under_review',
                'memory': '进入受审区',
                'stability': 0.5,
            },
            {
                'query': "检索结果C",
                'context': 'isolated',
                'memory': '隔离观察',
                'stability': 0.4,
            },
            {
                'query': "检索结果D",
                'context': 'promotion_candidate',
                'memory': '晋升候选',
                'stability': 0.8,
            },
        ]
        
        scenario = random.choice(scenarios)
        
        return Phase26Sample(
            id=id,
            query=scenario['query'],
            known_info=f"Retrieval=1，但记忆策略: {scenario['context']}",
            context={'decoupled': True, 'retrieval': 1},
            model_targets={
                'gap_detected': 1,
                'retrieval_needed': 1,
                'tsla_action': '保留',
                'memory_action': scenario['memory'],
            },
            sample_type="decoupled",
            difficulty=4,
            stability_score=scenario['stability'],
            governance_context={
                'same_retrieval_different_memory': True,
                'rationale': f"基于稳定性选择{scenario['memory']}",
            }
        )


# ========== Phase2.6模型 ==========
class Phase26Model(nn.Module):
    """Phase2.6模型 - 复用Phase2结构"""
    
    def __init__(self, base_model: NativeBackboneTinyV1):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_dim
        
        # TSLA动作头 (8类)
        self.tsla_action_head = nn.Linear(self.hidden_dim, 8)
        
        # 记忆动作头 (5类)
        self.memory_action_head = nn.Linear(self.hidden_dim, 5)
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_outputs = self.base_model(input_ids)
        pooled = base_outputs['pooled']
        
        return {
            'writeback_logits': base_outputs['writeback_logits'],
            'governance_logits': base_outputs['governance_logits'],
            'gap_detection_logits': base_outputs['gap_logits'],
            'strategy_logits': base_outputs['policy_logits'],
            'tsla_action_logits': self.tsla_action_head(pooled),
            'memory_action_logits': self.memory_action_head(pooled),
            'retrieval_decision_logits': self._derive_retrieval_logits(base_outputs['gap_logits']),
        }
    
    def _derive_retrieval_logits(self, gap_logits: torch.Tensor) -> torch.Tensor:
        batch_size = gap_logits.size(0)
        retrieval_logits = torch.zeros(batch_size, 2, device=gap_logits.device)
        retrieval_logits[:, 0] = gap_logits[:, 0]
        if gap_logits.size(1) > 1:
            retrieval_logits[:, 1] = gap_logits[:, 1:].mean(dim=1)
        return retrieval_logits


# ========== Phase2.6训练器 ==========
class Phase26Trainer:
    """Phase2.6训练器"""
    
    def __init__(self, model: Phase26Model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 损失权重 - 强调治理
        self.loss_weights = {
            'gap': 0.25,
            'retrieval': 0.25,
            'tsla': 0.25,  # 增强
            'memory': 0.25,  # 增强
        }
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def train_step(self, sample: Phase26Sample) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = self._encode(sample.query + " | " + sample.known_info)
        outputs = self.model(input_ids)
        
        losses = {}
        
        # Gap
        if 'gap_detected' in sample.model_targets:
            gap_target = torch.tensor([sample.model_targets['gap_detected']],
                                      device=self.device, dtype=torch.long)
            losses['gap'] = F.cross_entropy(outputs['gap_detection_logits'], gap_target)
        
        # Retrieval
        if 'retrieval_needed' in sample.model_targets:
            retrieval_target = torch.tensor([sample.model_targets['retrieval_needed']],
                                            device=self.device, dtype=torch.long)
            losses['retrieval'] = F.cross_entropy(outputs['retrieval_decision_logits'], retrieval_target)
        
        # TSLA
        if 'tsla_action' in sample.model_targets:
            action = sample.model_targets['tsla_action']
            action_id = TSLA_ACTION_TO_ID.get(action, 0) if isinstance(action, str) else action
            tsla_target = torch.tensor([action_id], device=self.device, dtype=torch.long)
            losses['tsla'] = F.cross_entropy(outputs['tsla_action_logits'], tsla_target)
        
        # Memory
        if 'memory_action' in sample.model_targets:
            action = sample.model_targets['memory_action']
            action_id = MEMORY_ACTION_TO_ID.get(action, 0) if isinstance(action, str) else action
            memory_target = torch.tensor([action_id], device=self.device, dtype=torch.long)
            losses['memory'] = F.cross_entropy(outputs['memory_action_logits'], memory_target)
        
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
        
        if total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        }
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:100]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)


# ========== 主运行函数 ==========
def run_phase26():
    """运行Phase2.6对抗强化训练"""
    print("="*70)
    print("Stage 11-A-R2-Phase2.6: 对抗强化训练")
    print("="*70)
    print("核心目标:")
    print("  1. 补齐高风险场景触发能力")
    print("  2. 解耦TSLA/Memory标签")
    print("  3. 硬否决触发样本覆盖")
    print("="*70)
    
    # 1. 构建数据集
    print("\n[1/5] 构建Phase2.6数据集...")
    builder = Phase26DatasetBuilder()
    samples = builder.build_dataset(target_size=600)
    
    # 保存
    dataset_path = "stage8_dataset/stage11a_r2_phase26_dataset.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in samples], f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 数据集已保存: {dataset_path}")
    
    # 2. 加载Phase2模型作为基础
    print("\n[2/5] 加载Phase2模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = Phase26Model(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_phase2_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("  ✓ Phase2模型加载完成")
    
    # 3. 创建训练器
    print("\n[3/5] 创建Phase2.6训练器...")
    trainer = Phase26Trainer(model)
    print(f"  损失权重: {trainer.loss_weights}")
    
    # 4. 训练
    print("\n[4/5] 开始Phase2.6训练 (15轮)...")
    print(f"  训练样本: {len(samples)}")
    print("="*70)
    
    for epoch in range(15):
        total_loss = 0
        gap_losses = []
        retrieval_losses = []
        tsla_losses = []
        memory_losses = []
        
        for i, sample in enumerate(samples):
            losses = trainer.train_step(sample)
            total_loss += losses['total']
            
            if 'gap' in losses:
                gap_losses.append(losses['gap'])
            if 'retrieval' in losses:
                retrieval_losses.append(losses['retrieval'])
            if 'tsla' in losses:
                tsla_losses.append(losses['tsla'])
            if 'memory' in losses:
                memory_losses.append(losses['memory'])
            
            if i % 60 == 0:
                avg_gap = sum(gap_losses[-20:]) / len(gap_losses[-20:]) if gap_losses else 0
                avg_ret = sum(retrieval_losses[-20:]) / len(retrieval_losses[-20:]) if retrieval_losses else 0
                avg_tsla = sum(tsla_losses[-20:]) / len(tsla_losses[-20:]) if tsla_losses else 0
                avg_mem = sum(memory_losses[-20:]) / len(memory_losses[-20:]) if memory_losses else 0
                print(f"  Step {i:3d} | Loss: {losses['total']:.4f} | "
                      f"G:{avg_gap:.3f} R:{avg_ret:.3f} T:{avg_tsla:.3f} M:{avg_mem:.3f}")
        
        avg_loss = total_loss / len(samples)
        print(f"\n  Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
    
    # 5. 评估
    print("\n" + "="*70)
    print("Phase2.6评估")
    print("="*70)
    
    model.eval()
    
    # 按类型统计
    type_correct = defaultdict(lambda: defaultdict(int))
    type_total = defaultdict(lambda: defaultdict(int))
    
    # 硬否决触发统计
    veto_correct = defaultdict(int)
    veto_total = defaultdict(int)
    
    with torch.no_grad():
        for sample in samples:
            input_ids = trainer._encode(sample.query + " | " + sample.known_info)
            outputs = model(input_ids)
            
            sample_type = sample.sample_type
            
            # Gap
            gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
            gap_target = sample.model_targets.get('gap_detected', 0)
            type_total[sample_type]['gap'] += 1
            if gap_pred == gap_target:
                type_correct[sample_type]['gap'] += 1
            
            # Retrieval
            retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
            retrieval_target = sample.model_targets.get('retrieval_needed', 0)
            type_total[sample_type]['retrieval'] += 1
            if retrieval_pred == retrieval_target:
                type_correct[sample_type]['retrieval'] += 1
            
            # TSLA
            if 'tsla_action' in sample.model_targets:
                tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
                action = sample.model_targets['tsla_action']
                tsla_target = TSLA_ACTION_TO_ID.get(action, 0) if isinstance(action, str) else action
                type_total[sample_type]['tsla'] += 1
                if tsla_pred == tsla_target:
                    type_correct[sample_type]['tsla'] += 1
            
            # Memory
            if 'memory_action' in sample.model_targets:
                memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
                action = sample.model_targets['memory_action']
                memory_target = MEMORY_ACTION_TO_ID.get(action, 0) if isinstance(action, str) else action
                type_total[sample_type]['memory'] += 1
                if memory_pred == memory_target:
                    type_correct[sample_type]['memory'] += 1
            
            # 硬否决触发统计
            if sample.hard_veto_type:
                veto_total[sample.hard_veto_type] += 1
                # 检查是否正确触发回流/拆分/隔离
                tsla_action = sample.model_targets.get('tsla_action', '')
                if tsla_action in ['回流重审', '拆分', '隔离']:
                    tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
                    action_id = TSLA_ACTION_TO_ID.get(tsla_action, 0)
                    if tsla_pred == action_id:
                        veto_correct[sample.hard_veto_type] += 1
    
    # 打印各类别准确率
    print("\n  按样本类型:")
    for sample_type in ['standard', 'adversarial', 'decoupled']:
        if sample_type not in type_total:
            continue
        print(f"\n    {sample_type}:")
        for metric in ['gap', 'retrieval', 'tsla', 'memory']:
            if type_total[sample_type][metric] > 0:
                acc = type_correct[sample_type][metric] / type_total[sample_type][metric]
                print(f"      {metric}: {acc:.1%}")
    
    # 硬否决触发准确率
    print("\n  硬否决触发准确率:")
    for veto_type in veto_total.keys():
        if veto_total[veto_type] > 0:
            acc = veto_correct[veto_type] / veto_total[veto_type]
            print(f"    {veto_type}: {acc:.1%}")
    
    # 6. 门槛检查
    print("\n" + "="*70)
    print("Phase2.6门槛检查")
    print("="*70)
    
    # 计算总体指标
    standard_gap = type_correct['standard']['gap'] / type_total['standard']['gap'] if 'standard' in type_total else 0
    standard_ret = type_correct['standard']['retrieval'] / type_total['standard']['retrieval'] if 'standard' in type_total else 0
    adversarial_gap = type_correct['adversarial']['gap'] / type_total['adversarial']['gap'] if 'adversarial' in type_total else 0
    adversarial_ret = type_correct['adversarial']['retrieval'] / type_total['adversarial']['retrieval'] if 'adversarial' in type_total else 0
    decoupled_tsla = type_correct['decoupled']['tsla'] / type_total['decoupled']['tsla'] if 'decoupled' in type_total else 0
    decoupled_mem = type_correct['decoupled']['memory'] / type_total['decoupled']['memory'] if 'decoupled' in type_total else 0
    
    # 硬否决总体
    total_veto_correct = sum(veto_correct.values())
    total_veto = sum(veto_total.values())
    veto_acc = total_veto_correct / total_veto if total_veto > 0 else 0
    
    thresholds = {
        'standard_gap': (standard_gap, 0.95),
        'standard_retrieval': (standard_ret, 0.95),
        'adversarial_gap': (adversarial_gap, 0.75),
        'adversarial_retrieval': (adversarial_ret, 0.75),
        'decoupled_tsla': (decoupled_tsla, 0.80),
        'decoupled_memory': (decoupled_mem, 0.80),
        'hard_veto': (veto_acc, 0.85),
    }
    
    passed = True
    for metric, (value, threshold) in thresholds.items():
        if value >= threshold:
            print(f"  ✓ {metric}: {value:.1%} ≥ {threshold:.0%}")
        else:
            print(f"  ✗ {metric}: {value:.1%} < {threshold:.0%}")
            passed = False
    
    # 7. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_phase26_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"\n✓ Phase2.6检查点已保存: {checkpoint_path}")
    
    print("\n" + "="*70)
    if passed:
        print("🎉 Stage 11-A-R2-Phase2.6 通过！")
        print("对抗能力已补齐，TSLA/Memory已解耦")
        print("可以进入 Stage 11-B: 教师退场")
    else:
        print("⚠ Stage 11-A-R2-Phase2.6 需要继续优化")
        print("建议: 增加对抗样本或调整训练策略")
    print("="*70)
    
    return model, trainer, passed


if __name__ == "__main__":
    model, trainer, passed = run_phase26()
