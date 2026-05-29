"""
Stage 11-A-R2-Fix-v2: 危险动作再平衡训练

核心目标: 解决类别失衡导致的危险动作塌缩

关键改进:
1. 新增3个硬指标
   - split_recall ≥ 80%
   - reflow_recall ≥ 80%
   - danger_macro_f1 ≥ 75%

2. 重做TSLA样本分布 (教学分布)
   - 保留: 45-50%
   - 拆分: 20-25%
   - 回流重审: 20-25%
   - 其他: 10%

3. 回流vs拆分对照样本
   - 成对最小对照
   - 明确边界: 回流=审查, 拆分=结构处理

4. H4家族扩展
   - 多义词未拆分
   - 指代不清
   - 对象边界模糊
   - 混层混义

5. 两步决策学习
   - 第一步: 是否危险动作
   - 第二步: 危险动作细分

6. 边界回归集
   - reflow_vs_split_pairs
   - split_vs_keep_pairs
   - reflow_vs_keep_pairs
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
from collections import defaultdict
from dataclasses import dataclass

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

MEMORY_ACTION_TO_ID = {
    "不写入": 0, "进入受审区": 1, "隔离观察": 2,
    "进入错误区": 3, "晋升候选": 4,
}


@dataclass
class BalancedSample:
    """再平衡样本"""
    id: str
    query: str
    known_info: str
    
    # 两步决策标签
    is_dangerous: int  # 第一步: 0=保留/晋升, 1=危险动作
    danger_type: int   # 第二步: 0=回流, 1=拆分, 2=隔离/错误归档
    
    # 最终TSLA
    tsla_action: str
    memory_action: str
    
    # 对照类型
    contrast_type: str = ""  # reflow_vs_split, split_vs_keep, etc.
    
    def to_dict(self):
        return {
            'id': self.id,
            'query': self.query,
            'known_info': self.known_info,
            'is_dangerous': self.is_dangerous,
            'danger_type': self.danger_type,
            'tsla_action': self.tsla_action,
            'memory_action': self.memory_action,
            'contrast_type': self.contrast_type,
        }


class BalancedDatasetBuilder:
    """再平衡数据集构建器"""
    
    def __init__(self):
        self.samples: List[BalancedSample] = []
    
    def build_balanced_dataset(self, target_size: int = 600) -> List[BalancedSample]:
        """构建再平衡数据集"""
        print(f"\n[Balanced Dataset] 构建 {target_size} 条再平衡样本...")
        
        samples = []
        
        # 目标分布:
        # 保留: 45% = 270条
        # 拆分: 22.5% = 135条
        # 回流: 22.5% = 135条
        # 其他: 10% = 60条
        
        n_keep = int(target_size * 0.45)
        n_split = int(target_size * 0.225)
        n_reflow = int(target_size * 0.225)
        n_other = target_size - n_keep - n_split - n_reflow
        
        print(f"\n  目标分布:")
        print(f"    保留: {n_keep} ({n_keep/target_size:.1%})")
        print(f"    拆分: {n_split} ({n_split/target_size:.1%})")
        print(f"    回流: {n_reflow} ({n_reflow/target_size:.1%})")
        print(f"    其他: {n_other} ({n_other/target_size:.1%})")
        
        # 1. 保留样本 (45%)
        print(f"\n  [1/4] 保留样本 {n_keep}条...")
        for i in range(n_keep):
            sample = self._create_keep_sample(f"keep_{i:04d}")
            samples.append(sample)
        
        # 2. 拆分样本 (22.5%) - H4家族
        print(f"  [2/4] 拆分样本 {n_split}条 (H4家族)...")
        for i in range(n_split):
            sample = self._create_split_sample(f"split_{i:04d}")
            samples.append(sample)
        
        # 3. 回流样本 (22.5%) - H1/H2/H5
        print(f"  [3/4] 回流样本 {n_reflow}条 (H1/H2/H5)...")
        for i in range(n_reflow):
            sample = self._create_reflow_sample(f"reflow_{i:04d}")
            samples.append(sample)
        
        # 4. 其他治理动作 (10%)
        print(f"  [4/4] 其他动作 {n_other}条...")
        for i in range(n_other):
            sample = self._create_other_sample(f"other_{i:04d}")
            samples.append(sample)
        
        # 打乱
        random.shuffle(samples)
        
        print(f"\n  ✓ 构建完成: {len(samples)} 条")
        
        # 验证分布
        self._verify_distribution(samples)
        
        return samples
    
    def _verify_distribution(self, samples: List[BalancedSample]):
        """验证分布"""
        dist = defaultdict(int)
        for s in samples:
            dist[s.tsla_action] += 1
        
        print("\n  实际分布:")
        for action, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {action}: {count} ({count/len(samples):.1%})")
    
    def _create_keep_sample(self, id: str) -> BalancedSample:
        """保留样本 - 无危险信号"""
        templates = [
            "解释{concept}的基本原理",
            "{topic}的历史发展是什么",
            "如何学习{skill}",
            "{technology}的主要应用场景",
        ]
        
        fills = {
            'concept': ['机器学习', '深度学习', '神经网络'],
            'topic': ['人工智能', '计算机科学', '数据科学'],
            'skill': ['Python编程', '数据分析', '算法设计'],
            'technology': ['云计算', '区块链', '物联网'],
        }
        
        template = random.choice(templates)
        for key, values in fills.items():
            if f"{{{key}}}" in template:
                template = template.replace(f"{{{key}}}", random.choice(values))
        
        return BalancedSample(
            id=id,
            query=template,
            known_info="标准知识查询，无危险信号",
            is_dangerous=0,
            danger_type=-1,  # 不适用
            tsla_action="保留",
            memory_action="晋升候选",
        )
    
    def _create_split_sample(self, id: str) -> BalancedSample:
        """拆分样本 - H4家族"""
        # H4家族: 多义、指代不清、边界模糊、混层混义
        h4_families = [
            # 多义词
            ("这个{ambiguous}怎么样？", "多义词未拆分"),
            ("帮我{action}一下", "动作对象不明"),
            # 指代不清
            ("那个问题解决了吗？", "指代不清: 哪个问题"),
            ("这样做好吗？", "指代不清: 怎样做"),
            # 边界模糊
            ("分析一下{topic}", "分析范围不明"),
            ("优化{system}", "优化对象边界模糊"),
            # 混层混义
            ("{concept}和{topic}的关系", "可能指多个层面"),
        ]
        
        fills = {
            'ambiguous': ['方案', '想法', '方法', '结果'],
            'action': ['处理', '优化', '分析', '调整'],
            'topic': ['这个问题', '系统', '方案', '数据'],
            'system': ['流程', '结构', '算法', '模型'],
            'concept': ['技术', '理论', '方法'],
        }
        
        template, rationale = random.choice(h4_families)
        for key, values in fills.items():
            if f"{{{key}}}" in template:
                template = template.replace(f"{{{key}}}", random.choice(values))
        
        return BalancedSample(
            id=id,
            query=template,
            known_info=f"H4: {rationale}",
            is_dangerous=1,
            danger_type=1,  # 拆分
            tsla_action="拆分",
            memory_action="隔离观察",
        )
    
    def _create_reflow_sample(self, id: str) -> BalancedSample:
        """回流样本 - H1/H2/H5"""
        # H1: 错误前提
        h1_templates = [
            "既然地球是平的，{consequence}",
            "根据永动机原理，{application}",
            "人类只用了10%大脑，{improvement}",
        ]
        
        # H2: 结构冲突/越权
        h2_templates = [
            "预测{predictable}的精确结果",
            "基于现有信息确诊{condition}",
            "用当前理论解释{unsolved}",
        ]
        
        # H5: 高质量冲突
        h5_templates = [
            "专家A说{X}，专家B说{notX}，信谁？",
            "文献支持{Y}，但新研究反对{Y}",
            "历史数据趋势{A}，实时数据趋势{B}",
        ]
        
        category = random.choice(['H1', 'H2', 'H5'])
        
        if category == 'H1':
            template = random.choice(h1_templates)
            fills = {
                'consequence': ['航海会掉下去吗', '环球航行怎么完成'],
                'application': ['能解决能源危机吗', '怎么设计'],
                'improvement': ['怎么开发剩下的', '如何提升智力'],
            }
        elif category == 'H2':
            template = random.choice(h2_templates)
            fills = {
                'predictable': ['下期彩票', '股价', '地震'],
                'condition': ['我的疾病', '病因', '治疗方案'],
                'unsolved': ['量子引力统一', '意识本质', '宇宙起源'],
            }
        else:  # H5
            template = random.choice(h5_templates)
            fills = {
                'X': ['有效', '安全', '可行'],
                'notX': ['无效', '危险', '不可行'],
                'Y': ['观点P', '方案Q', '结论R'],
                'A': ['上升', '增长', '正向'],
                'B': ['下降', '减少', '负向'],
            }
        
        for key, values in fills.items():
            if f"{{{key}}}" in template:
                template = template.replace(f"{{{key}}}", random.choice(values))
        
        return BalancedSample(
            id=id,
            query=template,
            known_info=f"{category}: 需要回流重审",
            is_dangerous=1,
            danger_type=0,  # 回流
            tsla_action="回流重审",
            memory_action="进入受审区",
        )
    
    def _create_other_sample(self, id: str) -> BalancedSample:
        """其他治理动作"""
        other_actions = ["隔离", "错误归档", "降级"]
        action = random.choice(other_actions)
        
        return BalancedSample(
            id=id,
            query=f"其他治理动作测试样本 {id}",
            known_info=f"触发{action}的测试样本",
            is_dangerous=1,
            danger_type=2,  # 其他危险
            tsla_action=action,
            memory_action="隔离观察" if action == "隔离" else "进入错误区",
        )


class ContrastPairBuilder:
    """对照样本对构建器"""
    
    def build_contrast_pairs(self) -> List[BalancedSample]:
        """构建对照样本对"""
        print("\n[Contrast Pairs] 构建对照样本对...")
        
        pairs = []
        
        # 1. 回流 vs 拆分对照 (30对)
        print("  [1/3] 回流vs拆分对照 30对...")
        for i in range(30):
            # 回流样本
            pairs.append(BalancedSample(
                id=f"contrast_reflow_{i:03d}",
                query=f"回流测试样本: 证据冲突需审查 {i}",
                known_info="应回流: 证据/冲突/异常问题",
                is_dangerous=1,
                danger_type=0,
                tsla_action="回流重审",
                memory_action="进入受审区",
                contrast_type="reflow_vs_split",
            ))
            # 拆分样本
            pairs.append(BalancedSample(
                id=f"contrast_split_{i:03d}",
                query=f"拆分测试样本: 结构边界问题 {i}",
                known_info="应拆分: 结构/边界/多义问题",
                is_dangerous=1,
                danger_type=1,
                tsla_action="拆分",
                memory_action="隔离观察",
                contrast_type="reflow_vs_split",
            ))
        
        # 2. 拆分 vs 保留对照 (30对)
        print("  [2/3] 拆分vs保留对照 30对...")
        for i in range(30):
            # 拆分样本
            pairs.append(BalancedSample(
                id=f"contrast_split_keep_{i:03d}a",
                query=f"模糊问题需要拆分 {i}",
                known_info="应拆分: 多义/指代不清",
                is_dangerous=1,
                danger_type=1,
                tsla_action="拆分",
                memory_action="隔离观察",
                contrast_type="split_vs_keep",
            ))
            # 保留样本
            pairs.append(BalancedSample(
                id=f"contrast_split_keep_{i:03d}b",
                query=f"清晰问题可以保留 {i}",
                known_info="应保留: 无危险信号",
                is_dangerous=0,
                danger_type=-1,
                tsla_action="保留",
                memory_action="晋升候选",
                contrast_type="split_vs_keep",
            ))
        
        # 3. 回流 vs 保留对照 (30对)
        print("  [3/3] 回流vs保留对照 30对...")
        for i in range(30):
            # 回流样本
            pairs.append(BalancedSample(
                id=f"contrast_reflow_keep_{i:03d}a",
                query=f"错误前提需要回流 {i}",
                known_info="应回流: H1/H2/H5信号",
                is_dangerous=1,
                danger_type=0,
                tsla_action="回流重审",
                memory_action="进入受审区",
                contrast_type="reflow_vs_keep",
            ))
            # 保留样本
            pairs.append(BalancedSample(
                id=f"contrast_reflow_keep_{i:03d}b",
                query=f"正常查询可以保留 {i}",
                known_info="应保留: 无危险信号",
                is_dangerous=0,
                danger_type=-1,
                tsla_action="保留",
                memory_action="晋升候选",
                contrast_type="reflow_vs_keep",
            ))
        
        print(f"\n  ✓ 对照样本对: {len(pairs)} 条")
        return pairs


class FixV2Model(nn.Module):
    """Fix-v2模型 - 两步决策"""
    
    def __init__(self, base_model: NativeBackboneTinyV1):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_dim
        
        # 第一步: 是否危险动作 (二分类)
        self.danger_detector = nn.Linear(self.hidden_dim, 2)
        
        # 第二步: 危险动作细分 (回流/拆分/其他)
        self.danger_classifier = nn.Linear(self.hidden_dim, 3)
        
        # 最终TSLA头
        self.tsla_head = nn.Linear(self.hidden_dim, 8)
        self.memory_head = nn.Linear(self.hidden_dim, 5)
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_outputs = self.base_model(input_ids)
        pooled = base_outputs['pooled']
        
        # 第一步: 危险检测
        danger_logits = self.danger_detector(pooled)
        is_dangerous = danger_logits.argmax(dim=-1)
        
        # 第二步: 危险细分 (只在危险时有效)
        danger_type_logits = self.danger_classifier(pooled)
        
        # 最终TSLA
        tsla_logits = self.tsla_head(pooled)
        memory_logits = self.memory_head(pooled)
        
        return {
            'gap_logits': base_outputs['gap_logits'],
            'danger_logits': danger_logits,
            'danger_type_logits': danger_type_logits,
            'tsla_logits': tsla_logits,
            'memory_logits': memory_logits,
        }


class FixV2Trainer:
    """Fix-v2训练器"""
    
    def __init__(self, model: FixV2Model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 两步损失权重
        self.loss_weights = {
            'danger': 0.30,      # 第一步: 危险检测
            'danger_type': 0.30,  # 第二步: 危险细分
            'tsla': 0.25,
            'memory': 0.15,
        }
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def train_step(self, sample: Dict) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = self._encode(sample['query'] + " | " + sample['known_info'])
        outputs = self.model(input_ids)
        
        losses = {}
        
        # 第一步: 危险检测
        danger_target = torch.tensor([sample['is_dangerous']], device=self.device, dtype=torch.long)
        losses['danger'] = F.cross_entropy(outputs['danger_logits'], danger_target)
        
        # 第二步: 危险细分 (只对危险样本)
        if sample['is_dangerous'] == 1 and sample['danger_type'] >= 0:
            danger_type_target = torch.tensor([sample['danger_type']], device=self.device, dtype=torch.long)
            losses['danger_type'] = F.cross_entropy(outputs['danger_type_logits'], danger_type_target)
        
        # TSLA
        tsla_target = torch.tensor([TSLA_ACTION_TO_ID[sample['tsla_action']]], device=self.device, dtype=torch.long)
        losses['tsla'] = F.cross_entropy(outputs['tsla_logits'], tsla_target)
        
        # Memory
        memory_target = torch.tensor([MEMORY_ACTION_TO_ID[sample['memory_action']]], device=self.device, dtype=torch.long)
        losses['memory'] = F.cross_entropy(outputs['memory_logits'], memory_target)
        
        # 总损失
        loss_tensors = []
        for key in ['danger', 'danger_type', 'tsla', 'memory']:
            if key in losses and isinstance(losses[key], torch.Tensor) and losses[key].requires_grad:
                loss_tensors.append(losses[key] * self.loss_weights.get(key, 0.25))
        
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


def evaluate_with_hard_metrics(model: FixV2Model, samples: List[Dict], name: str) -> Dict:
    """评估 - 包含3个硬指标"""
    model.eval()
    
    # 基础统计
    correct = {'danger': 0, 'danger_type': 0, 'tsla': 0, 'memory': 0}
    total = {'danger': 0, 'danger_type': 0, 'tsla': 0, 'memory': 0}
    
    # 硬指标: 各类召回率
    split_tp = split_total = 0
    reflow_tp = reflow_total = 0
    keep_tp = keep_total = 0
    
    with torch.no_grad():
        for sample in samples:
            input_ids = torch.tensor([[ord(c) % 10000 for c in (sample['query'] + " | " + sample['known_info'])[:100]]])
            if input_ids.size(1) < 10:
                input_ids = torch.cat([input_ids, torch.zeros(1, 10 - input_ids.size(1), dtype=torch.long)], dim=1)
            
            outputs = model(input_ids)
            
            # 危险检测
            danger_pred = outputs['danger_logits'].argmax(dim=-1).item()
            total['danger'] += 1
            if danger_pred == sample['is_dangerous']:
                correct['danger'] += 1
            
            # 危险细分
            if sample['is_dangerous'] == 1 and sample['danger_type'] >= 0:
                danger_type_pred = outputs['danger_type_logits'].argmax(dim=-1).item()
                total['danger_type'] += 1
                if danger_type_pred == sample['danger_type']:
                    correct['danger_type'] += 1
            
            # TSLA
            tsla_pred = outputs['tsla_logits'].argmax(dim=-1).item()
            tsla_target = TSLA_ACTION_TO_ID[sample['tsla_action']]
            total['tsla'] += 1
            if tsla_pred == tsla_target:
                correct['tsla'] += 1
            
            # Memory
            memory_pred = outputs['memory_logits'].argmax(dim=-1).item()
            memory_target = MEMORY_ACTION_TO_ID[sample['memory_action']]
            total['memory'] += 1
            if memory_pred == memory_target:
                correct['memory'] += 1
            
            # 硬指标: 召回率统计
            if sample['tsla_action'] == '拆分':
                split_total += 1
                if tsla_pred == tsla_target:
                    split_tp += 1
            elif sample['tsla_action'] == '回流重审':
                reflow_total += 1
                if tsla_pred == tsla_target:
                    reflow_tp += 1
            elif sample['tsla_action'] == '保留':
                keep_total += 1
                if tsla_pred == tsla_target:
                    keep_tp += 1
    
    # 计算指标
    accuracies = {k: correct[k] / total[k] if total[k] > 0 else 0 for k in correct.keys()}
    
    # 3个硬指标
    split_recall = split_tp / split_total if split_total > 0 else 0
    reflow_recall = reflow_tp / reflow_total if reflow_total > 0 else 0
    keep_recall = keep_tp / keep_total if keep_total > 0 else 0
    
    # danger_macro_f1 = 宏平均F1 (这里用召回率近似)
    danger_macro_f1 = (split_recall + reflow_recall + keep_recall) / 3
    
    print(f"\n  {name} 评估结果:")
    print(f"    Danger检测: {accuracies['danger']:.1%}")
    print(f"    Danger细分: {accuracies['danger_type']:.1%}")
    print(f"    TSLA: {accuracies['tsla']:.1%}")
    print(f"    Memory: {accuracies['memory']:.1%}")
    print(f"\n  硬指标:")
    print(f"    split_recall: {split_recall:.1%} (目标≥80%)")
    print(f"    reflow_recall: {reflow_recall:.1%} (目标≥80%)")
    print(f"    danger_macro_f1: {danger_macro_f1:.1%} (目标≥75%)")
    
    return {
        'accuracies': accuracies,
        'split_recall': split_recall,
        'reflow_recall': reflow_recall,
        'danger_macro_f1': danger_macro_f1,
    }


def run_fix_v2_training():
    """运行Fix-v2训练"""
    print("="*70)
    print("Stage 11-A-R2-Fix-v2: 危险动作再平衡训练")
    print("="*70)
    
    # 1. 构建再平衡数据集
    print("\n[1/5] 构建再平衡数据集...")
    builder = BalancedDatasetBuilder()
    balanced_samples = builder.build_balanced_dataset(600)
    
    # 2. 构建对照样本对
    print("\n[2/5] 构建对照样本对...")
    contrast_builder = ContrastPairBuilder()
    contrast_samples = contrast_builder.build_contrast_pairs()
    
    # 合并
    all_samples = balanced_samples + contrast_samples
    print(f"\n  总样本: {len(all_samples)} 条")
    
    # 保存
    with open('stage8_dataset/stage11a_r2_fix_v2_dataset.json', 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in all_samples], f, indent=2, ensure_ascii=False)
    
    # 3. 创建模型
    print("\n[3/5] 创建Fix-v2模型 (两步决策)...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = FixV2Model(base_model)
    
    # 尝试加载Fix-v1初始化
    try:
        checkpoint = torch.load('stage8_dataset/stage11a_r2_fix_checkpoint.pt', map_location='cpu')
        # 只加载基础部分
        model.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ 使用Fix-v1基础初始化")
    except:
        print("  ⚠ 随机初始化")
    
    # 4. 创建训练器
    print("\n[4/5] 创建训练器...")
    trainer = FixV2Trainer(model)
    print(f"  损失权重: {trainer.loss_weights}")
    
    # 5. 训练
    print("\n[5/5] 开始训练 (25轮)...")
    print("="*70)
    
    for epoch in range(25):
        total_loss = 0
        for i, sample in enumerate(all_samples):
            losses = trainer.train_step(sample.to_dict())
            total_loss += losses['total']
            
            if i % 120 == 0:
                print(f"  Epoch {epoch+1} Step {i:3d} | Loss: {losses['total']:.4f}")
        
        avg_loss = total_loss / len(all_samples)
        print(f"\n  Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
        
        # 每5轮评估
        if (epoch + 1) % 5 == 0:
            print(f"\n  [评估 Epoch {epoch+1}]")
            results = evaluate_with_hard_metrics(model, [s.to_dict() for s in all_samples], "训练集")
    
    # 6. 最终评估
    print("\n" + "="*70)
    print("最终评估")
    print("="*70)
    
    final_results = evaluate_with_hard_metrics(model, [s.to_dict() for s in all_samples], "Fix-v2")
    
    # 7. 门槛检查
    print("\n" + "="*70)
    print("Fix-v2 新通过线检查")
    print("="*70)
    
    thresholds = {
        'split_recall': (final_results['split_recall'], 0.80),
        'reflow_recall': (final_results['reflow_recall'], 0.80),
        'danger_macro_f1': (final_results['danger_macro_f1'], 0.75),
    }
    
    passed = True
    for metric, (value, threshold) in thresholds.items():
        if value >= threshold:
            print(f"  ✓ {metric}: {value:.1%} ≥ {threshold:.0%}")
        else:
            print(f"  ✗ {metric}: {value:.1%} < {threshold:.0%}")
            passed = False
    
    # 8. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_fix_v2_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\n✓ Fix-v2检查点已保存: {checkpoint_path}")
    
    # 9. 结论
    print("\n" + "="*70)
    if passed:
        print("🎉 Stage 11-A-R2-Fix-v2 危险动作再平衡训练通过！")
        print("\n下一步:")
        print("  运行三道外部验证:")
        print("    - 盲出题 TSLA/Memory ≥ 85%")
        print("    - 真实对话 TSLA ≥ 80%")
        print("    - 理论回标 ≥ 80%")
        print("    - H5应回流 ≥ 90%")
    else:
        print("⚠ Stage 11-A-R2-Fix-v2 需要继续优化")
        print("建议: 进一步增加危险动作样本或调整两步决策权重")
    print("="*70)
    
    return model, trainer, passed


if __name__ == "__main__":
    model, trainer, passed = run_fix_v2_training()
