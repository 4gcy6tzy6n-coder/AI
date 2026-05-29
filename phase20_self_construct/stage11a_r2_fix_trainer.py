"""
Stage 11-A-R2-Fix-Trainer: 理论对齐训练器

使用全量重标的600条理论对齐样本重新训练
目标: 让模型学会"理论治理逻辑"而非"标签风格"

新通过线:
- 理论回标 ≥ 80%
- 盲出题 TSLA/Memory ≥ 85%
- 真实对话 TSLA ≥ 80%
- H5应回流准确率 ≥ 90%
- 回归集全过
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, List
from collections import defaultdict

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

MEMORY_ACTION_TO_ID = {
    "不写入": 0, "进入受审区": 1, "隔离观察": 2,
    "进入错误区": 3, "晋升候选": 4,
}


class FixModel(nn.Module):
    """Fix模型 - 复用Phase2.6结构"""
    
    def __init__(self, base_model: NativeBackboneTinyV1):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = base_model.config.hidden_dim
        
        self.tsla_action_head = nn.Linear(self.hidden_dim, 8)
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


class FixTrainer:
    """Fix训练器"""
    
    def __init__(self, model: FixModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 损失权重 - 强调治理层
        self.loss_weights = {
            'gap': 0.20,
            'retrieval': 0.20,
            'tsla': 0.30,  # 增强治理
            'memory': 0.30,  # 增强治理
        }
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def train_step(self, sample: Dict) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = self._encode(sample['query'] + " | " + sample['known_info'])
        outputs = self.model(input_ids)
        
        losses = {}
        
        # Gap
        gap_target = torch.tensor([sample['theory_gap']], device=self.device, dtype=torch.long)
        losses['gap'] = F.cross_entropy(outputs['gap_detection_logits'], gap_target)
        
        # Retrieval
        retrieval_target = torch.tensor([sample['theory_retrieval']], device=self.device, dtype=torch.long)
        losses['retrieval'] = F.cross_entropy(outputs['retrieval_decision_logits'], retrieval_target)
        
        # TSLA (理论对齐标签)
        tsla_action = sample['theory_tsla']
        action_id = TSLA_ACTION_TO_ID.get(tsla_action, 0)
        tsla_target = torch.tensor([action_id], device=self.device, dtype=torch.long)
        losses['tsla'] = F.cross_entropy(outputs['tsla_action_logits'], tsla_target)
        
        # Memory (理论对齐标签)
        memory_action = sample['theory_memory']
        action_id = MEMORY_ACTION_TO_ID.get(memory_action, 0)
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


def evaluate_model(model: FixModel, samples: List[Dict], name: str) -> Dict:
    """评估模型"""
    model.eval()
    
    correct = {'gap': 0, 'retrieval': 0, 'tsla': 0, 'memory': 0}
    total = {'gap': 0, 'retrieval': 0, 'tsla': 0, 'memory': 0}
    
    # H5专项
    h5_total = 0
    h5_correct = 0
    
    # 回归集专项
    regression_correct = defaultdict(int)
    regression_total = defaultdict(int)
    
    with torch.no_grad():
        for sample in samples:
            input_ids = torch.tensor([[ord(c) % 10000 for c in (sample['query'] + " | " + sample['known_info'])[:100]]])
            if input_ids.size(1) < 10:
                input_ids = torch.cat([input_ids, torch.zeros(1, 10 - input_ids.size(1), dtype=torch.long)], dim=1)
            
            outputs = model(input_ids)
            
            # Gap
            gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
            total['gap'] += 1
            if gap_pred == sample['theory_gap']:
                correct['gap'] += 1
            
            # Retrieval
            retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
            total['retrieval'] += 1
            if retrieval_pred == sample['theory_retrieval']:
                correct['retrieval'] += 1
            
            # TSLA
            tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
            tsla_target = TSLA_ACTION_TO_ID.get(sample['theory_tsla'], 0)
            total['tsla'] += 1
            if tsla_pred == tsla_target:
                correct['tsla'] += 1
            
            # Memory
            memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
            memory_target = MEMORY_ACTION_TO_ID.get(sample['theory_memory'], 0)
            total['memory'] += 1
            if memory_pred == memory_target:
                correct['memory'] += 1
            
            # H5专项
            if 'H5' in sample.get('hard_veto_signals', []):
                h5_total += 1
                if tsla_pred == TSLA_ACTION_TO_ID['回流重审']:
                    h5_correct += 1
            
            # 回归集专项
            sample_type = sample.get('sample_type', '')
            if 'regression' in sample_type:
                regression_total[sample_type] += 1
                if tsla_pred == tsla_target:
                    regression_correct[sample_type] += 1
    
    accuracies = {k: correct[k] / total[k] if total[k] > 0 else 0 for k in correct.keys()}
    h5_acc = h5_correct / h5_total if h5_total > 0 else 0
    
    print(f"\n  {name} 评估结果:")
    print(f"    Gap: {accuracies['gap']:.1%}")
    print(f"    Retrieval: {accuracies['retrieval']:.1%}")
    print(f"    TSLA: {accuracies['tsla']:.1%}")
    print(f"    Memory: {accuracies['memory']:.1%}")
    if h5_total > 0:
        print(f"    H5应回流: {h5_acc:.1%}")
    
    if regression_total:
        print(f"\n    回归集:")
        for reg_type in regression_total.keys():
            acc = regression_correct[reg_type] / regression_total[reg_type]
            print(f"      {reg_type}: {acc:.1%}")
    
    return {
        'accuracies': accuracies,
        'h5_accuracy': h5_acc,
        'regression': {k: regression_correct[k] / regression_total[k] for k in regression_total.keys()},
    }


def run_fix_training():
    """运行Fix训练"""
    print("="*70)
    print("Stage 11-A-R2-Fix-Trainer: 理论对齐训练")
    print("="*70)
    
    # 1. 加载理论对齐数据
    print("\n[1/5] 加载理论对齐数据...")
    with open('stage8_dataset/stage11a_r2_fix_theory_aligned.json', 'r', encoding='utf-8') as f:
        theory_samples = json.load(f)
    print(f"  ✓ 加载 {len(theory_samples)} 条理论对齐样本")
    
    # 加载回归集
    with open('stage8_dataset/stage11a_r2_fix_regression_set.json', 'r', encoding='utf-8') as f:
        regression_samples = json.load(f)
    print(f"  ✓ 加载 {len(regression_samples)} 条回归集样本")
    
    # 2. 创建模型
    print("\n[2/5] 创建模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = FixModel(base_model)
    
    # 尝试加载Phase2.6作为初始化
    try:
        checkpoint = torch.load('stage8_dataset/stage11a_r2_phase26_checkpoint.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ 使用Phase2.6初始化")
    except:
        print("  ⚠ 无法加载Phase2.6，随机初始化")
    
    # 3. 创建训练器
    print("\n[3/5] 创建训练器...")
    trainer = FixTrainer(model)
    print(f"  损失权重: {trainer.loss_weights}")
    
    # 4. 训练
    print("\n[4/5] 开始训练 (20轮)...")
    print("="*70)
    
    for epoch in range(20):
        total_loss = 0
        gap_losses = []
        tsla_losses = []
        
        for i, sample in enumerate(theory_samples):
            losses = trainer.train_step(sample)
            total_loss += losses['total']
            
            if 'gap' in losses:
                gap_losses.append(losses['gap'])
            if 'tsla' in losses:
                tsla_losses.append(losses['tsla'])
            
            if i % 100 == 0:
                avg_gap = sum(gap_losses[-20:]) / len(gap_losses[-20:]) if gap_losses else 0
                avg_tsla = sum(tsla_losses[-20:]) / len(tsla_losses[-20:]) if tsla_losses else 0
                print(f"  Epoch {epoch+1} Step {i:3d} | Loss: {losses['total']:.4f} | G:{avg_gap:.3f} T:{avg_tsla:.3f}")
        
        avg_loss = total_loss / len(theory_samples)
        print(f"\n  Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
        
        # 每5轮评估回归集
        if (epoch + 1) % 5 == 0:
            print(f"\n  [回归集验证 Epoch {epoch+1}]")
            evaluate_model(model, regression_samples, "回归集")
    
    # 5. 最终评估
    print("\n" + "="*70)
    print("[5/5] 最终评估")
    print("="*70)
    
    # 5.1 理论对齐数据
    theory_results = evaluate_model(model, theory_samples, "理论对齐数据")
    
    # 5.2 回归集
    regression_results = evaluate_model(model, regression_samples, "回归集")
    
    # 5.3 门槛检查
    print("\n" + "="*70)
    print("新通过线检查")
    print("="*70)
    
    # 计算回归集总体准确率
    regression_acc = sum(regression_results['regression'].values()) / len(regression_results['regression']) if regression_results['regression'] else 0
    
    thresholds = {
        '理论对齐_TSLA': (theory_results['accuracies']['tsla'], 0.80),
        '理论对齐_Memory': (theory_results['accuracies']['memory'], 0.80),
        '回归集总体': (regression_acc, 0.90),
        'H5应回流': (theory_results['h5_accuracy'], 0.90),
    }
    
    passed = True
    for metric, (value, threshold) in thresholds.items():
        if value >= threshold:
            print(f"  ✓ {metric}: {value:.1%} ≥ {threshold:.0%}")
        else:
            print(f"  ✗ {metric}: {value:.1%} < {threshold:.0%}")
            passed = False
    
    # 6. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_fix_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\n✓ Fix检查点已保存: {checkpoint_path}")
    
    # 7. 结论
    print("\n" + "="*70)
    if passed:
        print("🎉 Stage 11-A-R2-Fix 理论对齐训练通过！")
        print("\n下一步:")
        print("  运行三道外部验证:")
        print("    - 盲出题 TSLA/Memory ≥ 85%")
        print("    - 真实对话 TSLA ≥ 80%")
        print("    - 理论回标 ≥ 80%")
    else:
        print("⚠ Stage 11-A-R2-Fix 需要继续优化")
        print("建议: 增加训练轮数或调整样本分布")
    print("="*70)
    
    return model, trainer, passed


if __name__ == "__main__":
    model, trainer, passed = run_fix_training()
