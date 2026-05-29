"""
Milestone 2.3b 优化冲刺 - 分阶段训练

改进点：
1. Writeback权重 0.5 → 0.3
2. 分阶段训练：
   Phase A: 训练 Policy + Gap + Governance + Response（冻结WritebackHead）
   Phase B: 只训练 WritebackHead（冻结其他头）
3. 使用改进的GapDetector架构
4. 关注长期候选Precision，而非Recall
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json

from phase17.enhanced_native_backbone import EnhancedNativeBackbone, EnhancedBackboneConfig
from phase17.enhanced_loss_runner import create_enhanced_loss_runner


@dataclass
class PhasedConfig:
    """分阶段训练配置"""
    # 模型配置
    hidden_dim: int = 256
    vocab_size: int = 10000
    
    # 训练配置
    phase_a_epochs: int = 5  # Phase A 轮数
    phase_b_epochs: int = 3  # Phase B 轮数
    batch_size: int = 16
    learning_rate: float = 5e-4
    device: str = 'cpu'
    
    # 权重配置（Writeback降低）
    policy_weight: float = 1.0
    gap_weight: float = 1.0  # 提高Gap权重
    governance_weight: float = 1.0
    writeback_weight: float = 0.3  # 从0.5降到0.3
    response_weight: float = 1.0


class PhasedDataset(Dataset):
    """分阶段训练数据集"""
    
    def __init__(self, size: int = 500, config: PhasedConfig = None, seed: int = 42):
        self.size = size
        self.config = config or PhasedConfig()
        torch.manual_seed(seed)
        
        self.data = []
        for i in range(size):
            # 输入（使用有意义的模式）
            seq_len = 20
            base = (i * 7) % (self.config.vocab_size // 2)
            
            # Gap标签
            r = i % 10
            if r < 3:
                gap_type = 0  # NO_GAP
                # 完整序列
                input_ids = torch.tensor([(base + j * 3) % self.config.vocab_size for j in range(seq_len)])
            elif r < 8:
                gap_type = 1  # RETRIEVABLE
                # 有缺失的序列
                input_ids = torch.tensor([
                    0 if j % 3 == 0 else (base + j * 5) % self.config.vocab_size
                    for j in range(seq_len)
                ])
            else:
                gap_type = 2  # HIGH_RISK
                # 异常序列
                input_ids = torch.tensor([
                    (base + 9999) % self.config.vocab_size if j % 4 == 0 else (base + j * 2) % self.config.vocab_size
                    for j in range(seq_len)
                ])
            
            # 策略标签（与gap相关）
            if gap_type == 0:
                policy_target = torch.tensor(0)  # DIRECT
            elif gap_type == 1:
                policy_target = torch.tensor(1)  # RETRIEVAL_FIRST
            else:
                policy_target = torch.tensor(3)  # DECLINE
            
            # 治理标签
            if gap_type == 2:
                governance_target = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0]).float()
            else:
                governance_target = torch.randint(0, 2, (8,)).float()
            
            # Writeback标签（与gap相关，更保守）
            if gap_type == 0:  # NO_GAP
                # 较少写回
                writeback_target = torch.tensor(0) if i % 3 != 0 else torch.tensor(1)
            elif gap_type == 1:  # RETRIEVABLE
                # 可能写回，但长期候选要精
                r2 = i % 5
                if r2 == 0:
                    writeback_target = torch.tensor(0)
                elif r2 <= 3:
                    writeback_target = torch.tensor(1)  # EPHEMERAL
                else:
                    writeback_target = torch.tensor(2)  # LONG_TERM（较少）
            else:  # HIGH_RISK
                # 高风险不写回
                writeback_target = torch.tensor(0)
            
            # 响应标签
            response_target = torch.randint(0, self.config.vocab_size, (1,))
            
            self.data.append({
                'input_ids': input_ids,
                'policy_target': policy_target,
                'gap_target': torch.tensor(gap_type),
                'governance_target': governance_target,
                'writeback_target': writeback_target,
                'response_target': response_target,
                'target': {
                    'strategy': policy_target.item(),
                    'gap': gap_type,
                    'governance': governance_target,
                    'writeback': writeback_target.item(),
                }
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_phase_a(
    model: nn.Module,
    train_loader: DataLoader,
    val_dataset: Dataset,
    config: PhasedConfig,
) -> Dict[str, List[float]]:
    """
    Phase A: 训练 Policy + Gap + Governance + Response
    冻结 WritebackHead
    """
    print("\n" + "=" * 70)
    print("Phase A: 训练 Policy + Gap + Governance + Response")
    print("=" * 70)
    print("  - WritebackHead 冻结")
    print("  - 目标: 让前四头先稳定")
    
    # 冻结 WritebackHead
    for param in model.writeback_head.parameters():
        param.requires_grad = False
    
    # 创建损失运行器（Phase A 不计算 Writeback loss）
    loss_runner = create_enhanced_loss_runner(
        policy_weight=config.policy_weight,
        gap_weight=config.gap_weight,
        governance_weight=config.governance_weight,
        writeback_weight=0.0,  # Phase A 不训练 Writeback
        response_weight=config.response_weight,
    )
    
    # 优化器（只优化非冻结参数）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    
    history = {'loss': [], 'gap_acc': [], 'policy_acc': []}
    
    for epoch in range(config.phase_a_epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            
            predictions = model(input_ids)
            targets = {
                'policy_target': batch['policy_target'].to(config.device),
                'gap_target': batch['gap_target'].to(config.device),
                'governance_target': batch['governance_target'].to(config.device),
                'writeback_target': batch['writeback_target'].to(config.device),
                'response_target': batch['response_target'].to(config.device),
            }
            
            loss = loss_runner.compute_loss(predictions, targets)
            
            optimizer.zero_grad()
            loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.total_loss.item())
        
        # 验证
        model.eval()
        gap_correct = policy_correct = total = 0
        
        with torch.no_grad():
            for sample in val_dataset:
                input_ids = sample['input_ids'].unsqueeze(0).to(config.device)
                outputs = model(input_ids)
                
                pred_gap = outputs['gap_logits'].argmax(dim=-1).item()
                true_gap = sample['target']['gap']
                if pred_gap == true_gap:
                    gap_correct += 1
                
                pred_policy = outputs['policy_logits'].argmax(dim=-1).item()
                true_policy = sample['target']['strategy']
                if pred_policy == true_policy:
                    policy_correct += 1
                
                total += 1
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        gap_acc = gap_correct / total
        policy_acc = policy_correct / total
        
        history['loss'].append(avg_loss)
        history['gap_acc'].append(gap_acc)
        history['policy_acc'].append(policy_acc)
        
        print(f"Epoch {epoch+1}/{config.phase_a_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Gap Acc: {gap_acc:.1%} | "
              f"Policy Acc: {policy_acc:.1%}")
    
    return history


def train_phase_b(
    model: nn.Module,
    train_loader: DataLoader,
    val_dataset: Dataset,
    config: PhasedConfig,
) -> Dict[str, List[float]]:
    """
    Phase B: 只训练 WritebackHead
    冻结其他所有头
    """
    print("\n" + "=" * 70)
    print("Phase B: 只训练 WritebackHead")
    print("=" * 70)
    print("  - 其他头冻结")
    print("  - 目标: 让写回决策稳定，不干扰前四头")
    
    # 冻结除 WritebackHead 外的所有参数
    for name, param in model.named_parameters():
        if 'writeback_head' not in name:
            param.requires_grad = False
    
    # 解冻 WritebackHead
    for param in model.writeback_head.parameters():
        param.requires_grad = True
    
    # 创建损失运行器（Phase B 主要关注 Writeback）
    loss_runner = create_enhanced_loss_runner(
        policy_weight=0.1,  # 降低其他头权重
        gap_weight=0.1,
        governance_weight=0.1,
        writeback_weight=config.writeback_weight,  # 0.3
        response_weight=0.1,
    )
    
    # 优化器（只优化 WritebackHead）
    optimizer = torch.optim.AdamW(
        model.writeback_head.parameters(),
        lr=config.learning_rate * 0.5,  # 更低的学习率
        weight_decay=1e-4,
    )
    
    history = {'loss': [], 'wb_acc': [], 'wb_fp': [], 'lt_prec': []}
    
    for epoch in range(config.phase_b_epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            
            predictions = model(input_ids)
            targets = {
                'policy_target': batch['policy_target'].to(config.device),
                'gap_target': batch['gap_target'].to(config.device),
                'governance_target': batch['governance_target'].to(config.device),
                'writeback_target': batch['writeback_target'].to(config.device),
                'response_target': batch['response_target'].to(config.device),
            }
            
            loss = loss_runner.compute_loss(predictions, targets)
            
            optimizer.zero_grad()
            loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.writeback_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.total_loss.item())
        
        # 验证
        model.eval()
        wb_correct = wb_fp_count = wb_fp_total = lt_pred_count = lt_correct = total = 0
        
        with torch.no_grad():
            for sample in val_dataset:
                input_ids = sample['input_ids'].unsqueeze(0).to(config.device)
                outputs = model(input_ids)
                
                pred_wb = outputs['writeback_logits'].argmax(dim=-1).item()
                true_wb = sample['target']['writeback']
                
                if pred_wb == true_wb:
                    wb_correct += 1
                
                # 写回误判率
                if true_wb == 0:  # 不该写回
                    wb_fp_total += 1
                    if pred_wb != 0:
                        wb_fp_count += 1
                
                # 长期候选精度
                if pred_wb == 2:  # 预测为长期候选
                    lt_pred_count += 1
                    if true_wb == 2:
                        lt_correct += 1
                
                total += 1
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        wb_acc = wb_correct / total
        wb_fp = wb_fp_count / wb_fp_total if wb_fp_total > 0 else 0
        lt_prec = lt_correct / lt_pred_count if lt_pred_count > 0 else 0
        
        history['loss'].append(avg_loss)
        history['wb_acc'].append(wb_acc)
        history['wb_fp'].append(wb_fp)
        history['lt_prec'].append(lt_prec)
        
        print(f"Epoch {epoch+1}/{config.phase_b_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"WB Acc: {wb_acc:.1%} | "
              f"WB FP: {wb_fp:.1%} | "
              f"LT Prec: {lt_prec:.1%}")
    
    return history


def run_phased_training():
    """运行分阶段训练"""
    print("=" * 70)
    print("Milestone 2.3b: 分阶段训练优化")
    print("=" * 70)
    print("\n关键改进:")
    print("  1. Writeback权重: 0.5 → 0.3")
    print("  2. Phase A: 训练 Policy+Gap+Governance+Response（冻结Writeback）")
    print("  3. Phase B: 只训练 WritebackHead（冻结其他）")
    print("  4. 关注长期候选Precision，而非Recall")
    
    config = PhasedConfig()
    
    # 创建模型
    backbone_config = EnhancedBackboneConfig(
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        device=config.device,
    )
    model = EnhancedNativeBackbone(backbone_config)
    model = model.to(config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    train_dataset = PhasedDataset(size=500, config=config, seed=42)
    val_dataset = PhasedDataset(size=100, config=config, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Phase A 训练
    phase_a_history = train_phase_a(model, train_loader, val_dataset, config)
    
    # Phase B 训练
    phase_b_history = train_phase_b(model, train_loader, val_dataset, config)
    
    # 最终评估
    print("\n" + "=" * 70)
    print("最终评估")
    print("=" * 70)
    
    model.eval()
    metrics = {
        'gap_correct': 0,
        'policy_correct': 0,
        'wb_correct': 0,
        'wb_fp_count': 0,
        'wb_fp_total': 0,
        'lt_pred_count': 0,
        'lt_correct': 0,
        'total': 0,
    }
    
    with torch.no_grad():
        for sample in val_dataset:
            input_ids = sample['input_ids'].unsqueeze(0).to(config.device)
            outputs = model(input_ids)
            
            # Gap
            pred_gap = outputs['gap_logits'].argmax(dim=-1).item()
            if pred_gap == sample['target']['gap']:
                metrics['gap_correct'] += 1
            
            # Policy
            pred_policy = outputs['policy_logits'].argmax(dim=-1).item()
            if pred_policy == sample['target']['strategy']:
                metrics['policy_correct'] += 1
            
            # Writeback
            pred_wb = outputs['writeback_logits'].argmax(dim=-1).item()
            true_wb = sample['target']['writeback']
            if pred_wb == true_wb:
                metrics['wb_correct'] += 1
            
            # WB False Positive
            if true_wb == 0:
                metrics['wb_fp_total'] += 1
                if pred_wb != 0:
                    metrics['wb_fp_count'] += 1
            
            # Long-term Precision
            if pred_wb == 2:
                metrics['lt_pred_count'] += 1
                if true_wb == 2:
                    metrics['lt_correct'] += 1
            
            metrics['total'] += 1
    
    gap_acc = metrics['gap_correct'] / metrics['total']
    policy_acc = metrics['policy_correct'] / metrics['total']
    wb_acc = metrics['wb_correct'] / metrics['total']
    wb_fp = metrics['wb_fp_count'] / metrics['wb_fp_total'] if metrics['wb_fp_total'] > 0 else 0
    lt_prec = metrics['lt_correct'] / metrics['lt_pred_count'] if metrics['lt_pred_count'] > 0 else 0
    
    print(f"\nGap准确率: {gap_acc:.1%}")
    print(f"策略准确率: {policy_acc:.1%}")
    print(f"写回准确率: {wb_acc:.1%}")
    print(f"写回误判率: {wb_fp:.1%}")
    print(f"长期候选精度: {lt_prec:.1%}")
    
    # 验收检查
    print("\n" + "=" * 70)
    print("Milestone 2.3b 验收标准")
    print("=" * 70)
    
    checks = [
        ("Gap准确率 >= 65%", gap_acc >= 0.65),
        ("策略准确率 >= 50%", policy_acc >= 0.50),
        ("写回误判率 <= 40%", wb_fp <= 0.40),
        ("长期候选精度 > 0", lt_prec > 0),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ Milestone 2.3b 验收通过！可以进入阶段3")
    else:
        print("\n✗ Milestone 2.3b 需要继续优化")
    
    # 保存结果
    results = {
        'config': asdict(config),
        'phase_a_history': phase_a_history,
        'phase_b_history': phase_b_history,
        'final_metrics': {
            'gap_accuracy': gap_acc,
            'policy_accuracy': policy_acc,
            'writeback_accuracy': wb_acc,
            'writeback_false_positive': wb_fp,
            'long_term_precision': lt_prec,
        },
        'passed': all_passed,
    }
    
    with open('phase17/milestone_2_3b_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_phased_training()
