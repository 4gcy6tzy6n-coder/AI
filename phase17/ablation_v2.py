"""
Ablation Study V2 - 消融实验V2

对比3组配置：
1. 基线组：最小闭环 (Policy + Governance + Response)
2. +GapDetector (使用优化后的架构)
3. +GapDetector + WritebackHead (分阶段训练，Writeback权重0.3)

验收标准：
- Gap准确率 >= 65%
- 策略准确率 >= 50%
- 写回误判率 <= 40%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json

from phase17.enhanced_native_backbone import EnhancedNativeBackbone, EnhancedBackboneConfig
from phase17.local_training_runner import MinimalNativeBackbone, TrainingConfig
from phase17.enhanced_loss_runner import create_enhanced_loss_runner
from phase17.multihead_loss_runner import create_multihead_loss_runner


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    use_gap: bool
    use_writeback: bool
    phased_training: bool
    writeback_weight: float
    
    # 通用配置
    hidden_dim: int = 256
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 5e-4
    device: str = 'cpu'


class AblationDataset(Dataset):
    """消融实验数据集"""
    
    def __init__(self, size: int = 300, config: AblationConfig = None, seed: int = 42):
        self.size = size
        self.config = config or AblationConfig("default", False, False, False, 0.3)
        torch.manual_seed(seed)
        
        self.data = []
        for i in range(size):
            seq_len = 20
            base = (i * 7) % 5000
            
            # Gap标签
            r = i % 10
            if r < 3:
                gap_type = 0
                input_ids = torch.tensor([(base + j * 3) % 10000 for j in range(seq_len)])
            elif r < 8:
                gap_type = 1
                input_ids = torch.tensor([0 if j % 3 == 0 else (base + j * 5) % 10000 for j in range(seq_len)])
            else:
                gap_type = 2
                input_ids = torch.tensor([(base + 9999) % 10000 if j % 4 == 0 else (base + j * 2) % 10000 for j in range(seq_len)])
            
            # 策略
            policy_target = torch.tensor([0, 1, 3][gap_type])
            
            # 治理
            governance_target = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0]).float() if gap_type == 2 else torch.randint(0, 2, (8,)).float()
            
            # Writeback（增加长期候选比例以便评估）
            if gap_type == 0:
                writeback_target = torch.tensor([0, 0, 1, 1, 2][i % 5])
            elif gap_type == 1:
                writeback_target = torch.tensor([0, 1, 1, 2, 2][i % 5])
            else:
                writeback_target = torch.tensor(0)
            
            response_target = torch.randint(0, 10000, (1,))
            
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
                    'writeback': writeback_target.item(),
                }
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


def run_ablation_experiment(config: AblationConfig) -> Dict[str, Any]:
    """运行单个消融实验"""
    print("\n" + "=" * 70)
    print(f"实验: {config.name}")
    print("=" * 70)
    print(f"配置: Gap={config.use_gap}, Writeback={config.use_writeback}, "
          f"Phased={config.phased_training}, WB_Weight={config.writeback_weight}")
    
    # 创建数据集
    train_dataset = AblationDataset(size=300, config=config, seed=42)
    val_dataset = AblationDataset(size=60, config=config, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 创建模型
    if config.use_gap:
        backbone_config = EnhancedBackboneConfig(
            hidden_dim=config.hidden_dim,
            vocab_size=10000,
            device=config.device,
        )
        model = EnhancedNativeBackbone(backbone_config)
    else:
        backbone_config = TrainingConfig(
            hidden_dim=config.hidden_dim,
            vocab_size=10000,
            device=config.device,
        )
        model = MinimalNativeBackbone(backbone_config)
    
    model = model.to(config.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    if config.phased_training and config.use_writeback:
        # 分阶段训练
        history = train_phased(model, train_loader, config)
    else:
        # 普通训练
        history = train_normal(model, train_loader, config)
    
    # 评估
    metrics = evaluate_model(model, val_dataset, config)
    
    print("\n评估结果:")
    print(f"  Gap准确率: {metrics['gap_acc']:.1%}")
    print(f"  策略准确率: {metrics['policy_acc']:.1%}")
    print(f"  写回准确率: {metrics['wb_acc']:.1%}")
    print(f"  写回误判率: {metrics['wb_fp']:.1%}")
    
    return {
        'config': asdict(config),
        'history': history,
        'metrics': metrics,
    }


def train_normal(model, train_loader, config):
    """普通训练"""
    if hasattr(model, 'gap_detector'):
        loss_runner = create_enhanced_loss_runner(
            writeback_weight=config.writeback_weight if config.use_writeback else 0.5
        )
    else:
        loss_runner = create_multihead_loss_runner()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    history = {'loss': []}
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            
            predictions = model(input_ids)
            targets = {
                'policy_target': batch['policy_target'].to(config.device),
                'gap_target': batch.get('gap_target', batch['policy_target']).to(config.device),
                'governance_target': batch['governance_target'].to(config.device),
                'writeback_target': batch.get('writeback_target', batch['policy_target']).to(config.device),
                'response_target': batch['response_target'].to(config.device),
            }
            
            loss = loss_runner.compute_loss(predictions, targets)
            
            optimizer.zero_grad()
            loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.total_loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        history['loss'].append(avg_loss)
        print(f"  Epoch {epoch+1}/{config.num_epochs}: Loss={avg_loss:.4f}")
    
    return history


def train_phased(model, train_loader, config):
    """分阶段训练"""
    # Phase A
    print("\n  Phase A: 训练前四头")
    for param in model.writeback_head.parameters():
        param.requires_grad = False
    
    loss_runner = create_enhanced_loss_runner(writeback_weight=0.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config.learning_rate, weight_decay=1e-4)
    
    for epoch in range(3):
        model.train()
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
            optimizer.step()
    
    # Phase B
    print("  Phase B: 训练WritebackHead")
    for name, param in model.named_parameters():
        param.requires_grad = 'writeback_head' in name
    
    loss_runner = create_enhanced_loss_runner(
        policy_weight=0.1, gap_weight=0.1, governance_weight=0.1,
        writeback_weight=config.writeback_weight, response_weight=0.1
    )
    optimizer = torch.optim.AdamW(model.writeback_head.parameters(),
                                   lr=config.learning_rate * 0.5, weight_decay=1e-4)
    
    for epoch in range(2):
        model.train()
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
            optimizer.step()
    
    return {'loss': []}


def evaluate_model(model, val_dataset, config):
    """评估模型"""
    model.eval()
    
    metrics = {
        'gap_correct': 0, 'policy_correct': 0, 'wb_correct': 0,
        'wb_fp_count': 0, 'wb_fp_total': 0, 'total': 0
    }
    
    with torch.no_grad():
        for sample in val_dataset:
            input_ids = sample['input_ids'].unsqueeze(0).to(config.device)
            outputs = model(input_ids)
            
            # Gap
            if 'gap_logits' in outputs:
                pred_gap = outputs['gap_logits'].argmax(dim=-1).item()
                if pred_gap == sample['target']['gap']:
                    metrics['gap_correct'] += 1
            
            # Policy
            pred_policy = outputs['policy_logits'].argmax(dim=-1).item()
            if pred_policy == sample['target']['strategy']:
                metrics['policy_correct'] += 1
            
            # Writeback
            if 'writeback_logits' in outputs:
                pred_wb = outputs['writeback_logits'].argmax(dim=-1).item()
                true_wb = sample['target']['writeback']
                if pred_wb == true_wb:
                    metrics['wb_correct'] += 1
                if true_wb == 0:
                    metrics['wb_fp_total'] += 1
                    if pred_wb != 0:
                        metrics['wb_fp_count'] += 1
            
            metrics['total'] += 1
    
    return {
        'gap_acc': metrics['gap_correct'] / metrics['total'] if metrics['gap_correct'] > 0 else 0,
        'policy_acc': metrics['policy_correct'] / metrics['total'],
        'wb_acc': metrics['wb_correct'] / metrics['total'] if metrics['wb_correct'] > 0 else 0,
        'wb_fp': metrics['wb_fp_count'] / metrics['wb_fp_total'] if metrics['wb_fp_total'] > 0 else 0,
    }


def run_all_ablations():
    """运行所有消融实验"""
    print("=" * 70)
    print("Ablation Study V2 - 消融实验V2")
    print("=" * 70)
    
    experiments = [
        AblationConfig(
            name="基线组 (最小闭环)",
            use_gap=False,
            use_writeback=False,
            phased_training=False,
            writeback_weight=0.3,
        ),
        AblationConfig(
            name="+GapDetector",
            use_gap=True,
            use_writeback=False,
            phased_training=False,
            writeback_weight=0.3,
        ),
        AblationConfig(
            name="+GapDetector +WritebackHead (分阶段, WB=0.3)",
            use_gap=True,
            use_writeback=True,
            phased_training=True,
            writeback_weight=0.3,
        ),
    ]
    
    results = []
    for exp_config in experiments:
        result = run_ablation_experiment(exp_config)
        results.append(result)
    
    # 对比总结
    print("\n" + "=" * 70)
    print("消融实验对比总结")
    print("=" * 70)
    print("\n| 实验组 | Gap准确率 | 策略准确率 | 写回准确率 | 写回误判率 |")
    print("|--------|-----------|------------|------------|------------|")
    
    for result in results:
        name = result['config']['name']
        m = result['metrics']
        print(f"| {name[:30]}... | {m['gap_acc']:.1%} | {m['policy_acc']:.1%} | "
              f"{m['wb_acc']:.1%} | {m['wb_fp']:.1%} |")
    
    # 保存结果
    with open('phase17/ablation_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 验收
    print("\n" + "=" * 70)
    print("阶段3进入门槛检查")
    print("=" * 70)
    
    final_result = results[-1]
    m = final_result['metrics']
    
    checks = [
        ("Gap准确率 >= 65%", m['gap_acc'] >= 0.65),
        ("策略准确率 >= 50%", m['policy_acc'] >= 0.50),
        ("写回误判率 <= 40%", m['wb_fp'] <= 0.40),
    ]
    
    all_passed = all(passed for _, passed in checks)
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print("\n✓ 满足阶段3进入门槛！")
    else:
        print("\n✗ 需要继续优化")
    
    return results


if __name__ == "__main__":
    results = run_all_ablations()
