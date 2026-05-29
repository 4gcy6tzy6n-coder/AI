"""
Stage 3A Training Runner

阶段 3A 训练运行器

训练目标：
- PolicyHead (策略选择)
- GapDetector (缺口识别)
- GovernanceHead (治理动作)

冻结：
- WritebackHead (写回头)

数据来源：
- D1: 高质量骨架 (60%)
- D2: 策略治理 (30%)
- D3: 少量对抗样本 (10%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from phase17.enhanced_native_backbone import EnhancedNativeBackbone, EnhancedBackboneConfig
from phase17.enhanced_loss_runner import create_enhanced_loss_runner
from phase18_stage3a.stage3a_dataloader import Stage3ADataset, Stage3AConfig, create_stage3a_dataloaders


@dataclass
class Stage3ATrainingConfig:
    """Stage 3A 训练配置"""
    # 模型配置
    hidden_dim: int = 256
    vocab_size: int = 10000
    max_seq_len: int = 100
    
    # 训练配置
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    device: str = 'cpu'
    
    # 损失权重
    policy_weight: float = 1.0
    gap_weight: float = 1.0
    governance_weight: float = 1.0
    writeback_weight: float = 0.0  # Stage 3A 冻结 Writeback
    response_weight: float = 0.5   # 降低响应权重
    
    # 数据配置
    total_samples: int = 500
    d1_ratio: float = 0.6
    d2_ratio: float = 0.3
    d3_ratio: float = 0.1
    
    # 检查点
    checkpoint_dir: str = "phase18_stage3a/checkpoints"
    save_best: bool = True
    
    # 早停
    patience: int = 3
    min_delta: float = 0.01


class Stage3ATrainingRunner:
    """Stage 3A 训练运行器"""
    
    def __init__(self, config: Stage3ATrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建模型
        backbone_config = EnhancedBackboneConfig(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            device=config.device,
        )
        self.model = EnhancedNativeBackbone(backbone_config)
        self.model = self.model.to(self.device)
        
        # 冻结 WritebackHead
        self._freeze_writeback_head()
        
        # 创建损失运行器
        self.loss_runner = create_enhanced_loss_runner(
            policy_weight=config.policy_weight,
            gap_weight=config.gap_weight,
            governance_weight=config.governance_weight,
            writeback_weight=config.writeback_weight,
            response_weight=config.response_weight,
        )
        
        # 优化器（只优化非冻结参数）
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'gap_acc': [],
            'policy_acc': [],
            'governance_acc': [],
        }
        
        self.best_gap_acc = 0.0
        self.patience_counter = 0
    
    def _freeze_writeback_head(self):
        """冻结 WritebackHead"""
        print("冻结 WritebackHead (Stage 3A 不训练)")
        for param in self.model.writeback_head.parameters():
            param.requires_grad = False
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            
            # 前向
            predictions = self.model(input_ids)
            
            # 目标
            targets = {
                'policy_target': batch['policy_target'].to(self.device),
                'gap_target': batch['gap_target'].to(self.device),
                'governance_target': batch['governance_target'].to(self.device),
                'writeback_target': torch.zeros(input_ids.size(0), dtype=torch.long).to(self.device),
                'response_target': torch.randint(0, self.config.vocab_size, (input_ids.size(0),)).to(self.device),
            }
            
            # 计算损失
            loss = self.loss_runner.compute_loss(predictions, targets)
            
            # 反向
            self.optimizer.zero_grad()
            loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            epoch_losses.append(loss.total_loss.item())
        
        return {'loss': sum(epoch_losses) / len(epoch_losses)}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        val_losses = []
        gap_correct = policy_correct = 0
        governance_correct = 0
        total = 0
        
        # 策略分布统计
        strategy_counts = [0] * 5
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                
                predictions = self.model(input_ids)
                
                targets = {
                    'policy_target': batch['policy_target'].to(self.device),
                    'gap_target': batch['gap_target'].to(self.device),
                    'governance_target': batch['governance_target'].to(self.device),
                    'writeback_target': torch.zeros(input_ids.size(0), dtype=torch.long).to(self.device),
                    'response_target': torch.randint(0, self.config.vocab_size, (input_ids.size(0),)).to(self.device),
                }
                
                loss = self.loss_runner.compute_loss(predictions, targets)
                val_losses.append(loss.total_loss.item())
                
                # 统计准确率
                pred_gap = predictions['gap_logits'].argmax(dim=-1)
                true_gap = targets['gap_target']
                gap_correct += (pred_gap == true_gap).sum().item()
                
                pred_policy = predictions['policy_logits'].argmax(dim=-1)
                true_policy = targets['policy_target']
                policy_correct += (pred_policy == true_policy).sum().item()
                
                # 统计策略分布
                for p in pred_policy:
                    strategy_counts[p.item()] += 1
                
                # 治理动作（简化：检查是否有任何动作匹配）
                pred_gov = (predictions['governance_logits'] > 0.5).float()
                true_gov = targets['governance_target']
                governance_correct += ((pred_gov == true_gov).all(dim=1)).sum().item()
                
                total += input_ids.size(0)
        
        return {
            'loss': sum(val_losses) / len(val_losses),
            'gap_acc': gap_correct / total,
            'policy_acc': policy_correct / total,
            'governance_acc': governance_correct / total,
            'strategy_distribution': strategy_counts,
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("Stage 3A 训练开始")
        print("=" * 70)
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 70)
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_metrics['gap_acc'])
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['gap_acc'].append(val_metrics['gap_acc'])
            self.history['policy_acc'].append(val_metrics['policy_acc'])
            self.history['governance_acc'].append(val_metrics['governance_acc'])
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Gap: {val_metrics['gap_acc']:.1%} | "
                  f"Policy: {val_metrics['policy_acc']:.1%} | "
                  f"Gov: {val_metrics['governance_acc']:.1%}")
            
            # 保存最佳模型
            if val_metrics['gap_acc'] > self.best_gap_acc:
                self.best_gap_acc = val_metrics['gap_acc']
                self.patience_counter = 0
                if self.config.save_best:
                    self._save_checkpoint(f"{self.config.checkpoint_dir}/best_model.pt", epoch)
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.config.patience:
                print(f"\n早停触发！最佳 Gap 准确率: {self.best_gap_acc:.1%}")
                break
        
        return self.history
    
    def _save_checkpoint(self, path: str, epoch: int):
        """保存检查点"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_gap_acc': self.best_gap_acc,
            'config': asdict(self.config),
        }, path)
    
    def save_report(self, path: str = "phase18_stage3a/eval/stage3a_report.json"):
        """保存训练报告"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'history': self.history,
            'final_metrics': {
                'best_gap_acc': self.best_gap_acc,
                'final_gap_acc': self.history['gap_acc'][-1] if self.history['gap_acc'] else 0,
                'final_policy_acc': self.history['policy_acc'][-1] if self.history['policy_acc'] else 0,
            },
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存: {path}")


def run_stage3a_training():
    """运行 Stage 3A 训练"""
    print("=" * 70)
    print("Stage 3A: 真实数据基础行为训练")
    print("=" * 70)
    
    # 配置
    config = Stage3ATrainingConfig()
    
    # 创建数据加载器
    data_config = Stage3AConfig(
        total_samples=config.total_samples,
        d1_ratio=config.d1_ratio,
        d2_ratio=config.d2_ratio,
        d3_ratio=config.d3_ratio,
    )
    train_loader, val_loader = create_stage3a_dataloaders(
        data_config,
        batch_size=config.batch_size,
    )
    
    # 创建训练器
    runner = Stage3ATrainingRunner(config)
    
    # 训练
    history = runner.train(train_loader, val_loader)
    
    # 最终评估
    print("\n" + "=" * 70)
    print("Stage 3A 最终评估")
    print("=" * 70)
    
    final_metrics = runner.evaluate(val_loader)
    
    print(f"\nGap 准确率: {final_metrics['gap_acc']:.1%}")
    print(f"策略准确率: {final_metrics['policy_acc']:.1%}")
    print(f"治理准确率: {final_metrics['governance_acc']:.1%}")
    
    print("\n策略分布:")
    strategy_names = ['DIRECT', 'RETRIEVAL_FIRST', 'CONSERVATIVE', 'DECLINE', 'REVIEW']
    for i, count in enumerate(final_metrics['strategy_distribution']):
        total = sum(final_metrics['strategy_distribution'])
        pct = count / total * 100 if total > 0 else 0
        print(f"  {strategy_names[i]}: {count} ({pct:.1f}%)")
    
    # 验收检查
    print("\n" + "=" * 70)
    print("Stage 3A 验收标准")
    print("=" * 70)
    
    checks = [
        ("Gap准确率 >= 70%", final_metrics['gap_acc'] >= 0.70),
        ("策略准确率 >= 50%", final_metrics['policy_acc'] >= 0.50),
        ("策略分布合理", max(final_metrics['strategy_distribution']) < len(val_loader.dataset) * 0.6),
    ]
    
    all_passed = all(passed for _, passed in checks)
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print("\n✓ Stage 3A 验收通过！可以进入 Stage 3B")
    else:
        print("\n✗ Stage 3A 需要继续优化")
    
    # 保存报告
    runner.save_report()
    
    return runner, final_metrics


if __name__ == "__main__":
    runner, metrics = run_stage3a_training()
