"""
GapDetector 单独优化训练

目标：将 Gap 准确率从 50% 提升到 65%+

Gap 类型：
- 0: NO_GAP (无缺口) - 30%
- 1: RETRIEVABLE_GAP (可检索补足) - 50%
- 2: HIGH_RISK_GAP (高风险缺口) - 20%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GapConfig:
    """Gap检测配置"""
    hidden_dim: int = 256
    vocab_size: int = 10000
    num_gap_types: int = 3
    seq_len: int = 10
    
    # 训练配置
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: str = 'cpu'


class GapDetectorOnly(nn.Module):
    """单独的 GapDetector 模型"""
    
    def __init__(self, config: GapConfig):
        super().__init__()
        self.config = config
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        # Gap 分类器（增强版）
        self.gap_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, config.num_gap_types),
        )
        
        # 严重程度估计
        self.severity_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        encoded = self.encoder(input_ids)  # [batch, seq, hidden]
        pooled = encoded.mean(dim=1)  # [batch, hidden]
        
        # Gap 预测
        gap_logits = self.gap_classifier(pooled)
        gap_probs = F.softmax(gap_logits, dim=-1)
        gap_type = gap_logits.argmax(dim=-1)
        severity = self.severity_estimator(pooled).squeeze(-1)
        
        return {
            'gap_logits': gap_logits,
            'gap_probs': gap_probs,
            'gap_type': gap_type,
            'gap_severity': severity,
        }


class GapDataset(Dataset):
    """Gap检测数据集"""
    
    def __init__(self, size: int = 500, config: GapConfig = None):
        self.size = size
        self.config = config or GapConfig()
        
        self.data = []
        for i in range(size):
            # 输入
            input_ids = torch.randint(0, self.config.vocab_size, (self.config.seq_len,))
            
            # Gap 标签（保持合理分布）
            r = i % 10
            if r < 3:
                gap_type = 0  # NO_GAP (30%)
            elif r < 8:
                gap_type = 1  # RETRIEVABLE (50%)
            else:
                gap_type = 2  # HIGH_RISK (20%)
            
            self.data.append({
                'input_ids': input_ids,
                'gap_target': torch.tensor(gap_type),
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_gap_detector():
    """训练 GapDetector"""
    print("=" * 70)
    print("GapDetector 单独优化训练")
    print("=" * 70)
    
    config = GapConfig()
    
    # 创建模型
    model = GapDetectorOnly(config)
    model = model.to(config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    train_dataset = GapDataset(size=500, config=config)
    val_dataset = GapDataset(size=100, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 训练历史
    best_acc = 0.0
    
    print("\n开始训练...")
    print("-" * 70)
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            gap_target = batch['gap_target'].to(config.device)
            
            # 前向
            outputs = model(input_ids)
            
            # 损失
            loss = F.cross_entropy(outputs['gap_logits'], gap_target)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            epoch_losses.append(loss.item())
            preds = outputs['gap_logits'].argmax(dim=-1)
            epoch_correct += (preds == gap_target).sum().item()
            epoch_total += gap_target.size(0)
        
        scheduler.step()
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = epoch_correct / epoch_total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        # 各类别统计
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        
        with torch.no_grad():
            for sample in val_dataset:
                input_ids = sample['input_ids'].unsqueeze(0).to(config.device)
                gap_target = sample['gap_target'].unsqueeze(0).to(config.device)
                
                outputs = model(input_ids)
                pred = outputs['gap_logits'].argmax(dim=-1)
                
                val_correct += (pred == gap_target).sum().item()
                val_total += 1
                
                # 类别统计
                true_class = gap_target.item()
                class_total[true_class] += 1
                if pred.item() == true_class:
                    class_correct[true_class] += 1
        
        val_acc = val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.1%} | "
              f"Val Acc: {val_acc:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 每3轮打印详细类别准确率
        if (epoch + 1) % 3 == 0:
            print(f"  类别准确率:")
            class_names = ['NO_GAP', 'RETRIEVABLE', 'HIGH_RISK']
            for i, name in enumerate(class_names):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i]
                    print(f"    {name}: {acc:.1%} ({class_correct[i]}/{class_total[i]})")
    
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"\n最佳验证准确率: {best_acc:.1%}")
    
    # 最终评估
    print("\n最终类别准确率:")
    class_names = ['NO_GAP (无缺口)', 'RETRIEVABLE (可检索)', 'HIGH_RISK (高风险)']
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {name}: {acc:.1%}")
    
    # 判断是否达标
    print("\n验收标准:")
    if best_acc >= 0.65:
        print(f"  ✓ Gap准确率达标: {best_acc:.1%} >= 65%")
    else:
        print(f"  ✗ Gap准确率未达标: {best_acc:.1%} < 65%")
    
    return model, best_acc


if __name__ == "__main__":
    model, accuracy = train_gap_detector()
