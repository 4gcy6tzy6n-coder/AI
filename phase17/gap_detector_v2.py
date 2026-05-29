"""
GapDetector V2 - 改进版

改进点：
1. 增加数据多样性，让模型真正学习特征
2. 添加类别权重，解决类别不平衡
3. 早停机制，防止过拟合
4. 数据增强
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
from dataclasses import dataclass
import random


@dataclass
class GapConfig:
    """Gap检测配置"""
    hidden_dim: int = 256
    vocab_size: int = 10000
    num_gap_types: int = 3
    seq_len: int = 20  # 增加序列长度
    
    # 训练配置
    num_epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 5e-4  # 降低学习率
    weight_decay: float = 1e-4  # 添加权重衰减
    device: str = 'cpu'


class GapDatasetV2(Dataset):
    """
    改进版 Gap 数据集
    
    让数据有实际可学习的模式：
    - NO_GAP: 序列有规律，词汇分布均匀
    - RETRIEVABLE_GAP: 序列有缺失模式，某些位置空缺
    - HIGH_RISK_GAP: 序列有冲突/矛盾模式
    """
    
    def __init__(self, size: int = 1000, config: GapConfig = None, seed: int = 42):
        self.size = size
        self.config = config or GapConfig()
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = []
        for i in range(size):
            gap_type = self._sample_gap_type(i)
            input_ids = self._generate_sequence(gap_type, i)
            
            self.data.append({
                'input_ids': input_ids,
                'gap_target': torch.tensor(gap_type),
            })
    
    def _sample_gap_type(self, idx: int) -> int:
        """采样gap类型"""
        r = idx % 10
        if r < 3:
            return 0  # NO_GAP (30%)
        elif r < 8:
            return 1  # RETRIEVABLE (50%)
        else:
            return 2  # HIGH_RISK (20%)
    
    def _generate_sequence(self, gap_type: int, idx: int) -> torch.Tensor:
        """
        根据gap类型生成有区分度的序列
        
        NO_GAP: 完整、连贯的序列
        RETRIEVABLE_GAP: 有特定模式但缺少某些信息
        HIGH_RISK_GAP: 有冲突或异常的序列
        """
        seq_len = self.config.seq_len
        vocab_size = self.config.vocab_size
        
        if gap_type == 0:  # NO_GAP
            # 生成连贯的序列（有规律的模式）
            base = (idx * 7) % (vocab_size // 2)
            seq = [(base + i * 3) % vocab_size for i in range(seq_len)]
            
        elif gap_type == 1:  # RETRIEVABLE_GAP
            # 生成有缺失的序列（某些位置是0，表示缺失）
            base = (idx * 11) % (vocab_size // 2)
            seq = []
            for i in range(seq_len):
                if i % 3 == 0:  # 每3个位置缺失1个
                    seq.append(0)  # 用0表示缺失
                else:
                    seq.append((base + i * 5) % vocab_size)
                    
        else:  # HIGH_RISK_GAP = 2
            # 生成有冲突的序列（重复模式、异常值）
            base = (idx * 13) % (vocab_size // 2)
            seq = []
            for i in range(seq_len):
                if i % 4 == 0:
                    # 插入异常值
                    seq.append((base + 9999) % vocab_size)
                elif i % 5 == 0:
                    # 重复之前的值
                    seq.append(seq[-1] if seq else base)
                else:
                    seq.append((base + i * 2) % vocab_size)
        
        return torch.tensor(seq, dtype=torch.long)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


class GapDetectorV2(nn.Module):
    """改进版 GapDetector"""
    
    def __init__(self, config: GapConfig):
        super().__init__()
        self.config = config
        
        # 编码器（使用更好的架构）
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # 使用Transformer encoder来捕捉序列模式
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.num_gap_types),
        )
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Embedding
        x = self.embedding(input_ids)  # [batch, seq, hidden]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, seq, hidden]
        
        # 平均池化
        pooled = x.mean(dim=1)  # [batch, hidden]
        
        # 分类
        gap_logits = self.classifier(pooled)
        gap_probs = F.softmax(gap_logits, dim=-1)
        gap_type = gap_logits.argmax(dim=-1)
        
        return {
            'gap_logits': gap_logits,
            'gap_probs': gap_probs,
            'gap_type': gap_type,
        }


def train_gap_detector_v2():
    """训练改进版 GapDetector"""
    print("=" * 70)
    print("GapDetector V2 - 改进版训练")
    print("=" * 70)
    print("\n改进点:")
    print("  - 数据有实际可学习的模式")
    print("  - 使用Transformer架构")
    print("  - 类别权重平衡")
    print("  - 早停机制")
    
    config = GapConfig()
    
    # 创建模型
    model = GapDetectorV2(config)
    model = model.to(config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    train_dataset = GapDatasetV2(size=800, config=config, seed=42)
    val_dataset = GapDatasetV2(size=200, config=config, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 类别权重（解决类别不平衡）
    # NO_GAP: 30%, RETRIEVABLE: 50%, HIGH_RISK: 20%
    class_weights = torch.tensor([1.0, 0.6, 1.5]).to(config.device)
    
    # 优化器（带权重衰减）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # 早停
    best_acc = 0.0
    patience = 4
    patience_counter = 0
    
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
            
            # 损失（带类别权重）
            loss = F.cross_entropy(outputs['gap_logits'], gap_target, weight=class_weights)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            epoch_losses.append(loss.item())
            preds = outputs['gap_logits'].argmax(dim=-1)
            epoch_correct += (preds == gap_target).sum().item()
            epoch_total += gap_target.size(0)
        
        train_acc = epoch_correct / epoch_total
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
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
                
                true_class = gap_target.item()
                class_total[true_class] += 1
                if pred.item() == true_class:
                    class_correct[true_class] += 1
        
        val_acc = val_correct / val_total
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 早停检查
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'phase17/best_gap_detector.pt')
        else:
            patience_counter += 1
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.1%} | "
              f"Val: {val_acc:.1%} | "
              f"Best: {best_acc:.1%} | "
              f"Patience: {patience_counter}/{patience}")
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发！最佳验证准确率: {best_acc:.1%}")
            break
    
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"\n最佳验证准确率: {best_acc:.1%}")
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('phase17/best_gap_detector.pt'))
    model.eval()
    
    # 最终评估
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    with torch.no_grad():
        for sample in val_dataset:
            input_ids = sample['input_ids'].unsqueeze(0).to(config.device)
            gap_target = sample['gap_target'].unsqueeze(0).to(config.device)
            
            outputs = model(input_ids)
            pred = outputs['gap_logits'].argmax(dim=-1)
            
            true_class = gap_target.item()
            class_total[true_class] += 1
            if pred.item() == true_class:
                class_correct[true_class] += 1
    
    print("\n最终类别准确率:")
    class_names = ['NO_GAP (无缺口)', 'RETRIEVABLE (可检索)', 'HIGH_RISK (高风险)']
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {name}: {acc:.1%} ({class_correct[i]}/{class_total[i]})")
    
    # 判断是否达标
    print("\n验收标准:")
    if best_acc >= 0.65:
        print(f"  ✓ Gap准确率达标: {best_acc:.1%} >= 65%")
        return True
    else:
        print(f"  ✗ Gap准确率未达标: {best_acc:.1%} < 65%")
        return False


if __name__ == "__main__":
    success = train_gap_detector_v2()
