"""
Train Memory Writeback - 记忆写回训练

Phase 14 Stage 14.3: 记忆治理与写回训练
目标：训练"不是所有内容都该记住"
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MemoryLayer:
    """记忆层级"""
    EPHEMERAL = 0        # 瞬时层
    LONG_TERM = 1        # 长期层
    DEEP_PERMANENT = 2   # 深层永久


class MemoryWritebackDataset(Dataset):
    """记忆写回训练数据集"""
    
    def __init__(self, data_path: str):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 提取特征
        features = self._extract_features(sample)
        
        # 标签
        should_writeback = 1.0 if sample['output']['memory_decision']['should_writeback'] else 0.0
        target_layer = getattr(MemoryLayer, sample['output']['memory_decision']['target_layer'].upper())
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'should_writeback': torch.tensor(should_writeback, dtype=torch.float32),
            'target_layer': torch.tensor(target_layer, dtype=torch.long)
        }
    
    def _extract_features(self, sample: Dict) -> List[float]:
        """提取特征"""
        query = sample['input']['user_query']
        response = sample['output']['final_response']
        history = sample['input'].get('conversation_history', [])
        
        features = []
        
        # 1. 内容价值特征
        content_len = len(query) + len(response)
        features.append(content_len / 200.0)
        
        # 2. 信息密度
        info_keywords = ['项目', '目标', '计划', '决定', '重要', '关键']
        info_density = sum(1 for kw in info_keywords if kw in query + response) / len(info_keywords)
        features.append(info_density)
        
        # 3. 事实性
        factual_keywords = ['是', '为', '等于', '定义', '目标', '计划']
        is_factual = any(kw in query + response for kw in factual_keywords)
        features.append(1.0 if is_factual else 0.0)
        
        # 4. 持久性指标
        persistent_keywords = ['项目', '用户', '配置', '规则', '策略']
        is_persistent = any(kw in query + response for kw in persistent_keywords)
        features.append(1.0 if is_persistent else 0.0)
        
        # 5. 对话轮次
        turn_number = sample['input'].get('current_turn', len(history) + 1)
        features.append(turn_number / 20.0)
        
        # 6. 上下文依赖
        has_dependency = len(sample['input'].get('context_dependencies', [])) > 0
        features.append(1.0 if has_dependency else 0.0)
        
        # 7. 用户确认
        confirmation_keywords = ['是的', '对的', '正确', '确认', 'yes', 'correct']
        has_confirmation = any(kw in response.lower() for kw in confirmation_keywords)
        features.append(1.0 if has_confirmation else 0.0)
        
        # 8. 风险内容
        risk_keywords = ['错误', '问题', '失败', 'bug', 'error']
        has_risk = any(kw in query + response for kw in risk_keywords)
        features.append(1.0 if has_risk else 0.0)
        
        return features


class MemoryWritebackHead(nn.Module):
    """记忆写回决策头"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 写回决策
        self.writeback_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 层级选择
        self.layer_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3个层级
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        writeback_prob = self.writeback_head(shared_features).squeeze(-1)
        layer_logits = self.layer_head(shared_features)
        
        return writeback_prob, layer_logits


class MemoryWritebackTrainer:
    """记忆写回训练器"""
    
    def __init__(
        self,
        model: MemoryWritebackHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'writeback_accuracy': [],
            'layer_accuracy': [],
            'false_positive': [],  # 误写回率
            'false_negative': []   # 漏写回率
        }
    
    def compute_loss(self, outputs, targets):
        """计算损失"""
        writeback_prob, layer_logits = outputs
        
        # 写回决策损失
        loss_writeback = nn.BCELoss()(writeback_prob, targets['should_writeback'])
        
        # 层级选择损失（只对应该写回的样本计算）
        mask = targets['should_writeback'] > 0.5
        if mask.sum() > 0:
            loss_layer = nn.CrossEntropyLoss()(
                layer_logits[mask],
                targets['target_layer'][mask]
            )
        else:
            loss_layer = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = loss_writeback + 0.5 * loss_layer
        
        return total_loss, {
            'writeback': loss_writeback.item(),
            'layer': loss_layer.item()
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_breakdown = {'writeback': 0, 'layer': 0}
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            targets = {
                'should_writeback': batch['should_writeback'].to(self.device),
                'target_layer': batch['target_layer'].to(self.device)
            }
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss, breakdown = self.compute_loss(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_breakdown:
                loss_breakdown[key] += breakdown[key]
        
        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in loss_breakdown.items()}
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        all_wb_preds = []
        all_wb_targets = []
        all_layer_preds = []
        all_layer_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                targets = {
                    'should_writeback': batch['should_writeback'].to(self.device),
                    'target_layer': batch['target_layer'].to(self.device)
                }
                
                outputs = self.model(features)
                loss, _ = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                
                writeback_prob, layer_logits = outputs
                
                # 统计
                all_wb_preds.extend((writeback_prob > 0.5).cpu().numpy())
                all_wb_targets.extend(targets['should_writeback'].cpu().numpy())
                all_layer_preds.extend(torch.argmax(layer_logits, dim=1).cpu().numpy())
                all_layer_targets.extend(targets['target_layer'].cpu().numpy())
        
        # 计算指标
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        wb_accuracy = accuracy_score(all_wb_targets, all_wb_preds)
        layer_accuracy = accuracy_score(all_layer_targets, all_layer_preds)
        
        # 误写回率和漏写回率
        cm = confusion_matrix(all_wb_targets, all_wb_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            false_positive = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            false_positive = false_negative = 0
        
        return {
            'loss': total_loss / len(self.val_loader),
            'writeback_accuracy': wb_accuracy,
            'layer_accuracy': layer_accuracy,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
    
    def train(self, epochs: int = 15):
        """完整训练"""
        print("="*70)
        print("Memory Writeback Training - 记忆写回训练")
        print("="*70)
        
        best_accuracy = 0
        
        for epoch in range(epochs):
            train_loss, breakdown = self.train_epoch()
            val_metrics = self.validate()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['writeback_accuracy'].append(val_metrics['writeback_accuracy'])
            self.history['layer_accuracy'].append(val_metrics['layer_accuracy'])
            self.history['false_positive'].append(val_metrics['false_positive'])
            self.history['false_negative'].append(val_metrics['false_negative'])
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} (W:{breakdown['writeback']:.3f} L:{breakdown['layer']:.3f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Writeback Acc: {val_metrics['writeback_accuracy']:.2%}")
            print(f"  Layer Acc: {val_metrics['layer_accuracy']:.2%}")
            print(f"  False Positive: {val_metrics['false_positive']:.2%} (误写回)")
            print(f"  False Negative: {val_metrics['false_negative']:.2%} (漏写回)")
            
            # 保存最佳
            if val_metrics['writeback_accuracy'] > best_accuracy:
                best_accuracy = val_metrics['writeback_accuracy']
                self.save_checkpoint('memory_writeback_best.pt')
                print(f"  [保存最佳模型，准确率: {best_accuracy:.2%}]")
        
        print("\n" + "="*70)
        print(f"训练完成！最佳写回决策准确率: {best_accuracy:.2%}")
        print("="*70)
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)


def main():
    """主函数"""
    # 使用 D4 数据
    data_path = '../datasets/d4_multiturn_memory.jsonl'
    
    if not Path(data_path).exists():
        print(f"数据文件不存在: {data_path}")
        return
    
    dataset = MemoryWritebackDataset(data_path)
    
    # 划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    
    # 创建模型
    model = MemoryWritebackHead(input_dim=8, hidden_dim=64)
    
    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = MemoryWritebackTrainer(model, train_loader, val_loader, lr=1e-3, device=device)
    history = trainer.train(epochs=15)
    
    # 测试
    print("\n" + "="*70)
    print("测试记忆写回决策")
    print("="*70)
    
    test_cases = [
        {"query": "项目目标是Q3完成", "response": "已记录，项目目标Q3完成", "turn": 3, "persistent": True},
        {"query": "今天天气怎么样？", "response": "我无法获取实时天气", "turn": 1, "persistent": False},
        {"query": "我叫Alice", "response": "你好Alice！", "turn": 1, "persistent": True},
        {"query": "谢谢", "response": "不客气", "turn": 5, "persistent": False},
    ]
    
    layer_names = ['EPHEMERAL', 'LONG_TERM', 'DEEP_PERMANENT']
    
    model.eval()
    with torch.no_grad():
        for case in test_cases:
            features = [
                (len(case['query']) + len(case['response'])) / 200.0,
                0.5 if case['persistent'] else 0.1,
                1.0 if case['persistent'] else 0.0,
                1.0 if case['persistent'] else 0.0,
                case['turn'] / 20.0,
                0.0,
                0.0,
                0.0
            ]
            
            x = torch.tensor([features], dtype=torch.float32).to(device)
            writeback_prob, layer_logits = model(x)
            
            predicted_layer = torch.argmax(layer_logits, dim=1).item()
            
            print(f"\n查询: {case['query']}")
            print(f"  写回概率: {writeback_prob.item():.2%}")
            print(f"  建议层级: {layer_names[predicted_layer]}")
            print(f"  决策: {'写回' if writeback_prob > 0.5 else '不写回'}")


if __name__ == "__main__":
    main()
