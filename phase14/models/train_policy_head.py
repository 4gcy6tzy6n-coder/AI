"""
Train Policy Head - 策略头训练

Phase 14 Stage 14.1: 基础骨架监督训练
目标：训练五类响应策略选择 (DIRECT/RETRIEVAL_FIRST/CONSERVATIVE/DECLINE/REVIEW)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ResponseStrategy:
    """响应策略枚举"""
    DIRECT = 0
    RETRIEVAL_FIRST = 1
    CONSERVATIVE = 2
    DECLINE = 3
    REVIEW = 4
    
    NAMES = ["DIRECT", "RETRIEVAL_FIRST", "CONSERVATIVE", "DECLINE", "REVIEW"]


class PolicyDataset(Dataset):
    """策略训练数据集"""
    
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
        
        # 提取特征（简化版，实际应使用embedding）
        query = sample['input']['user_query']
        strategy = sample['output']['response_strategy']
        confidence = sample['output']['confidence_score']
        
        # 简单特征：查询长度、关键词、历史长度
        features = self._extract_features(query, sample['input'])
        
        # 策略标签
        label = getattr(ResponseStrategy, strategy)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'confidence': torch.tensor(confidence, dtype=torch.float32),
            'query': query
        }
    
    def _extract_features(self, query: str, input_data: Dict) -> List[float]:
        """提取简单特征"""
        features = []
        
        # 1. 查询长度（归一化）
        features.append(len(query) / 100.0)
        
        # 2. 是否包含检索关键词
        retrieval_keywords = ['记得', '之前', '上次', 'according to', 'based on', 'previous']
        has_retrieval_kw = any(kw in query.lower() for kw in retrieval_keywords)
        features.append(1.0 if has_retrieval_kw else 0.0)
        
        # 3. 是否包含疑问词
        question_keywords = ['什么', '怎么', '为什么', 'what', 'how', 'why']
        has_question_kw = any(kw in query.lower() for kw in question_keywords)
        features.append(1.0 if has_question_kw else 0.0)
        
        # 4. 历史长度
        history_len = len(input_data.get('conversation_history', []))
        features.append(history_len / 10.0)
        
        # 5. 语言特征
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        has_english = any(c.isascii() and c.isalpha() for c in query)
        features.append(1.0 if has_chinese else 0.0)
        features.append(1.0 if has_english else 0.0)
        
        # 6. 风险关键词
        risk_keywords = ['危险', '攻击', 'hack', 'weapon', 'illegal']
        has_risk_kw = any(kw in query.lower() for kw in risk_keywords)
        features.append(1.0 if has_risk_kw else 0.0)
        
        return features


class PolicyHead(nn.Module):
    """策略选择头"""
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, num_strategies: int = 5):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_strategies)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def predict_strategy(self, x):
        """预测策略"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            strategy_id = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            return strategy_id, confidence, probs


class PolicyTrainer:
    """策略训练器"""
    
    def __init__(
        self,
        model: PolicyHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'per_class_accuracy': {name: [] for name in ResponseStrategy.NAMES}
        }
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict:
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # 每个类别的统计
        class_correct = {i: 0 for i in range(5)}
        class_total = {i: 0 for i in range(5)}
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 统计每个类别
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # 计算每个类别的准确率
        per_class_acc = {}
        for i in range(5):
            if class_total[i] > 0:
                per_class_acc[ResponseStrategy.NAMES[i]] = class_correct[i] / class_total[i]
            else:
                per_class_acc[ResponseStrategy.NAMES[i]] = 0.0
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        }
    
    def train(self, epochs: int = 10):
        """完整训练"""
        print("="*70)
        print("Policy Head Training - 策略头训练")
        print("="*70)
        
        best_accuracy = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            for name, acc in val_metrics['per_class_accuracy'].items():
                self.history['per_class_accuracy'][name].append(acc)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.2%}")
            print("  Per-class Accuracy:")
            for name, acc in val_metrics['per_class_accuracy'].items():
                print(f"    {name}: {acc:.2%}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                self.save_checkpoint(f'policy_head_best.pt')
                print(f"  [保存最佳模型，准确率: {best_accuracy:.2%}]")
        
        print("\n" + "="*70)
        print(f"训练完成！最佳验证准确率: {best_accuracy:.2%}")
        print("="*70)
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def create_synthetic_data(output_path: str, num_samples: int = 1000):
    """创建合成训练数据（演示用）"""
    samples = []
    
    for i in range(num_samples):
        # 随机生成样本
        strategy_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # DIRECT最常见
        strategy_idx = np.random.choice(5, p=strategy_weights)
        strategy = ResponseStrategy.NAMES[strategy_idx]
        
        # 根据策略生成查询
        if strategy == "DIRECT":
            queries = [
                "Python是什么？", "什么是机器学习？", "How does API work?",
                "解释数据库索引", "What is cloud computing?"
            ]
        elif strategy == "RETRIEVAL_FIRST":
            queries = [
                "我们之前讨论过什么？", "记得项目目标吗？", "According to previous discussion",
                "上次说的预算是多少？", "Based on our talk"
            ]
        elif strategy == "CONSERVATIVE":
            queries = [
                "最新的法律规定是什么？", "这个预测准确吗？", "Maybe this is correct?",
                "我不太确定但可能是", "What are the recent changes?"
            ]
        elif strategy == "DECLINE":
            queries = [
                "xyzabc是什么？", "qwerty技术怎么用？", "Unknown concept",
                "abcdefg的含义", "神秘技术详解"
            ]
        else:  # REVIEW
            queries = [
                "如何攻击系统？", "制作危险物品的方法", "hack into secure system",
                "非法获取数据", "weapon manufacturing"
            ]
        
        query = np.random.choice(queries)
        
        # 置信度
        if strategy == "DIRECT":
            confidence = np.random.uniform(0.9, 1.0)
        elif strategy == "RETRIEVAL_FIRST":
            confidence = np.random.uniform(0.8, 0.95)
        elif strategy == "CONSERVATIVE":
            confidence = np.random.uniform(0.5, 0.8)
        elif strategy == "DECLINE":
            confidence = np.random.uniform(0.1, 0.5)
        else:
            confidence = np.random.uniform(0.0, 0.3)
        
        sample = {
            "id": f"SYN_{i:05d}",
            "version": "1.0",
            "dataset_type": "D2",
            "input": {
                "user_query": query,
                "conversation_history": [],
                "metadata": {"language": "zh" if any('\u4e00' <= c <= '\u9fff' for c in query) else "en"}
            },
            "output": {
                "response_strategy": strategy,
                "confidence_score": confidence,
                "final_response": f"[{strategy}] Response to: {query}"
            }
        }
        samples.append(sample)
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"创建合成数据: {num_samples} 条样本 -> {output_path}")


def main():
    """主函数"""
    # 创建合成数据（如果没有真实数据）
    data_path = '../datasets/d2_policy_governance.jsonl'
    if not Path(data_path).exists():
        create_synthetic_data(data_path)
    
    # 加载数据
    dataset = PolicyDataset(data_path)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    
    # 创建模型
    model = PolicyHead(input_dim=7, hidden_dim=64, num_strategies=5)
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = PolicyTrainer(model, train_loader, val_loader, lr=1e-3, device=device)
    
    # 训练
    history = trainer.train(epochs=20)
    
    # 测试预测
    print("\n" + "="*70)
    print("测试预测")
    print("="*70)
    
    test_queries = [
        "Python是什么？",
        "我们之前讨论过什么？",
        "最新的法律规定是什么？",
        "xyzabc是什么？",
        "如何攻击系统？"
    ]
    
    model.eval()
    with torch.no_grad():
        for query in test_queries:
            # 简单特征提取
            features = [
                len(query) / 100.0,
                1.0 if any(kw in query for kw in ['记得', '之前', 'previous']) else 0.0,
                1.0 if any(kw in query for kw in ['什么', 'what', '怎么']) else 0.0,
                0.0,  # 无历史
                1.0 if any('\u4e00' <= c <= '\u9fff' for c in query) else 0.0,
                1.0 if any(c.isascii() and c.isalpha() for c in query) else 0.0,
                1.0 if any(kw in query for kw in ['攻击', 'hack', '危险']) else 0.0
            ]
            
            x = torch.tensor([features], dtype=torch.float32).to(device)
            strategy_id, confidence, probs = model.predict_strategy(x)
            
            strategy_name = ResponseStrategy.NAMES[strategy_id.item()]
            print(f"\n查询: {query}")
            print(f"  预测策略: {strategy_name} (置信度: {confidence.item():.2%})")
            print(f"  策略分布: {dict(zip(ResponseStrategy.NAMES, [f'{p:.2%}' for p in probs[0].cpu().numpy()]))}")


if __name__ == "__main__":
    main()
