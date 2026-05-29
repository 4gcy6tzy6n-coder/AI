"""
Train Retrieval & Governance - 检索与治理训练

Phase 14 Stage 14.2: 检索习惯与治理决策训练
目标：训练"检索不是工具，而是本能"
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


class RetrievalGovernanceDataset(Dataset):
    """检索与治理训练数据集"""
    
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
        should_retrieve = 1.0 if sample['output']['retrieval_triggered'] else 0.0
        necessity_score = sample['output']['retrieval_necessity_score']
        
        # 治理动作（多标签）
        governance_actions = sample['output'].get('governance_actions', [])
        governance_labels = self._encode_governance_actions(governance_actions)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'should_retrieve': torch.tensor(should_retrieve, dtype=torch.float32),
            'necessity_score': torch.tensor(necessity_score, dtype=torch.float32),
            'governance_labels': torch.tensor(governance_labels, dtype=torch.float32)
        }
    
    def _extract_features(self, sample: Dict) -> List[float]:
        """提取特征"""
        query = sample['input']['user_query']
        evidence = sample['input'].get('evidence_status', {})
        risk = sample['input'].get('risk_indicators', {})
        
        features = []
        
        # 1. 查询特征
        features.append(len(query) / 100.0)
        
        # 2. 证据状态
        has_evidence = 1.0 if evidence.get('has_sufficient_evidence', False) else 0.0
        evidence_confidence = evidence.get('evidence_confidence', 0.0)
        features.extend([has_evidence, evidence_confidence])
        
        # 3. 风险指标
        sensitive = 1.0 if risk.get('sensitive_topic', False) else 0.0
        factual = 1.0 if risk.get('factual_claim', False) else 0.0
        high_stakes = 1.0 if risk.get('high_stakes', False) else 0.0
        features.extend([sensitive, factual, high_stakes])
        
        # 4. 查询类型特征
        need_memory_kw = ['记得', '之前', '上次', 'previous', 'earlier', 'before']
        needs_memory = any(kw in query.lower() for kw in need_memory_kw)
        features.append(1.0 if needs_memory else 0.0)
        
        factual_kw = ['是什么', '什么是', 'how to', 'what is', '为什么', 'why']
        is_factual = any(kw in query.lower() for kw in factual_kw)
        features.append(1.0 if is_factual else 0.0)
        
        # 5. 历史长度
        history_len = len(sample['input'].get('conversation_history', []))
        features.append(history_len / 10.0)
        
        return features
    
    def _encode_governance_actions(self, actions: List[str]) -> List[float]:
        """编码治理动作"""
        action_set = ['TSLA', 'STRONG_REVIEW', 'VALIDATION', 'ISOLATE', 'ARCHIVE', 'PROMOTE']
        labels = [1.0 if action in actions else 0.0 for action in action_set]
        return labels


class RetrievalGovernanceHead(nn.Module):
    """检索与治理决策头"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 检索决策头
        self.retrieval_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 必要性评分头
        self.necessity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 治理动作头（多标签）
        self.governance_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6个治理动作
            nn.Sigmoid()
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        retrieve_prob = self.retrieval_head(shared_features).squeeze(-1)
        necessity = self.necessity_head(shared_features).squeeze(-1)
        governance_probs = self.governance_head(shared_features)
        
        return retrieve_prob, necessity, governance_probs


class RetrievalGovernanceTrainer:
    """检索与治理训练器"""
    
    def __init__(
        self,
        model: RetrievalGovernanceHead,
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
            'retrieval_accuracy': [],
            'retrieval_f1': [],
            'governance_accuracy': []
        }
    
    def compute_loss(self, outputs, targets):
        """计算多任务损失"""
        retrieve_prob, necessity, governance_probs = outputs
        
        # 检索触发损失
        loss_retrieve = nn.BCELoss()(retrieve_prob, targets['should_retrieve'])
        
        # 必要性评分损失
        loss_necessity = nn.MSELoss()(necessity, targets['necessity_score'])
        
        # 治理动作损失（多标签BCE）
        loss_governance = nn.BCELoss()(governance_probs, targets['governance_labels'])
        
        # 总损失
        total_loss = loss_retrieve + 0.5 * loss_necessity + 0.3 * loss_governance
        
        return total_loss, {
            'retrieve': loss_retrieve.item(),
            'necessity': loss_necessity.item(),
            'governance': loss_governance.item()
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_breakdown = {'retrieve': 0, 'necessity': 0, 'governance': 0}
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            targets = {
                'should_retrieve': batch['should_retrieve'].to(self.device),
                'necessity_score': batch['necessity_score'].to(self.device),
                'governance_labels': batch['governance_labels'].to(self.device)
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
        
        # 统计
        all_preds = []
        all_targets = []
        all_necessity_preds = []
        all_necessity_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                targets = {
                    'should_retrieve': batch['should_retrieve'].to(self.device),
                    'necessity_score': batch['necessity_score'].to(self.device),
                    'governance_labels': batch['governance_labels'].to(self.device)
                }
                
                outputs = self.model(features)
                loss, _ = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                
                retrieve_prob, necessity, _ = outputs
                all_preds.extend((retrieve_prob > 0.5).cpu().numpy())
                all_targets.extend(targets['should_retrieve'].cpu().numpy())
                all_necessity_preds.extend(necessity.cpu().numpy())
                all_necessity_targets.extend(targets['necessity_score'].cpu().numpy())
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'necessity_mae': np.mean(np.abs(np.array(all_necessity_preds) - np.array(all_necessity_targets)))
        }
    
    def train(self, epochs: int = 15):
        """完整训练"""
        print("="*70)
        print("Retrieval & Governance Training - 检索与治理训练")
        print("="*70)
        
        best_f1 = 0
        
        for epoch in range(epochs):
            train_loss, breakdown = self.train_epoch()
            val_metrics = self.validate()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['retrieval_accuracy'].append(val_metrics['accuracy'])
            self.history['retrieval_f1'].append(val_metrics['f1'])
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} (R:{breakdown['retrieve']:.3f} N:{breakdown['necessity']:.3f} G:{breakdown['governance']:.3f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Retrieval - Acc: {val_metrics['accuracy']:.2%}, P: {val_metrics['precision']:.2%}, R: {val_metrics['recall']:.2%}, F1: {val_metrics['f1']:.2%}")
            print(f"  Necessity MAE: {val_metrics['necessity_mae']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint('retrieval_governance_best.pt')
                print(f"  [保存最佳模型，F1: {best_f1:.2%}]")
        
        print("\n" + "="*70)
        print(f"训练完成！最佳检索 F1: {best_f1:.2%}")
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
    # 使用 D2 数据
    data_path = '../datasets/d2_policy_governance.jsonl'
    
    if not Path(data_path).exists():
        print(f"数据文件不存在: {data_path}")
        print("请先运行 train_policy_head.py 创建数据")
        return
    
    dataset = RetrievalGovernanceDataset(data_path)
    
    # 划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    
    # 创建模型
    model = RetrievalGovernanceHead(input_dim=10, hidden_dim=64)
    
    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = RetrievalGovernanceTrainer(model, train_loader, val_loader, lr=1e-3, device=device)
    history = trainer.train(epochs=15)
    
    # 测试
    print("\n" + "="*70)
    print("测试检索决策")
    print("="*70)
    
    test_cases = [
        {"query": "Python是什么？", "has_evidence": True, "evidence_conf": 0.95},
        {"query": "我们之前讨论过什么？", "has_evidence": False, "evidence_conf": 0.1},
        {"query": "最新的法律规定？", "has_evidence": False, "evidence_conf": 0.3, "sensitive": True},
        {"query": "2+2=？", "has_evidence": True, "evidence_conf": 1.0},
    ]
    
    model.eval()
    with torch.no_grad():
        for case in test_cases:
            features = [
                len(case['query']) / 100.0,
                1.0 if case['has_evidence'] else 0.0,
                case['evidence_conf'],
                1.0 if case.get('sensitive', False) else 0.0,
                1.0,  # factual
                0.0,  # high_stakes
                1.0 if '之前' in case['query'] else 0.0,
                1.0 if '是什么' in case['query'] else 0.0,
                0.0   # no history
            ]
            
            x = torch.tensor([features], dtype=torch.float32).to(device)
            retrieve_prob, necessity, governance = model(x)
            
            print(f"\n查询: {case['query']}")
            print(f"  检索概率: {retrieve_prob.item():.2%}")
            print(f"  必要性评分: {necessity.item():.2f}")
            print(f"  建议动作: {'检索' if retrieve_prob > 0.5 else '直接回答'}")


if __name__ == "__main__":
    main()
