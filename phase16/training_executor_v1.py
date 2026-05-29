"""
Training Executor v1 - 训练执行器 v1

Phase 16: 真实训练执行
目标：跑第一轮真实训练，验证框架能力是否能被模型内化
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    training_time: float


@dataclass
class TrainingResult:
    """训练结果"""
    head_name: str
    epochs_completed: int
    final_loss: float
    final_accuracy: float
    best_val_accuracy: float
    metrics_history: List[TrainingMetrics]
    model_path: str


class SimplePolicyHead(nn.Module):
    """简化版策略头"""
    
    def __init__(self, input_dim: int = 768, num_strategies: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_strategies)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleRetrievalGovernanceHead(nn.Module):
    """简化版检索治理头"""
    
    def __init__(self, input_dim: int = 768):
        super().__init__()
        # 检索触发判断
        self.retrieval_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 治理动作预测
        self.governance_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8),  # 8种治理动作
            nn.Sigmoid()
        )
    
    def forward(self, x):
        retrieval_prob = self.retrieval_fc(x)
        governance_probs = self.governance_fc(x)
        return retrieval_prob, governance_probs


class SimpleMemoryWritebackHead(nn.Module):
    """简化版记忆写回头"""
    
    def __init__(self, input_dim: int = 768):
        super().__init__()
        # 是否写回
        self.writeback_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 目标层级
        self.layer_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3个层级
        )
    
    def forward(self, x):
        writeback_prob = self.writeback_fc(x)
        layer_logits = self.layer_fc(x)
        return writeback_prob, layer_logits


class TrainingExecutor:
    """训练执行器"""
    
    def __init__(self, output_dir: str = "phase16/training_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def train_policy_head(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> TrainingResult:
        """训练策略头"""
        
        print("\n" + "="*70)
        print("开始训练: Policy Head (策略头)")
        print("="*70)
        
        # 加载数据
        train_data = self._load_jsonl(train_data_path)
        val_data = self._load_jsonl(val_data_path) if val_data_path else None
        
        print(f"训练样本数: {len(train_data)}")
        if val_data:
            print(f"验证样本数: {len(val_data)}")
        
        # 创建模型
        model = SimplePolicyHead(input_dim=768, num_strategies=5).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练循环
        metrics_history = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, sample in enumerate(train_data):
                # 模拟特征（实际应使用真实embedding）
                features = self._extract_simple_features(sample['input']['user_query'])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 标签
                strategy_map = {"DIRECT": 0, "RETRIEVAL_FIRST": 1, "CONSERVATIVE": 2, "DECLINE": 3, "REVIEW": 4}
                label = strategy_map.get(sample['output']['response_strategy'], 0)
                label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, label_tensor)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += 1
                train_correct += (predicted == label_tensor).sum().item()
            
            train_loss /= len(train_data)
            train_acc = train_correct / train_total
            
            # 验证
            val_loss, val_acc = 0.0, 0.0
            if val_data:
                val_loss, val_acc = self._validate_policy(model, val_data, criterion)
                best_val_acc = max(best_val_acc, val_acc)
            
            training_time = time.time() - start_time
            
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                training_time=training_time
            )
            metrics_history.append(metrics)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                  f"Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}, "
                  f"Time={training_time:.1f}s")
        
        # 保存模型
        model_path = self.output_dir / "policy_head.pt"
        torch.save(model.state_dict(), model_path)
        
        return TrainingResult(
            head_name="Policy Head",
            epochs_completed=epochs,
            final_loss=train_loss,
            final_accuracy=train_acc,
            best_val_accuracy=best_val_acc,
            metrics_history=metrics_history,
            model_path=str(model_path)
        )
    
    def train_retrieval_governance_head(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> TrainingResult:
        """训练检索治理头"""
        
        print("\n" + "="*70)
        print("开始训练: Retrieval & Governance Head (检索治理头)")
        print("="*70)
        
        train_data = self._load_jsonl(train_data_path)
        val_data = self._load_jsonl(val_data_path) if val_data_path else None
        
        print(f"训练样本数: {len(train_data)}")
        
        model = SimpleRetrievalGovernanceHead(input_dim=768).to(self.device)
        retrieval_criterion = nn.BCELoss()
        governance_criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        metrics_history = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for sample in train_data:
                features = self._extract_simple_features(sample['input']['user_query'])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 检索标签
                retrieval_label = torch.tensor([[1.0 if sample['output']['retrieval_triggered'] else 0.0]], 
                                               dtype=torch.float32).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                retrieval_prob, governance_probs = model(features)
                
                # 计算损失
                loss = retrieval_criterion(retrieval_prob, retrieval_label)
                
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                predicted = (retrieval_prob > 0.5).float()
                train_total += 1
                train_correct += (predicted == retrieval_label).sum().item()
            
            train_loss /= len(train_data)
            train_acc = train_correct / train_total
            
            val_loss, val_acc = 0.0, 0.0
            if val_data:
                val_loss, val_acc = self._validate_retrieval(model, val_data, retrieval_criterion)
                best_val_acc = max(best_val_acc, val_acc)
            
            training_time = time.time() - start_time
            
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                training_time=training_time
            )
            metrics_history.append(metrics)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                  f"Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}")
        
        model_path = self.output_dir / "retrieval_governance_head.pt"
        torch.save(model.state_dict(), model_path)
        
        return TrainingResult(
            head_name="Retrieval & Governance Head",
            epochs_completed=epochs,
            final_loss=train_loss,
            final_accuracy=train_acc,
            best_val_accuracy=best_val_acc,
            metrics_history=metrics_history,
            model_path=str(model_path)
        )
    
    def train_memory_writeback_head(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> TrainingResult:
        """训练记忆写回头"""
        
        print("\n" + "="*70)
        print("开始训练: Memory Writeback Head (记忆写回头)")
        print("="*70)
        
        train_data = self._load_jsonl(train_data_path)
        val_data = self._load_jsonl(val_data_path) if val_data_path else None
        
        print(f"训练样本数: {len(train_data)}")
        
        model = SimpleMemoryWritebackHead(input_dim=768).to(self.device)
        writeback_criterion = nn.BCELoss()
        layer_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        metrics_history = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for sample in train_data:
                features = self._extract_simple_features(sample['input']['user_query'])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 写回标签
                writeback_label = torch.tensor([[1.0 if sample['output']['memory_decision']['should_writeback'] else 0.0]], 
                                               dtype=torch.float32).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                writeback_prob, layer_logits = model(features)
                
                # 计算损失
                loss = writeback_criterion(writeback_prob, writeback_label)
                
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                predicted = (writeback_prob > 0.5).float()
                train_total += 1
                train_correct += (predicted == writeback_label).sum().item()
            
            train_loss /= len(train_data)
            train_acc = train_correct / train_total
            
            val_loss, val_acc = 0.0, 0.0
            if val_data:
                val_loss, val_acc = self._validate_memory(model, val_data, writeback_criterion)
                best_val_acc = max(best_val_acc, val_acc)
            
            training_time = time.time() - start_time
            
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                training_time=training_time
            )
            metrics_history.append(metrics)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        
        model_path = self.output_dir / "memory_writeback_head.pt"
        torch.save(model.state_dict(), model_path)
        
        return TrainingResult(
            head_name="Memory Writeback Head",
            epochs_completed=epochs,
            final_loss=train_loss,
            final_accuracy=train_acc,
            best_val_accuracy=best_val_acc,
            metrics_history=metrics_history,
            model_path=str(model_path)
        )
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载JSONL数据"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _extract_simple_features(self, text: str, dim: int = 768) -> np.ndarray:
        """提取简单特征（模拟embedding）"""
        # 使用简单的hash特征作为模拟
        features = np.zeros(dim)
        for i, char in enumerate(text):
            idx = hash(char) % dim
            features[idx] += 1.0
        # 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
    
    def _validate_policy(self, model, val_data, criterion):
        """验证策略头"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sample in val_data:
                features = self._extract_simple_features(sample['input']['user_query'])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                strategy_map = {"DIRECT": 0, "RETRIEVAL_FIRST": 1, "CONSERVATIVE": 2, "DECLINE": 3, "REVIEW": 4}
                label = strategy_map.get(sample['output']['response_strategy'], 0)
                label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)
                
                outputs = model(features)
                loss = criterion(outputs, label_tensor)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += 1
                val_correct += (predicted == label_tensor).sum().item()
        
        val_loss /= len(val_data)
        val_acc = val_correct / val_total
        return val_loss, val_acc
    
    def _validate_retrieval(self, model, val_data, criterion):
        """验证检索头"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sample in val_data:
                features = self._extract_simple_features(sample['input']['user_query'])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                retrieval_label = torch.tensor([[1.0 if sample['output']['retrieval_triggered'] else 0.0]], 
                                               dtype=torch.float32).to(self.device)
                
                retrieval_prob, _ = model(features)
                loss = criterion(retrieval_prob, retrieval_label)
                
                val_loss += loss.item()
                predicted = (retrieval_prob > 0.5).float()
                val_total += 1
                val_correct += (predicted == retrieval_label).sum().item()
        
        val_loss /= len(val_data)
        val_acc = val_correct / val_total
        return val_loss, val_acc
    
    def _validate_memory(self, model, val_data, criterion):
        """验证记忆头"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sample in val_data:
                features = self._extract_simple_features(sample['input']['user_query'])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                writeback_label = torch.tensor([[1.0 if sample['output']['memory_decision']['should_writeback'] else 0.0]], 
                                               dtype=torch.float32).to(self.device)
                
                writeback_prob, _ = model(features)
                loss = criterion(writeback_prob, writeback_label)
                
                val_loss += loss.item()
                predicted = (writeback_prob > 0.5).float()
                val_total += 1
                val_correct += (predicted == writeback_label).sum().item()
        
        val_loss /= len(val_data)
        val_acc = val_correct / val_total
        return val_loss, val_acc


# 便捷函数
def create_training_executor(output_dir: str = "phase16/training_output") -> TrainingExecutor:
    """创建训练执行器"""
    return TrainingExecutor(output_dir=output_dir)


# 测试
if __name__ == "__main__":
    print("="*70)
    print("Training Executor v1 - 测试模式")
    print("="*70)
    
    executor = create_training_executor()
    
    # 创建模拟训练数据
    print("\n创建模拟训练数据...")
    
    # 策略头训练数据
    policy_train = []
    for i in range(100):
        policy_train.append({
            "input": {"user_query": f"问题{i}"},
            "output": {"response_strategy": ["DIRECT", "RETRIEVAL_FIRST", "CONSERVATIVE", "DECLINE", "REVIEW"][i % 5]}
        })
    
    policy_val = []
    for i in range(20):
        policy_val.append({
            "input": {"user_query": f"验证问题{i}"},
            "output": {"response_strategy": "RETRIEVAL_FIRST"}
        })
    
    # 保存模拟数据
    import json
    with open("phase16/policy_train.jsonl", "w") as f:
        for item in policy_train:
            f.write(json.dumps(item) + "\n")
    
    with open("phase16/policy_val.jsonl", "w") as f:
        for item in policy_val:
            f.write(json.dumps(item) + "\n")
    
    # 训练策略头
    result = executor.train_policy_head(
        train_data_path="phase16/policy_train.jsonl",
        val_data_path="phase16/policy_val.jsonl",
        epochs=5
    )
    
    print(f"\n训练完成: {result.head_name}")
    print(f"最终准确率: {result.final_accuracy:.4f}")
    print(f"最佳验证准确率: {result.best_val_accuracy:.4f}")
    print(f"模型保存路径: {result.model_path}")
