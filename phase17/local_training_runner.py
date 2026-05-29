"""
Local Training Runner v1 - 本地训练运行器 v1

Phase 17 Stage 2: 本地真实训练基础设施
目标：训练总入口，调度前向、损失、保存、评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase17.multihead_loss_runner import create_multihead_loss_runner, MultiHeadLoss
from phase17.checkpoint_manager import create_checkpoint_manager
from phase17.offline_eval_suite import create_offline_eval_suite


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    hidden_dim: int = 768
    num_strategies: int = 5
    num_governance_actions: int = 8
    vocab_size: int = 50000
    
    # 训练配置
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # 损失权重
    policy_weight: float = 1.0
    governance_weight: float = 1.0
    memory_weight: float = 0.5
    response_weight: float = 1.0
    
    # 检查点配置
    checkpoint_dir: str = "phase17/checkpoints"
    save_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MinimalNativeBackbone(nn.Module):
    """
    最小原生主干
    
    架构: UnitEncoder → PolicyHead → GovernanceHead → ResponseDecoder
    目标: 验证最小训练闭环
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # 1. Unit Encoder (简化版)
        self.unit_encoder = nn.Sequential(
            nn.Embedding(config.vocab_size, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        # 2. Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_strategies),
        )
        
        # 3. Governance Head
        self.governance_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_governance_actions),
        )
        
        # 4. Memory Head (简化版 WritebackHead)
        self.memory_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 写回: yes/no
        )
        
        # 5. Response Decoder (简化版)
        self.response_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.num_strategies, 512),
            nn.ReLU(),
            nn.Linear(512, config.vocab_size),
        )
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 1. Unit Encoding
        unit_state = self.unit_encoder(input_ids)  # [batch, seq, hidden]
        unit_pooled = unit_state.mean(dim=1)  # [batch, hidden]
        
        # 2. Policy Prediction
        policy_logits = self.policy_head(unit_pooled)  # [batch, num_strategies]
        
        # 3. Governance Prediction
        governance_logits = self.governance_head(unit_pooled)  # [batch, num_actions]
        
        # 4. Memory Prediction
        memory_logits = self.memory_head(unit_pooled)  # [batch, 2]
        
        # 5. Response Prediction (使用策略信息)
        strategy_onehot = torch.softmax(policy_logits, dim=-1)
        response_input = torch.cat([unit_pooled, strategy_onehot], dim=-1)
        response_logits = self.response_decoder(response_input)  # [batch, vocab_size]
        
        return {
            'policy_logits': policy_logits,
            'governance_logits': governance_logits,
            'memory_logits': memory_logits,
            'response_logits': response_logits.unsqueeze(1),  # [batch, 1, vocab]
        }


class MockDataset(Dataset):
    """模拟数据集"""
    
    def __init__(self, num_samples: int = 1000, config: TrainingConfig = None):
        self.num_samples = num_samples
        self.config = config or TrainingConfig()
        
        # 生成模拟数据
        self.data = []
        for i in range(num_samples):
            self.data.append({
                'input_ids': torch.randint(0, self.config.vocab_size, (10,)),
                'policy_target': torch.tensor(i % self.config.num_strategies),
                'governance_target': torch.randint(0, 2, (self.config.num_governance_actions,)).float(),
                'memory_target': torch.tensor(i % 2),
                'response_target': torch.randint(0, self.config.vocab_size, (1,)),
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class LocalTrainingRunner:
    """本地训练运行器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.loss_runner = create_multihead_loss_runner(
            policy_weight=config.policy_weight,
            governance_weight=config.governance_weight,
            memory_weight=config.memory_weight,
            response_weight=config.response_weight,
        )
        
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir=config.checkpoint_dir,
        )
        
        self.eval_suite = create_offline_eval_suite()
    
    def train(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Returns:
            training_history: 训练历史
        """
        print("\n" + "="*70)
        print("开始训练")
        print("="*70)
        
        # 移动模型到设备
        model = model.to(self.device)
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # 训练历史
        history = {
            'train_losses': [],
            'val_metrics': [],
        }
        
        global_step = 0
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-"*50)
            
            model.train()
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                
                # 前向传播
                predictions = model(input_ids)
                
                # 准备目标
                targets = {
                    'policy_target': batch['policy_target'].to(self.device),
                    'governance_target': batch['governance_target'].to(self.device),
                    'memory_target': batch['memory_target'].to(self.device),
                    'response_target': batch['response_target'].to(self.device),
                }
                
                # 计算损失
                loss = self.loss_runner.compute_loss(predictions, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.total_loss.backward()
                optimizer.step()
                
                # 记录
                epoch_losses.append(loss.total_loss.item())
                history['train_losses'].append(loss.total_loss.item())
                
                global_step += 1
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"  Step {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss.total_loss.item():.4f} "
                          f"(P:{loss.policy_loss_value:.3f} "
                          f"G:{loss.governance_loss_value:.3f} "
                          f"M:{loss.memory_loss_value:.3f} "
                          f"R:{loss.response_loss_value:.3f})")
                
                # 保存检查点
                if global_step % self.config.save_every_n_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                        step=global_step,
                        metrics={'loss': loss.total_loss.item()},
                        config=self.config.__dict__,
                        checkpoint_type="experimental",
                    )
            
            #  epoch 平均损失
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch + 1} 平均损失: {avg_loss:.4f}")
            
            # 评估
            if val_dataset is not None and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                print("\n运行评估...")
                val_metrics = self.eval_suite.evaluate_behavior(
                    model,
                    [val_dataset[i] for i in range(min(100, len(val_dataset)))],
                    evaluation_name=f"epoch_{epoch + 1}",
                )
                history['val_metrics'].append(val_metrics)
                
                # 保存最佳检查点
                self.checkpoint_manager.save_best_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    step=global_step,
                    metric_value=val_metrics.overall_behavior_score,
                    metric_name="overall_behavior_score",
                    config=self.config.__dict__,
                )
        
        # 保存最终检查点
        self.checkpoint_manager.save_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=self.config.num_epochs - 1,
            step=global_step,
            metrics={'final_loss': avg_loss},
            config=self.config.__dict__,
            checkpoint_type="latest",
        )
        
        print("\n" + "="*70)
        print("训练完成")
        print("="*70)
        
        return history


# 便捷函数
def create_training_runner(config: Optional[TrainingConfig] = None) -> LocalTrainingRunner:
    """创建训练运行器"""
    config = config or TrainingConfig()
    return LocalTrainingRunner(config)


# 测试
if __name__ == "__main__":
    print("="*70)
    print("Local Training Runner v1 - 测试")
    print("="*70)
    
    # 配置
    config = TrainingConfig(
        num_epochs=2,
        batch_size=8,
        save_every_n_steps=50,
    )
    
    # 创建组件
    runner = create_training_runner(config)
    model = MinimalNativeBackbone(config)
    train_dataset = MockDataset(num_samples=100, config=config)
    val_dataset = MockDataset(num_samples=20, config=config)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    history = runner.train(model, train_dataset, val_dataset)
    
    print("\n✓ 本地训练运行器工作正常")
    print(f"  训练损失记录数: {len(history['train_losses'])}")
    print(f"  评估记录数: {len(history['val_metrics'])}")
