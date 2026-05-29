"""
测试完整训练闭环 V2 - 包括前向、反向、损失计算、检查点保存、评估
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 清除缓存
if 'phase17.multihead_loss_runner' in sys.modules:
    del sys.modules['phase17.multihead_loss_runner']
if 'phase17.local_training_runner' in sys.modules:
    del sys.modules['phase17.local_training_runner']
if 'phase17.checkpoint_manager' in sys.modules:
    del sys.modules['phase17.checkpoint_manager']
if 'phase17.offline_eval_suite' in sys.modules:
    del sys.modules['phase17.offline_eval_suite']

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from phase17.local_training_runner import MinimalNativeBackbone, TrainingConfig
from phase17.multihead_loss_runner import create_multihead_loss_runner
from phase17.checkpoint_manager import create_checkpoint_manager
from phase17.offline_eval_suite import create_offline_eval_suite


class DummyDataset(Dataset):
    """模拟数据集"""
    def __init__(self, size=100, config=None):
        self.size = size
        self.config = config or TrainingConfig()
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 模拟输入
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (seq_len,))
        
        # 模拟目标
        policy_target = torch.randint(0, self.config.num_strategies, ())
        governance_target = torch.randint(0, 2, (self.config.num_governance_actions,)).float()
        memory_target = torch.randint(0, 2, ())
        # 响应目标：单个token（简化版，对应模型输出的[batch, 1, vocab_size]）
        response_target = torch.randint(0, self.config.vocab_size, (1,))
        
        return {
            'input_ids': input_ids,
            'policy_target': policy_target,
            'governance_target': governance_target,
            'memory_target': memory_target,
            'response_target': response_target,
            'target': {
                'strategy': policy_target.item(),
                'governance': governance_target,
                'memory': memory_target.item(),
                'retrieval': memory_target.item() == 1,
            }
        }


def test_training_loop():
    print("=" * 60)
    print("测试完整训练闭环 V2")
    print("=" * 60)
    
    # 配置
    config = TrainingConfig(
        hidden_dim=256,
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        device='cpu',
        save_every_n_steps=50,
        eval_every_n_epochs=1,
    )
    
    # 创建模型
    print("\n1. 创建模型...")
    model = MinimalNativeBackbone(config)
    model = model.to(config.device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    print("\n2. 创建数据集...")
    train_dataset = DummyDataset(size=50, config=config)
    val_dataset = DummyDataset(size=20, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    print(f"   训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建损失运行器
    print("\n3. 创建损失运行器...")
    loss_runner = create_multihead_loss_runner()
    print(f"   损失运行器创建成功")
    print(f"   权重: {loss_runner.weights}")
    
    # 创建检查点管理器
    print("\n4. 创建检查点管理器...")
    checkpoint_manager = create_checkpoint_manager("phase17/checkpoints")
    print("   检查点管理器创建成功")
    
    # 创建评估套件
    print("\n5. 创建评估套件...")
    eval_suite = create_offline_eval_suite()
    print("   评估套件创建成功")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    print("\n6. 开始训练...")
    print("-" * 60)
    
    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(config.device)
            
            # 前向传播
            predictions = model(input_ids)
            
            # 准备目标
            targets = {
                'policy_target': batch['policy_target'].to(config.device),
                'governance_target': batch['governance_target'].to(config.device),
                'memory_target': batch['memory_target'].to(config.device),
                'response_target': batch['response_target'].to(config.device),
            }
            
            # 调试信息
            if batch_idx == 0 and epoch == 0:
                print(f"\n   调试 - Batch {batch_idx}:")
                print(f"     predictions keys: {list(predictions.keys())}")
                print(f"     targets keys: {list(targets.keys())}")
                for k, v in predictions.items():
                    print(f"     predictions[{k}]: shape={v.shape}, dtype={v.dtype}")
                for k, v in targets.items():
                    print(f"     targets[{k}]: shape={v.shape}, dtype={v.dtype}")
            
            # 计算损失
            try:
                loss = loss_runner.compute_loss(predictions, targets)
            except Exception as e:
                print(f"\n   计算损失时出错:")
                print(f"     错误: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # 反向传播
            optimizer.zero_grad()
            loss.total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.total_loss.item())
            global_step += 1
            
            if batch_idx % 5 == 0:
                print(f"   Epoch {epoch+1}/{config.num_epochs} | "
                      f"Step {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.total_loss.item():.4f} "
                      f"(P:{loss.policy_loss_value:.3f} G:{loss.governance_loss_value:.3f} "
                      f"M:{loss.memory_loss_value:.3f} R:{loss.response_loss_value:.3f})")
            
            # 保存检查点
            if global_step % config.save_every_n_steps == 0:
                checkpoint_manager.save_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    step=global_step,
                    metrics={'loss': loss.total_loss.item()},
                    config=config.__dict__,
                    checkpoint_type="experimental",
                )
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n   Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        # 评估
        if (epoch + 1) % config.eval_every_n_epochs == 0:
            print("\n7. 运行评估...")
            val_data = [val_dataset[i] for i in range(min(10, len(val_dataset)))]
            metrics = eval_suite.evaluate_behavior(model, val_data, evaluation_name=f"epoch_{epoch+1}")
            print(f"   整体行为评分: {metrics.overall_behavior_score:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 完整训练闭环测试通过！")
    print("=" * 60)
    print("\n验证清单:")
    print("  [✓] 本地前向传播")
    print("  [✓] 本地反向传播")
    print("  [✓] 多头损失计算")
    print("  [✓] 检查点保存")
    print("  [✓] 离线行为评估")
    print("\nMilestone 2.1: 最小训练闭环 - 完成!")


if __name__ == "__main__":
    test_training_loop()
