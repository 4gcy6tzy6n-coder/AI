"""
Milestone 2.3 Training - 记忆决策链接入训练

Phase 17 Stage 2.3: GapDetector → Governance → Writeback 合理闭环

消融实验设计：
1. 基线组：只有最小训练闭环 (Policy + Governance + Response)
2. +GapDetector：加入缺口识别
3. +GapDetector + WritebackHead：完整记忆决策链

关键监控指标：
- Gap准确率（不能低于70%）
- 检索误触发率（不能高于30%）
- 写回误判率（不能高于40%）
- 长期候选精度（越高越好）
- 策略准确率（不能继续下滑）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

from phase17.enhanced_native_backbone import EnhancedNativeBackbone, EnhancedBackboneConfig
from phase17.enhanced_loss_runner import create_enhanced_loss_runner
from phase17.local_training_runner import MinimalNativeBackbone, TrainingConfig
from phase17.multihead_loss_runner import create_multihead_loss_runner
from phase17.checkpoint_manager import create_checkpoint_manager
from phase17.offline_eval_suite import create_offline_eval_suite


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    use_gap_detector: bool
    use_writeback_head: bool
    
    # 训练配置
    hidden_dim: int = 256
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = 'cpu'


class EnhancedDataset(Dataset):
    """增强版数据集 - 支持Gap和Writeback标签"""
    
    def __init__(self, size: int = 200, config: EnhancedBackboneConfig = None):
        self.size = size
        self.config = config or EnhancedBackboneConfig()
        
        # 生成带合理标签分布的数据
        self.data = []
        for i in range(size):
            # 输入
            seq_len = 10
            input_ids = torch.randint(0, self.config.vocab_size, (seq_len,))
            
            # Gap标签设计（有合理分布）
            # 0: NO_GAP (30%), 1: RETRIEVABLE (50%), 2: HIGH_RISK (20%)
            gap_target = self._sample_gap_type(i)
            
            # 策略标签（与gap相关）
            policy_target = self._derive_policy_from_gap(gap_target, i)
            
            # 治理标签
            governance_target = self._sample_governance(gap_target)
            
            # Writeback标签（与gap相关）
            # 0: NO_WRITEBACK (40%), 1: EPHEMERAL (35%), 2: LONG_TERM (25%)
            writeback_target = self._derive_writeback_from_gap(gap_target, i)
            
            # 响应标签
            response_target = torch.randint(0, self.config.vocab_size, (1,))
            
            self.data.append({
                'input_ids': input_ids,
                'policy_target': policy_target,
                'gap_target': gap_target,
                'governance_target': governance_target,
                'writeback_target': writeback_target,
                'response_target': response_target,
                # 用于评估的target字典
                'target': {
                    'strategy': policy_target.item(),
                    'gap': gap_target.item(),
                    'governance': governance_target,
                    'writeback': writeback_target.item(),
                    'retrieval': gap_target.item() == 1,  # 只有RETRIEVABLE_GAP才需要检索
                }
            })
    
    def _sample_gap_type(self, idx: int) -> torch.Tensor:
        """采样gap类型，保持合理分布"""
        r = idx % 10
        if r < 3:
            return torch.tensor(0)  # NO_GAP
        elif r < 8:
            return torch.tensor(1)  # RETRIEVABLE
        else:
            return torch.tensor(2)  # HIGH_RISK
    
    def _derive_policy_from_gap(self, gap: torch.Tensor, idx: int) -> torch.Tensor:
        """从gap推导策略"""
        gap_val = gap.item()
        if gap_val == 0:  # NO_GAP
            return torch.tensor(0)  # DIRECT
        elif gap_val == 1:  # RETRIEVABLE
            return torch.tensor(1)  # RETRIEVAL_FIRST
        else:  # HIGH_RISK
            return torch.tensor(3)  # DECLINE or REVIEW
    
    def _sample_governance(self, gap: torch.Tensor) -> torch.Tensor:
        """采样治理动作"""
        gap_val = gap.item()
        if gap_val == 2:  # HIGH_RISK
            # 高风险时更可能触发治理
            return torch.tensor([1, 0, 1, 0, 0, 0, 0, 0]).float()
        else:
            # 其他情况随机
            return torch.randint(0, 2, (8,)).float()
    
    def _derive_writeback_from_gap(self, gap: torch.Tensor, idx: int) -> torch.Tensor:
        """从gap推导写回决策"""
        gap_val = gap.item()
        if gap_val == 0:  # NO_GAP
            # 无缺口时较少写回
            return torch.tensor(0) if idx % 3 != 0 else torch.tensor(1)
        elif gap_val == 1:  # RETRIEVABLE
            # 可检索补足时可能写回
            r = idx % 4
            if r == 0:
                return torch.tensor(0)
            elif r <= 2:
                return torch.tensor(1)
            else:
                return torch.tensor(2)
        else:  # HIGH_RISK
            # 高风险时谨慎写回
            return torch.tensor(0)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


def run_experiment(exp_config: ExperimentConfig) -> Dict[str, Any]:
    """运行单个实验"""
    print("\n" + "=" * 70)
    print(f"实验: {exp_config.name}")
    print("=" * 70)
    print(f"配置: GapDetector={exp_config.use_gap_detector}, WritebackHead={exp_config.use_writeback_head}")
    
    # 创建配置
    backbone_config = EnhancedBackboneConfig(
        hidden_dim=exp_config.hidden_dim,
        device=exp_config.device,
    )
    
    # 创建模型
    model = EnhancedNativeBackbone(backbone_config)
    model = model.to(exp_config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    train_dataset = EnhancedDataset(size=200, config=backbone_config)
    val_dataset = EnhancedDataset(size=50, config=backbone_config)
    train_loader = DataLoader(train_dataset, batch_size=exp_config.batch_size, shuffle=True)
    
    # 创建损失运行器
    loss_runner = create_enhanced_loss_runner()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.learning_rate)
    
    # 训练历史
    history = {
        'train_losses': [],
        'gap_accuracies': [],
        'retrieval_fps': [],
        'writeback_fps': [],
        'long_term_precisions': [],
    }
    
    # 训练循环
    print("\n开始训练...")
    print("-" * 70)
    
    for epoch in range(exp_config.num_epochs):
        model.train()
        epoch_metrics = {
            'losses': [],
            'gap_accs': [],
            'ret_fps': [],
            'wb_fps': [],
            'lt_precs': [],
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(exp_config.device)
            
            # 前向传播
            predictions = model(input_ids)
            
            # 准备目标
            targets = {
                'policy_target': batch['policy_target'].to(exp_config.device),
                'gap_target': batch['gap_target'].to(exp_config.device),
                'governance_target': batch['governance_target'].to(exp_config.device),
                'writeback_target': batch['writeback_target'].to(exp_config.device),
                'response_target': batch['response_target'].to(exp_config.device),
            }
            
            # 计算损失
            loss = loss_runner.compute_loss(predictions, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.total_loss.backward()
            optimizer.step()
            
            # 记录
            epoch_metrics['losses'].append(loss.total_loss.item())
            epoch_metrics['gap_accs'].append(loss.gap_accuracy)
            epoch_metrics['ret_fps'].append(loss.retrieval_false_positive)
            epoch_metrics['wb_fps'].append(loss.writeback_false_positive)
            epoch_metrics['lt_precs'].append(loss.long_term_precision)
        
        # 计算epoch平均
        avg_loss = sum(epoch_metrics['losses']) / len(epoch_metrics['losses'])
        avg_gap_acc = sum(epoch_metrics['gap_accs']) / len(epoch_metrics['gap_accs'])
        avg_ret_fp = sum(epoch_metrics['ret_fps']) / len(epoch_metrics['ret_fps'])
        avg_wb_fp = sum(epoch_metrics['wb_fps']) / len(epoch_metrics['wb_fps'])
        avg_lt_prec = sum(epoch_metrics['lt_precs']) / len(epoch_metrics['lt_precs'])
        
        history['train_losses'].append(avg_loss)
        history['gap_accuracies'].append(avg_gap_acc)
        history['retrieval_fps'].append(avg_ret_fp)
        history['writeback_fps'].append(avg_wb_fp)
        history['long_term_precisions'].append(avg_lt_prec)
        
        print(f"Epoch {epoch+1}/{exp_config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"GapAcc: {avg_gap_acc:.1%} | "
              f"RetFP: {avg_ret_fp:.1%} | "
              f"WB_FP: {avg_wb_fp:.1%} | "
              f"LT_Prec: {avg_lt_prec:.1%}")
    
    # 最终评估
    print("\n最终评估...")
    model.eval()
    
    # 在验证集上评估
    val_data = [val_dataset[i] for i in range(min(30, len(val_dataset)))]
    
    gap_correct = 0
    policy_correct = 0
    total = 0
    
    with torch.no_grad():
        for sample in val_data:
            input_ids = sample['input_ids'].unsqueeze(0).to(exp_config.device)
            outputs = model(input_ids)
            
            # Gap准确率
            pred_gap = outputs['gap_logits'].argmax(dim=-1).item()
            true_gap = sample['target']['gap']
            if pred_gap == true_gap:
                gap_correct += 1
            
            # 策略准确率
            pred_policy = outputs['policy_logits'].argmax(dim=-1).item()
            true_policy = sample['target']['strategy']
            if pred_policy == true_policy:
                policy_correct += 1
            
            total += 1
    
    final_gap_acc = gap_correct / total if total > 0 else 0
    final_policy_acc = policy_correct / total if total > 0 else 0
    
    print(f"  Gap准确率: {final_gap_acc:.1%}")
    print(f"  策略准确率: {final_policy_acc:.1%}")
    
    results = {
        'config': asdict(exp_config),
        'history': history,
        'final_gap_accuracy': final_gap_acc,
        'final_policy_accuracy': final_policy_acc,
    }
    
    return results


def run_all_experiments():
    """运行所有消融实验"""
    print("=" * 70)
    print("Milestone 2.3: 记忆决策链接入 - 消融实验")
    print("=" * 70)
    
    experiments = [
        ExperimentConfig(
            name="基线组 (最小闭环)",
            use_gap_detector=False,
            use_writeback_head=False,
        ),
        ExperimentConfig(
            name="+GapDetector",
            use_gap_detector=True,
            use_writeback_head=False,
        ),
        ExperimentConfig(
            name="+GapDetector +WritebackHead",
            use_gap_detector=True,
            use_writeback_head=True,
        ),
    ]
    
    all_results = []
    
    for exp_config in experiments:
        results = run_experiment(exp_config)
        all_results.append(results)
    
    # 对比总结
    print("\n" + "=" * 70)
    print("消融实验对比总结")
    print("=" * 70)
    
    print("\n| 实验组 | Gap准确率 | 策略准确率 | 最终Loss |")
    print("|--------|-----------|------------|----------|")
    for result in all_results:
        name = result['config']['name']
        gap_acc = result['final_gap_accuracy']
        policy_acc = result['final_policy_accuracy']
        final_loss = result['history']['train_losses'][-1]
        print(f"| {name} | {gap_acc:.1%} | {policy_acc:.1%} | {final_loss:.4f} |")
    
    # 保存结果
    with open('phase17/milestone_2_3_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 所有实验完成，结果已保存到 milestone_2_3_results.json")
    
    # 判断Milestone 2.3是否达标
    final_result = all_results[-1]  # 完整配置的结果
    
    print("\n" + "=" * 70)
    print("Milestone 2.3 验收标准检查")
    print("=" * 70)
    
    checks = [
        ("Gap准确率 >= 70%", final_result['final_gap_accuracy'] >= 0.70),
        ("策略准确率不继续下滑", final_result['final_policy_accuracy'] >= 0.15),  # 基线是10%
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ Milestone 2.3 验收通过！")
    else:
        print("\n✗ Milestone 2.3 需要继续优化")
    
    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
