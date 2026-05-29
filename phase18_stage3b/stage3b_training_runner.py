"""
Stage 3B Training Runner

阶段 3B 训练运行器 - 多轮记忆与写回联合训练

关键设计：
1. 加载 Stage 3A 最佳模型
2. 部分解冻 WritebackHead
3. Policy/Gap/Governance 使用低学习率
4. 关注多轮记忆稳定性和写回误判率

数据来源：
- D4: 多轮记忆 (85%)
- D5: 真实样本回放 (15%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from phase17.enhanced_native_backbone import EnhancedNativeBackbone, EnhancedBackboneConfig
from phase17.enhanced_loss_runner import create_enhanced_loss_runner
from phase18_stage3b.stage3b_dataloader import Stage3BDataset, Stage3BConfig, create_stage3b_dataloaders


@dataclass
class Stage3BTrainingConfig:
    """Stage 3B 训练配置"""
    # 模型配置
    hidden_dim: int = 256
    vocab_size: int = 10000
    max_seq_len: int = 150
    
    # 训练配置
    num_epochs: int = 8
    batch_size: int = 16
    
    # 分层学习率
    base_lr: float = 1e-4      # Policy/Gap/Governance 低学习率
    writeback_lr: float = 5e-4  # WritebackHead 较高学习率
    
    weight_decay: float = 1e-4
    device: str = 'cpu'
    
    # 损失权重
    policy_weight: float = 1.0
    gap_weight: float = 1.0
    governance_weight: float = 1.0
    writeback_weight: float = 0.5  # Stage 3B 恢复写回训练
    response_weight: float = 0.3
    
    # 数据配置
    total_samples: int = 300
    d4_ratio: float = 0.85
    d5_ratio: float = 0.15
    
    # 检查点
    checkpoint_dir: str = "phase18_stage3b/checkpoints"
    stage3a_checkpoint: str = "phase18_stage3a/checkpoints/best_model.pt"
    save_best: bool = True
    
    # 早停
    patience: int = 3
    min_delta: float = 0.005


class Stage3BTrainingRunner:
    """Stage 3B 训练运行器"""
    
    def __init__(self, config: Stage3BTrainingConfig):
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
        
        # 加载 Stage 3A 检查点
        self._load_stage3a_checkpoint()
        
        # 部分解冻参数
        self._setup_partial_freeze()
        
        # 创建分层优化器
        self.optimizer = self._create_layered_optimizer()
        
        # 创建损失运行器
        self.loss_runner = create_enhanced_loss_runner(
            policy_weight=config.policy_weight,
            gap_weight=config.gap_weight,
            governance_weight=config.governance_weight,
            writeback_weight=config.writeback_weight,
            response_weight=config.response_weight,
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
            'wb_acc': [],
            'wb_fp': [],
            'lt_prec': [],
            'lt_recall': [],
        }
        
        self.best_gap_acc = 0.0
        self.patience_counter = 0
    
    def _load_stage3a_checkpoint(self):
        """加载 Stage 3A 检查点"""
        try:
            checkpoint = torch.load(self.config.stage3a_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 已加载 Stage 3A 模型: {self.config.stage3a_checkpoint}")
            print(f"  原训练 epoch: {checkpoint.get('epoch', 'unknown')}")
        except FileNotFoundError:
            print(f"⚠ Stage 3A 检查点未找到，使用随机初始化")
    
    def _setup_partial_freeze(self):
        """
        部分解冻策略（修正版）：
        - WritebackHead: 完全解冻
        - PolicyHead: 完全解冻（需要适应多轮场景）
        - GapDetector: 完全解冻
        - GovernanceHead: 完全解冻
        - UnitEncoder: 冻结（保持语义编码稳定）
        - ResponseDecoder: 冻结
        """
        print("\n设置部分解冻...")
        
        # 1. WritebackHead 完全解冻
        for param in self.model.writeback_head.parameters():
            param.requires_grad = True
        
        # 2. PolicyHead: 完全解冻（需要学习多轮策略）
        for param in self.model.policy_head.parameters():
            param.requires_grad = True
        
        # 3. GapDetector: 完全解冻
        for param in self.model.gap_detector.parameters():
            param.requires_grad = True
        
        # 4. GovernanceHead: 完全解冻
        for param in self.model.governance_head.parameters():
            param.requires_grad = True
        
        # 5. UnitEncoder: 冻结（保持语义编码稳定）
        for param in self.model.unit_encoder.parameters():
            param.requires_grad = False
        
        # 6. ResponseDecoder: 冻结
        for param in self.model.response_decoder.parameters():
            param.requires_grad = False
        
        # 统计
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        print(f"  - WritebackHead: 完全解冻")
        print(f"  - Policy/Gap/Gov: 完全解冻（适应多轮场景）")
        print(f"  - UnitEncoder/ResponseDecoder: 冻结")
    
    def _create_layered_optimizer(self) -> torch.optim.Optimizer:
        """创建分层优化器，不同参数组使用不同学习率"""
        
        # 参数分组
        writeback_params = []
        base_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'writeback_head' in name:
                writeback_params.append(param)
            else:
                base_params.append(param)
        
        # 分层优化器
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.config.base_lr, 'weight_decay': self.config.weight_decay},
            {'params': writeback_params, 'lr': self.config.writeback_lr, 'weight_decay': self.config.weight_decay},
        ])
        
        print(f"\n分层学习率:")
        print(f"  基础参数 (Policy/Gap/Gov): lr={self.config.base_lr}")
        print(f"  WritebackHead: lr={self.config.writeback_lr}")
        
        return optimizer
    
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
                'writeback_target': batch['writeback_target'].to(self.device),
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
        
        # 统计指标
        gap_correct = policy_correct = gov_correct = 0
        wb_correct = 0
        wb_fp_count = wb_fp_total = 0  # 写回误判
        lt_pred_count = lt_correct = 0  # 长期候选精度
        lt_total = 0  # 长期候选 recall
        
        total = 0
        
        # 多轮稳定性统计
        turn_correct = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                
                predictions = self.model(input_ids)
                
                targets = {
                    'policy_target': batch['policy_target'].to(self.device),
                    'gap_target': batch['gap_target'].to(self.device),
                    'governance_target': batch['governance_target'].to(self.device),
                    'writeback_target': batch['writeback_target'].to(self.device),
                    'response_target': torch.randint(0, self.config.vocab_size, (input_ids.size(0),)).to(self.device),
                }
                
                loss = self.loss_runner.compute_loss(predictions, targets)
                val_losses.append(loss.total_loss.item())
                
                batch_size = input_ids.size(0)
                
                # Gap 准确率
                pred_gap = predictions['gap_logits'].argmax(dim=-1)
                gap_correct += (pred_gap == targets['gap_target']).sum().item()
                
                # Policy 准确率
                pred_policy = predictions['policy_logits'].argmax(dim=-1)
                policy_correct += (pred_policy == targets['policy_target']).sum().item()
                
                # 多轮稳定性
                for i in range(batch_size):
                    turn = batch['turn_number'][i].item()
                    if turn in turn_correct:
                        turn_correct[turn][0] += 1
                        if pred_policy[i] == targets['policy_target'][i]:
                            turn_correct[turn][1] += 1
                
                # Governance 准确率
                pred_gov = (predictions['governance_logits'] > 0.5).float()
                gov_correct += ((pred_gov == targets['governance_target']).all(dim=1)).sum().item()
                
                # Writeback 准确率
                pred_wb = predictions['writeback_logits'].argmax(dim=-1)
                true_wb = targets['writeback_target']
                wb_correct += (pred_wb == true_wb).sum().item()
                
                # 写回误判率 (False Positive)
                for i in range(batch_size):
                    if true_wb[i] == 0:  # 不该写回
                        wb_fp_total += 1
                        if pred_wb[i] != 0:  # 但预测要写回
                            wb_fp_count += 1
                
                # 长期候选 Precision
                for i in range(batch_size):
                    if pred_wb[i] == 2:  # 预测为长期候选
                        lt_pred_count += 1
                        if true_wb[i] == 2:
                            lt_correct += 1
                
                # 长期候选 Recall
                for i in range(batch_size):
                    if true_wb[i] == 2:  # 真实是长期候选
                        lt_total += 1
                
                total += batch_size
        
        # 计算多轮稳定性
        turn_stability = {}
        for turn, (count, correct) in turn_correct.items():
            if count > 0:
                turn_stability[turn] = correct / count
        
        return {
            'loss': sum(val_losses) / len(val_losses),
            'gap_acc': gap_correct / total,
            'policy_acc': policy_correct / total,
            'governance_acc': gov_correct / total,
            'wb_acc': wb_correct / total,
            'wb_fp': wb_fp_count / wb_fp_total if wb_fp_total > 0 else 0,
            'lt_prec': lt_correct / lt_pred_count if lt_pred_count > 0 else 0,
            'lt_recall': lt_correct / lt_total if lt_total > 0 else 0,
            'turn_stability': turn_stability,
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("Stage 3B 训练开始")
        print("=" * 70)
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"Epochs: {self.config.num_epochs}")
        print("-" * 70)
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            
            # 学习率调度（基于 Gap 准确率）
            self.scheduler.step(val_metrics['gap_acc'])
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['gap_acc'].append(val_metrics['gap_acc'])
            self.history['policy_acc'].append(val_metrics['policy_acc'])
            self.history['governance_acc'].append(val_metrics['governance_acc'])
            self.history['wb_acc'].append(val_metrics['wb_acc'])
            self.history['wb_fp'].append(val_metrics['wb_fp'])
            self.history['lt_prec'].append(val_metrics['lt_prec'])
            self.history['lt_recall'].append(val_metrics['lt_recall'])
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                  f"Gap: {val_metrics['gap_acc']:.1%} | "
                  f"Policy: {val_metrics['policy_acc']:.1%} | "
                  f"WB: {val_metrics['wb_acc']:.1%} | "
                  f"WB_FP: {val_metrics['wb_fp']:.1%}")
            
            # 保存最佳模型
            if val_metrics['gap_acc'] > self.best_gap_acc + self.config.min_delta:
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
    
    def save_report(self, path: str = "phase18_stage3b/eval/stage3b_report.json"):
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
                'final_wb_fp': self.history['wb_fp'][-1] if self.history['wb_fp'] else 0,
                'final_lt_prec': self.history['lt_prec'][-1] if self.history['lt_prec'] else 0,
            },
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存: {path}")


def run_stage3b_training():
    """运行 Stage 3B 训练"""
    print("=" * 70)
    print("Stage 3B: 多轮记忆与写回联合训练")
    print("=" * 70)
    print("\n关键目标:")
    print("  1. 多轮记忆稳定性")
    print("  2. WritebackHead 可控性")
    print("  3. 真实样本回放行为一致性")
    
    # 配置
    config = Stage3BTrainingConfig()
    
    # 创建数据加载器
    data_config = Stage3BConfig(
        total_samples=config.total_samples,
        d4_ratio=config.d4_ratio,
        d5_ratio=config.d5_ratio,
    )
    train_loader, val_loader = create_stage3b_dataloaders(
        data_config,
        batch_size=config.batch_size,
    )
    
    # 创建训练器
    runner = Stage3BTrainingRunner(config)
    
    # 训练
    history = runner.train(train_loader, val_loader)
    
    # 最终评估
    print("\n" + "=" * 70)
    print("Stage 3B 最终评估")
    print("=" * 70)
    
    final_metrics = runner.evaluate(val_loader)
    
    print(f"\n核心指标:")
    print(f"  Gap 准确率: {final_metrics['gap_acc']:.1%}")
    print(f"  策略准确率: {final_metrics['policy_acc']:.1%}")
    print(f"  治理准确率: {final_metrics['governance_acc']:.1%}")
    print(f"  写回准确率: {final_metrics['wb_acc']:.1%}")
    
    print(f"\n写回控制指标:")
    print(f"  写回误判率: {final_metrics['wb_fp']:.1%}")
    print(f"  长期候选 Precision: {final_metrics['lt_prec']:.1%}")
    print(f"  长期候选 Recall: {final_metrics['lt_recall']:.1%}")
    
    print(f"\n多轮稳定性:")
    for turn, acc in final_metrics['turn_stability'].items():
        print(f"  Turn {turn}: {acc:.1%}")
    
    # 验收检查
    print("\n" + "=" * 70)
    print("Stage 3B 验收标准")
    print("=" * 70)
    
    checks = [
        ("写回误判率 <= 25%", final_metrics['wb_fp'] <= 0.25),
        ("Gap 准确率 >= 70%", final_metrics['gap_acc'] >= 0.70),
        ("策略准确率 >= 50%", final_metrics['policy_acc'] >= 0.50),
        ("多轮稳定性可接受", min(final_metrics['turn_stability'].values()) >= 0.6 if final_metrics['turn_stability'] else False),
    ]
    
    all_passed = all(passed for _, passed in checks)
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print("\n✓ Stage 3B 验收通过！框架在多轮记忆下仍然稳定")
    else:
        print("\n✗ Stage 3B 需要继续优化")
    
    # 保存报告
    runner.save_report()
    
    return runner, final_metrics


if __name__ == "__main__":
    runner, metrics = run_stage3b_training()
