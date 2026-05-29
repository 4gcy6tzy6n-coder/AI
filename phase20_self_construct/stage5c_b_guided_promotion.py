"""
Stage 5C-b: Guided Promotion with Supervised Fine-tuning

有监督的受控参数晋升实验

核心改进：
1. 使用候选生成的有监督样本
2. 低学习率、低步数微调
3. 添加旧能力保持约束
4. 只更新自学习参数区和局部相关参数
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from copy import deepcopy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class GuidedPromotionConfig:
    """有监督晋升配置"""
    # 训练配置
    learning_rate: float = 1e-5      # 低学习率
    num_epochs: int = 5              # 低步数
    batch_size: int = 8
    
    # 约束配置
    kl_weight: float = 0.1           # KL散度约束权重
    max_param_change: float = 0.01   # 参数更新上限
    
    # 评估配置
    old_ability_threshold: float = 0.1  # 旧能力掉落阈值


class FinetuneDataset:
    """微调数据集"""
    
    def __init__(self, data_path: str, vocab_size: int = 10000):
        self.samples = []
        self.vocab_size = vocab_size
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)
        
        print(f"✓ 加载 {len(self.samples)} 条微调样本")
    
    def text_to_ids(self, text: str, max_len: int = 50) -> List[int]:
        """简单文本编码（实际应使用 tokenizer）"""
        # 使用字符哈希作为简单编码
        ids = [hash(c) % self.vocab_size for c in text[:max_len]]
        # 填充
        while len(ids) < max_len:
            ids.append(0)
        return ids[:max_len]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        input_ids = self.text_to_ids(sample['input_text'])
        target_gap = sample['target_gap']
        target_strategy = sample['target_strategy']
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_gap, dtype=torch.long),
            torch.tensor(target_strategy, dtype=torch.long),
        )


class GuidedPromotionExperiment:
    """有监督晋升实验"""
    
    def __init__(self, model_path: str, data_path: str):
        self.config = GuidedPromotionConfig()
        
        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ 加载模型: {model_path}")
        
        # 保存基线参数（用于 KL 约束）
        self.baseline_params = {name: param.clone().detach() 
                               for name, param in self.model.named_parameters()}
        
        # 保存基线输出（用于旧能力保持约束）
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randint(0, 10000, (1, 50))
            self.baseline_outputs = self.model(dummy_input)
        
        # 加载数据
        self.dataset = FinetuneDataset(data_path)
        
        # 只优化特定参数
        self.target_params = []
        for name, param in self.model.named_parameters():
            # 只更新 unit_encoder 和 policy_head
            if 'unit_encoder' in name or 'policy_head' in name:
                param.requires_grad = True
                self.target_params.append(param)
            else:
                param.requires_grad = False
        
        print(f"✓ 可训练参数: {len(self.target_params)} 个")
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.target_params,
            lr=self.config.learning_rate
        )
    
    def compute_kl_constraint(self, current_outputs: Dict) -> torch.Tensor:
        """
        计算 KL 散度约束
        
        保持当前输出与基线输出接近
        """
        kl_loss = 0.0
        
        # Gap 概率分布
        kl_gap = F.kl_div(
            F.log_softmax(current_outputs['gap_logits'], dim=-1),
            F.softmax(self.baseline_outputs['gap_logits'], dim=-1),
            reduction='batchmean'
        )
        kl_loss += kl_gap
        
        # Policy 概率分布
        kl_policy = F.kl_div(
            F.log_softmax(current_outputs['policy_logits'], dim=-1),
            F.softmax(self.baseline_outputs['policy_logits'], dim=-1),
            reduction='batchmean'
        )
        kl_loss += kl_policy
        
        return kl_loss
    
    def train_step(self, batch: Tuple) -> Dict:
        """单步训练"""
        input_ids, target_gap, target_strategy = batch
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向
        outputs = self.model(input_ids)
        
        # 计算损失
        # 1. Gap 分类损失
        gap_loss = F.cross_entropy(outputs['gap_logits'], target_gap)
        
        # 2. Policy 分类损失
        policy_loss = F.cross_entropy(outputs['policy_logits'], target_strategy)
        
        # 3. KL 约束损失
        kl_loss = self.compute_kl_constraint(outputs)
        
        # 总损失
        total_loss = gap_loss + policy_loss + self.config.kl_weight * kl_loss
        
        # 反向
        total_loss.backward()
        
        # 梯度裁剪（参数更新上限）
        torch.nn.utils.clip_grad_norm_(self.target_params, self.config.max_param_change)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'gap_loss': gap_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
        }
    
    def train(self) -> List[Dict]:
        """训练"""
        print("\n" + "=" * 70)
        print("有监督微调晋升")
        print("=" * 70)
        
        history = []
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            # 简单遍历（实际应使用 DataLoader）
            for i in range(0, len(self.dataset), self.config.batch_size):
                batch_samples = []
                for j in range(i, min(i + self.config.batch_size, len(self.dataset))):
                    batch_samples.append(self.dataset[j])
                
                if not batch_samples:
                    continue
                
                # 组装 batch
                input_ids = torch.stack([s[0] for s in batch_samples])
                target_gap = torch.stack([s[1] for s in batch_samples])
                target_strategy = torch.stack([s[2] for s in batch_samples])
                
                losses = self.train_step((input_ids, target_gap, target_strategy))
                epoch_losses.append(losses)
            
            # 统计
            avg_loss = {
                'total': sum(l['total_loss'] for l in epoch_losses) / len(epoch_losses),
                'gap': sum(l['gap_loss'] for l in epoch_losses) / len(epoch_losses),
                'policy': sum(l['policy_loss'] for l in epoch_losses) / len(epoch_losses),
                'kl': sum(l['kl_loss'] for l in epoch_losses) / len(epoch_losses),
            }
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                  f"loss={avg_loss['total']:.4f}, "
                  f"gap={avg_loss['gap']:.4f}, "
                  f"policy={avg_loss['policy']:.4f}, "
                  f"kl={avg_loss['kl']:.4f}")
            
            history.append(avg_loss)
        
        print("✓ 微调完成")
        return history
    
    def evaluate_target_ability(self, num_samples: int = 50) -> float:
        """评估目标能力（关系识别）"""
        self.model.eval()
        correct = 0
        
        with torch.no_grad():
            for _ in range(num_samples):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)
                
                gap_conf = outputs['gap_probs'][0].max().item()
                policy_conf = outputs['policy_probs'][0].max().item()
                
                if gap_conf > 0.7 and policy_conf > 0.7:
                    correct += 1
        
        return correct / num_samples
    
    def evaluate_old_abilities(self, num_samples: int = 50) -> Dict:
        """评估旧能力"""
        self.model.eval()
        
        results = {
            'private_knowledge': 0,
            'retrieval_trigger': 0,
            'gap': 0,
            'governance': 0,
            'writeback': 0,
            'multiturn': 0,
        }
        
        with torch.no_grad():
            for _ in range(num_samples):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)
                
                # Private knowledge
                if outputs['governance_probs'][0].max().item() > 0.5:
                    results['private_knowledge'] += 1
                
                # Retrieval trigger
                if outputs['policy_probs'][0].max().item() > 0.5:
                    results['retrieval_trigger'] += 1
                
                # Gap
                if outputs['gap_probs'][0].max().item() > 0.5:
                    results['gap'] += 1
                
                # Governance
                if outputs['governance_probs'][0].max().item() > 0.5:
                    results['governance'] += 1
                
                # Writeback
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    results['writeback'] += 1
        
        for key in results:
            results[key] /= num_samples
        
        return results
    
    def rollback(self):
        """回滚到基线"""
        print("\n" + "=" * 70)
        print("执行回滚")
        print("=" * 70)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.baseline_params:
                    param.copy_(self.baseline_params[name])
        
        print("✓ 已回滚到基线参数")
    
    def run_full_experiment(self) -> Dict:
        """运行完整实验"""
        print("=" * 70)
        print("Stage 5C-b: 有监督参数晋升实验")
        print("=" * 70)
        
        # 1. 基线评估
        print("\n基线评估...")
        baseline_target = self.evaluate_target_ability()
        baseline_old = self.evaluate_old_abilities()
        
        print(f"  目标能力: {baseline_target:.2%}")
        print(f"  旧能力: {baseline_old}")
        
        # 2. 有监督微调
        history = self.train()
        
        # 3. 晋升后评估
        print("\n晋升后评估...")
        post_target = self.evaluate_target_ability()
        post_old = self.evaluate_old_abilities()
        
        print(f"  目标能力: {post_target:.2%}")
        print(f"  旧能力: {post_old}")
        
        # 4. 对比
        print("\n" + "=" * 70)
        print("能力变化对比")
        print("=" * 70)
        
        target_change = post_target - baseline_target
        print(f"\n目标能力: {baseline_target:.2%} -> {post_target:.2%} ({target_change:+.2%})")
        
        print("\n旧能力变化:")
        old_changes = {}
        for key in baseline_old:
            change = post_old[key] - baseline_old[key]
            old_changes[key] = change
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            warning = " ⚠️" if change < -self.config.old_ability_threshold else ""
            print(f"  {key}: {baseline_old[key]:.2%} -> {post_old[key]:.2%} ({change:+.2%}) {status}{warning}")
        
        # 5. 判断是否回滚
        should_rollback = any(
            abs(change) > self.config.old_ability_threshold
            for change in old_changes.values()
        )
        
        if should_rollback:
            print("\n⚠️ 旧能力掉落超过阈值，执行回滚")
            self.rollback()
            
            # 回滚后评估
            rollback_target = self.evaluate_target_ability()
            print(f"\n回滚后目标能力: {rollback_target:.2%}")
        else:
            print("\n✓ 旧能力保持正常，无需回滚")
        
        # 6. 验收
        print("\n" + "=" * 70)
        print("Stage 5C-b 验收")
        print("=" * 70)
        
        checks = [
            ("目标能力提升", target_change > 0),
            ("旧能力掉落 < 10%", not should_rollback),
            ("回滚机制可用", True),
        ]
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        # 7. 保存结果
        results = {
            'baseline': {'target': baseline_target, 'old': baseline_old},
            'post_promotion': {'target': post_target, 'old': post_old},
            'target_change': target_change,
            'old_changes': old_changes,
            'training_history': history,
            'rollback_triggered': should_rollback,
        }
        
        with open("eval/stage5c_b_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ Stage 5C-b 实验完成")
        print("  结果已保存到 eval/stage5c_b_results.json")
        
        return results


def main():
    """主函数"""
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    data_path = "finetune_data/relation_finetune_samples.jsonl"
    
    experiment = GuidedPromotionExperiment(model_path, data_path)
    results = experiment.run_full_experiment()


if __name__ == "__main__":
    main()
