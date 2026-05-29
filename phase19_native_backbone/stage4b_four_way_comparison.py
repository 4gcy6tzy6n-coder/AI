"""
Stage 4b: Four-Way Comparison

4 组对照实验：
1. Native-128: hidden_dim=128 (原版)
2. Native-64: hidden_dim=64 (小容量)
3. Transformer-Matched: 参数量接近 Native-128
4. Transformer-Small: 参数量接近 Native-64

目标：
- 验证增加数据后原生主干是否更稳
- 验证减小容量后原生主干是否更适配小样本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import random
from typing import Dict, List
from dataclasses import dataclass
import time

from native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from transformer_baseline_tiny import TransformerBaselineTiny, TransformerTinyConfig


@dataclass
class Stage4bConfig:
    """Stage 4b 配置"""
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    dropout: float = 0.3
    patience: int = 7
    device: str = 'cpu'
    seed: int = 42


class Stage4bDataset(Dataset):
    """Stage 4b 数据集"""
    
    def __init__(self, data_path: str, vocab_size: int = 10000, max_len: int = 100):
        self.samples = []
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def _encode_text(self, text: str) -> List[int]:
        ids = []
        for char in text[:self.max_len]:
            hash_val = hash(char) % self.vocab_size
            ids.append(abs(hash_val))
        while len(ids) < self.max_len:
            ids.append(0)
        return ids[:self.max_len]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_data = sample.get('input', {})
        if 'user_query' in input_data:
            text = input_data['user_query']
        elif 'conversation_history' in input_data:
            history = input_data['conversation_history']
            texts = []
            for turn in history[-5:]:
                if isinstance(turn, dict):
                    texts.append(f"{turn.get('role', '')}: {turn.get('content', '')}")
            texts.append(f"user: {input_data.get('user_query', '')}")
            text = " | ".join(texts)
        else:
            text = ""
        
        input_ids = self._encode_text(text)
        
        output = sample.get('output', {})
        
        gap_map = {'NO_GAP': 0, 'RETRIEVABLE': 1, 'HIGH_RISK': 2}
        gap_label = gap_map.get(output.get('gap_type', 'NO_GAP'), 0)
        
        policy_map = {'DIRECT': 0, 'RETRIEVAL_FIRST': 1, 'CONSERVATIVE': 2, 'DECLINE': 3, 'REVIEW': 4}
        policy_label = policy_map.get(output.get('response_strategy', 'DIRECT'), 0)
        
        memory_decision = output.get('memory_decision', {})
        if not memory_decision.get('should_writeback', False):
            wb_label = 0
        elif memory_decision.get('target_layer') == 'long_term':
            wb_label = 2
        else:
            wb_label = 1
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'gap_label': torch.tensor(gap_label, dtype=torch.long),
            'policy_label': torch.tensor(policy_label, dtype=torch.long),
            'writeback_label': torch.tensor(wb_label, dtype=torch.long),
        }


def create_model(model_type: str, hidden_dim: int, dropout: float):
    """创建模型"""
    if model_type == 'native':
        config = NativeTinyConfig(hidden_dim=hidden_dim, dropout=dropout)
        return NativeBackboneTinyV1(config)
    else:  # transformer
        config = TransformerTinyConfig(hidden_dim=hidden_dim, dropout=dropout)
        return TransformerBaselineTiny(config)


def train_model(model, train_loader, val_loader, config: Stage4bConfig, model_name: str):
    """训练模型"""
    print(f"\n{'='*70}")
    print(f"训练 {model_name}")
    print(f"{'='*70}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    gap_criterion = nn.CrossEntropyLoss()
    policy_criterion = nn.CrossEntropyLoss()
    wb_criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(config.num_epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids']
            gap_labels = batch['gap_label']
            policy_labels = batch['policy_label']
            wb_labels = batch['writeback_label']
            
            optimizer.zero_grad()
            
            outputs = model(input_ids)
            
            loss_gap = gap_criterion(outputs['gap_logits'], gap_labels)
            loss_policy = policy_criterion(outputs['policy_logits'], policy_labels)
            loss_wb = wb_criterion(outputs['writeback_logits'], wb_labels)
            
            loss = loss_gap + loss_policy + loss_wb
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # 验证
        model.eval()
        val_gap_correct = 0
        val_policy_correct = 0
        val_wb_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                gap_labels = batch['gap_label']
                policy_labels = batch['policy_label']
                wb_labels = batch['writeback_label']
                
                outputs = model(input_ids)
                
                val_gap_correct += (outputs['gap_type'] == gap_labels).sum().item()
                val_policy_correct += (outputs['strategy'] == policy_labels).sum().item()
                val_wb_correct += (outputs['writeback_decision'] == wb_labels).sum().item()
                val_total += input_ids.size(0)
        
        val_gap_acc = val_gap_correct / val_total if val_total > 0 else 0
        val_policy_acc = val_policy_correct / val_total if val_total > 0 else 0
        val_wb_acc = val_wb_correct / val_total if val_total > 0 else 0
        avg_val_acc = (val_gap_acc + val_policy_acc + val_wb_acc) / 3
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_gap_acc': val_gap_acc,
            'val_policy_acc': val_policy_acc,
            'val_wb_acc': val_wb_acc,
        })
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 4 == 0 or patience_counter == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Gap: {val_gap_acc:.1%} | "
                  f"Policy: {val_policy_acc:.1%} | "
                  f"WB: {val_wb_acc:.1%}")
        
        if patience_counter >= config.patience:
            print(f"早停于 Epoch {epoch+1}")
            break
    
    return best_val_acc, history


def evaluate_model(model, test_loader, model_name: str):
    """评估模型"""
    model.eval()
    
    gap_correct = 0
    policy_correct = 0
    wb_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            gap_labels = batch['gap_label']
            policy_labels = batch['policy_label']
            wb_labels = batch['writeback_label']
            
            outputs = model(input_ids)
            
            gap_correct += (outputs['gap_type'] == gap_labels).sum().item()
            policy_correct += (outputs['strategy'] == policy_labels).sum().item()
            wb_correct += (outputs['writeback_decision'] == wb_labels).sum().item()
            total += input_ids.size(0)
    
    gap_acc = gap_correct / total if total > 0 else 0
    policy_acc = policy_correct / total if total > 0 else 0
    wb_acc = wb_correct / total if total > 0 else 0
    
    print(f"\n{model_name}:")
    print(f"  Gap Acc: {gap_acc:.1%}")
    print(f"  Policy Acc: {policy_acc:.1%}")
    print(f"  Writeback Acc: {wb_acc:.1%}")
    
    return {'gap_acc': gap_acc, 'policy_acc': policy_acc, 'writeback_acc': wb_acc}


def run_stage4b_comparison():
    """运行 Stage 4b 四组对照实验"""
    print("=" * 70)
    print("Stage 4b: 四组对照实验")
    print("=" * 70)
    
    config = Stage4bConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # 加载数据
    print("\n加载 Stage 4b 数据...")
    train_dataset = Stage4bDataset("datasets/stage4b_train.jsonl")
    val_dataset = Stage4bDataset("datasets/stage4b_val.jsonl")
    test_dataset = Stage4bDataset("datasets/stage4b_test.jsonl")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    print(f"测试集: {len(test_dataset)} 条")
    
    # 定义 4 组实验
    experiments = [
        ('Native-128', 'native', 128),
        ('Native-64', 'native', 64),
        ('Transformer-Matched', 'transformer', 128),
        ('Transformer-Small', 'transformer', 64),
    ]
    
    results = {}
    
    for exp_name, model_type, hidden_dim in experiments:
        print(f"\n{'='*70}")
        print(f"实验: {exp_name} (hidden_dim={hidden_dim})")
        print(f"{'='*70}")
        
        # 创建模型
        model = create_model(model_type, hidden_dim, config.dropout)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")
        
        # 训练
        best_val_acc, history = train_model(model, train_loader, val_loader, config, exp_name)
        
        # 测试
        test_results = evaluate_model(model, test_loader, exp_name)
        
        # 保存结果
        results[exp_name] = {
            'model_type': model_type,
            'hidden_dim': hidden_dim,
            'params': params,
            'best_val_acc': best_val_acc,
            'test_results': test_results,
            'history': history,
        }
        
        # 保存模型
        torch.save(model.state_dict(), f"checkpoints/{exp_name.replace('-', '_').lower()}.pt")
    
    # 对比总结
    print("\n" + "=" * 70)
    print("Stage 4b 对比总结")
    print("=" * 70)
    
    print("\n参数量对比:")
    for exp_name, result in results.items():
        print(f"  {exp_name}: {result['params']:,}")
    
    print("\n验证集最佳准确率:")
    for exp_name, result in results.items():
        print(f"  {exp_name}: {result['best_val_acc']:.1%}")
    
    print("\n测试集 Policy 准确率:")
    for exp_name, result in results.items():
        print(f"  {exp_name}: {result['test_results']['policy_acc']:.1%}")
    
    print("\n测试集 Gap 准确率:")
    for exp_name, result in results.items():
        print(f"  {exp_name}: {result['test_results']['gap_acc']:.1%}")
    
    # 保存结果
    with open("eval/stage4b_results.json", 'w') as f:
        # 移除 history 后再保存（太大）
        save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} 
                       for k, v in results.items()}
        json.dump(save_results, f, indent=2)
    
    print("\n✓ 所有实验完成")
    print("✓ 结果已保存到 eval/stage4b_results.json")
    
    return results


if __name__ == "__main__":
    results = run_stage4b_comparison()
