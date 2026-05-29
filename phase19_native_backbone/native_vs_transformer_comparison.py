"""
Native vs Transformer Comparison

阶段 4 核心对比实验：
1. Native Backbone Tiny V1 (GRU-based)
2. Transformer Baseline Tiny

对比维度：
- 参数量
- 训练稳定性
- 核心行为准确率
- 长对话稳定性
- 同义改写鲁棒性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

from native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from transformer_baseline_tiny import TransformerBaselineTiny, TransformerTinyConfig


@dataclass
class ComparisonConfig:
    """对比实验配置"""
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-3
    device: str = 'cpu'
    seed: int = 42


class SimpleDataset(Dataset):
    """简单数据集"""
    
    def __init__(self, data_path: str, vocab_size: int = 10000, max_len: int = 50):
        self.samples = []
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def _encode_text(self, text: str) -> List[int]:
        """简单编码"""
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
        
        # 编码输入
        input_data = sample.get('input', {})
        if 'user_query' in input_data:
            text = input_data['user_query']
        elif 'conversation_history' in input_data:
            # 多轮对话
            history = input_data['conversation_history']
            texts = []
            for turn in history[-5:]:  # 取最近5轮
                if isinstance(turn, dict):
                    texts.append(f"{turn.get('role', '')}: {turn.get('content', '')}")
            texts.append(f"user: {input_data.get('user_query', '')}")
            text = " | ".join(texts)
        else:
            text = ""
        
        input_ids = self._encode_text(text)
        
        # 标签
        output = sample.get('output', {})
        
        # Gap 标签
        gap_map = {'NO_GAP': 0, 'RETRIEVABLE': 1, 'HIGH_RISK': 2}
        gap_label = gap_map.get(output.get('gap_type', 'NO_GAP'), 0)
        
        # Policy 标签
        policy_map = {
            'DIRECT': 0, 'RETRIEVAL_FIRST': 1, 'CONSERVATIVE': 2,
            'DECLINE': 3, 'REVIEW': 4
        }
        policy_label = policy_map.get(output.get('response_strategy', 'DIRECT'), 0)
        
        # Writeback 标签
        memory_decision = output.get('memory_decision', {})
        if not memory_decision.get('should_writeback', False):
            wb_label = 0  # NO_WRITEBACK
        elif memory_decision.get('target_layer') == 'long_term':
            wb_label = 2  # LONG_TERM
        else:
            wb_label = 1  # EPHEMERAL
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'gap_label': torch.tensor(gap_label, dtype=torch.long),
            'policy_label': torch.tensor(policy_label, dtype=torch.long),
            'writeback_label': torch.tensor(wb_label, dtype=torch.long),
        }


def train_model(model, train_loader, val_loader, config: ComparisonConfig, model_name: str):
    """训练模型"""
    print(f"\n{'='*70}")
    print(f"训练 {model_name}")
    print(f"{'='*70}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    gap_criterion = nn.CrossEntropyLoss()
    policy_criterion = nn.CrossEntropyLoss()
    wb_criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_gap_acc': [], 'val_policy_acc': []}
    
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
            
            # 计算损失
            loss_gap = gap_criterion(outputs['gap_logits'], gap_labels)
            loss_policy = policy_criterion(outputs['policy_logits'], policy_labels)
            loss_wb = wb_criterion(outputs['writeback_logits'], wb_labels)
            
            loss = loss_gap + loss_policy + loss_wb
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
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
        
        val_gap_acc = val_gap_correct / val_total
        val_policy_acc = val_policy_correct / val_total
        val_wb_acc = val_wb_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_gap_acc'].append(val_gap_acc)
        history['val_policy_acc'].append(val_policy_acc)
        
        avg_val_acc = (val_gap_acc + val_policy_acc + val_wb_acc) / 3
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Gap: {val_gap_acc:.1%} | "
                  f"Policy: {val_policy_acc:.1%} | "
                  f"WB: {val_wb_acc:.1%}")
    
    return history, best_val_acc


def evaluate_on_test(model, test_loader, test_name: str):
    """在测试集上评估"""
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
    
    gap_acc = gap_correct / total
    policy_acc = policy_correct / total
    wb_acc = wb_correct / total
    
    print(f"\n{test_name}:")
    print(f"  Gap Acc: {gap_acc:.1%}")
    print(f"  Policy Acc: {policy_acc:.1%}")
    print(f"  Writeback Acc: {wb_acc:.1%}")
    
    return {
        'gap_acc': gap_acc,
        'policy_acc': policy_acc,
        'writeback_acc': wb_acc,
    }


def measure_inference_speed(model, input_shape: Tuple[int, int], num_runs: int = 100):
    """测量推理速度"""
    model.eval()
    
    # 预热
    dummy_input = torch.randint(0, 10000, input_shape)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 正式测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    return avg_time


def run_comparison():
    """运行对比实验"""
    print("=" * 70)
    print("阶段 4: Native vs Transformer 对比实验")
    print("=" * 70)
    
    config = ComparisonConfig()
    torch.manual_seed(config.seed)
    
    # 加载数据
    print("\n加载数据集...")
    train_dataset = SimpleDataset("datasets/train.jsonl")
    val_dataset = SimpleDataset("datasets/val.jsonl")
    holdout_dataset = SimpleDataset("datasets/test_holdout.jsonl")
    paraphrase_dataset = SimpleDataset("datasets/test_paraphrase.jsonl")
    longdialog_dataset = SimpleDataset("datasets/test_longdialog.jsonl")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size)
    paraphrase_loader = DataLoader(paraphrase_dataset, batch_size=config.batch_size)
    longdialog_loader = DataLoader(longdialog_dataset, batch_size=config.batch_size)
    
    # 创建模型
    print("\n创建模型...")
    native_config = NativeTinyConfig()
    transformer_config = TransformerTinyConfig()
    
    native_model = NativeBackboneTinyV1(native_config)
    transformer_model = TransformerBaselineTiny(transformer_config)
    
    # 统计参数量
    native_params = native_model.count_parameters()
    transformer_params = transformer_model.count_parameters()
    
    print(f"\n参数量对比:")
    print(f"  Native (GRU): {native_params:,}")
    print(f"  Transformer:  {transformer_params:,}")
    print(f"  比例: {native_params/transformer_params:.2f}x")
    
    # 训练 Native
    native_history, native_best = train_model(
        native_model, train_loader, val_loader, config, "Native Backbone (GRU)"
    )
    
    # 训练 Transformer
    transformer_history, transformer_best = train_model(
        transformer_model, train_loader, val_loader, config, "Transformer Baseline"
    )
    
    # 测试集评估
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    
    print("\n--- Native (GRU) ---")
    native_holdout = evaluate_on_test(native_model, holdout_loader, "留出集")
    native_paraphrase = evaluate_on_test(native_model, paraphrase_loader, "同义改写")
    native_longdialog = evaluate_on_test(native_model, longdialog_loader, "长对话")
    
    print("\n--- Transformer ---")
    transformer_holdout = evaluate_on_test(transformer_model, holdout_loader, "留出集")
    transformer_paraphrase = evaluate_on_test(transformer_model, paraphrase_loader, "同义改写")
    transformer_longdialog = evaluate_on_test(transformer_model, longdialog_loader, "长对话")
    
    # 推理速度
    print("\n" + "=" * 70)
    print("推理速度对比 (batch=8, seq=50, 100 runs)")
    print("=" * 70)
    
    native_speed = measure_inference_speed(native_model, (8, 50))
    transformer_speed = measure_inference_speed(transformer_model, (8, 50))
    
    print(f"  Native (GRU): {native_speed:.2f} ms/batch")
    print(f"  Transformer:  {transformer_speed:.2f} ms/batch")
    print(f"  速度比: {transformer_speed/native_speed:.2f}x")
    
    # 保存结果
    results = {
        'native': {
            'params': native_params,
            'best_val_acc': native_best,
            'holdout': native_holdout,
            'paraphrase': native_paraphrase,
            'longdialog': native_longdialog,
            'inference_speed_ms': native_speed,
        },
        'transformer': {
            'params': transformer_params,
            'best_val_acc': transformer_best,
            'holdout': transformer_holdout,
            'paraphrase': transformer_paraphrase,
            'longdialog': transformer_longdialog,
            'inference_speed_ms': transformer_speed,
        }
    }
    
    with open("eval/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    
    print(f"\n参数量: Native 是 Transformer 的 {native_params/transformer_params:.2f}x")
    print(f"推理速度: Native 是 Transformer 的 {transformer_speed/native_speed:.2f}x")
    
    print("\n留出集表现:")
    print(f"  Native Gap Acc: {native_holdout['gap_acc']:.1%}")
    print(f"  Transformer Gap Acc: {transformer_holdout['gap_acc']:.1%}")
    
    print("\n同义改写鲁棒性:")
    print(f"  Native Policy Acc: {native_paraphrase['policy_acc']:.1%}")
    print(f"  Transformer Policy Acc: {transformer_paraphrase['policy_acc']:.1%}")
    
    print("\n长对话稳定性:")
    print(f"  Native Policy Acc: {native_longdialog['policy_acc']:.1%}")
    print(f"  Transformer Policy Acc: {transformer_longdialog['policy_acc']:.1%}")
    
    # 保存模型
    torch.save(native_model.state_dict(), "checkpoints/native_tiny_best.pt")
    torch.save(transformer_model.state_dict(), "checkpoints/transformer_tiny_best.pt")
    print("\n✓ 模型已保存到 checkpoints/")
    print("✓ 结果已保存到 eval/comparison_results.json")
    
    return results


if __name__ == "__main__":
    results = run_comparison()
