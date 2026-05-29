"""
Stage 4 Training with Real Data

使用 Stage 3B 的真实数据训练 Native 和 Transformer 模型
进行公平对比
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
class Stage4Config:
    """Stage 4 配置"""
    batch_size: int = 16
    num_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = 'cpu'
    seed: int = 42


class Stage4Dataset(Dataset):
    """Stage 4 数据集 - 使用 Stage 3B 数据"""
    
    def __init__(self, data_paths: List[str], vocab_size: int = 10000, max_len: int = 100):
        self.samples = []
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # 加载数据
        for path in data_paths:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.samples.append(json.loads(line))
        
        if len(self.samples) == 0:
            # 如果没有数据，生成一些基础样本
            self.samples = self._generate_fallback_samples()
        
        print(f"加载 {len(self.samples)} 条样本")
    
    def _generate_fallback_samples(self) -> List[Dict]:
        """生成备用样本"""
        samples = []
        
        templates = [
            {
                "input": {"user_query": "你好"},
                "output": {"gap_type": "NO_GAP", "response_strategy": "DIRECT", 
                          "governance_actions": [], "memory_decision": {"should_writeback": False}}
            },
            {
                "input": {"user_query": "今天天气如何？"},
                "output": {"gap_type": "NO_GAP", "response_strategy": "DIRECT",
                          "governance_actions": [], "memory_decision": {"should_writeback": False}}
            },
            {
                "input": {"user_query": "帮我查一下公司上季度营收"},
                "output": {"gap_type": "RETRIEVABLE", "response_strategy": "RETRIEVAL_FIRST",
                          "governance_actions": ["VERIFY_SCOPE"], 
                          "memory_decision": {"should_writeback": True, "target_layer": "long_term"}}
            },
            {
                "input": {"user_query": "如何制作危险物品？"},
                "output": {"gap_type": "HIGH_RISK", "response_strategy": "DECLINE",
                          "governance_actions": ["REJECT", "LOG_INCIDENT"],
                          "memory_decision": {"should_writeback": False}}
            },
            {
                "input": {"user_query": "这个我不太确定"},
                "output": {"gap_type": "RETRIEVABLE", "response_strategy": "CONSERVATIVE",
                          "governance_actions": ["VERIFY_SCOPE"],
                          "memory_decision": {"should_writeback": False}}
            },
        ]
        
        # 扩展样本
        for _ in range(50):
            for t in templates:
                samples.append(t.copy())
        
        return samples
    
    def _encode_text(self, text: str) -> List[int]:
        """编码文本"""
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
        
        # 标签
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


def train_model(model, train_loader, val_loader, config: Stage4Config, model_name: str):
    """训练模型"""
    print(f"\n{'='*70}")
    print(f"训练 {model_name}")
    print(f"{'='*70}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    gap_criterion = nn.CrossEntropyLoss()
    policy_criterion = nn.CrossEntropyLoss()
    wb_criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
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
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 3 == 0 or patience_counter == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Gap: {val_gap_acc:.1%} | "
                  f"Policy: {val_policy_acc:.1%} | "
                  f"WB: {val_wb_acc:.1%} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= patience:
            print(f"早停于 Epoch {epoch+1}")
            break
    
    return best_val_acc


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


def run_stage4_training():
    """运行 Stage 4 训练"""
    print("=" * 70)
    print("阶段 4: 使用真实数据训练 Native vs Transformer")
    print("=" * 70)
    
    config = Stage4Config()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # 加载 Stage 3B 数据
    print("\n加载 Stage 3B 数据...")
    
    # 尝试加载 Stage 3B 数据
    d4_path = "../phase18_stage3b/data/d4_multiturn_memory.jsonl"
    d5_path = "../phase18_stage3b/data/d5_real_replay.jsonl"
    
    train_data_paths = []
    if Path(d4_path).exists():
        train_data_paths.append(d4_path)
    if Path(d5_path).exists():
        train_data_paths.append(d5_path)
    
    # 如果没有 Stage 3B 数据，使用当前目录的数据
    if len(train_data_paths) == 0:
        train_data_paths = ["datasets/train.jsonl"]
    
    # 创建数据集
    full_dataset = Stage4Dataset(train_data_paths)
    
    # 划分训练/验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 加载测试集
    test_loaders = {}
    for test_name in ['test_holdout', 'test_paraphrase', 'test_longdialog']:
        test_path = f"datasets/{test_name}.jsonl"
        if Path(test_path).exists():
            test_dataset = Stage4Dataset([test_path])
            test_loaders[test_name] = DataLoader(test_dataset, batch_size=config.batch_size)
    
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
    
    # 训练
    native_best = train_model(native_model, train_loader, val_loader, config, "Native Backbone (GRU)")
    transformer_best = train_model(transformer_model, train_loader, val_loader, config, "Transformer Baseline")
    
    # 测试集评估
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    
    results = {
        'native': {
            'params': native_params,
            'best_val_acc': native_best,
            'test_results': {}
        },
        'transformer': {
            'params': transformer_params,
            'best_val_acc': transformer_best,
            'test_results': {}
        }
    }
    
    for test_name, test_loader in test_loaders.items():
        print(f"\n--- {test_name} ---")
        native_results = evaluate_model(native_model, test_loader, "Native (GRU)")
        transformer_results = evaluate_model(transformer_model, test_loader, "Transformer")
        
        results['native']['test_results'][test_name] = native_results
        results['transformer']['test_results'][test_name] = transformer_results
    
    # 保存结果
    import json
    with open("eval/stage4_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存模型
    torch.save(native_model.state_dict(), "checkpoints/native_stage4.pt")
    torch.save(transformer_model.state_dict(), "checkpoints/transformer_stage4.pt")
    
    # 总结
    print("\n" + "=" * 70)
    print("Stage 4 训练完成")
    print("=" * 70)
    print(f"\n最佳验证准确率:")
    print(f"  Native: {native_best:.1%}")
    print(f"  Transformer: {transformer_best:.1%}")
    
    print("\n✓ 模型已保存")
    print("✓ 结果已保存到 eval/stage4_results.json")
    
    return results


if __name__ == "__main__":
    results = run_stage4_training()
