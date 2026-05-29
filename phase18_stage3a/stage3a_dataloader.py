"""
Stage 3A DataLoader

加载 D1 + D2 + 少量 D3 进行真实数据训练

数据分布：
- D1: 高质量骨架 (约 60%)
- D2: 策略治理 (约 30%)
- D3: 少量对抗样本 (约 10%)

训练目标：
- PolicyHead
- GapDetector
- GovernanceHead

WritebackHead 冻结/低权重
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class Stage3AConfig:
    """Stage 3A 数据配置"""
    # 数据路径
    d1_path: str = "phase14/datasets/d1_seed_quality.jsonl"
    d2_path: str = "phase14/datasets/d2_policy_governance.jsonl"
    d3_path: str = "phase14/datasets/d3_conflict_noise.jsonl"
    
    # 数据比例
    d1_ratio: float = 0.6
    d2_ratio: float = 0.3
    d3_ratio: float = 0.1
    
    # 总样本数
    total_samples: int = 500
    
    # 词汇表
    vocab_size: int = 10000
    max_seq_len: int = 100
    
    # 标签映射
    strategy_to_idx = {
        'DIRECT': 0,
        'RETRIEVAL_FIRST': 1,
        'CONSERVATIVE': 2,
        'DECLINE': 3,
        'REVIEW': 4,
    }
    
    gap_to_idx = {
        'NO_GAP': 0,
        'RETRIEVABLE': 1,
        'HIGH_RISK': 2,
    }
    
    # TSLA 动作映射 (8个动作)
    tsla_actions = [
        'TSLA',           # 0
        'STRONG_REVIEW',  # 1
        'ISOLATE',        # 2
        'ARCHIVE',        # 3
        'ROLLBACK',       # 4
        'PROMOTE',        # 5
        'REJECT',         # 6
        'NO_ACTION',      # 7
    ]


class Stage3ADataset(Dataset):
    """
    Stage 3A 数据集
    
    混合 D1 + D2 + D3，专注于 Policy + Gap + Governance 训练
    """
    
    def __init__(self, config: Stage3AConfig, split: str = 'train', seed: int = 42):
        self.config = config
        self.split = split
        random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据
        self.samples = self._load_and_mix_data()
        
        # 划分训练/验证集
        split_idx = int(len(self.samples) * 0.8)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"Stage3A {split} dataset: {len(self.samples)} samples")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载 jsonl 文件"""
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Warning: {path} not found, using mock data")
            return []
        return samples
    
    def _load_and_mix_data(self) -> List[Dict]:
        """加载并混合 D1/D2/D3 数据，平衡策略分布"""
        # 加载数据
        d1_data = self._load_jsonl(self.config.d1_path)
        d2_data = self._load_jsonl(self.config.d2_path)
        d3_data = self._load_jsonl(self.config.d3_path)
        
        # 按策略分类 D2 数据（D2 包含各种策略）
        strategy_groups = {s: [] for s in self.config.strategy_to_idx.keys()}
        for sample in d2_data:
            strategy = sample.get('output', {}).get('response_strategy', 'DIRECT')
            if strategy in strategy_groups:
                strategy_groups[strategy].append(sample)
        
        # 平衡采样：每个策略至少有一定比例
        samples_per_strategy = self.config.total_samples // 5  # 5种策略
        balanced_samples = []
        
        for strategy, samples in strategy_groups.items():
            if len(samples) > 0:
                # 循环使用样本直到达到目标数量
                extended = []
                while len(extended) < samples_per_strategy:
                    extended.extend(samples)
                balanced_samples.extend(extended[:samples_per_strategy])
        
        # 添加 D1 数据（DIRECT 为主）
        n_d1 = int(self.config.total_samples * 0.2)  # 减少 D1 比例
        d1_extended = []
        while len(d1_extended) < n_d1:
            d1_extended.extend(d1_data)
        balanced_samples.extend(d1_extended[:n_d1])
        
        # 添加 D3 数据（对抗样本）
        n_d3 = int(self.config.total_samples * 0.1)
        d3_extended = []
        while len(d3_extended) < n_d3:
            d3_extended.extend(d3_data)
        balanced_samples.extend(d3_extended[:n_d3])
        
        # 打乱
        random.shuffle(balanced_samples)
        
        return balanced_samples[:self.config.total_samples]
    
    def _text_to_ids(self, text: str) -> List[int]:
        """将文本转换为 ID 序列（简化版）"""
        # 使用简单的哈希方式生成词汇表索引
        ids = []
        for char in text[:self.config.max_seq_len]:
            # 中文字符和英文单词分别处理
            hash_val = hash(char) % self.config.vocab_size
            ids.append(abs(hash_val))
        
        # 填充到固定长度
        while len(ids) < self.config.max_seq_len:
            ids.append(0)  # PAD token
        
        return ids[:self.config.max_seq_len]
    
    def _extract_gap_type(self, sample: Dict) -> int:
        """从样本中提取 gap 类型"""
        output = sample.get('output', {})
        
        # 基于策略推断 gap 类型
        strategy = output.get('response_strategy', 'DIRECT')
        
        if strategy == 'DIRECT':
            return self.config.gap_to_idx['NO_GAP']
        elif strategy in ['RETRIEVAL_FIRST', 'CONSERVATIVE']:
            return self.config.gap_to_idx['RETRIEVABLE']
        else:  # DECLINE, REVIEW
            return self.config.gap_to_idx['HIGH_RISK']
    
    def _extract_governance_actions(self, sample: Dict) -> List[float]:
        """提取治理动作标签（8维向量）"""
        output = sample.get('output', {})
        actions = output.get('governance_actions', [])
        
        # 转换为 one-hot 向量
        action_vec = [0.0] * 8
        for action in actions:
            if action in self.config.tsla_actions:
                idx = self.config.tsla_actions.index(action)
                action_vec[idx] = 1.0
        
        return action_vec
    
    def _extract_features(self, sample: Dict) -> Dict:
        """从原始样本提取训练特征"""
        # 输入文本
        input_data = sample.get('input', {})
        user_query = input_data.get('user_query', '')
        
        # 输出标签
        output = sample.get('output', {})
        strategy = output.get('response_strategy', 'DIRECT')
        
        # 转换为索引
        policy_target = self.config.strategy_to_idx.get(strategy, 0)
        gap_target = self._extract_gap_type(sample)
        governance_target = self._extract_governance_actions(sample)
        
        # 文本编码
        input_ids = self._text_to_ids(user_query)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'policy_target': torch.tensor(policy_target, dtype=torch.long),
            'gap_target': torch.tensor(gap_target, dtype=torch.long),
            'governance_target': torch.tensor(governance_target, dtype=torch.float),
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._extract_features(sample)


def create_stage3a_dataloaders(
    config: Stage3AConfig = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """创建 Stage 3A 训练/验证 dataloader"""
    config = config or Stage3AConfig()
    
    train_dataset = Stage3ADataset(config, split='train', seed=42)
    val_dataset = Stage3ADataset(config, split='val', seed=42)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def test_dataloader():
    """测试 dataloader"""
    print("=" * 70)
    print("Stage 3A DataLoader 测试")
    print("=" * 70)
    
    config = Stage3AConfig(total_samples=100)
    train_loader, val_loader = create_stage3a_dataloaders(config, batch_size=4)
    
    print(f"\n训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print("\n批次数据结构:")
    for key, value in batch.items():
        if key != 'raw_sample':
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 统计策略分布
    print("\n训练集策略分布:")
    strategy_counts = [0] * 5
    for batch in train_loader:
        for target in batch['policy_target']:
            strategy_counts[target.item()] += 1
    
    idx_to_strategy = {v: k for k, v in config.strategy_to_idx.items()}
    for idx, count in enumerate(strategy_counts):
        strategy = idx_to_strategy.get(idx, 'UNKNOWN')
        print(f"  {strategy}: {count}")
    
    print("\n✓ DataLoader 测试通过")


if __name__ == "__main__":
    test_dataloader()
