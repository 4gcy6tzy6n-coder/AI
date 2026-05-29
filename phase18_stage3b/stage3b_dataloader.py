"""
Stage 3B DataLoader

加载 D4 (多轮记忆) + 少量 D5 (真实样本回放)

Stage 3B 特点：
- 引入多轮对话上下文
- 包含记忆写回决策标签
- 关注多轮记忆稳定性
- 包含真实失败案例

训练目标：
- PolicyHead
- GapDetector
- GovernanceHead
- WritebackHead (解冻)
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
class Stage3BConfig:
    """Stage 3B 数据配置"""
    # 数据路径
    d4_path: str = "phase14/datasets/d4_multiturn_memory.jsonl"
    d5_path: str = "phase14/datasets/d5_real_case_replay.jsonl"
    
    # 数据比例
    d4_ratio: float = 0.85  # 多轮记忆为主
    d5_ratio: float = 0.15  # 少量真实样本
    
    # 总样本数
    total_samples: int = 300
    
    # 词汇表
    vocab_size: int = 10000
    max_seq_len: int = 150  # 多轮需要更长序列
    max_turns: int = 5      # 最大对话轮数
    
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
    
    # Writeback 类型
    writeback_to_idx = {
        'NO_WRITEBACK': 0,
        'EPHEMERAL': 1,
        'LONG_TERM': 2,
    }
    
    # TSLA 动作
    tsla_actions = [
        'TSLA', 'STRONG_REVIEW', 'ISOLATE', 'ARCHIVE',
        'ROLLBACK', 'PROMOTE', 'REJECT', 'NO_ACTION',
    ]


class Stage3BDataset(Dataset):
    """
    Stage 3B 数据集
    
    支持多轮对话和记忆写回决策
    """
    
    def __init__(self, config: Stage3BConfig, split: str = 'train', seed: int = 42):
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
        
        print(f"Stage3B {split} dataset: {len(self.samples)} samples")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载 jsonl 文件"""
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Warning: {path} not found")
            return []
        return samples
    
    def _load_and_mix_data(self) -> List[Dict]:
        """加载并混合 D4/D5 数据"""
        d4_data = self._load_jsonl(self.config.d4_path)
        d5_data = self._load_jsonl(self.config.d5_path)
        
        # 计算样本数
        n_d4 = int(self.config.total_samples * self.config.d4_ratio)
        n_d5 = int(self.config.total_samples * self.config.d5_ratio)
        
        # 扩展数据
        def extend_data(data, target_len):
            if len(data) == 0:
                return []
            result = []
            while len(result) < target_len:
                result.extend(data)
            return result[:target_len]
        
        d4_extended = extend_data(d4_data, n_d4)
        d5_extended = extend_data(d5_data, n_d5)
        
        # 混合并打乱
        all_data = d4_extended + d5_extended
        random.shuffle(all_data)
        
        return all_data[:self.config.total_samples]
    
    def _encode_conversation(self, sample: Dict) -> List[int]:
        """编码多轮对话为 ID 序列"""
        input_data = sample.get('input', {})
        
        # 获取对话历史
        history = input_data.get('conversation_history', [])
        current_query = input_data.get('user_query', '')
        
        # 构建完整文本
        texts = []
        for turn in history[-self.config.max_turns:]:  # 只取最近几轮
            if isinstance(turn, dict):
                role = turn.get('role', '')
                content = turn.get('content', '')
                texts.append(f"{role}: {content}")
        
        # 添加当前查询
        texts.append(f"user: {current_query}")
        
        full_text = " | ".join(texts)
        
        # 编码为 IDs
        ids = []
        for char in full_text[:self.config.max_seq_len]:
            hash_val = hash(char) % self.config.vocab_size
            ids.append(abs(hash_val))
        
        # 填充
        while len(ids) < self.config.max_seq_len:
            ids.append(0)
        
        return ids[:self.config.max_seq_len]
    
    def _extract_writeback_label(self, sample: Dict) -> int:
        """提取写回决策标签"""
        output = sample.get('output', {})
        memory_decision = output.get('memory_decision', {})
        
        should_writeback = memory_decision.get('should_writeback', False)
        if not should_writeback:
            return self.config.writeback_to_idx['NO_WRITEBACK']
        
        target_layer = memory_decision.get('target_layer', 'ephemeral')
        if target_layer == 'long_term':
            return self.config.writeback_to_idx['LONG_TERM']
        else:
            return self.config.writeback_to_idx['EPHEMERAL']
    
    def _extract_features(self, sample: Dict) -> Dict:
        """从样本提取训练特征"""
        # 编码对话
        input_ids = self._encode_conversation(sample)
        
        # 输出标签
        output = sample.get('output', {})
        strategy = output.get('response_strategy', 'DIRECT')
        
        # 策略标签
        policy_target = self.config.strategy_to_idx.get(strategy, 0)
        
        # Gap 标签（从策略推断）
        if strategy == 'DIRECT':
            gap_target = self.config.gap_to_idx['NO_GAP']
        elif strategy in ['RETRIEVAL_FIRST', 'CONSERVATIVE']:
            gap_target = self.config.gap_to_idx['RETRIEVABLE']
        else:
            gap_target = self.config.gap_to_idx['HIGH_RISK']
        
        # 治理标签
        governance_actions = output.get('governance_actions', [])
        action_vec = [0.0] * 8
        for action in governance_actions:
            if action in self.config.tsla_actions:
                idx = self.config.tsla_actions.index(action)
                action_vec[idx] = 1.0
        
        # 写回标签（Stage 3B 新增）
        writeback_target = self._extract_writeback_label(sample)
        
        # 多轮特征
        input_data = sample.get('input', {})
        current_turn = input_data.get('current_turn', 1)
        has_context_deps = len(input_data.get('context_dependencies', [])) > 0
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'policy_target': torch.tensor(policy_target, dtype=torch.long),
            'gap_target': torch.tensor(gap_target, dtype=torch.long),
            'governance_target': torch.tensor(action_vec, dtype=torch.float),
            'writeback_target': torch.tensor(writeback_target, dtype=torch.long),
            'turn_number': torch.tensor(current_turn, dtype=torch.long),
            'has_context_deps': torch.tensor(has_context_deps, dtype=torch.float),
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._extract_features(sample)


def create_stage3b_dataloaders(
    config: Stage3BConfig = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """创建 Stage 3B 训练/验证 dataloader"""
    config = config or Stage3BConfig()
    
    train_dataset = Stage3BDataset(config, split='train', seed=42)
    val_dataset = Stage3BDataset(config, split='val', seed=42)
    
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
    print("Stage 3B DataLoader 测试")
    print("=" * 70)
    
    config = Stage3BConfig(total_samples=100)
    train_loader, val_loader = create_stage3b_dataloaders(config, batch_size=4)
    
    print(f"\n训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print("\n批次数据结构:")
    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 统计写回分布
    print("\n训练集写回决策分布:")
    wb_counts = [0, 0, 0]
    for batch in train_loader:
        for target in batch['writeback_target']:
            wb_counts[target.item()] += 1
    
    wb_names = ['NO_WRITEBACK', 'EPHEMERAL', 'LONG_TERM']
    for i, count in enumerate(wb_counts):
        print(f"  {wb_names[i]}: {count}")
    
    print("\n✓ Stage 3B DataLoader 测试通过")


if __name__ == "__main__":
    test_dataloader()
