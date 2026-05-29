"""
Stage 14.1: 基础语言训练 - Decoder训练

目标:
1. Loss下降是否正常
2. 输出是否从乱码变成可读句子
3. 简单问答是否开始成形

策略:
- 冻结Encoder和其他分类头
- 只训练ResponseDecoder
- 使用交叉熵损失
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import random

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage11a_r2_fix_v2_balanced import FixV2Model
from stage14_language_generation import RecurrentResponseDecoder
from stage14_1_dataset import DIALOGUE_TRAINING_DATA, STRUCTURED_TRAINING_DATA, GOVERNANCE_TRAINING_DATA


class DialogueDataset(Dataset):
    """对话数据集"""

    def __init__(self, data: List[Dict], vocab_size: int = 10000, max_in: int = 100, max_out: int = 50):
        self.data = data
        self.vocab_size = vocab_size
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.data)

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = [ord(c) % self.vocab_size for c in text[:max_len]]
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        return tokens

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        input_ids = self.encode(item['query'], self.max_in)
        target_ids = self.encode(item['response'], self.max_out)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


class Stage14_1Trainer:
    """Stage 14.1 训练器"""

    VERSION = "Stage 14.1: Decoder基础语言训练"

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None
        self.decoder = None
        self.optimizer = None
        self.vocab_size = 10000
        self.max_in = 100
        self.max_out = 50

    def load_model(self):
        """加载Stage 13 v3检查点"""
        print(f"\n[模型] 加载Stage 13 v3检查点...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.base_model = FixV2Model(base_model)

        checkpoint = torch.load('stage8_dataset/stage13_v3_final_checkpoint.pt', map_location='cpu')
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.base_model.eval()

        self.decoder = RecurrentResponseDecoder(config)

        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  ✓ Decoder参数量: {total_params:,}")
        print(f"  ✓ Encoder已冻结")

    def build_dataloader(self) -> DataLoader:
        """构建数据加载器"""
        all_data = DIALOGUE_TRAINING_DATA + STRUCTURED_TRAINING_DATA + GOVERNANCE_TRAINING_DATA
        random.shuffle(all_data)

        dataset = DialogueDataset(all_data, self.vocab_size, self.max_in, self.max_out)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        print(f"\n[数据] 训练样本: {len(all_data)}条")
        print(f"  基础表达: {len(DIALOGUE_TRAINING_DATA)}条")
        print(f"  结构化解释: {len(STRUCTURED_TRAINING_DATA)}条")
        print(f"  治理兼容: {len(GOVERNANCE_TRAINING_DATA)}条")

        return loader

    def get_policy_gap_probs(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从base_model获取policy和gap概率分布"""
        with torch.no_grad():
            base_out = self.base_model.base_model.base_model if hasattr(self.base_model.base_model, 'base_model') else self.base_model.base_model

        gap_logits = base_out.gap_detector(pooled)
        gap_probs = F.softmax(gap_logits, dim=-1)

        policy_logits = base_out.policy_head(pooled, gap_probs)
        policy_probs = F.softmax(policy_logits, dim=-1)

        return policy_probs, gap_probs

    def forward_decoder(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> Dict:
        """前向传播"""
        base_out = self.base_model.base_model(input_ids)
        pooled = base_out['pooled']
        policy_probs = base_out['policy_probs']
        gap_probs = base_out['gap_probs']

        decoder_out = self.decoder(pooled, policy_probs, gap_probs, target_ids)

        return {
            'response_logits': decoder_out['response_logits'],
            'target_ids': target_ids,
        }

    def compute_loss(self, response_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """计算交叉熵损失"""
        batch_size, seq_len, vocab_size = response_logits.shape
        response_logits = response_logits.view(batch_size * seq_len, vocab_size)
        target_ids = target_ids.view(batch_size * seq_len)
        loss = F.cross_entropy(response_logits, target_ids, ignore_index=0)
        return loss

    def train_epoch(self, loader: DataLoader) -> float:
        """训练一个epoch"""
        self.decoder.train()
        total_loss = 0
        num_batches = 0

        for input_ids, target_ids in loader:
            self.optimizer.zero_grad()
            outputs = self.forward_decoder(input_ids, target_ids)
            loss = self.compute_loss(outputs['response_logits'], target_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def generate_sample(self, query: str) -> str:
        """生成样本"""
        self.decoder.eval()
        with torch.no_grad():
            tokens = [ord(c) % self.vocab_size for c in query[:self.max_in]]
            if len(tokens) < self.max_in:
                tokens.extend([0] * (self.max_in - len(tokens)))
            input_ids = torch.tensor([tokens])

            base_out = self.base_model.base_model(input_ids)
            pooled = base_out['pooled']
            policy_probs = base_out['policy_probs']
            gap_probs = base_out['gap_probs']

            out = self.decoder(pooled, policy_probs, gap_probs, None)
            generated_ids = out['generated_ids'][0]

            text = ''.join([chr(max(1, min(10000, t.item()))) for t in generated_ids if t.item() > 0])
            return text

    def run_training(self, num_epochs: int = 50):
        """运行训练"""
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=1e-3)
        loader = self.build_dataloader()

        print(f"\n{'='*70}")
        print(f"开始训练 (最多{num_epochs}轮)")
        print(f"{'='*70}")

        best_loss = float('inf')
        best_state = None
        loss_history = []

        for epoch in range(num_epochs):
            loss = self.train_epoch(loader)
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1:3d}: loss={loss:.4f} (best={best_loss:.4f})")

                if epoch == 0 or epoch >= num_epochs - 5:
                    sample = self.generate_sample("你好")
                    print(f"  样本: {sample[:40]}...")

        if best_state:
            self.decoder.load_state_dict(best_state)

        print(f"\n{'='*70}")
        print(f"训练完成")
        print(f"最终loss: {best_loss:.4f}")
        print(f"{'='*70}")

        print(f"\n[生成测试]")
        test_queries = ["你好", "Python是什么", "怎么学习编程"]
        for q in test_queries:
            gen = self.generate_sample(q)
            print(f"  输入: {q}")
            print(f"  生成: {gen[:50]}...")
            print()

        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'version': self.VERSION,
            'loss_history': loss_history,
        }, 'stage8_dataset/stage14_1_decoder.pt')

        print(f"  补丁已保存: stage8_dataset/stage14_1_decoder.pt")

        return best_loss, loss_history


def main():
    trainer = Stage14_1Trainer()
    trainer.load_model()
    best_loss, history = trainer.run_training(num_epochs=50)
    return best_loss, history


if __name__ == "__main__":
    main()
