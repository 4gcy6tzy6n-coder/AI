"""
Stage 14.2: 生成层 tokenization 重构

问题: 字符级 ord(c) % 10000 映射不可逆
方案: Bigram分词 + 固定词表 (不依赖外部库)

设计:
- 治理层保持不变 (Unit/TSLA)
- 生成层使用更合理的tokenization
- 编码可逆，解码准确
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
from collections import Counter


SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<BOS>': 2,
    '<EOS>': 3,
}

MAX_VOCAB = 5000
PAD_ID = SPECIAL_TOKENS['<PAD>']
UNK_ID = SPECIAL_TOKENS['<UNK>']
BOS_ID = SPECIAL_TOKENS['<BOS>']
EOS_ID = SPECIAL_TOKENS['<EOS>']


class SimpleBigramTokenizer:
    """简单bigram分词器 - 不依赖外部库"""

    def __init__(self):
        self.word2id = SPECIAL_TOKENS.copy()
        self.id2word = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.next_id = len(SPECIAL_TOKENS)
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """简单bigram分词"""
        tokens = []
        i = 0
        text = text.strip()
        while i < len(text):
            if i + 1 < len(text):
                bigram = text[i:i+2]
                tokens.append(bigram)
                i += 2
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def fit(self, texts: List[str]):
        """从文本集合构建词表"""
        word_freq = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            for t in tokens:
                if t.strip():
                    word_freq[t] += 1

        for word, freq in word_freq.most_common(MAX_VOCAB - len(SPECIAL_TOKENS)):
            self.word2id[word] = self.next_id
            self.id2word[self.next_id] = word
            self.next_id += 1

        self.fitted = True
        print(f"  词表大小: {len(self.word2id)}")

    def encode(self, text: str, max_len: int) -> List[int]:
        """编码"""
        tokens = self._tokenize(text)
        ids = [BOS_ID]
        for t in tokens[:max_len-2]:
            ids.append(self.word2id.get(t, UNK_ID))
        ids.append(EOS_ID)

        if len(ids) < max_len:
            ids.extend([PAD_ID] * (max_len - len(ids)))

        return ids

    def decode(self, ids: List[int]) -> str:
        """解码"""
        words = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            if i in self.id2word:
                words.append(self.id2word[i])
            else:
                words.append('<UNK>')
        return ''.join(words)


class Stage142Tokenizer(SimpleBigramTokenizer):
    """Stage 14.2 专用tokenizer"""
    pass


def build_tokenizer(train_data: List[Dict]) -> Tuple[SimpleBigramTokenizer, int, int]:
    """构建tokenizer"""
    texts = []
    for item in train_data:
        texts.append(item['query'])
        texts.append(item['response'])

    tokenizer = Stage142Tokenizer()
    tokenizer.fit(texts)

    return tokenizer


class RecurrentResponseDecoderV2(nn.Module):
    """改进的循环Decoder - V2"""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, config.hidden_dim)

        self.gru = nn.GRU(
            input_size=config.hidden_dim + config.num_strategies + config.num_gap_types,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_proj = nn.Linear(config.hidden_dim, vocab_size)
        self.max_len = 50

    def forward_step(self, prev_token: torch.Tensor, hidden: torch.Tensor,
                     policy_probs: torch.Tensor, gap_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步生成"""
        token_emb = self.token_embedding(prev_token)
        combined = torch.cat([token_emb, policy_probs, gap_probs], dim=-1)
        combined = combined.unsqueeze(1)
        gru_out, hidden = self.gru(combined, hidden)
        logits = self.output_proj(gru_out.squeeze(1))
        return logits, hidden

    def forward(self, pooled: torch.Tensor, policy_probs: torch.Tensor,
                gap_probs: torch.Tensor, target_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """前向"""
        batch_size = pooled.size(0)

        if target_ids is not None:
            return self._forward_train(pooled, policy_probs, gap_probs, target_ids)
        else:
            return self._forward_generate(pooled, policy_probs, gap_probs)

    def _forward_train(self, pooled: torch.Tensor, policy_probs: torch.Tensor,
                       gap_probs: torch.Tensor, target_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """训练前向"""
        batch_size = target_ids.size(0)
        seq_len = target_ids.size(1)
        hidden = pooled.unsqueeze(0)
        logits_list = []
        input_token = torch.full((batch_size,), BOS_ID, dtype=torch.long, device=target_ids.device)

        for t in range(seq_len):
            logits, hidden = self.forward_step(input_token, hidden, policy_probs, gap_probs)
            logits_list.append(logits)
            input_token = target_ids[:, t]

        all_logits = torch.stack(logits_list, dim=1)
        return {'response_logits': all_logits, 'target_ids': target_ids}

    def _forward_generate(self, pooled: torch.Tensor, policy_probs: torch.Tensor,
                         gap_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成前向"""
        batch_size = pooled.size(0)
        hidden = pooled.unsqueeze(0)
        generated_ids = []
        input_token = torch.full((batch_size,), BOS_ID, dtype=torch.long, device=pooled.device)

        for _ in range(self.max_len):
            logits, hidden = self.forward_step(input_token, hidden, policy_probs, gap_probs)
            probs = F.softmax(logits, dim=-1)
            next_token = probs.argmax(dim=-1)
            generated_ids.append(next_token)
            input_token = next_token

        generated_ids = torch.stack(generated_ids, dim=1)
        return {'generated_ids': generated_ids}


class DialogueDatasetV2(Dataset):
    """V2对话数据集"""

    def __init__(self, data: List[Dict], tokenizer: SimpleBigramTokenizer, max_in: int = 100, max_out: int = 30):
        self.data = data
        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        input_ids = self.tokenizer.encode(item['query'], self.max_in)
        target_ids = self.tokenizer.encode(item['response'], self.max_out)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


class Stage14_2Trainer:
    """Stage 14.2 训练器"""

    VERSION = "Stage 14.2: Tokenization重构"

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.tokenizer = None
        self.decoder = None
        self.optimizer = None

    def load_components(self):
        """加载组件"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
        from stage11a_r2_fix_v2_balanced import FixV2Model

        print(f"\n[模型] 加载Stage 13 v3检查点...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.base_model = FixV2Model(base_model)

        checkpoint = torch.load('stage8_dataset/stage13_v3_final_checkpoint.pt', map_location='cpu')
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.base_model.eval()

        for param in self.base_model.parameters():
            param.requires_grad = False

        print(f"  ✓ 基础模型已加载")

    def prepare_tokenizer(self, train_data: List[Dict]):
        """准备tokenizer"""
        print(f"\n[Tokenization] 构建bigram分词器...")
        self.tokenizer = build_tokenizer(train_data)
        print(f"  ✓ Tokenizer构建完成")

    def build_decoder(self):
        """构建Decoder"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeTinyConfig

        config = NativeTinyConfig()
        vocab_size = len(self.tokenizer.word2id)

        self.decoder = RecurrentResponseDecoderV2(config, vocab_size)

        params = sum(p.numel() for p in self.decoder.parameters())
        print(f"\n[Decoder] V2参数量: {params:,}")

    def build_dataloader(self, data: List[Dict]) -> DataLoader:
        """构建数据加载器"""
        dataset = DialogueDatasetV2(data, self.tokenizer)
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def train_epoch(self, loader: DataLoader) -> float:
        """训练一个epoch"""
        self.decoder.train()
        total_loss = 0
        num_batches = 0

        for input_ids, target_ids in loader:
            self.optimizer.zero_grad()

            base_out = self.base_model.base_model(input_ids)
            pooled = base_out['pooled']
            policy_probs = base_out['policy_probs']
            gap_probs = base_out['gap_probs']

            outputs = self.decoder(pooled, policy_probs, gap_probs, target_ids)

            logits = outputs['response_logits']
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = target_ids.view(batch_size * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_ID)
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
            input_ids = self.tokenizer.encode(query, 50)
            input_tensor = torch.tensor([input_ids])

            base_out = self.base_model.base_model(input_tensor)
            pooled = base_out['pooled']
            policy_probs = base_out['policy_probs']
            gap_probs = base_out['gap_probs']

            out = self.decoder(pooled, policy_probs, gap_probs, None)
            generated_ids = out['generated_ids'][0].tolist()

            return self.tokenizer.decode(generated_ids)

    def run_training(self, train_data: List[Dict], num_epochs: int = 50):
        """运行训练"""
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=1e-3)
        loader = self.build_dataloader(train_data)

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

                if epoch >= num_epochs - 5 or epoch == 0:
                    sample = self.generate_sample("你好")
                    print(f"  样本: {sample[:50]}...")

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
            print(f"  生成: {gen[:60]}")
            print()

        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'tokenizer_word2id': self.tokenizer.word2id,
            'tokenizer_id2word': self.tokenizer.id2word,
            'version': self.VERSION,
            'loss_history': loss_history,
        }, 'stage8_dataset/stage14_2_decoder.pt')

        print(f"  补丁已保存: stage8_dataset/stage14_2_decoder.pt")

        return best_loss, loss_history


def main():
    from stage14_1_dataset import DIALOGUE_TRAINING_DATA, STRUCTURED_TRAINING_DATA, GOVERNANCE_TRAINING_DATA

    trainer = Stage14_2Trainer()
    trainer.load_components()

    all_data = DIALOGUE_TRAINING_DATA + STRUCTURED_TRAINING_DATA + GOVERNANCE_TRAINING_DATA
    random.shuffle(all_data)

    trainer.prepare_tokenizer(all_data)
    trainer.build_decoder()
    best_loss, history = trainer.run_training(all_data, num_epochs=50)

    return best_loss, history


if __name__ == "__main__":
    main()
