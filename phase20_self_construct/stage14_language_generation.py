"""
Stage 14: 主体能力建设 - 端到端对话生成

目标: 让模型具备真正的文本生成能力

现状分析:
- 当前模型: 输入 → TSLA动作分类 (分类能力 ✓)
- 缺失能力: 输入 → 自然语言输出 (生成能力 ✗)

NativeTinyResponseDecoder 当前实现:
- 只是MLP: concat(pooled, policy_probs, gap_probs) → vocab_size
- 输出单个token的logits，不是序列生成

Stage 14 目标:
1. 添加循环Decoder实现真正的序列生成
2. 训练生成能力而非只训练分类
3. 与TSLA-v2治理骨架融合

架构设计:
Input → GRUEncoder → [Gap, Policy, Governance, Writeback] → ResponseDecoder(循环)
                                        ↓
                               TSLA-v2 双层判定
                                        ↓
                                  生成的响应
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random


class RecurrentResponseDecoder(nn.Module):
    """
    循环Response Decoder

    替代原NativeTinyResponseDecoder的MLP结构
    实现真正的序列生成能力
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.gru = nn.GRU(
            input_size=config.hidden_dim + config.num_strategies + config.num_gap_types,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

        self.start_token = 0
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
                gap_probs: torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            pooled: [batch, hidden_dim]
            policy_probs: [batch, num_strategies]
            gap_probs: [batch, num_gap_types]
            target_ids: [batch, seq_len] (训练时提供)

        Returns:
            dict with 'response_logits' or 'generated_ids'
        """
        batch_size = pooled.size(0)

        if target_ids is not None:
            return self._forward_train(pooled, policy_probs, gap_probs, target_ids)
        else:
            return self._forward_generate(pooled, policy_probs, gap_probs)

    def _forward_train(self, pooled: torch.Tensor, policy_probs: torch.Tensor,
                       gap_probs: torch.Tensor, target_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """训练时前向"""
        batch_size = target_ids.size(0)
        seq_len = target_ids.size(1)

        hidden = pooled.unsqueeze(0)

        logits_list = []
        input_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=target_ids.device)

        for t in range(seq_len):
            logits, hidden = self.forward_step(input_token, hidden, policy_probs, gap_probs)
            logits_list.append(logits)
            input_token = target_ids[:, t]

        all_logits = torch.stack(logits_list, dim=1)
        return {'response_logits': all_logits, 'target_ids': target_ids}

    def _forward_generate(self, pooled: torch.Tensor, policy_probs: torch.Tensor,
                         gap_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成时前向"""
        batch_size = pooled.size(0)

        hidden = pooled.unsqueeze(0)

        generated_ids = []
        input_token = torch.full((batch_size,), self.start_token, dtype=torch.long, device=pooled.device)

        for _ in range(self.max_len):
            logits, hidden = self.forward_step(input_token, hidden, policy_probs, gap_probs)
            probs = F.softmax(logits, dim=-1)
            next_token = probs.argmax(dim=-1)
            generated_ids.append(next_token)
            input_token = next_token

        generated_ids = torch.stack(generated_ids, dim=1)
        return {'generated_ids': generated_ids}


class Stage14LanguageModel(nn.Module):
    """
    Stage 14: 带语言生成能力的完整模型

    在NativeBackboneTinyV1基础上替换ResponseDecoder
    """

    def __init__(self, base_model, config):
        super().__init__()
        self.base = base_model
        self.config = config

        self.response_decoder = RecurrentResponseDecoder(config)

    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """完整前向"""
        base_outputs = self.base.base_model(input_ids)
        pooled = base_outputs['pooled']
        gap_probs = base_outputs['gap_probs']
        policy_probs = base_outputs['policy_probs']

        decoder_outputs = self.response_decoder(pooled, policy_probs, gap_probs, target_ids)

        return {
            **base_outputs,
            **decoder_outputs,
        }

    def generate(self, input_ids: torch.Tensor, max_len: int = 50) -> List[str]:
        """生成文本"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, target_ids=None)
            generated_ids = outputs['generated_ids']

        return generated_ids


class Stage14Trainer:
    """Stage 14 训练器"""

    def __init__(self):
        print(f"\n{'='*70}")
        print("Stage 14: 主体能力建设 - 端到端对话生成")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None
        self.optimizer = None

    def load_base_model(self):
        """加载基础模型"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
        from stage11a_r2_fix_v2_balanced import FixV2Model

        print(f"\n[模型] 加载Stage 13 v3检查点...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.base_model = FixV2Model(base_model)

        checkpoint = torch.load('stage8_dataset/stage13_v3_final_checkpoint.pt', map_location='cpu')
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print(f"  ✓ 基础模型已加载")

    def build_model(self):
        """构建Stage 14模型"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeTinyConfig

        config = NativeTinyConfig()
        self.model = Stage14LanguageModel(self.base_model, config)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n[模型] Stage 14 模型已构建")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")

    def test_generation(self):
        """测试生成能力"""
        print(f"\n{'='*70}")
        print("测试: 端到端文本生成")
        print(f"{'='*70}")

        if self.model is None:
            print("  ⚠️  模型未构建")
            return

        test_queries = [
            "你好，今天天气不错",
            "给我讲个笑话",
            "解释一下什么是机器学习",
        ]

        for query in test_queries:
            tokens = [ord(c) % 10000 for c in query[:100]]
            if len(tokens) < 100:
                tokens.extend([0] * (100 - len(tokens)))

            input_ids = torch.tensor([tokens])

            generated_ids = self.model.generate(input_ids, max_len=20)

            generated_text = ''.join([chr(max(1, min(10000, t.item()))) for t in generated_ids[0] if t.item() > 0])

            print(f"\n  输入: {query}")
            print(f"  生成: {generated_text[:50]}...")

    def run(self):
        """运行Stage 14"""
        self.load_base_model()
        self.build_model()
        self.test_generation()

        print(f"\n{'='*70}")
        print("Stage 14 框架验证完成")
        print(f"{'='*70}")
        print("\n后续步骤:")
        print("1. 准备对话训练数据")
        print("2. 设计生成损失函数")
        print("3. 实现TSLA-v2融合训练")
        print("4. 执行端到端训练")


def main():
    trainer = Stage14Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
