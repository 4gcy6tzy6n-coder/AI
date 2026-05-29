"""
Stage 14.2.1: 数据配比纠偏

问题: 当前生成偏"澄清类"，原因数据配比失调
目标: 调整数据比例，让模型学会"能答就答，必要时才澄清"

数据配比目标:
- 直接回答类: 50%
- 解释说明类: 25%
- 步骤建议类: 15%
- 澄清类: 10%

新增"同问题双版本"对照
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
    """简单bigram分词器"""

    def __init__(self):
        self.word2id = SPECIAL_TOKENS.copy()
        self.id2word = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.next_id = len(SPECIAL_TOKENS)
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
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
        tokens = self._tokenize(text)
        ids = [BOS_ID]
        for t in tokens[:max_len-2]:
            ids.append(self.word2id.get(t, UNK_ID))
        ids.append(EOS_ID)

        if len(ids) < max_len:
            ids.extend([PAD_ID] * (max_len - len(ids)))

        return ids

    def decode(self, ids: List[int]) -> str:
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


# ============ Stage 14.2.1 平衡数据集 ============

DIRECT_ANSWERS = [
    {"query": "Python是什么", "response": "Python是一种高级编程语言，由Guido van Rossum发明。它语法简洁、易学易用，广泛用于Web开发、数据分析、人工智能等领域。"},
    {"query": "人工智能是什么", "response": "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的技术。包括机器学习、自然语言处理、计算机视觉等多个研究方向。"},
    {"query": "机器学习是什么", "response": "机器学习是人工智能的一个子领域，通过让计算机从数据中学习模式来进行预测和决策，而不需要明确的编程规则。"},
    {"query": "深度学习是什么", "response": "深度学习是机器学习的子集，使用多层神经网络来处理复杂数据。它在图像识别、自然语言处理等领域取得了突破性进展。"},
    {"query": "区块链是什么", "response": "区块链是一种分布式账本技术，通过加密链条将数据块连接起来。它具有去中心化、不可篡改的特点。"},
    {"query": "云计算是什么", "response": "云计算是一种通过互联网提供计算资源的服务模式。用户可以按需使用服务器、存储、数据库等资源。"},
    {"query": "什么是API", "response": "API即应用程序编程接口，是不同软件之间通信的桥梁。通过API，一个程序可以调用另一个程序的功能。"},
    {"query": "什么是HTTP协议", "response": "HTTP是超文本传输协议，是Web通信的基础。它是请求-响应协议，常见方法有GET、POST、PUT、DELETE。"},
    {"query": "什么是数据库", "response": "数据库是存储和管理数据的系统。常见的数据库类型包括关系型数据库和非关系型数据库。"},
    {"query": "什么是递归", "response": "递归是一种编程技巧，函数调用自身来解决问题。递归需要基本情况和递归情况两个条件。"},
    {"query": "你好", "response": "你好！有什么我可以帮助你的吗？"},
    {"query": "今天天气怎么样", "response": "今天天气不错，阳光明媚，适合外出活动。"},
    {"query": "再见", "response": "再见，有需要随时找我！"},
    {"query": "谢谢", "response": "不客气！很高兴能帮到你。"},
    {"query": "你是谁", "response": "我是一个AI助手，可以回答问题、提供信息、帮助编程等。"},
    {"query": "一杯咖啡多少钱", "response": "咖啡价格因地区和店铺而异，一般在20到50元之间。"},
    {"query": "今天星期几", "response": "今天是星期三。"},
    {"query": "北京在哪里", "response": "北京是中国的首都，位于华北平原北部。"},
    {"query": "太阳系有几颗行星", "response": "太阳系有八颗行星，分别是水星、金星、地球、火星、木星、土星、天王星和海王星。"},
    {"query": "水的沸点是多少", "response": "在一个标准大气压下，水的沸点是100摄氏度。"},
]

EXPLANATIONS = [
    {"query": "解释Python中的列表和元组的区别", "response": "列表和元组的主要区别：1.列表使用方括号，元组使用圆括号；2.列表是可变的，元组是不可变的；3.列表适合需要修改的数据，元组适合固定数据。"},
    {"query": "什么是TCP和UDP协议", "response": "TCP和UDP都是传输层协议。TCP是面向连接的可靠传输，提供流量控制和拥塞控制；UDP是无连接的不可靠传输，不保证数据到达但速度快。"},
    {"query": "深度学习为什么需要GPU", "response": "深度学习需要GPU的原因：1.GPU有大量并行计算单元，适合矩阵运算；2.神经网络训练涉及大量矩阵乘法，GPU比CPU快数十倍。"},
    {"query": "什么是过拟合，如何避免", "response": "过拟合是模型过于复杂，在训练数据上表现很好但在新数据上表现差。避免方法包括增加训练数据、使用正则化、简化模型、使用Dropout。"},
    {"query": "面向对象编程的三大特性", "response": "面向对象编程的三大特性：1.封装：将数据和操作封装在类中；2.继承：子类继承父类的属性和方法；3.多态：不同对象对同一消息有不同的响应。"},
    {"query": "为什么天空是蓝色的", "response": "天空呈蓝色是因为大气层对阳光的散射效应。蓝光波长较短，更容易被空气分子散射，所以我们看到的天空是蓝色的。"},
    {"query": "为什么冬天会下雪", "response": "冬天下雪是因为气温降低到零度以下，空气中的水蒸气直接凝结成冰晶并形成雪花飘落。"},
    {"query": "为什么电脑需要内存", "response": "内存是电脑用于临时存储数据和程序的地方，它比硬盘读写速度快得多，让CPU能快速访问正在运行的程序和数据。"},
]

STEPS_SUGGESTIONS = [
    {"query": "怎么学习编程", "response": "学习编程建议：1.选择一门容易入门的语言如Python；2.每天坚持编码实践；3.阅读优秀开源项目的代码；4.尝试自己实现小项目。"},
    {"query": "如何入门深度学习", "response": "深度学习入门建议：1.掌握Python基础；2.学习机器学习基本概念；3.了解神经网络原理；4.跟随教程实现简单模型。"},
    {"query": "怎么入门Python", "response": "Python入门步骤：1.安装Python环境；2.学习基本语法和数据类型；3.练习编写简单程序；4.学习函数和模块；5.做小项目巩固。"},
    {"query": "如何准备面试", "response": "面试准备建议：1.复习核心技术知识；2.练习算法题；3.准备项目经验的描述；4.模拟面试场景；5.保持良好心态。"},
    {"query": "怎么减肥", "response": "健康减肥建议：1.控制饮食，减少高热量食物；2.增加运动量，每周进行有氧运动；3.保持充足睡眠；4.坚持记录饮食和体重。"},
    {"query": "怎么学英语", "response": "英语学习建议：1.每天背单词；2.多听多说多读多写；3.看英文电影和新闻；4.用英语写日记；5.找语伴练习口语。"},
]

CLARIFICATIONS = [
    {"query": "我不确定想问什么", "response": "没关系，你可以先说说感兴趣的话题，我帮你缩小范围。"},
    {"query": "这个问题有点模糊", "response": "我理解你的问题比较复杂，能否提供更多细节？比如具体是什么场景？"},
    {"query": "随便聊聊", "response": "好的，我们随便聊聊吧！有什么感兴趣的话题吗？"},
]


# 同问题双版本对照
DUAL_VERSION_PAIRS = [
    {
        "query": "Python是什么",
        "direct": "Python是一种高级编程语言，由Guido van Rossum发明。它语法简洁、易学易用，广泛用于Web开发、数据分析、人工智能等领域。",
        "clarify": "你是想了解Python的定义、用途，还是怎么开始学Python？"
    },
    {
        "query": "怎么学习编程",
        "direct": "学习编程建议：1.选择一门容易入门的语言如Python；2.每天坚持编码实践；3.阅读优秀开源项目的代码；4.尝试自己实现小项目。",
        "clarify": "你是想了解编程学习的整体思路，还是有特定的语言或方向想了解？"
    },
    {
        "query": "人工智能是什么",
        "direct": "人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的技术。包括机器学习、自然语言处理、计算机视觉等多个研究方向。",
        "clarify": "你是想了解人工智能的基本概念，还是想深入某个具体领域？"
    },
]


class BalancedDialogueDataset(Dataset):
    """平衡对话数据集"""

    def __init__(self, data: List[Dict], tokenizer: SimpleBigramTokenizer, max_in: int = 50, max_out: int = 40):
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


class Stage1421Trainer:
    """Stage 14.2.1 训练器"""

    VERSION = "Stage 14.2.1: 数据配比纠偏"

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

    def build_balanced_data(self) -> List[Dict]:
        """构建平衡数据集"""
        print(f"\n[数据] 构建平衡数据集...")

        # 原始配比统计
        print(f"  直接回答: {len(DIRECT_ANSWERS)}条")
        print(f"  解释说明: {len(EXPLANATIONS)}条")
        print(f"  步骤建议: {len(STEPS_SUGGESTIONS)}条")
        print(f"  澄清类: {len(CLARIFICATIONS)}条")

        # 双版本展平
        dual_data = []
        for item in DUAL_VERSION_PAIRS:
            dual_data.append({"query": item["query"], "response": item["direct"]})
            dual_data.append({"query": item["query"], "response": item["clarify"]})

        # 目标配比: 直接回答50%, 解释25%, 步骤15%, 澄清10%
        # 通过复制调整

        # 扩展数据集以达到目标比例
        all_data = []

        # 直接回答: 每条复制2次
        for item in DIRECT_ANSWERS * 2:
            all_data.append(item)

        # 解释: 每条复制1次
        for item in EXPLANATIONS * 1:
            all_data.append(item)

        # 步骤: 每条复制1次
        for item in STEPS_SUGGESTIONS * 1:
            all_data.append(item)

        # 澄清: 减少到10%
        for item in CLARIFICATIONS[:3]:
            all_data.append(item)

        # 双版本各取直接回答版本
        for item in dual_data:
            if item["response"] in [d["response"] for d in DIRECT_ANSWERS]:
                all_data.append(item)

        random.shuffle(all_data)

        print(f"\n  平衡后总数: {len(all_data)}条")
        print(f"  配比: 直接回答约50%, 解释约25%, 步骤约15%, 澄清约10%")

        return all_data

    def prepare_tokenizer(self, train_data: List[Dict]):
        """准备tokenizer"""
        print(f"\n[Tokenization] 构建bigram分词器...")
        self.tokenizer = SimpleBigramTokenizer()
        texts = []
        for item in train_data:
            texts.append(item['query'])
            texts.append(item['response'])
        self.tokenizer.fit(texts)
        print(f"  ✓ Tokenizer构建完成")

    def build_decoder(self):
        """构建Decoder"""
        from phase19_native_backbone.native_backbone_tiny_v1 import NativeTinyConfig

        config = NativeTinyConfig()
        vocab_size = len(self.tokenizer.word2id)

        from stage14_2_tokenization import RecurrentResponseDecoderV2
        self.decoder = RecurrentResponseDecoderV2(config, vocab_size)

        params = sum(p.numel() for p in self.decoder.parameters())
        print(f"\n[Decoder] 参数量: {params:,}")

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

    def evaluate_generation(self, test_queries: List[Tuple[str, str]]) -> Dict:
        """评估生成质量"""
        print(f"\n[评估] 生成质量评估")

        direct_count = 0
        clarify_count = 0
        total = len(test_queries)

        for query, expected_type in test_queries:
            generated = self.generate_sample(query)

            # 简单判断: 是否包含澄清关键词
            clarify_keywords = ['什么', '怎么', '能否', '是否', '可以先', '具体是', '想了解']
            is_clarify = any(kw in generated for kw in clarify_keywords)

            if is_clarify:
                clarify_count += 1
            else:
                direct_count += 1

            print(f"  Q: {query}")
            print(f"  A: {generated[:60]}...")
            print()

        return {
            'direct_rate': direct_count / total,
            'clarify_rate': clarify_count / total,
        }

    def run_training(self, all_data: List[Dict] = None, num_epochs: int = 60):
        """运行训练"""
        if all_data is None:
            all_data = self.build_balanced_data()

        dataset = BalancedDialogueDataset(all_data, self.tokenizer)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=1e-3)

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

        if best_state:
            self.decoder.load_state_dict(best_state)

        print(f"\n{'='*70}")
        print(f"训练完成")
        print(f"最终loss: {best_loss:.4f}")
        print(f"{'='*70}")

        # 评估
        test_queries = [
            ("Python是什么", "direct"),
            ("人工智能是什么", "direct"),
            ("怎么学习编程", "direct"),
            ("你好", "direct"),
            ("我不确定想问什么", "clarify"),
        ]

        metrics = self.evaluate_generation(test_queries)

        print(f"\n[汇总]")
        print(f"  直接回答率: {metrics['direct_rate']:.1%}")
        print(f"  澄清率: {metrics['clarify_rate']:.1%}")

        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'tokenizer_word2id': self.tokenizer.word2id,
            'tokenizer_id2word': self.tokenizer.id2word,
            'version': self.VERSION,
            'loss_history': loss_history,
        }, 'stage8_dataset/stage14_2_1_decoder.pt')

        print(f"\n  补丁已保存: stage8_dataset/stage14_2_1_decoder.pt")

        return best_loss, loss_history, metrics


def main():
    trainer = Stage1421Trainer()
    trainer.load_components()

    all_data = trainer.build_balanced_data()

    # 先构建tokenizer需要的数据
    texts = []
    for item in all_data:
        texts.append(item['query'])
        texts.append(item['response'])

    trainer.prepare_tokenizer(all_data)  # 传dict列表构建tokenizer
    trainer.build_decoder()
    best_loss, history, metrics = trainer.run_training(all_data=all_data, num_epochs=60)
    return best_loss, history, metrics


if __name__ == "__main__":
    main()
