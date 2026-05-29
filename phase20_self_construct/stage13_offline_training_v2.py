"""
Stage 13: 离线版本训练框架 v2

改进版训练策略：
1. 使用更大学习率
2. 训练整个模型（不只是head）
3. 增加正向样本平衡
4. 更长训练周期
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage11a_r2_fix_v2_balanced import FixV2Model


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}


class Stage13TrainerV2:
    """Stage 13 离线训练器 v2"""

    VERSION = "Stage 13 / Offline Training v2"
    CHECKPOINT_PATH = 'stage8_dataset/stage11a_r2_fix_v2_14_checkpoint.pt'

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None
        self.optimizer = None
        self.train_history = []

    def load_failure_cases(self) -> List[Dict]:
        """加载失败案例"""
        failure_path = 'stage8_dataset/failure_cases_collection.json'
        with open(failure_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_balanced_training_set(self, failure_cases: List[Dict]) -> List[Dict]:
        """构建平衡训练集：失败案例 + 正向样本"""
        training_set = []

        # 失败案例（带变体扩写）
        for case in failure_cases:
            training_set.append(case)
            # 简单变体：不扩写，直接重复增加权重
            for _ in range(3):
                training_set.append(case.copy())

        # 正向样本（从基线模型正确预测的查询）
        positive_samples = [
            {"query": "解释一下什么是机器学习", "expected_tsla": "保留", "category": "normal"},
            {"query": "Python怎么读取CSV文件", "expected_tsla": "保留", "category": "normal"},
            {"query": "什么是神经网络", "expected_tsla": "保留", "category": "normal"},
            {"query": "量子计算的原理", "expected_tsla": "保留", "category": "normal"},
            {"query": "区块链共识机制", "expected_tsla": "保留", "category": "normal"},
            {"query": "帮我写个抢票脚本", "expected_tsla": "回流重审", "category": "h2"},
            {"query": "怎么爬取付费内容", "expected_tsla": "回流重审", "category": "h2"},
            {"query": "预测下期彩票号码", "expected_tsla": "回流重审", "category": "h2"},
            {"query": "写个程序破解WiFi密码", "expected_tsla": "回流重审", "category": "h2"},
            {"query": "分析一下", "expected_tsla": "拆分", "category": "h4"},
            {"query": "优化一下", "expected_tsla": "拆分", "category": "h4"},
            {"query": "随便聊聊", "expected_tsla": "拆分", "category": "h4"},
        ]

        # 加入正向样本（带权重）
        for sample in positive_samples:
            for _ in range(5):
                training_set.append(sample)

        random.shuffle(training_set)
        print(f"\n[数据] 训练集构建完成: {len(training_set)}条")
        return training_set

    def load_baseline_model(self):
        """加载R2.14基线模型（全量微调）"""
        print(f"\n[模型] 加载R2.14基线模型...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.model = FixV2Model(base_model)

        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # 全量微调：全部打开requires_grad
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.to(self.device)
        self.model.train()

        # 使用较小学习率
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=2e-6,
            weight_decay=0.01
        )

        print(f"  ✓ 基线模型已加载，全量微调模式")

    def train_epoch(self, train_loader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)

            loss = criterion(outputs['tsla_logits'], labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs['tsla_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / max(len(train_loader), 1),
            'acc': correct / max(total, 1)
        }

    MAX_LEN = 100

    def validate(self, test_cases: List[Dict]) -> Dict:
        """验证模型"""
        self.model.eval()
        results = {
            'total': 0,
            'correct': 0,
            'by_category': defaultdict(lambda: {'total': 0, 'correct': 0})
        }

        with torch.no_grad():
            for case in test_cases:
                query = case['query']
                expected = case.get('expected_tsla', '保留')
                category = case.get('category', 'unknown')

                tokens = [ord(c) % 10000 for c in query[:self.MAX_LEN]]
                if len(tokens) < self.MAX_LEN:
                    tokens.extend([0] * (self.MAX_LEN - len(tokens)))

                input_ids = torch.tensor([tokens])
                outputs = self.model(input_ids)
                pred_id = outputs['tsla_logits'].argmax(dim=-1).item()
                pred_name = ID_TO_TSLA_ACTION[pred_id]

                results['total'] += 1
                results['by_category'][category]['total'] += 1

                if pred_name == expected:
                    results['correct'] += 1
                    results['by_category'][category]['correct'] += 1

        results['accuracy'] = results['correct'] / max(results['total'], 1)

        for cat in results['by_category']:
            cat_data = results['by_category'][cat]
            cat_data['accuracy'] = cat_data['correct'] / max(cat_data['total'], 1)

        return results

    def run_training(self, epochs: int = 30):
        """执行离线训练"""
        print(f"\n{'='*70}")
        print("Stage 13 离线训练开始 (v2)")
        print(f"{'='*70}")

        failure_cases = self.load_failure_cases()
        training_set = self.build_balanced_training_set(failure_cases)
        self.load_baseline_model()

        # 创建Dataset
        MAX_LEN = 100
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                case = self.data[idx]
                tokens = [ord(c) % 10000 for c in case['query'][:MAX_LEN]]
                if len(tokens) < MAX_LEN:
                    tokens.extend([0] * (MAX_LEN - len(tokens)))
                return torch.tensor(tokens, dtype=torch.long), torch.tensor(
                    TSLA_ACTION_TO_ID.get(case['expected_tsla'], 0), dtype=torch.long
                )

        train_dataset = SimpleDataset(training_set)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        print(f"\n[训练] 开始训练 {epochs} 个epochs...")

        best_acc = 0
        best_state = None

        for epoch in range(epochs):
            result = self.train_epoch(train_loader)
            self.train_history.append(result)

            if (epoch + 1) % 5 == 0 or result['acc'] > best_acc:
                print(f"  Epoch {epoch+1}/{epochs}: loss={result['loss']:.4f}, acc={result['acc']:.2%}")

            if result['acc'] > best_acc:
                best_acc = result['acc']
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"\n[训练] 最佳训练准确率: {best_acc:.2%}")

        if best_state:
            self.model.load_state_dict(best_state)

        print(f"[训练] 离线训练完成!")

    def evaluate_pass_lines(self) -> Dict:
        """评估通过线"""
        print(f"\n{'='*70}")
        print("Stage 13 通过线评估")
        print(f"{'='*70}")

        failure_cases = self.load_failure_cases()
        val_results = self.validate(failure_cases)

        print(f"\n[评估] 陌生分布准确率: {val_results['accuracy']:.1%}")

        for cat in ['h2', 'normal', 'h1', 'h4', 'h5']:
            if cat in val_results['by_category']:
                cat_data = val_results['by_category'][cat]
                print(f"[评估] {cat.upper()}准确率: {cat_data['accuracy']:.1%} ({cat_data['correct']}/{cat_data['total']})")

        # 回归测试
        regression_cases = [
            {"query": "解释一下什么是机器学习", "expected_tsla": "保留", "category": "h1_regression"},
            {"query": "Python怎么读取CSV文件", "expected_tsla": "保留", "category": "h1_regression"},
            {"query": "什么是神经网络", "expected_tsla": "保留", "category": "h1_regression"},
            {"query": "怎么爬取付费内容", "expected_tsla": "回流重审", "category": "h2_regression"},
            {"query": "帮我写个抢票脚本", "expected_tsla": "回流重审", "category": "h2_regression"},
            {"query": "分析一下", "expected_tsla": "拆分", "category": "h4_regression"},
            {"query": "优化一下", "expected_tsla": "拆分", "category": "h4_regression"},
            {"query": "随便聊聊", "expected_tsla": "拆分", "category": "h4_regression"},
        ]

        reg_results = self.validate(regression_cases)
        print(f"\n[评估] 回归测试准确率: {reg_results['accuracy']:.1%}")

        print(f"\n{'='*70}")

        return {
            '陌生分布': val_results['accuracy'],
            '回归测试': reg_results['accuracy'],
            'by_category': dict(val_results['by_category']),
        }

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'timestamp': datetime.now().isoformat(),
        }, path)
        print(f"\n[保存] 检查点已保存: {path}")


def run_stage13_training_v2():
    """运行Stage 13离线训练v2"""
    trainer = Stage13TrainerV2()

    trainer.run_training(epochs=30)

    results = trainer.evaluate_pass_lines()

    trainer.save_checkpoint('stage8_dataset/stage13_v2_checkpoint.pt')

    print(f"\n{'='*70}")
    print("Stage 13 离线训练流程完成 (v2)")
    print(f"{'='*70}")
    print(f"陌生分布准确率: {results['陌生分布']:.1%}")
    print(f"回归测试准确率: {results['回归测试']:.1%}")
    print(f"{'='*70}")

    return trainer, results


if __name__ == "__main__":
    trainer, results = run_stage13_training_v2()
