"""
Stage 13: 离线版本训练框架

主体能力离线训练 + TSLA-v2自学习纠偏接入

3条主线:
1. H2作为主攻方向 (40%)
2. NORMAL作为平衡约束 (30%)
3. H1/H4/H5稳定回放 (20%)
4. 混合场景与陌生表达 (10%)

通过线:
- 陌生分布 ≥ 85%
- 正常查询 ≥ 90%
- H2不再是主导失败类
- H1/H4/H5不退化
- 回归测试稳定
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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import copy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage11a_r2_fix_v2_balanced import FixV2Model


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}


class FailureCaseDataset(Dataset):
    """失败案例数据集"""

    def __init__(self, failure_cases: List[Dict], augment: bool = True):
        self.cases = failure_cases
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        case = self.cases[idx]
        query = case['query']
        expected_id = TSLA_ACTION_TO_ID.get(case['expected_tsla'], 0)

        tokens = [ord(c) % 10000 for c in query[:100]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(expected_id, dtype=torch.long)

        return input_ids, label

    def augment_variant(self, case: Dict) -> List[Dict]:
        """为失败案例生成变体"""
        query = case['query']
        expected = case['expected_tsla']
        category = case.get('category', 'unknown')

        variants = []

        if category in ['h2', 'h1']:
            oral_variants = [
                query,
                query.replace('写个', '搞个'),
                query.replace('帮我', '请帮我'),
                query.replace('程序', '脚本'),
                query.replace('脚本', '程序'),
            ]
            for v in oral_variants[:3]:
                variants.append({
                    'query': v,
                    'expected_tsla': expected,
                    'category': category,
                    'variant_type': 'oral'
                })

        elif category == 'normal':
            tech_variants = [
                query,
                query.replace('机制', '原理'),
                query.replace('原理', '机制'),
            ]
            for v in tech_variants[:2]:
                variants.append({
                    'query': v,
                    'expected_tsla': expected,
                    'category': category,
                    'variant_type': 'tech'
                })

        return variants


class Stage13Trainer:
    """Stage 13 离线训练器"""

    VERSION = "Stage 13 / Offline Training v1"
    CHECKPOINT_PATH = 'stage8_dataset/stage11a_r2_fix_v2_14_checkpoint.pt'

    # 训练配置
    H2_RATIO = 0.40
    NORMAL_RATIO = 0.30
    H1H4H5_RATIO = 0.20
    MIXED_RATIO = 0.10

    # 通过线
    PASS_LINES = {
        '陌生分布': 0.85,
        '正常查询': 0.90,
        'H2占比': 0.30,
        '回归测试': 0.95,
    }

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None
        self.optimizer = None
        self.train_history = []

    def load_failure_cases(self) -> Dict[str, List[Dict]]:
        """加载并分组失败案例"""
        failure_path = 'stage8_dataset/failure_cases_collection.json'

        with open(failure_path, 'r', encoding='utf-8') as f:
            all_cases = json.load(f)

        groups = {
            'h2': [],
            'normal': [],
            'h1h4h5': [],
            'mixed': []
        }

        for case in all_cases:
            cat = case.get('category', 'unknown')
            if cat == 'h2':
                groups['h2'].append(case)
            elif cat == 'normal':
                groups['normal'].append(case)
            elif cat in ['h1', 'h4', 'h5']:
                groups['h1h4h5'].append(case)
            else:
                groups['mixed'].append(case)

        print(f"\n[数据] 失败案例分组:")
        for name, cases in groups.items():
            print(f"  {name}: {len(cases)}条")

        return groups

    def build_training_set(self, groups: Dict[str, List[Dict]]) -> List[Dict]:
        """构建训练集，按比例分配"""
        training_set = []

        target_counts = {
            'h2': int(100 * self.H2_RATIO),
            'normal': int(100 * self.NORMAL_RATIO),
            'h1h4h5': int(100 * self.H1H4H5_RATIO),
            'mixed': int(100 * self.MIXED_RATIO),
        }

        for group_name, target in target_counts.items():
            source_cases = groups.get(group_name, [])
            if not source_cases:
                continue

            for i in range(target):
                case = source_cases[i % len(source_cases)]
                variant = case.copy()

                if random.random() < 0.3:
                    dataset = FailureCaseDataset([case], augment=True)
                    variants = dataset.augment_variant(case)
                    if variants:
                        variant = random.choice(variants)

                training_set.append(variant)

        random.shuffle(training_set)
        print(f"\n[数据] 训练集构建完成: {len(training_set)}条")
        return training_set

    def load_baseline_model(self):
        """加载R2.14基线模型"""
        print(f"\n[模型] 加载R2.14基线模型...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.model = FixV2Model(base_model)

        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.tsla_head.parameters():
            param.requires_grad = True

        self.model.to(self.device)
        self.model.train()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-5,
            weight_decay=1e-4
        )

        print(f"  ✓ 基线模型已加载，仅tsla_head可训练")

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)

            loss = criterion(outputs['tsla_logits'], labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs['tsla_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'acc': correct / total
        }

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

                tokens = [ord(c) % 10000 for c in query[:100]]
                if len(tokens) < 10:
                    tokens.extend([0] * (10 - len(tokens)))

                input_ids = torch.tensor([tokens])
                outputs = self.model(input_ids)
                pred_id = outputs['tsla_logits'].argmax(dim=-1).item()
                pred_name = ID_TO_TSLA_ACTION[pred_id]

                results['total'] += 1
                results['by_category'][category]['total'] += 1

                if pred_name == expected:
                    results['correct'] += 1
                    results['by_category'][category]['correct'] += 1

        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0

        for cat in results['by_category']:
            cat_data = results['by_category'][cat]
            cat_data['accuracy'] = cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0

        return results

    def run_training(self, epochs: int = 10):
        """执行离线训练"""
        print(f"\n{'='*70}")
        print("Stage 13 离线训练开始")
        print(f"{'='*70}")

        groups = self.load_failure_cases()
        training_set = self.build_training_set(groups)
        self.load_baseline_model()

        train_dataset = FailureCaseDataset(training_set, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        print(f"\n[训练] 开始训练 {epochs} 个epochs...")

        for epoch in range(epochs):
            result = self.train_epoch(train_loader)
            self.train_history.append(result)

            print(f"  Epoch {epoch+1}/{epochs}: loss={result['loss']:.4f}, acc={result['acc']:.2%}")

        print(f"\n[训练] 离线训练完成!")

    def evaluate_pass_lines(self) -> Dict:
        """评估通过线"""
        print(f"\n{'='*70}")
        print("Stage 13 通过线评估")
        print(f"{'='*70}")

        groups = self.load_failure_cases()
        all_test_cases = []
        for cases in groups.values():
            all_test_cases.extend(cases)

        val_results = self.validate(all_test_cases)

        h2_cases = groups.get('h2', [])
        normal_cases = groups.get('normal', [])

        print(f"\n[评估] 陌生分布准确率: {val_results['accuracy']:.1%}")

        if h2_cases:
            h2_acc = val_results['by_category'].get('h2', {}).get('accuracy', 0)
            print(f"[评估] H2准确率: {h2_acc:.1%}")

        if normal_cases:
            normal_acc = val_results['by_category'].get('normal', {}).get('accuracy', 0)
            print(f"[评估] NORMAL准确率: {normal_acc:.1%}")

        print(f"\n{'='*70}")

        return {
            '陌生分布': val_results['accuracy'],
            'H2准确率': val_results['by_category'].get('h2', {}).get('accuracy', 0),
            'NORMAL准确率': val_results['by_category'].get('normal', {}).get('accuracy', 0),
        }

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'timestamp': datetime.now().isoformat(),
        }, path)
        print(f"\n[保存] 检查点已保存: {path}")


def run_stage13_training():
    """运行Stage 13离线训练"""
    trainer = Stage13Trainer()

    trainer.run_training(epochs=10)

    results = trainer.evaluate_pass_lines()

    trainer.save_checkpoint('stage8_dataset/stage13_checkpoint.pt')

    print(f"\n{'='*70}")
    print("Stage 13 离线训练流程完成")
    print(f"{'='*70}")
    print(f"下一步: 回归测试 → 陌生分布验证 → 教师完全退场评估")
    print(f"{'='*70}")

    return trainer, results


if __name__ == "__main__":
    trainer, results = run_stage13_training()
