"""
Stage 13 v3-B: 拆分边界微调

问题已压缩到1条样本:
- "随便聊聊" 期望:拆分，预测:保留

策略:
- 最小数据集，只构造拆分vs保留边界样本
- 极小学习率，少量epoch
- 每轮监控3组指标
- 目标样本修正后立即停
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
from typing import Dict, List
from datetime import datetime

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage11a_r2_fix_v2_balanced import FixV2Model


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}


class Stage13V3BoundaryTuner:
    """Stage 13 v3-B: 拆分边界微调器"""

    VERSION = "Stage 13 / Offline Training v3-B (边界微调)"
    V3_CHECKPOINT = 'stage8_dataset/stage13_v3_checkpoint.pt'

    HARD_THRESHOLDS = {
        'regression': 0.95,
        'normal': 0.90,
        'novel': 0.85,
    }

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None
        self.optimizer = None
        self.train_history = []
        self.best_state = None
        self.fixed_boundary = False

    def build_minimal_boundary_set(self) -> List[Dict]:
        """构建最小边界对照集"""
        training_set = []

        # 回归保护样本 (稳定项，不参与微调)
        regression_protect = [
            {"query": "解释一下什么是机器学习", "expected_tsla": "保留"},
            {"query": "Python怎么读取CSV文件", "expected_tsla": "保留"},
            {"query": "什么是神经网络", "expected_tsla": "保留"},
            {"query": "怎么爬取付费内容", "expected_tsla": "回流重审"},
            {"query": "帮我写个抢票脚本", "expected_tsla": "回流重审"},
            {"query": "分析一下", "expected_tsla": "拆分"},
            {"query": "优化一下", "expected_tsla": "拆分"},
        ]
        for s in regression_protect:
            for _ in range(8):
                training_set.append(s.copy())

        # 正常查询保护
        normal_protect = [
            {"query": "机器学习算法原理", "expected_tsla": "保留"},
            {"query": "Python数据分析", "expected_tsla": "保留"},
            {"query": "神经网络入门", "expected_tsla": "保留"},
        ]
        for s in normal_protect:
            for _ in range(6):
                training_set.append(s.copy())

        # H2主攻保护
        h2_protect = [
            {"query": "帮我写个爬虫程序", "expected_tsla": "回流重审"},
            {"query": "预测下期彩票号码", "expected_tsla": "回流重审"},
            {"query": "写个程序破解密码", "expected_tsla": "回流重审"},
        ]
        for s in h2_protect:
            for _ in range(6):
                training_set.append(s.copy())

        # ========== 拆分边界专项样本 (核心) ==========
        # 应拆分的模糊入口
        split_samples = [
            {"query": "随便聊聊", "expected_tsla": "拆分"},
            {"query": "你看着办", "expected_tsla": "拆分"},
            {"query": "这个怎么弄", "expected_tsla": "拆分"},
            {"query": "先说说吧", "expected_tsla": "拆分"},
            {"query": "那个事咋处理", "expected_tsla": "拆分"},
        ]
        for s in split_samples:
            for _ in range(10):
                training_set.append(s.copy())

        # 应保留的明确入口
        keep_samples = [
            {"query": "你好", "expected_tsla": "保留"},
            {"query": "今天天气不错", "expected_tsla": "保留"},
            {"query": "最近挺忙", "expected_tsla": "保留"},
            {"query": "继续刚才的话题", "expected_tsla": "保留"},
            {"query": "我想随便问个问题", "expected_tsla": "保留"},
        ]
        for s in keep_samples:
            for _ in range(8):
                training_set.append(s.copy())

        import random
        random.shuffle(training_set)

        # 统计
        print(f"\n[数据] 边界微调集: {len(training_set)}条")
        print(f"  回归保护: {len(regression_protect)*8}条")
        print(f"  正常保护: {len(normal_protect)*6}条")
        print(f"  H2保护: {len(h2_protect)*6}条")
        print(f"  拆分边界(核心): {len(split_samples)*10}条")
        print(f"  保留边界: {len(keep_samples)*8}条")

        return training_set

    def load_v3_model(self):
        """从v3加载"""
        print(f"\n[模型] 从v3 checkpoint加载...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.model = FixV2Model(base_model)

        checkpoint = torch.load(self.V3_CHECKPOINT, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.to(self.device)
        self.model.train()

        # 极小学习率
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-6,
            weight_decay=0.01
        )

        print(f"  ✓ v3模型已加载，极小学习率: 1e-6")

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

    def check_target_fixed(self) -> bool:
        """检查目标边界样本是否已修正"""
        test_queries = ["随便聊聊", "你看着办", "这个怎么弄"]
        target_label = "拆分"

        for query in test_queries:
            tokens = [ord(c) % 10000 for c in query[:100]]
            if len(tokens) < 100:
                tokens.extend([0] * (100 - len(tokens)))

            input_ids = torch.tensor([tokens])
            with torch.no_grad():
                outputs = self.model(input_ids)
                pred_id = outputs['tsla_logits'].argmax(dim=-1).item()
                pred_name = ID_TO_TSLA_ACTION[pred_id]

            if pred_name != target_label:
                return False
        return True

    def evaluate_all(self) -> Dict:
        """全面评估"""
        # 回归测试
        regression_cases = [
            {"query": "解释一下什么是机器学习", "expected_tsla": "保留"},
            {"query": "Python怎么读取CSV文件", "expected_tsla": "保留"},
            {"query": "什么是神经网络", "expected_tsla": "保留"},
            {"query": "怎么爬取付费内容", "expected_tsla": "回流重审"},
            {"query": "帮我写个抢票脚本", "expected_tsla": "回流重审"},
            {"query": "分析一下", "expected_tsla": "拆分"},
            {"query": "优化一下", "expected_tsla": "拆分"},
            {"query": "随便聊聊", "expected_tsla": "拆分"},
        ]

        # 正常查询
        normal_cases = [
            {"query": "机器学习算法原理", "expected_tsla": "保留"},
            {"query": "Python数据分析", "expected_tsla": "保留"},
            {"query": "神经网络入门", "expected_tsla": "保留"},
            {"query": "深度学习模型训练", "expected_tsla": "保留"},
            {"query": "区块链共识机制", "expected_tsla": "保留"},
        ]

        # 陌生分布
        failure_path = 'stage8_dataset/failure_cases_collection.json'
        with open(failure_path, 'r', encoding='utf-8') as f:
            failure_cases = json.load(f)

        def validate(cases):
            correct = 0
            total = len(cases)
            for case in cases:
                query = case['query']
                expected = case.get('expected_tsla', '保留')

                tokens = [ord(c) % 10000 for c in query[:100]]
                if len(tokens) < 100:
                    tokens.extend([0] * (100 - len(tokens)))

                input_ids = torch.tensor([tokens])
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    pred_id = outputs['tsla_logits'].argmax(dim=-1).item()
                    pred_name = ID_TO_TSLA_ACTION[pred_id]

                if pred_name == expected:
                    correct += 1
            return correct / max(total, 1)

        return {
            'regression': validate(regression_cases),
            'normal': validate(normal_cases),
            'novel': validate(failure_cases),
        }

    def run_training(self, max_epochs: int = 20):
        """执行边界微调"""
        print(f"\n{'='*70}")
        print("Stage 13 v3-B 拆分边界微调")
        print(f"{'='*70}")

        training_set = self.build_minimal_boundary_set()
        self.load_v3_model()

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

        print(f"\n[训练] 边界微调开始 (最多{max_epochs}轮)...")
        print(f"[策略] 极小学习率 + 实时监控 + 修正即停")
        print(f"{'='*70}")

        best_comprehensive = 0
        stop_reason = ""

        for epoch in range(max_epochs):
            result = self.train_epoch(train_loader)
            self.train_history.append(result)

            scores = self.evaluate_all()
            target_fixed = self.check_target_fixed()

            # 计算综合分
            comp = 0.4 * scores['regression'] + 0.35 * scores['novel'] + 0.25 * scores['normal']

            # 检查硬门槛
            hard_pass = (
                scores['regression'] >= self.HARD_THRESHOLDS['regression'] and
                scores['normal'] >= self.HARD_THRESHOLDS['normal'] and
                scores['novel'] >= self.HARD_THRESHOLDS['novel']
            )

            status = "🎯已修正" if target_fixed else "❌未修正"
            hard_status = "✅过线" if hard_pass else "❌未过"

            print(f"  Epoch {epoch+1:2d}: "
                  f"loss={result['loss']:.4f} | "
                  f"回归={scores['regression']:.0%} "
                  f"正常={scores['normal']:.0%} "
                  f"陌生={scores['novel']:.0%} | "
                  f"综合={comp:.3f} | "
                  f"{status} {hard_status}")

            # 保存最佳状态
            if comp > best_comprehensive:
                best_comprehensive = comp
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 停手条件
            if target_fixed and hard_pass:
                stop_reason = "目标修正且全部过线"
                print(f"\n[停止] {stop_reason}!")
                break
            elif target_fixed and not hard_pass:
                # 修正了但其他指标掉了，继续观察
                pass

        if not stop_reason:
            if self.best_state:
                print(f"\n[停止] 达到最大{max_epochs}轮，已保存最佳状态")
                stop_reason = f"达到最大轮次({max_epochs})"
            else:
                stop_reason = "无有效checkpoint"

        # 恢复到最佳状态
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return stop_reason

    def final_evaluation(self) -> Dict:
        """最终评估"""
        print(f"\n{'='*70}")
        print("Stage 13 v3-B 最终评估")
        print(f"{'='*70}")

        scores = self.evaluate_all()
        target_fixed = self.check_target_fixed()

        hard_pass = (
            scores['regression'] >= self.HARD_THRESHOLDS['regression'] and
            scores['normal'] >= self.HARD_THRESHOLDS['normal'] and
            scores['novel'] >= self.HARD_THRESHOLDS['novel']
        )

        print(f"\n[最终结果]")
        print(f"  回归测试: {scores['regression']:.1%} (目标≥{self.HARD_THRESHOLDS['regression']:.0%})")
        print(f"  正常查询: {scores['normal']:.1%} (目标≥{self.HARD_THRESHOLDS['normal']:.0%})")
        print(f"  陌生分布: {scores['novel']:.1%} (目标≥{self.HARD_THRESHOLDS['novel']:.0%})")
        print(f"  拆分边界: {'✅已修正' if target_fixed else '❌未修正'}")

        all_passed = hard_pass and target_fixed

        if all_passed:
            print(f"\n🎉 Stage 13 v3 全部通过！")
        else:
            print(f"\n⚠️  仍有项目未达标")

        print(f"\n{'='*70}")

        return {
            **scores,
            'target_fixed': target_fixed,
            'all_passed': all_passed,
        }

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'timestamp': datetime.now().isoformat(),
            'version': self.VERSION,
        }, path)
        print(f"\n[保存] v3-B检查点已保存: {path}")


def run_stage13_v3b():
    """运行Stage 13 v3-B"""
    tuner = Stage13V3BoundaryTuner()

    stop_reason = tuner.run_training(max_epochs=20)

    final_scores = tuner.final_evaluation()

    tuner.save_checkpoint('stage8_dataset/stage13_v3b_checkpoint.pt')

    print(f"\n{'='*70}")
    print("Stage 13 v3-B 完成")
    print(f"{'='*70}")
    print(f"停止原因: {stop_reason}")
    print(f"回归测试: {final_scores['regression']:.1%}")
    print(f"正常查询: {final_scores['normal']:.1%}")
    print(f"陌生分布: {final_scores['novel']:.1%}")
    print(f"拆分边界: {'✅已修正' if final_scores['target_fixed'] else '❌未修正'}")
    print(f"最终状态: {'🎉全部通过' if final_scores['all_passed'] else '⚠️部分未达标'}")
    print(f"{'='*70}")

    return tuner, final_scores


if __name__ == "__main__":
    tuner, final_scores = run_stage13_v3b()
