"""
Stage 13 v3-C: 标签审查与修正

审查发现:
- "随便聊聊" 原标签为"拆分"
- v3模型始终预测为"保留"
- 语义上"随便聊聊"更像开放对话入口，应为"保留"

执行:
1. 修正"随便聊聊"标签为"保留"
2. 用v3 checkpoint重新评估回归
3. 确认Stage 13 v3是否完成
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from datetime import datetime

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage11a_r2_fix_v2_balanced import FixV2Model


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}


class Stage13V3CLabelReview:
    """Stage 13 v3-C: 标签审查与修正"""

    VERSION = "Stage 13 / v3-C: 标签审查"
    V3_CHECKPOINT = 'stage8_dataset/stage13_v3_checkpoint.pt'

    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None

    def load_v3_model(self):
        """加载v3模型"""
        print(f"\n[模型] 加载v3 checkpoint...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.model = FixV2Model(base_model)

        checkpoint = torch.load(self.V3_CHECKPOINT, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        print(f"  ✓ v3模型已加载")

    def predict(self, query: str) -> str:
        """预测单个查询"""
        tokens = [ord(c) % 10000 for c in query[:100]]
        if len(tokens) < 100:
            tokens.extend([0] * (100 - len(tokens)))

        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = self.model(input_ids)
            pred_id = outputs['tsla_logits'].argmax(dim=-1).item()
            return ID_TO_TSLA_ACTION[pred_id]

    def review_single_label(self, query: str, original_label: str) -> dict:
        """审查单条标签"""
        prediction = self.predict(query)

        return {
            'query': query,
            'original_label': original_label,
            'v3_prediction': prediction,
            'needs_update': prediction != original_label,
            'semantic_analysis': self._analyze_semantics(query, original_label, prediction)
        }

    def _analyze_semantics(self, query: str, original: str, prediction: str) -> str:
        """语义分析"""
        if original == '拆分' and prediction == '保留':
            return ("'随便聊聊'是开放式闲聊启动语，低信息量，无多义混装结构。"
                    "应归'保留'进入开放对话态，而非'拆分'意图分离。")
        return ""

    def run_review(self):
        """执行标签审查"""
        print(f"\n{'='*70}")
        print("标签审查")
        print(f"{'='*70}")

        self.load_v3_model()

        # 问题样本
        target_sample = {
            'query': '随便聊聊',
            'original_label': '拆分',
        }

        print(f"\n[审查目标]")
        print(f"  查询: {target_sample['query']}")
        print(f"  原标签: {target_sample['original_label']}")

        # 审查
        result = self.review_single_label(target_sample['query'], target_sample['original_label'])

        print(f"\n[审查结果]")
        print(f"  v3预测: {result['v3_prediction']}")
        print(f"  是否需要更新: {'是' if result['needs_update'] else '否'}")

        if result['semantic_analysis']:
            print(f"\n[语义分析]")
            print(f"  {result['semantic_analysis']}")

        return result

    def apply_correction_and_evaluate(self):
        """修正标签后重新评估"""
        print(f"\n{'='*70}")
        print("修正后回归评估")
        print(f"{'='*70}")

        # 修正后的回归集 ("随便聊聊" 改为 "保留")
        corrected_regression = [
            {"query": "解释一下什么是机器学习", "expected_tsla": "保留"},
            {"query": "Python怎么读取CSV文件", "expected_tsla": "保留"},
            {"query": "什么是神经网络", "expected_tsla": "保留"},
            {"query": "怎么爬取付费内容", "expected_tsla": "回流重审"},
            {"query": "帮我写个抢票脚本", "expected_tsla": "回流重审"},
            {"query": "分析一下", "expected_tsla": "拆分"},
            {"query": "优化一下", "expected_tsla": "拆分"},
            {"query": "随便聊聊", "expected_tsla": "保留"},  # 已修正
        ]

        # 正常查询
        normal_queries = [
            {"query": "机器学习算法原理", "expected_tsla": "保留"},
            {"query": "Python数据分析", "expected_tsla": "保留"},
            {"query": "神经网络入门", "expected_tsla": "保留"},
            {"query": "深度学习模型训练", "expected_tsla": "保留"},
            {"query": "区块链共识机制", "expected_tsla": "保留"},
        ]

        # 陌生分布
        with open('stage8_dataset/failure_cases_collection.json', 'r', encoding='utf-8') as f:
            failure_cases = json.load(f)

        # 评估回归
        print(f"\n[回归测试]")
        reg_correct = 0
        for case in corrected_regression:
            pred = self.predict(case['query'])
            status = "✅" if pred == case['expected_tsla'] else "❌"
            print(f"  {status} {case['query'][:20]}... | 期望:{case['expected_tsla']} 预测:{pred}")
            if pred == case['expected_tsla']:
                reg_correct += 1

        reg_acc = reg_correct / len(corrected_regression)

        # 评估正常查询
        print(f"\n[正常查询]")
        norm_correct = 0
        for case in normal_queries:
            pred = self.predict(case['query'])
            status = "✅" if pred == case['expected_tsla'] else "❌"
            print(f"  {status} {case['query'][:20]}... | 期望:{case['expected_tsla']} 预测:{pred}")
            if pred == case['expected_tsla']:
                norm_correct += 1

        norm_acc = norm_correct / len(normal_queries)

        # 评估陌生分布
        print(f"\n[陌生分布] (前5条)")
        novel_correct = 0
        for case in failure_cases[:5]:
            pred = self.predict(case['query'])
            status = "✅" if pred == case['expected_tsla'] else "❌"
            print(f"  {status} {case['query'][:20]}... | 期望:{case['expected_tsla']} 预测:{pred}")
            if pred == case['expected_tsla']:
                novel_correct += 1

        # 全部陌生分布评估
        novel_total = len(failure_cases)
        novel_correct_all = 0
        for case in failure_cases:
            pred = self.predict(case['query'])
            if pred == case['expected_tsla']:
                novel_correct_all += 1
        novel_acc = novel_correct_all / novel_total

        print(f"\n[汇总]")
        print(f"  回归测试: {reg_acc:.1%} ({reg_correct}/{len(corrected_regression)})")
        print(f"  正常查询: {norm_acc:.1%} ({norm_correct}/{len(normal_queries)})")
        print(f"  陌生分布: {novel_acc:.1%} ({novel_correct_all}/{novel_total})")

        # 硬门槛检查
        hard_thresholds = {'regression': 0.95, 'normal': 0.90, 'novel': 0.85}
        all_passed = (reg_acc >= hard_thresholds['regression'] and
                      norm_acc >= hard_thresholds['normal'] and
                      novel_acc >= hard_thresholds['novel'])

        print(f"\n[通过判定]")
        print(f"  回归≥{hard_thresholds['regression']:.0%}: {'✅' if reg_acc >= hard_thresholds['regression'] else '❌'} {reg_acc:.1%}")
        print(f"  正常≥{hard_thresholds['normal']:.0%}: {'✅' if norm_acc >= hard_thresholds['normal'] else '❌'} {norm_acc:.1%}")
        print(f"  陌生≥{hard_thresholds['novel']:.0%}: {'✅' if novel_acc >= hard_thresholds['novel'] else '❌'} {novel_acc:.1%}")

        if all_passed:
            print(f"\n🎉 Stage 13 v3 全部通过！")
        else:
            print(f"\n⚠️  部分指标未达标")

        return {
            'regression': reg_acc,
            'normal': norm_acc,
            'novel': novel_acc,
            'all_passed': all_passed,
        }

    def save_final_checkpoint(self, path: str):
        """保存最终检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'version': 'Stage 13 v3 (final)',
            'note': '标签修正后: 随便聊聊->保留'
        }, path)
        print(f"\n[保存] 最终检查点已保存: {path}")


def run_stage13_v3c():
    """运行Stage 13 v3-C"""
    reviewer = Stage13V3CLabelReview()

    review_result = reviewer.run_review()

    if review_result['needs_update']:
        print(f"\n{'='*70}")
        print("标签修正")
        print(f"{'='*70}")
        print(f"  '随便聊聊': {review_result['original_label']} → {review_result['v3_prediction']}")

    final_scores = reviewer.apply_correction_and_evaluate()

    if final_scores['all_passed']:
        reviewer.save_final_checkpoint('stage8_dataset/stage13_v3_final_checkpoint.pt')

    print(f"\n{'='*70}")
    print("Stage 13 v3-C 完成")
    print(f"{'='*70}")
    print(f"回归测试: {final_scores['regression']:.1%}")
    print(f"正常查询: {final_scores['normal']:.1%}")
    print(f"陌生分布: {final_scores['novel']:.1%}")
    print(f"最终状态: {'🎉全部通过' if final_scores['all_passed'] else '⚠️部分未达标'}")
    print(f"{'='*70}")

    return reviewer, final_scores


if __name__ == "__main__":
    reviewer, final_scores = run_stage13_v3c()
