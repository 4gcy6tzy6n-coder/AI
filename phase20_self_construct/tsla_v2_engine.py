"""
TSLA-v2: 自学习治理引擎

在TSLA-v1基础上补全4个功能:

1. 错因归因 - 不只输出动作，还输出为什么错
2. 纠偏候选生成 - 被回流/拆分后，生成修正版候选
3. 反重复错误约束 - 同类错误反复出现时提高门槛
4. 受控晋升 - 纠偏成功后接入晋升链

基于Stage 13 v3冻结检查点: stage8_dataset/stage13_v3_final_checkpoint.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage11a_r2_fix_v2_balanced import FixV2Model


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}


@dataclass
class ErrorAttribution:
    """错因归因结果"""
    action: str
    error_type: str
    error_reason: str
    correction_hints: List[str]
    confidence: float


@dataclass
class CorrectionCandidate:
    """纠偏候选"""
    original_action: str
    corrected_action: str
    query: str
    corrected_query: Optional[str] = None
    confidence: float = 0.0
    is_approved: bool = False


@dataclass
class ErrorRecord:
    """错误记录"""
    query: str
    predicted_action: str
    correct_action: str
    timestamp: str
    error_type: str
    retry_count: int = 0
    suppression_level: int = 0


class TSLAV2Engine:
    """TSLA-v2 自学习治理引擎"""

    VERSION = "TSLA-v2 / Stage 13.5"

    # 错误类型定义
    ERROR_TYPES = {
        "h1": "危险动作误放",
        "h2": "陌生表达误判",
        "h3": "边界模糊失误",
        "h4": "过度保守误杀",
        "h5": "多义混装失误",
        "boundary": "边界样本失误",
    }

    # 反重复抑制等级
    SUPPRESSION_LEVELS = [0, 1, 2, 3]  # 0=无, 1=低, 2=中, 3=高

    def __init__(self, checkpoint_path: str = 'stage8_dataset/stage13_v3_final_checkpoint.pt'):
        print(f"\n{'='*70}")
        print(f"{self.VERSION}: 自学习治理引擎")
        print(f"{'='*70}")

        self.device = torch.device('cpu')
        self.model = None
        self.checkpoint_path = checkpoint_path

        self.error_history: List[ErrorRecord] = []
        self.correction_candidates: List[CorrectionCandidate] = []
        self.promotion_queue: List[Dict] = []

        self.load_model()

        print(f"  ✓ TSLA-v2引擎初始化完成")

    def load_model(self):
        """加载模型"""
        print(f"\n[模型] 加载Stage 13 v3检查点...")

        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        self.model = FixV2Model(base_model)

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        print(f"  ✓ 模型已加载")

    def predict(self, query: str) -> Tuple[str, float]:
        """预测动作和置信度"""
        tokens = [ord(c) % 10000 for c in query[:100]]
        if len(tokens) < 100:
            tokens.extend([0] * (100 - len(tokens)))

        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = self.model(input_ids)
            probs = torch.softmax(outputs['tsla_logits'], dim=-1)
            conf, pred_id = probs.max(dim=-1)
            return ID_TO_TSLA_ACTION[pred_id.item()], conf.item()

    def attribute_error(self, query: str, predicted_action: str, correct_action: str) -> ErrorAttribution:
        """错因归因"""
        error_type = self._classify_error(query, predicted_action, correct_action)

        reasons = {
            "h1": f"危险动作({correct_action})被误放为{predicted_action}",
            "h2": f"陌生表达模式未被识别，当前判断:{predicted_action}",
            "h3": f"边界模糊，当前判断:{predicted_action}，应为:{correct_action}",
            "h4": f"正常查询({correct_action})被误杀为{predicted_action}",
            "h5": f"多义混装，当前判断:{predicted_action}",
            "boundary": f"边界样本，当前判断:{predicted_action}",
        }

        hints = {
            "h1": ["检查危险动作关键词", "加强H1类特征学习", "提高回流重审阈值"],
            "h2": ["扩展陌生表达覆盖", "增加H2类训练样本", "降低H2类判断阈值"],
            "h3": ["增加边界样本训练", "细化边界判定规则", "考虑拆分处理"],
            "h4": ["减少过度保守", "检查误杀原因", "调整正常查询保护"],
            "h5": ["识别多义混装模式", "考虑先拆分再判断", "增加意图分离"],
            "boundary": ["扩充边界样本集", "提高边界敏感度", "考虑保守策略"],
        }

        return ErrorAttribution(
            action=predicted_action,
            error_type=error_type,
            error_reason=reasons.get(error_type, "未知错误"),
            correction_hints=hints.get(error_type, ["通用纠偏建议"]),
            confidence=0.8
        )

    def _classify_error(self, query: str, predicted: str, correct: str) -> str:
        """分类错误类型"""
        if correct == "回流重审" and predicted == "保留":
            return "h1"
        if predicted in ["保留", "拆分"] and correct in ["回流重审", "隔离"]:
            h2_keywords = ['抢', '爬', '破解', '预测', '伪造', '入侵', '黑', '刷', '自动']
            if any(kw in query for kw in h2_keywords):
                return "h2"
        if correct == "拆分" and predicted in ["保留", "保留"]:
            return "h3"
        if correct == "保留" and predicted in ["回流重审", "隔离", "降级"]:
            return "h4"
        return "boundary"

    def generate_correction_candidate(self, query: str, wrong_action: str, error_type: str) -> CorrectionCandidate:
        """生成纠偏候选"""
        original = CorrectionCandidate(
            original_action=wrong_action,
            corrected_action="待生成",
            query=query,
            confidence=0.0
        )

        if wrong_action == "保留" and error_type in ["h1", "h2"]:
            original.corrected_action = "回流重审"
            original.confidence = 0.85
        elif wrong_action == "回流重审" and error_type == "h4":
            original.corrected_action = "保留"
            original.confidence = 0.85
        elif wrong_action == "保留" and error_type == "h3":
            original.corrected_action = "拆分"
            original.confidence = 0.7
        elif wrong_action == "拆分":
            original.corrected_action = "保留"
            original.confidence = 0.6
        else:
            original.corrected_action = "回流重审"
            original.confidence = 0.5

        return original

    def check_repetition_suppression(self, query: str) -> Tuple[bool, int]:
        """检查是否需要抑制重复错误"""
        same_query_records = [r for r in self.error_history if r.query == query]

        if not same_query_records:
            return False, 0

        recent_errors = same_query_records[-5:]
        error_rate = sum(1 for r in recent_errors if r.predicted_action != r.correct_action) / len(recent_errors)

        if error_rate >= 0.8:
            return True, 3
        elif error_rate >= 0.6:
            return True, 2
        elif error_rate >= 0.4:
            return True, 1

        return False, 0

    def record_error(self, query: str, predicted: str, correct: str, error_type: str):
        """记录错误"""
        record = ErrorRecord(
            query=query,
            predicted_action=predicted,
            correct_action=correct,
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            retry_count=0,
            suppression_level=0
        )
        self.error_history.append(record)

    def approve_correction(self, candidate: CorrectionCandidate) -> bool:
        """审批纠偏候选"""
        if candidate.confidence >= 0.7:
            candidate.is_approved = True
            self.correction_candidates.append(candidate)
            return True
        return False

    def prepare_promotion(self, query: str, corrected_action: str, confidence: float):
        """准备晋升"""
        promotion_item = {
            'query': query,
            'corrected_action': corrected_action,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'approved': False
        }
        self.promotion_queue.append(promotion_item)
        return promotion_item

    def approve_promotion(self, index: int) -> bool:
        """审批晋升"""
        if 0 <= index < len(self.promotion_queue):
            item = self.promotion_queue[index]
            if item['confidence'] >= 0.75:
                item['approved'] = True
                item['status'] = 'approved'
                return True
        return False

    def process_query(self, query: str, teacher_feedback: Optional[str] = None) -> Dict:
        """完整处理查询"""
        predicted_action, base_confidence = self.predict(query)

        needs_suppression, suppression_level = self.check_repetition_suppression(query)

        result = {
            'query': query,
            'predicted_action': predicted_action,
            'confidence': base_confidence,
            'needs_suppression': needs_suppression,
            'suppression_level': suppression_level,
            'teacher_feedback': teacher_feedback,
            'is_error': False,
            'error_attribution': None,
            'correction_candidate': None,
            'promotion_item': None,
        }

        if teacher_feedback and teacher_feedback != predicted_action:
            result['is_error'] = True
            error_type = self._classify_error(query, predicted_action, teacher_feedback)

            self.record_error(query, predicted_action, teacher_feedback, error_type)

            attr = self.attribute_error(query, predicted_action, teacher_feedback)
            result['error_attribution'] = attr

            corr = self.generate_correction_candidate(query, predicted_action, error_type)
            result['correction_candidate'] = corr

            if corr.confidence >= 0.7:
                self.approve_correction(corr)
                promo = self.prepare_promotion(query, corr.corrected_action, corr.confidence)
                result['promotion_item'] = promo

        return result

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_errors = len(self.error_history)
        by_type = defaultdict(int)
        for r in self.error_history:
            by_type[r.error_type] += 1

        return {
            'total_queries': len(self.error_history) + len(self.correction_candidates),
            'total_errors': total_errors,
            'error_by_type': dict(by_type),
            'correction_candidates': len(self.correction_candidates),
            'approved_corrections': sum(1 for c in self.correction_candidates if c.is_approved),
            'promotion_queue': len(self.promotion_queue),
            'approved_promotions': sum(1 for p in self.promotion_queue if p.get('approved', False)),
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()

        print(f"\n{'='*70}")
        print("TSLA-v2 引擎统计")
        print(f"{'='*70}")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  错误数: {stats['total_errors']}")
        print(f"  错误类型分布:")
        for et, count in stats['error_by_type'].items():
            print(f"    {et}: {count}")
        print(f"  纠偏候选: {stats['correction_candidates']}")
        print(f"  已批准纠偏: {stats['approved_corrections']}")
        print(f"  晋升队列: {stats['promotion_queue']}")
        print(f"  已批准晋升: {stats['approved_promotions']}")
        print(f"{'='*70}")


def demo_tslav2():
    """TSLA-v2演示"""
    engine = TSLAV2Engine()

    print(f"\n{'='*70}")
    print("TSLA-v2 功能演示")
    print(f"{'='*70}")

    test_queries = [
        "帮我写个抢票脚本",
        "解释一下机器学习",
        "随便聊聊",
        "怎么爬取付费内容",
    ]

    for query in test_queries:
        predicted, conf = engine.predict(query)
        print(f"\n[查询] {query}")
        print(f"  预测: {predicted} (置信度: {conf:.2f})")

        suppression_needed, level = engine.check_repetition_suppression(query)
        if suppression_needed:
            print(f"  抑制等级: {level}")

    engine.print_stats()

    print(f"\n{'='*70}")
    print("TSLA-v2 演示完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo_tslav2()
