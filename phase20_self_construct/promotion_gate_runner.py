"""
Promotion Gate Runner

Stage 5B: 知识库晋升实验

目标：验证候选内容能否经 TSLA → 强审查 → 验证 → 门控分流，进入知识库体系

门控分流结果：
- 不写入
- 瞬时保留
- 长期候选
- 隔离观察
- 错误归档

关键指标：
- 晋升通过率
- 隔离率
- 回流率
- 错误归档率
- 晋升后调用稳定性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class PromotionDecision(Enum):
    """晋升决策"""
    NO_WRITE = "NO_WRITE"           # 不写入
    EPHEMERAL = "EPHEMERAL"         # 瞬时保留
    LONG_TERM_CANDIDATE = "LONG_TERM_CANDIDATE"  # 长期候选
    ISOLATION = "ISOLATION"         # 隔离观察
    ERROR_ARCHIVE = "ERROR_ARCHIVE" # 错误归档


@dataclass
class GateConfig:
    """门控配置"""
    # TSLA 审查阈值
    tsla_verify_threshold: float = 0.8
    tsla_reject_threshold: float = 0.3
    
    # 强审查阈值
    strong_review_threshold: float = 0.7
    
    # 验证阈值
    validation_threshold: float = 0.75
    
    # 门控分流阈值
    promotion_threshold: float = 0.8    # 晋升到长期候选
    ephemeral_threshold: float = 0.5    # 瞬时保留
    isolation_threshold: float = 0.3    # 隔离观察
    # < 0.3: 错误归档或不写入


class TSLAReview:
    """TSLA 八动作审查"""
    
    ACTIONS = [
        "VERIFY_SCOPE",      # 验证范围
        "CHECK_PERMISSION",  # 检查权限
        "REJECT",           # 拒绝
        "LOG_INCIDENT",     # 记录事件
        "REQUEST_CLARIFICATION",  # 请求澄清
        "ESCALATE",         # 升级
        "ISOLATE",          # 隔离
        "ARCHIVE",          # 归档
    ]
    
    def __init__(self, config: GateConfig):
        self.config = config
    
    def review(self, candidate: Dict) -> Dict:
        """
        TSLA 审查
        
        返回审查结果和置信度
        """
        candidate_type = candidate.get('candidate_type', '')
        hallucination_score = candidate.get('metadata', {}).get('hallucination_score', 1.0)
        confidence = candidate.get('generated_content', {}).get('confidence', 0.5)
        
        # 如果是单个值，转换为字典
        if isinstance(confidence, (int, float)):
            confidence = {'overall': confidence}
        
        avg_confidence = confidence.get('overall', 0.5)
        if isinstance(avg_confidence, dict):
            avg_confidence = sum(avg_confidence.values()) / len(avg_confidence)
        
        # 基于候选类型和置信度决定动作
        actions = []
        
        # 高幻觉 -> 拒绝或隔离
        if hallucination_score > 0.5:
            actions.append("REJECT")
            actions.append("LOG_INCIDENT")
        
        # 低置信度 -> 请求澄清
        if avg_confidence < 0.6:
            actions.append("REQUEST_CLARIFICATION")
        
        # 高风险类型 -> 验证范围
        if candidate_type in ['RULE', 'PATTERN']:
            actions.append("VERIFY_SCOPE")
            actions.append("CHECK_PERMISSION")
        
        # 如果没有触发任何动作，默认验证范围
        if not actions:
            actions.append("VERIFY_SCOPE")
        
        # 计算审查分数
        review_score = self._calculate_review_score(
            candidate_type, hallucination_score, avg_confidence, actions
        )
        
        return {
            'actions': actions,
            'score': review_score,
            'passed': review_score >= self.config.tsla_verify_threshold,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _calculate_review_score(self, candidate_type: str, hallucination: float, 
                                confidence: float, actions: List[str]) -> float:
        """计算审查分数"""
        # 基础分数
        score = confidence
        
        # 幻觉惩罚
        score -= hallucination * 0.5
        
        # 动作惩罚
        if "REJECT" in actions:
            score -= 0.3
        if "ISOLATE" in actions:
            score -= 0.2
        if "REQUEST_CLARIFICATION" in actions:
            score -= 0.1
        
        # 类型加成
        if candidate_type == 'EXPLANATION':
            score += 0.05  # 解释类风险较低
        elif candidate_type == 'RULE':
            score -= 0.05  # 规则类风险较高
        
        return max(0.0, min(1.0, score))


class StrongReview:
    """强审查"""
    
    def __init__(self, config: GateConfig):
        self.config = config
    
    def review(self, candidate: Dict, tsla_result: Dict) -> Dict:
        """
        强审查
        
        基于 TSLA 结果进行深度审查
        """
        # 如果 TSLA 未通过，直接不通过
        if not tsla_result['passed']:
            return {
                'passed': False,
                'score': tsla_result['score'],
                'reason': 'TSLA review failed',
                'timestamp': datetime.now().isoformat(),
            }
        
        # 结构完整性检查
        structure_valid = self._check_structure(candidate)
        
        # 内容一致性检查
        content_consistent = self._check_consistency(candidate)
        
        # 计算强审查分数
        score = tsla_result['score']
        if not structure_valid:
            score -= 0.2
        if not content_consistent:
            score -= 0.15
        
        score = max(0.0, min(1.0, score))
        
        return {
            'passed': score >= self.config.strong_review_threshold,
            'score': score,
            'structure_valid': structure_valid,
            'content_consistent': content_consistent,
            'timestamp': datetime.now().isoformat(),
        }
    
    def _check_structure(self, candidate: Dict) -> bool:
        """检查结构完整性"""
        required = ['candidate_id', 'candidate_type', 'generated_content', 'metadata']
        return all(field in candidate for field in required)
    
    def _check_consistency(self, candidate: Dict) -> bool:
        """检查内容一致性"""
        content = candidate.get('generated_content', {})
        
        # 检查是否有矛盾
        if 'gap_type' in content and 'strategy' in content:
            # 高风险缺口应该对应 DECLINE 策略
            if content['gap_type'] == 'HIGH_RISK' and content['strategy'] != 'DECLINE':
                return False
        
        return True


class Validation:
    """验证"""
    
    def __init__(self, config: GateConfig):
        self.config = config
    
    def validate(self, candidate: Dict, strong_review_result: Dict) -> Dict:
        """
        验证候选
        
        模拟外部验证（如人工审核、对照测试等）
        """
        if not strong_review_result['passed']:
            return {
                'passed': False,
                'score': strong_review_result['score'],
                'method': 'skipped',
                'timestamp': datetime.now().isoformat(),
            }
        
        # 模拟验证过程
        # 实际应用中这里可能是：
        # - 人工审核
        # - A/B 测试
        # - 对照实验
        # - 专家评估
        
        base_score = strong_review_result['score']
        
        # 模拟验证噪声
        validation_noise = random.uniform(-0.05, 0.05)
        validation_score = max(0.0, min(1.0, base_score + validation_noise))
        
        return {
            'passed': validation_score >= self.config.validation_threshold,
            'score': validation_score,
            'method': 'simulated',
            'timestamp': datetime.now().isoformat(),
        }


class PromotionGate:
    """晋升门控"""
    
    def __init__(self, config: GateConfig):
        self.config = config
        self.tsla = TSLAReview(config)
        self.strong_review = StrongReview(config)
        self.validation = Validation(config)
    
    def process(self, candidate: Dict) -> Dict:
        """
        处理候选，决定晋升路径
        
        返回完整的处理结果和决策
        """
        result = {
            'candidate_id': candidate.get('candidate_id'),
            'timestamp': datetime.now().isoformat(),
            'stages': {},
        }
        
        # Stage 1: TSLA 审查
        tsla_result = self.tsla.review(candidate)
        result['stages']['tsla'] = tsla_result
        
        if not tsla_result['passed']:
            result['decision'] = self._decide_no_write(tsla_result['score'])
            return result
        
        # Stage 2: 强审查
        strong_result = self.strong_review.review(candidate, tsla_result)
        result['stages']['strong_review'] = strong_result
        
        if not strong_result['passed']:
            result['decision'] = self._decide_isolation(strong_result['score'])
            return result
        
        # Stage 3: 验证
        validation_result = self.validation.validate(candidate, strong_result)
        result['stages']['validation'] = validation_result
        
        if not validation_result['passed']:
            result['decision'] = self._decide_ephemeral(validation_result['score'])
            return result
        
        # Stage 4: 门控分流
        final_score = validation_result['score']
        result['decision'] = self._gate_decision(final_score)
        
        return result
    
    def _decide_no_write(self, score: float) -> Dict:
        """决定不写入"""
        if score < 0.2:
            return {
                'decision': PromotionDecision.ERROR_ARCHIVE.value,
                'score': score,
                'reason': 'Too low score, archive as error',
            }
        else:
            return {
                'decision': PromotionDecision.NO_WRITE.value,
                'score': score,
                'reason': 'TSLA review failed',
            }
    
    def _decide_isolation(self, score: float) -> Dict:
        """决定隔离"""
        return {
            'decision': PromotionDecision.ISOLATION.value,
            'score': score,
            'reason': 'Strong review failed',
        }
    
    def _decide_ephemeral(self, score: float) -> Dict:
        """决定瞬时保留"""
        return {
            'decision': PromotionDecision.EPHEMERAL.value,
            'score': score,
            'reason': 'Validation failed but acceptable',
        }
    
    def _gate_decision(self, score: float) -> Dict:
        """门控分流决策"""
        if score >= self.config.promotion_threshold:
            return {
                'decision': PromotionDecision.LONG_TERM_CANDIDATE.value,
                'score': score,
                'reason': 'High quality, promote to long-term candidate',
            }
        elif score >= self.config.ephemeral_threshold:
            return {
                'decision': PromotionDecision.EPHEMERAL.value,
                'score': score,
                'reason': 'Medium quality, keep as ephemeral',
            }
        elif score >= self.config.isolation_threshold:
            return {
                'decision': PromotionDecision.ISOLATION.value,
                'score': score,
                'reason': 'Low quality, isolate for observation',
            }
        else:
            return {
                'decision': PromotionDecision.ERROR_ARCHIVE.value,
                'score': score,
                'reason': 'Too low quality, archive as error',
            }


class Stage5BExperiment:
    """Stage 5B 实验运行器"""
    
    def __init__(self, use_high_quality_batch: bool = True):
        self.config = GateConfig()
        self.gate = PromotionGate(self.config)
        self.results = []
        self.use_high_quality_batch = use_high_quality_batch
    
    def load_candidates(self, path: str = None) -> List[Dict]:
        """加载候选"""
        # 使用高质量小批次
        if self.use_high_quality_batch:
            path = "candidates/stage5b_high_quality_batch.jsonl"
            print("✓ 使用高质量小批次候选 (解释3 + 关系2 + 规则2)")
        else:
            path = path or "candidates/stage5a_candidates.jsonl"
        
        candidates = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                candidates.append(json.loads(line))
        print(f"✓ 加载 {len(candidates)} 个候选")
        return candidates
    
    def run_promotion_experiment(self, candidates: List[Dict]) -> Dict:
        """运行晋升实验"""
        print("\n" + "=" * 70)
        print("Stage 5B: 知识库晋升实验")
        print("=" * 70)
        
        decisions = {
            PromotionDecision.NO_WRITE.value: [],
            PromotionDecision.EPHEMERAL.value: [],
            PromotionDecision.LONG_TERM_CANDIDATE.value: [],
            PromotionDecision.ISOLATION.value: [],
            PromotionDecision.ERROR_ARCHIVE.value: [],
        }
        
        for i, candidate in enumerate(candidates):
            print(f"\n--- 候选 {i+1}/{len(candidates)} ---")
            print(f"  ID: {candidate.get('candidate_id', 'unknown')}")
            print(f"  类型: {candidate.get('candidate_type', 'unknown')}")
            print(f"  来源: {candidate.get('source_query', 'unknown')[:50]}...")
            
            result = self.gate.process(candidate)
            self.results.append(result)
            
            decision = result['decision']['decision']
            decisions[decision].append(result)
            
            # 打印审查流程
            print(f"\n  审查流程:")
            if 'tsla' in result['stages']:
                tsla = result['stages']['tsla']
                print(f"    TSLA: {'✓通过' if tsla['passed'] else '✗未通过'} (分数: {tsla['score']:.2f})")
                print(f"      动作: {', '.join(tsla['actions'])}")
            
            if 'strong_review' in result['stages']:
                sr = result['stages']['strong_review']
                print(f"    强审查: {'✓通过' if sr['passed'] else '✗未通过'} (分数: {sr['score']:.2f})")
            
            if 'validation' in result['stages']:
                val = result['stages']['validation']
                print(f"    验证: {'✓通过' if val['passed'] else '✗未通过'} (分数: {val['score']:.2f})")
            
            print(f"\n  最终决策: {decision}")
            print(f"  原因: {result['decision']['reason']}")
        
        # 统计
        total = len(candidates)
        stats = {
            'total': total,
            'no_write': len(decisions[PromotionDecision.NO_WRITE.value]),
            'ephemeral': len(decisions[PromotionDecision.EPHEMERAL.value]),
            'long_term': len(decisions[PromotionDecision.LONG_TERM_CANDIDATE.value]),
            'isolation': len(decisions[PromotionDecision.ISOLATION.value]),
            'error_archive': len(decisions[PromotionDecision.ERROR_ARCHIVE.value]),
        }
        
        print(f"\n晋升统计:")
        print(f"  总数: {stats['total']}")
        print(f"  不写入: {stats['no_write']} ({stats['no_write']/total:.1%})")
        print(f"  瞬时保留: {stats['ephemeral']} ({stats['ephemeral']/total:.1%})")
        print(f"  长期候选: {stats['long_term']} ({stats['long_term']/total:.1%})")
        print(f"  隔离观察: {stats['isolation']} ({stats['isolation']/total:.1%})")
        print(f"  错误归档: {stats['error_archive']} ({stats['error_archive']/total:.1%})")
        
        # 关键指标
        promotion_rate = stats['long_term'] / total if total else 0
        isolation_rate = stats['isolation'] / total if total else 0
        error_rate = stats['error_archive'] / total if total else 0
        
        print(f"\n关键指标:")
        print(f"  晋升通过率: {promotion_rate:.1%}")
        print(f"  隔离率: {isolation_rate:.1%}")
        print(f"  错误归档率: {error_rate:.1%}")
        
        return {
            'stats': stats,
            'promotion_rate': promotion_rate,
            'isolation_rate': isolation_rate,
            'error_rate': error_rate,
            'decisions': decisions,
        }
    
    def save_knowledge_base(self, path: str = "knowledge_base/"):
        """保存知识库"""
        Path(path).mkdir(exist_ok=True)
        
        # 按决策分类保存
        for result in self.results:
            decision = result['decision']['decision']
            filename = f"{path}/{decision.lower()}.jsonl"
            
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\n✓ 知识库已保存到 {path}")
    
    def run_full_experiment(self):
        """运行完整实验"""
        print("=" * 70)
        print("Stage 5B: 知识库晋升实验")
        print("=" * 70)
        
        # 加载候选
        candidates = self.load_candidates()
        
        if not candidates:
            print("⚠ 没有候选，请先运行 Stage 5A")
            return None
        
        # 运行晋升实验
        results = self.run_promotion_experiment(candidates)
        
        # 保存知识库
        self.save_knowledge_base()
        
        # 保存详细结果
        with open("eval/stage5b_results.json", 'w') as f:
            # 只保存统计信息，不保存完整的 decisions（太大）
            save_results = {
                'stats': results['stats'],
                'promotion_rate': results['promotion_rate'],
                'isolation_rate': results['isolation_rate'],
                'error_rate': results['error_rate'],
            }
            json.dump(save_results, f, indent=2)
        
        # 验收标准
        print("\n" + "=" * 70)
        print("Stage 5B 验收标准")
        print("=" * 70)
        
        checks = [
            ("晋升率 < 50% (precision 优先)", results['promotion_rate'] < 0.5),
            ("错误归档率 < 30%", results['error_rate'] < 0.3),
            ("有候选成功晋升", results['stats']['long_term'] > 0),
        ]
        
        all_passed = all(passed for _, passed in checks)
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        if all_passed:
            print("\n✓ Stage 5B 验收通过！可以进入 Stage 5C")
        else:
            print("\n✗ Stage 5B 需要继续优化")
        
        results['passed'] = all_passed
        return results


def main():
    """主函数"""
    experiment = Stage5BExperiment()
    results = experiment.run_full_experiment()
    
    if results:
        print("\n✓ 结果已保存到 eval/stage5b_results.json")


if __name__ == "__main__":
    main()
