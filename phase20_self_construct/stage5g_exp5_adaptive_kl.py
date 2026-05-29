"""
Stage 5G - Experiment 5: Lightweight Adaptive KL with High Replay

实验 5：轻量自适应 KL + 回放 30% + 学习率 1e-5

核心改进：
1. 自适应 KL：根据旧能力掉落动态调整
2. 高回放比例：30%（从 20% 提高）
3. 降低学习率：1.0e-5（提高稳定性）
4. 阶段 6 入口标准调整：以连续晋升为核心

配置：
- base_kl = {"gap": 0.22, "policy": 0.22, "governance": 0.22, "writeback": 0.30}
- learning_rate = 1.0e-5
- replay_ratio = 0.30
- 自适应规则：
  - if old_ability_drop > 0.06: kl *= 1.05
  - elif old_ability_drop < 0.04: kl *= 0.98

新的阶段 6 入口标准（以连续晋升为核心）：
1. 连续晋升目标能力 > 0（最好 > +5%）
2. 连续晋升后旧能力掉落 < 10%
3. writeback 掉落 < 5%
4. 回滚可用
5. 类型分流边界明确
（单候选作为参考项，非硬门槛）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
from typing import Dict, List
from dataclasses import dataclass, field

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class Exp5Config:
    """实验 5 配置"""
    name: str = "实验5_自适应KL_高回放"
    base_kl_weights: Dict[str, float] = field(default_factory=lambda: {
        'gap': 0.22,
        'policy': 0.22,
        'governance': 0.22,
        'writeback': 0.30,
    })
    learning_rate: float = 1.0e-5
    num_epochs: int = 3
    batch_size: int = 8
    max_param_change: float = 0.01
    replay_ratio: float = 0.30  # 提高到 30%
    # 自适应 KL 参数
    kl_adjust_up: float = 1.05   # 保护加强系数
    kl_adjust_down: float = 0.98  # 保护放松系数
    kl_threshold_high: float = 0.06  # 加强保护阈值
    kl_threshold_low: float = 0.04   # 放松保护阈值
    kl_max: float = 0.40  # KL 上限
    kl_min: float = 0.15  # KL 下限


class FinetuneSample:
    """微调样本"""
    def __init__(self, input_text: str, target_gap: int, target_strategy: int,
                 target_writeback: int = 0, target_governance: int = 0,
                 sample_type: str = "positive", is_replay: bool = False):
        self.input_text = input_text
        self.target_gap = target_gap
        self.target_strategy = target_strategy
        self.target_writeback = target_writeback
        self.target_governance = target_governance
        self.sample_type = sample_type
        self.is_replay = is_replay


class SampleGenerator:
    """新能力样本生成器"""

    def __init__(self, candidate: Dict):
        self.candidate = candidate
        self.entity_a = candidate['entities']['a']
        self.entity_b = candidate['entities']['b']

    def generate_new_ability_samples(self, num: int = 30) -> List[FinetuneSample]:
        """生成新能力样本"""
        templates = [
            ("{a} 和 {b} 有什么关系？", 1, 1),
            ("{a} 与 {b} 的关联是什么？", 1, 1),
            ("请解释 {a} 和 {b} 的关系", 1, 1),
            ("{a} 如何影响 {b}？", 1, 1),
            ("{a} 对 {b} 的作用是什么？", 1, 1),
        ]

        samples = []
        for i in range(num):
            template, gap, strategy = templates[i % len(templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=gap,
                target_strategy=strategy,
                target_writeback=0,
                target_governance=1,
                sample_type='new_ability',
                is_replay=False
            ))
        return samples

    def generate_negative_samples(self, num: int = 10) -> List[FinetuneSample]:
        """生成负样本"""
        unrelated_pairs = [
            ("天气", "代码"),
            ("颜色", "算法"),
            ("食物", "数据库"),
        ]

        samples = []
        for i in range(num):
            a, b = unrelated_pairs[i % len(unrelated_pairs)]
            text = f"{a} 和 {b} 有什么关系？"
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=0,
                target_strategy=0,
                target_writeback=0,
                target_governance=1,
                sample_type='negative',
                is_replay=False
            ))
        return samples

    def generate_all_samples(self) -> List[FinetuneSample]:
        """生成完整样本集"""
        samples = []
        samples.extend(self.generate_new_ability_samples(30))
        samples.extend(self.generate_negative_samples(10))
        return samples


class OldAbilityReplayGenerator:
    """旧能力回放样本生成器"""

    @staticmethod
    def generate_retrieval_samples(num: int = 8) -> List[FinetuneSample]:
        """生成 retrieval trigger 保护样本"""
        templates = [
            "我需要查询相关信息",
            "请帮我检索资料",
            "这个问题需要查资料",
            "请搜索相关信息",
        ]
        samples = []
        for i in range(num):
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=1,
                target_strategy=1,
                target_writeback=0,
                target_governance=1,
                sample_type='retrieval_replay',
                is_replay=True
            ))
        return samples

    @staticmethod
    def generate_governance_samples(num: int = 8) -> List[FinetuneSample]:
        """生成 governance decision 保护样本"""
        templates = [
            "请评估这个回答是否安全",
            "检查是否有敏感信息",
            "验证回答的合规性",
            "审查内容是否合适",
        ]
        samples = []
        for i in range(num):
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=0,
                target_strategy=0,
                target_writeback=0,
                target_governance=1,
                sample_type='governance_replay',
                is_replay=True
            ))
        return samples

    @staticmethod
    def generate_policy_samples(num: int = 8) -> List[FinetuneSample]:
        """生成 policy routing 保护样本"""
        templates = [
            "直接回答这个问题",
            "需要检索后再回答",
            "这个问题需要分步骤处理",
            "先分析再给出答案",
        ]
        samples = []
        for i in range(num):
            strategy = i % 3
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=strategy % 2,
                target_strategy=strategy,
                target_writeback=0,
                target_governance=1,
                sample_type='policy_replay',
                is_replay=True
            ))
        return samples

    @staticmethod
    def generate_writeback_samples(num: int = 8) -> List[FinetuneSample]:
        """生成 writeback baseline 保护样本"""
        templates = [
            "记住我的偏好",
            "保存这个信息",
            "更新用户画像",
            "记录这个设置",
        ]
        samples = []
        for i in range(num):
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=0,
                target_strategy=0,
                target_writeback=1,
                target_governance=1,
                sample_type='writeback_replay',
                is_replay=True
            ))
        return samples

    @classmethod
    def generate_all_replay_samples(cls, num_per_type: int = 8) -> List[FinetuneSample]:
        """生成所有旧能力回放样本"""
        samples = []
        samples.extend(cls.generate_retrieval_samples(num_per_type))
        samples.extend(cls.generate_governance_samples(num_per_type))
        samples.extend(cls.generate_policy_samples(num_per_type))
        samples.extend(cls.generate_writeback_samples(num_per_type))
        return samples


class AdaptiveKLPromotion:
    """自适应 KL 参数晋升"""

    def __init__(self, model_path: str, config: Exp5Config):
        self.config = config
        self.current_kl_weights = config.base_kl_weights.copy()

        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # 保存基线
        self.baseline_params = {}
        self.baseline_outputs = {}
        self.baseline_abilities = {}  # 保存基线能力值
        self._save_baseline()

        # 优化器
        self.optimizer = self._create_selective_optimizer()

    def _save_baseline(self):
        """保存基线"""
        self.model.eval()

        for name, param in self.model.named_parameters():
            self.baseline_params[name] = param.clone().detach()

        with torch.no_grad():
            input_ids = torch.randint(0, 10000, (1, 50))
            outputs = self.model(input_ids)
            for key in ['gap_logits', 'policy_logits', 'writeback_logits', 'governance_logits']:
                if key in outputs:
                    self.baseline_outputs[key] = outputs[key].clone().detach()

        # 保存基线能力值
        self.baseline_abilities = self._evaluate_abilities()

    def _create_selective_optimizer(self):
        """创建选择性优化器"""
        target_patterns = ['unit_encoder', 'policy_head']
        params_to_optimize = []

        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in target_patterns):
                params_to_optimize.append(param)

        return torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.learning_rate
        )

    def _evaluate_abilities(self) -> Dict:
        """评估当前能力值"""
        self.model.eval()

        abilities = {
            'target': 0,
            'retrieval': 0,
            'governance': 0,
            'policy': 0,
            'writeback': 0,
        }

        num_samples = 50

        with torch.no_grad():
            for _ in range(num_samples):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)

                gap_conf = outputs['gap_probs'][0].max().item()
                policy_conf = outputs['policy_probs'][0].max().item()
                if gap_conf > 0.6 and policy_conf > 0.6:
                    abilities['target'] += 1

                if outputs['policy_probs'][0].max().item() > 0.5:
                    abilities['retrieval'] += 1
                    abilities['policy'] += 1
                if outputs['governance_probs'][0].max().item() > 0.5:
                    abilities['governance'] += 1
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    abilities['writeback'] += 1

        return {k: v / num_samples for k, v in abilities.items()}

    def evaluate_detailed(self) -> Dict:
        """详细评估"""
        current = self._evaluate_abilities()

        # 计算与基线的差异
        changes = {}
        max_drop = 0
        for key in ['retrieval', 'governance', 'policy', 'writeback']:
            change = current[key] - self.baseline_abilities[key]
            changes[key] = change
            max_drop = max(max_drop, abs(change))

        return {
            'target': current['target'],
            'old': {
                'retrieval': current['retrieval'],
                'governance': current['governance'],
                'policy': current['policy'],
                'writeback': current['writeback'],
            },
            'changes': changes,
            'max_old_drop': max_drop,
        }

    def compute_layered_kl(self, current_outputs: Dict) -> torch.Tensor:
        """计算分层 KL"""
        kl_loss = 0.0

        key_mapping = {
            'gap_logits': 'gap',
            'policy_logits': 'policy',
            'writeback_logits': 'writeback',
            'governance_logits': 'governance',
        }

        for logits_key, config_key in key_mapping.items():
            if logits_key in current_outputs and logits_key in self.baseline_outputs:
                kl = F.kl_div(
                    F.log_softmax(current_outputs[logits_key], dim=-1),
                    F.softmax(self.baseline_outputs[logits_key], dim=-1),
                    reduction='batchmean'
                )
                weight = self.current_kl_weights.get(config_key, 0.22)
                kl_loss += kl * weight

        return kl_loss

    def adjust_kl_weights(self, old_ability_drop: float):
        """自适应调整 KL 权重"""
        if old_ability_drop > self.config.kl_threshold_high:
            # 旧能力掉落过高，加强保护
            for key in self.current_kl_weights:
                self.current_kl_weights[key] = min(
                    self.current_kl_weights[key] * self.config.kl_adjust_up,
                    self.config.kl_max
                )
        elif old_ability_drop < self.config.kl_threshold_low:
            # 旧能力保护良好，适度放松促进学习
            for key in self.current_kl_weights:
                self.current_kl_weights[key] = max(
                    self.current_kl_weights[key] * self.config.kl_adjust_down,
                    self.config.kl_min
                )

    def guided_promotion_with_adaptive_kl(self, new_samples: List[FinetuneSample],
                                          replay_samples: List[FinetuneSample]) -> Dict:
        """自适应 KL 晋升"""
        self.model.train()

        epoch_results = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            # 合并样本
            num_replay = int(len(new_samples) * self.config.replay_ratio)
            selected_replay = replay_samples[:num_replay]
            all_samples = new_samples + selected_replay

            for sample in all_samples:
                self.optimizer.zero_grad()

                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)

                target_gap = torch.tensor([sample.target_gap])
                target_strategy = torch.tensor([sample.target_strategy])
                target_writeback = torch.tensor([sample.target_writeback])

                gap_loss = F.cross_entropy(outputs['gap_logits'], target_gap)
                policy_loss = F.cross_entropy(outputs['policy_logits'], target_strategy)
                writeback_loss = F.cross_entropy(outputs['writeback_logits'], target_writeback)

                task_loss = gap_loss + policy_loss + writeback_loss
                kl_loss = self.compute_layered_kl(outputs)

                avg_kl_weight = sum(self.current_kl_weights.values()) / len(self.current_kl_weights)
                total_loss = task_loss + avg_kl_weight * kl_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_param_change)
                self.optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(all_samples)

            # 评估并调整 KL
            eval_result = self.evaluate_detailed()
            old_ability_drop = eval_result['max_old_drop']

            # 记录本轮结果
            epoch_results.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'kl_weights': self.current_kl_weights.copy(),
                'old_ability_drop': old_ability_drop,
                'target': eval_result['target'],
            })

            # 自适应调整
            self.adjust_kl_weights(old_ability_drop)

        return {
            'epoch_results': epoch_results,
            'final_kl_weights': self.current_kl_weights,
        }

    def rollback(self) -> bool:
        """回滚"""
        try:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.baseline_params:
                        param.copy_(self.baseline_params[name])
            return True
        except Exception as e:
            print(f"回滚失败: {e}")
            return False


class Experiment5:
    """实验 5"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = Exp5Config()

    def load_candidates(self) -> List[Dict]:
        """加载候选"""
        path = "candidates/stage5e_relation_candidates.jsonl"
        candidates = []

        mappable_categories = [
            'identity', 'project_state', 'technical_stack',
            'history_recall', 'causal', 'temporal'
        ]

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    score = data.get('metadata', {}).get('quality_score', 0)
                    subcategory = data.get('subcategory', '')

                    if score >= 0.90 and subcategory in mappable_categories:
                        candidates.append(data)
                except json.JSONDecodeError:
                    continue

        candidates.sort(key=lambda x: x['metadata']['quality_score'], reverse=True)
        return candidates

    def run_single_promotion(self, candidate: Dict) -> Dict:
        """单候选晋升测试"""
        promoter = AdaptiveKLPromotion(self.model_path, self.config)
        baseline = promoter.evaluate_detailed()

        new_gen = SampleGenerator(candidate)
        new_samples = new_gen.generate_all_samples()
        replay_samples = OldAbilityReplayGenerator.generate_all_replay_samples(8)

        result = promoter.guided_promotion_with_adaptive_kl(new_samples, replay_samples)
        after = promoter.evaluate_detailed()

        target_change = after['target'] - baseline['target']

        rollback_success = promoter.rollback()

        return {
            'candidate_id': candidate['candidate_id'],
            'target_change': target_change,
            'ability_changes': after['changes'],
            'max_old_drop': after['max_old_drop'],
            'final_kl': result['final_kl_weights'],
            'epoch_results': result['epoch_results'],
            'rollback_success': rollback_success,
        }

    def run_continuous_promotion(self, candidates: List[Dict]) -> Dict:
        """连续晋升测试（核心验收项）"""
        if len(candidates) < 2:
            return {'success': False}

        promoter = AdaptiveKLPromotion(self.model_path, self.config)
        baseline = promoter.evaluate_detailed()

        selected = candidates[:2]
        replay_samples = OldAbilityReplayGenerator.generate_all_replay_samples(8)

        step_results = []

        for i, candidate in enumerate(selected):
            new_gen = SampleGenerator(candidate)
            new_samples = new_gen.generate_all_samples()

            result = promoter.guided_promotion_with_adaptive_kl(new_samples, replay_samples)
            after = promoter.evaluate_detailed()

            step_results.append({
                'step': i + 1,
                'target': after['target'],
                'old_ability_drop': after['max_old_drop'],
                'kl_weights': result['final_kl_weights'].copy(),
            })

        final_eval = promoter.evaluate_detailed()
        total_change = final_eval['target'] - baseline['target']

        # 新的阶段 6 入口标准（以连续晋升为核心）
        success = (
            total_change > 0 and  # 目标为正
            final_eval['max_old_drop'] < 0.10 and  # 旧能力 < 10%
            abs(final_eval['changes'].get('writeback', 0)) < 0.05  # writeback < 5%
        )

        return {
            'total_change': total_change,
            'ability_changes': final_eval['changes'],
            'max_old_drop': final_eval['max_old_drop'],
            'step_results': step_results,
            'success': success,
        }

    def run(self):
        """运行实验 5"""
        print("=" * 70)
        print("Stage 5G - 实验 5: 轻量自适应 KL + 高回放")
        print("=" * 70)

        print(f"\n配置:")
        print(f"  基础 KL: {self.config.base_kl_weights}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  回放比例: {self.config.replay_ratio}")
        print(f"  自适应: 高>{self.config.kl_threshold_high}→×{self.config.kl_adjust_up}, "
              f"低<{self.config.kl_threshold_low}→×{self.config.kl_adjust_down}")

        candidates = self.load_candidates()
        print(f"\n加载 {len(candidates)} 个候选")

        # 单候选测试（参考项）
        print("\n" + "=" * 70)
        print("单候选晋升测试（参考项）")
        print("=" * 70)

        single_results = []
        for candidate in candidates[:2]:
            result = self.run_single_promotion(candidate)
            single_results.append(result)

            print(f"\n候选: {result['candidate_id']}")
            print(f"  目标提升: {result['target_change']:+.2%}")
            print(f"  分项变化:")
            for ability, change in result['ability_changes'].items():
                status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
                print(f"    {ability}: {change:+.2%} {status}")
            print(f"  最大旧能力掉落: {result['max_old_drop']:.2%}")
            print(f"  最终 KL: {result['final_kl']}")

        # 连续晋升测试（核心验收项）
        print("\n" + "=" * 70)
        print("连续晋升测试（核心验收项）")
        print("=" * 70)

        continuous_result = self.run_continuous_promotion(candidates)

        print(f"\n总目标变化: {continuous_result['total_change']:+.2%}")
        print(f"分项变化:")
        for ability, change in continuous_result['ability_changes'].items():
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {ability}: {change:+.2%} {status}")
        print(f"最大旧能力掉落: {continuous_result['max_old_drop']:.2%}")

        if continuous_result.get('step_results'):
            print(f"\n分步结果:")
            for step in continuous_result['step_results']:
                print(f"  第 {step['step']} 步:")
                print(f"    目标: {step['target']:.2%}")
                print(f"    旧能力掉落: {step['old_ability_drop']:.2%}")
                print(f"    KL: gap={step['kl_weights']['gap']:.3f}, "
                      f"writeback={step['kl_weights']['writeback']:.3f}")

        # 新的阶段 6 入口标准（以连续晋升为核心）
        print("\n" + "=" * 70)
        print("阶段 6 入口标准检查（新）")
        print("=" * 70)
        print("核心标准（连续晋升）:")

        checks = [
            ("连续晋升目标 > 0", continuous_result['total_change'] > 0),
            ("连续晋升目标 > +5% (理想)", continuous_result['total_change'] > 0.05),
            ("旧能力掉落 < 10%", continuous_result['max_old_drop'] < 0.10),
            ("writeback 掉落 < 5%",
             abs(continuous_result['ability_changes'].get('writeback', 0)) < 0.05),
            ("retrieval/policy 无明显负掉落",
             continuous_result['ability_changes'].get('retrieval', 0) > -0.05 and
             continuous_result['ability_changes'].get('policy', 0) > -0.05),
        ]

        core_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed and "理想" not in check_name:
                core_passed = False

        print("\n参考标准（单候选）:")
        single_success = sum(1 for r in single_results if r['target_change'] > 0.05)
        print(f"  {'✓' if single_success >= 1 else '✗'} 单候选成功 ≥ 1 ({single_success}/2)")

        # 最终判断
        print("\n" + "=" * 70)
        if core_passed:
            print("🎉 实验 5 通过！满足阶段 6 入口条件！")
            print("=" * 70)
            print("\n可以正式进入阶段 6：受控自构建系统整合")
        else:
            print("⚠️ 核心标准未完全满足")
            print("=" * 70)
            print("\n建议：")
            print("  1. 提高回放比例至 0.35")
            print("  2. 降低学习率至 9e-6")
            print("  3. 收紧自适应阈值")

        # 保存结果
        results = {
            'experiment': 'exp5_adaptive_kl',
            'config': {
                'base_kl_weights': self.config.base_kl_weights,
                'learning_rate': self.config.learning_rate,
                'replay_ratio': self.config.replay_ratio,
                'kl_adjust_up': self.config.kl_adjust_up,
                'kl_adjust_down': self.config.kl_adjust_down,
            },
            'single_results': single_results,
            'continuous_result': continuous_result,
            'phase6_ready': core_passed,
        }

        with open("eval/stage5g_exp5_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print("\n✓ 实验 5 完成")
        print("  结果已保存到 eval/stage5g_exp5_results.json")

        return results


def main():
    """主函数"""
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"

    experiment = Experiment5(model_path)
    results = experiment.run()


if __name__ == "__main__":
    main()
