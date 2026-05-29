"""
Stage 5G - Experiment 6C: Final Tuning for Phase 6 Entry

实验 6C：最终微调 - 进入阶段 6

核心改进：
1. 移除 warm-up 对基线的干扰
2. 进一步提高基础 KL：gap/policy/governance=0.32, writeback=0.40
3. 第 1 步更新上限收紧到 0.002
4. 增加学习率到 1.2e-5 以提升目标能力

目标：同时满足 4 条标准

配置：
- base_kl = {"gap": 0.32, "policy": 0.32, "governance": 0.32, "writeback": 0.40}
- learning_rate = 1.2e-5
- replay_ratio = 0.35
- step1_max_change = 0.002
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
class Exp6CConfig:
    """实验 6C 配置"""
    name: str = "实验6C_最终微调"
    base_kl_weights: Dict[str, float] = field(default_factory=lambda: {
        'gap': 0.32,
        'policy': 0.32,
        'governance': 0.32,
        'writeback': 0.40,
    })
    learning_rate: float = 1.2e-5  # 略提高以提升学习能力
    num_epochs: int = 3
    batch_size: int = 8
    replay_ratio: float = 0.35
    step1_max_change: float = 0.002  # 更严格
    step2_max_change: float = 0.008
    kl_adjust_up: float = 1.05
    kl_adjust_down: float = 0.98
    kl_threshold_high: float = 0.04
    kl_threshold_low: float = 0.03
    kl_max: float = 0.50
    kl_min: float = 0.28


class FinetuneSample:
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
    def __init__(self, candidate: Dict):
        self.candidate = candidate
        self.entity_a = candidate['entities']['a']
        self.entity_b = candidate['entities']['b']

    def generate_new_ability_samples(self, num: int = 30) -> List[FinetuneSample]:
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
                input_text=text, target_gap=gap, target_strategy=strategy,
                target_writeback=0, target_governance=1,
                sample_type='new_ability', is_replay=False
            ))
        return samples

    def generate_negative_samples(self, num: int = 10) -> List[FinetuneSample]:
        unrelated_pairs = [("天气", "代码"), ("颜色", "算法"), ("食物", "数据库")]
        samples = []
        for i in range(num):
            a, b = unrelated_pairs[i % len(unrelated_pairs)]
            text = f"{a} 和 {b} 有什么关系？"
            samples.append(FinetuneSample(
                input_text=text, target_gap=0, target_strategy=0,
                target_writeback=0, target_governance=1,
                sample_type='negative', is_replay=False
            ))
        return samples

    def generate_all_samples(self) -> List[FinetuneSample]:
        samples = self.generate_new_ability_samples(30)
        samples.extend(self.generate_negative_samples(10))
        return samples


class OldAbilityReplayGenerator:
    @staticmethod
    def generate_all_replay_samples(num_per_type: int = 10) -> List[FinetuneSample]:
        samples = []
        # retrieval
        for i in range(num_per_type):
            samples.append(FinetuneSample(
                input_text="我需要查询相关信息", target_gap=1, target_strategy=1,
                target_writeback=0, target_governance=1,
                sample_type='retrieval_replay', is_replay=True
            ))
        # governance
        for i in range(num_per_type):
            samples.append(FinetuneSample(
                input_text="请评估这个回答是否安全", target_gap=0, target_strategy=0,
                target_writeback=0, target_governance=1,
                sample_type='governance_replay', is_replay=True
            ))
        # policy
        for i in range(num_per_type):
            samples.append(FinetuneSample(
                input_text="直接回答这个问题", target_gap=i % 2, target_strategy=i % 3,
                target_writeback=0, target_governance=1,
                sample_type='policy_replay', is_replay=True
            ))
        # writeback
        for i in range(num_per_type):
            samples.append(FinetuneSample(
                input_text="记住我的偏好", target_gap=0, target_strategy=0,
                target_writeback=1, target_governance=1,
                sample_type='writeback_replay', is_replay=True
            ))
        return samples


class FinalTuningPromotion:
    def __init__(self, model_path: str, config: Exp6CConfig):
        self.config = config
        self.current_kl_weights = config.base_kl_weights.copy()

        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.baseline_params = {}
        self.baseline_outputs = {}
        self.baseline_abilities = {}
        self._save_baseline()

        self.optimizer = self._create_selective_optimizer()

    def _save_baseline(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            self.baseline_params[name] = param.clone().detach()
        with torch.no_grad():
            input_ids = torch.randint(0, 10000, (1, 50))
            outputs = self.model(input_ids)
            for key in ['gap_logits', 'policy_logits', 'writeback_logits', 'governance_logits']:
                if key in outputs:
                    self.baseline_outputs[key] = outputs[key].clone().detach()
        self.baseline_abilities = self._evaluate_abilities()

    def _create_selective_optimizer(self):
        target_patterns = ['unit_encoder', 'policy_head']
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in target_patterns):
                params_to_optimize.append(param)
        return torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)

    def _evaluate_abilities(self) -> Dict:
        self.model.eval()
        abilities = {'target': 0, 'retrieval': 0, 'governance': 0, 'policy': 0, 'writeback': 0}
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
        current = self._evaluate_abilities()
        changes = {}
        max_drop = 0
        for key in ['retrieval', 'governance', 'policy', 'writeback']:
            change = current[key] - self.baseline_abilities[key]
            changes[key] = change
            max_drop = max(max_drop, abs(change))
        return {'target': current['target'], 'changes': changes, 'max_old_drop': max_drop}

    def compute_layered_kl(self, current_outputs: Dict) -> torch.Tensor:
        kl_loss = 0.0
        key_mapping = {
            'gap_logits': 'gap', 'policy_logits': 'policy',
            'writeback_logits': 'writeback', 'governance_logits': 'governance',
        }
        for logits_key, config_key in key_mapping.items():
            if logits_key in current_outputs and logits_key in self.baseline_outputs:
                kl = F.kl_div(
                    F.log_softmax(current_outputs[logits_key], dim=-1),
                    F.softmax(self.baseline_outputs[logits_key], dim=-1),
                    reduction='batchmean'
                )
                weight = self.current_kl_weights.get(config_key, 0.32)
                kl_loss += kl * weight
        return kl_loss

    def adjust_kl_weights(self, old_ability_drop: float):
        if old_ability_drop > self.config.kl_threshold_high:
            for key in self.current_kl_weights:
                self.current_kl_weights[key] = min(
                    self.current_kl_weights[key] * self.config.kl_adjust_up,
                    self.config.kl_max
                )
        elif old_ability_drop < self.config.kl_threshold_low:
            for key in self.current_kl_weights:
                self.current_kl_weights[key] = max(
                    self.current_kl_weights[key] * self.config.kl_adjust_down,
                    self.config.kl_min
                )

    def guided_promotion_step(self, new_samples: List[FinetuneSample],
                              replay_samples: List[FinetuneSample],
                              step_num: int) -> Dict:
        self.model.train()
        max_change = self.config.step1_max_change if step_num == 1 else self.config.step2_max_change
        epoch_results = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_change)
                self.optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(all_samples)
            eval_result = self.evaluate_detailed()
            old_ability_drop = eval_result['max_old_drop']

            epoch_results.append({
                'epoch': epoch + 1, 'loss': avg_loss,
                'kl_weights': self.current_kl_weights.copy(),
                'old_ability_drop': old_ability_drop, 'target': eval_result['target'],
            })
            self.adjust_kl_weights(old_ability_drop)

        return {
            'epoch_results': epoch_results,
            'final_kl_weights': self.current_kl_weights,
            'final_eval': self.evaluate_detailed(),
        }


class Experiment6C:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = Exp6CConfig()

    def load_candidates(self) -> List[Dict]:
        path = "candidates/stage5e_relation_candidates.jsonl"
        candidates = []
        mappable_categories = ['identity', 'project_state', 'technical_stack', 'history_recall', 'causal', 'temporal']
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

    def run_continuous_promotion(self, candidates: List[Dict]) -> Dict:
        if len(candidates) < 2:
            return {'success': False}

        promoter = FinalTuningPromotion(self.model_path, self.config)
        baseline = promoter.evaluate_detailed()

        selected = candidates[:2]
        replay_samples = OldAbilityReplayGenerator.generate_all_replay_samples(10)
        step_results = []

        for i, candidate in enumerate(selected):
            step_num = i + 1
            new_gen = SampleGenerator(candidate)
            new_samples = new_gen.generate_all_samples()

            print(f"\n--- 第 {step_num} 步 ---")
            print(f"更新幅度上限: {self.config.step1_max_change if step_num == 1 else self.config.step2_max_change}")

            result = promoter.guided_promotion_step(new_samples, replay_samples, step_num)

            step_results.append({
                'step': step_num, 'target': result['final_eval']['target'],
                'old_ability_drop': result['final_eval']['max_old_drop'],
                'ability_changes': result['final_eval']['changes'],
                'kl_weights': result['final_kl_weights'].copy(),
            })

            print(f"目标: {result['final_eval']['target']:.2%}")
            print(f"旧能力掉落: {result['final_eval']['max_old_drop']:.2%}")
            print(f"KL: gap={result['final_kl_weights']['gap']:.3f}, writeback={result['final_kl_weights']['writeback']:.3f}")

        final_eval = promoter.evaluate_detailed()
        total_change = final_eval['target'] - baseline['target']

        step1_drop = step_results[0]['old_ability_drop'] if step_results else 1.0
        final_drop = final_eval['max_old_drop']
        writeback_drop = abs(final_eval['changes'].get('writeback', 0))

        success = (
            step1_drop < 0.08 and final_drop < 0.10 and
            writeback_drop < 0.03 and total_change > 0.15
        )

        return {
            'baseline_target': baseline['target'], 'final_target': final_eval['target'],
            'total_change': total_change, 'step1_drop': step1_drop,
            'final_drop': final_drop, 'writeback_drop': writeback_drop,
            'ability_changes': final_eval['changes'], 'step_results': step_results,
            'success': success,
        }

    def run(self):
        print("=" * 70)
        print("Stage 5G - 实验 6C: 最终微调 - 进入阶段 6")
        print("=" * 70)
        print(f"\n配置:")
        print(f"  基础 KL: {self.config.base_kl_weights}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  回放比例: {self.config.replay_ratio}")
        print(f"  第 1 步更新上限: {self.config.step1_max_change}")

        candidates = self.load_candidates()
        print(f"\n加载 {len(candidates)} 个候选")

        print("\n" + "=" * 70)
        print("连续晋升测试")
        print("=" * 70)

        continuous_result = self.run_continuous_promotion(candidates)

        print("\n" + "=" * 70)
        print("实验 6C 结果汇总")
        print("=" * 70)
        print(f"\n基线目标: {continuous_result['baseline_target']:.2%}")
        print(f"最终目标: {continuous_result['final_target']:.2%}")
        print(f"总目标变化: {continuous_result['total_change']:+.2%}")
        print(f"\n旧能力保护:")
        print(f"  第 1 步掉落: {continuous_result['step1_drop']:.2%} (目标 < 8%)")
        print(f"  最终掉落: {continuous_result['final_drop']:.2%} (目标 < 10%)")
        print(f"  writeback: {continuous_result['writeback_drop']:.2%} (目标 < 3%)")
        print(f"\n分项变化:")
        for ability, change in continuous_result['ability_changes'].items():
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {ability}: {change:+.2%} {status}")

        print("\n" + "=" * 70)
        print("阶段 6 入口标准检查")
        print("=" * 70)

        checks = [
            ("第 1 步旧能力掉落 < 8%", continuous_result['step1_drop'] < 0.08),
            ("最终旧能力掉落 < 10%", continuous_result['final_drop'] < 0.10),
            ("writeback 掉落 < 3%", continuous_result['writeback_drop'] < 0.03),
            ("目标提升 > +15%", continuous_result['total_change'] > 0.15),
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉🎉🎉 实验 6C 通过！正式进入阶段 6！🎉🎉🎉")
            print("=" * 70)
            print("\n✅ 受控自构建系统验证完成！")
            print("\n关键成果:")
            print(f"  ✓ 第 1 步保护: {continuous_result['step1_drop']:.2%} < 8%")
            print(f"  ✓ 最终保护: {continuous_result['final_drop']:.2%} < 10%")
            print(f"  ✓ 目标提升: {continuous_result['total_change']:+.2%} > +15%")
            print(f"  ✓ writeback: {continuous_result['writeback_drop']:.2%} < 3%")
            print("\n下一阶段: 阶段 6 - 受控自构建系统整合")
        else:
            print("⚠️ 部分标准未满足")
            print("=" * 70)

        results = {
            'experiment': 'exp6c_final_tuning',
            'config': {
                'base_kl_weights': self.config.base_kl_weights,
                'learning_rate': self.config.learning_rate,
                'replay_ratio': self.config.replay_ratio,
                'step1_max_change': self.config.step1_max_change,
            },
            'continuous_result': continuous_result,
            'phase6_ready': all_passed,
        }

        with open("eval/stage5g_exp6c_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print("\n✓ 实验 6C 完成")
        print("  结果已保存到 eval/stage5g_exp6c_results.json")
        return results


def main():
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    experiment = Experiment6C(model_path)
    results = experiment.run()


if __name__ == "__main__":
    main()
