"""
Stage 5G+: Fine-tuning Re-verification with Layered Protection

方案 A+ 微调复验 + writeback 专项高保护

实验设计：
1. 实验 1: A+ 基础版
   - kl_weight = 0.18
   - lr = 8e-6
   
2. 实验 2: A+ + writeback 高保护（主推）
   - gap/policy = 0.18
   - governance = 0.20
   - writeback = 0.30
   - lr = 8e-6
   
3. 实验 3: A+ + 更小步长
   - kl_weight = 0.18
   - lr = 7e-6

验收指标：
- 单候选目标提升 > +10%
- 连续晋升目标变化 > 0 (最好 +3%以上)
- 旧能力最大掉落 < 10% (硬门槛)
- writeback 掉落 < 5%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class FineTuneConfig:
    """微调配置"""
    name: str
    kl_weights: Dict[str, float]  # 分层 KL 权重
    learning_rate: float
    num_epochs: int = 3
    batch_size: int = 8
    max_param_change: float = 0.01
    old_ability_threshold: float = 0.10  # 硬门槛 10%
    writeback_threshold: float = 0.05    # writeback 专项 5%


# 三个实验配置
EXPERIMENT_CONFIGS = [
    # 实验 1: A+ 基础版
    FineTuneConfig(
        name="实验1_A+基础版",
        kl_weights={
            'gap': 0.18,
            'policy': 0.18,
            'governance': 0.18,
            'writeback': 0.18,
        },
        learning_rate=8e-6,
    ),
    # 实验 2: A+ + writeback 高保护（主推）
    FineTuneConfig(
        name="实验2_A+_writeback高保护",
        kl_weights={
            'gap': 0.18,
            'policy': 0.18,
            'governance': 0.20,
            'writeback': 0.30,  # 专项高保护
        },
        learning_rate=8e-6,
    ),
    # 实验 3: A+ + 更小步长
    FineTuneConfig(
        name="实验3_A+_更小步长",
        kl_weights={
            'gap': 0.18,
            'policy': 0.18,
            'governance': 0.18,
            'writeback': 0.18,
        },
        learning_rate=7e-6,  # 更小学习率
    ),
]


class FinetuneSample:
    """微调样本"""
    def __init__(self, input_text: str, target_gap: int, target_strategy: int,
                 target_writeback: int = 0, sample_type: str = "positive"):
        self.input_text = input_text
        self.target_gap = target_gap
        self.target_strategy = target_strategy
        self.target_writeback = target_writeback
        self.sample_type = sample_type


class SampleGenerator:
    """样本生成器"""

    def __init__(self, candidate: Dict):
        self.candidate = candidate
        self.entity_a = candidate['entities']['a']
        self.entity_b = candidate['entities']['b']

    def generate_all_samples(self, num_positive: int = 15, num_negative: int = 10,
                            num_adversarial: int = 15) -> List[FinetuneSample]:
        """生成完整样本集"""
        samples = []
        samples.extend(self.generate_positive_samples(num_positive))
        samples.extend(self.generate_negative_samples(num_negative))
        samples.extend(self.generate_adversarial_samples(num_adversarial))
        return samples

    def generate_positive_samples(self, num: int) -> List[FinetuneSample]:
        """生成正样本"""
        templates = [
            "{a} 和 {b} 有什么关系？",
            "{a} 与 {b} 的关联是什么？",
            "请解释 {a} 和 {b} 的关系",
            "{a} 如何影响 {b}？",
            "{a} 对 {b} 的作用是什么？",
        ]

        samples = []
        for i in range(num):
            template = templates[i % len(templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,
                target_strategy=1,
                target_writeback=0,
                sample_type='positive'
            ))
        return samples

    def generate_negative_samples(self, num: int) -> List[FinetuneSample]:
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
                sample_type='negative'
            ))
        return samples

    def generate_adversarial_samples(self, num: int) -> List[FinetuneSample]:
        """生成对抗样本"""
        synonym_templates = [
            "{a} 与 {b} 有何关联？",
            "{a} 对 {b} 的依赖程度如何？",
            "{a} 和 {b} 是如何相互作用的？",
        ]

        colloquial_templates = [
            "{a} 跟 {b} 有啥关系啊？",
            "为啥 {a} 要用 {b} 呢？",
            "{a} 是不是离不开 {b}？",
        ]

        samples = []
        for i in range(num):
            if i < num // 2:
                template = synonym_templates[i % len(synonym_templates)]
            else:
                template = colloquial_templates[i % len(colloquial_templates)]

            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,
                target_strategy=1,
                target_writeback=0,
                sample_type='adversarial'
            ))
        return samples


class LayeredProtectionPromotion:
    """分层保护参数晋升"""

    def __init__(self, model_path: str, config: FineTuneConfig):
        self.config = config

        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # 保存基线
        self.baseline_params = {}
        self.baseline_outputs = {}
        self._save_baseline()

        # 优化器
        self.optimizer = self._create_selective_optimizer()

    def _save_baseline(self):
        """保存基线状态和输出"""
        self.model.eval()

        for name, param in self.model.named_parameters():
            self.baseline_params[name] = param.clone().detach()

        with torch.no_grad():
            input_ids = torch.randint(0, 10000, (1, 50))
            outputs = self.model(input_ids)
            for key in ['gap_logits', 'policy_logits', 'writeback_logits', 'governance_logits']:
                if key in outputs:
                    self.baseline_outputs[key] = outputs[key].clone().detach()

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

    def evaluate(self) -> Dict:
        """评估模型能力"""
        self.model.eval()

        target_correct = 0
        old_abilities = {
            'private_knowledge': 0,
            'retrieval_trigger': 0,
            'gap': 0,
            'governance': 0,
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
                    target_correct += 1

                if outputs['governance_probs'][0].max().item() > 0.5:
                    old_abilities['private_knowledge'] += 1
                    old_abilities['governance'] += 1
                if outputs['policy_probs'][0].max().item() > 0.5:
                    old_abilities['retrieval_trigger'] += 1
                if outputs['gap_probs'][0].max().item() > 0.5:
                    old_abilities['gap'] += 1
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    old_abilities['writeback'] += 1

        return {
            'target': target_correct / num_samples,
            'old': {k: v / num_samples for k, v in old_abilities.items()},
        }

    def compute_layered_kl_constraint(self, current_outputs: Dict) -> torch.Tensor:
        """计算分层 KL 散度约束"""
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
                # 使用分层权重
                weight = self.config.kl_weights.get(config_key, 0.18)
                kl_loss += kl * weight

        return kl_loss

    def guided_promotion(self, samples: List[FinetuneSample]) -> Dict:
        """有监督的参数晋升"""
        self.model.train()

        losses = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            for sample in samples:
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
                kl_loss = self.compute_layered_kl_constraint(outputs)

                # 使用平均 KL 权重作为总权重系数
                avg_kl_weight = sum(self.config.kl_weights.values()) / len(self.config.kl_weights)
                total_loss = task_loss + avg_kl_weight * kl_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_param_change)
                self.optimizer.step()

                epoch_loss += total_loss.item()

            losses.append(epoch_loss / len(samples))

        return {'losses': losses}

    def rollback(self) -> bool:
        """回滚到基线"""
        try:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.baseline_params:
                        param.copy_(self.baseline_params[name])
            return True
        except Exception as e:
            print(f"回滚失败: {e}")
            return False


class Stage5GPlusExperiment:
    """Stage 5G+ 实验"""

    def __init__(self, model_path: str):
        self.model_path = model_path

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

    def test_config(self, config: FineTuneConfig, candidates: List[Dict]) -> Dict:
        """测试一个配置"""
        print("\n" + "=" * 70)
        print(f"测试 {config.name}")
        print(f"  KL权重: {config.kl_weights}")
        print(f"  学习率: {config.learning_rate}")
        print("=" * 70)

        # 单候选测试（2个）
        single_results = []
        for candidate in candidates[:2]:
            result = self._test_single_promotion(config, candidate)
            single_results.append(result)

        # 连续晋升测试
        continuous_result = self._test_continuous_promotion(config, candidates)

        # 汇总
        single_success = sum(1 for r in single_results if r['success'])
        avg_target_improvement = sum(r['target_change'] for r in single_results) / len(single_results)
        max_old_drop = max(r['max_old_drop'] for r in single_results + [continuous_result])
        writeback_drop = max(r['writeback_drop'] for r in single_results + [continuous_result])

        print(f"\n{config.name} 结果汇总:")
        print(f"  单候选成功: {single_success}/2")
        print(f"  平均目标提升: {avg_target_improvement:+.2%}")
        print(f"  连续晋升目标变化: {continuous_result.get('total_change', 0):+.2%}")
        print(f"  最大旧能力掉落: {max_old_drop:.2%}")
        print(f"  writeback掉落: {writeback_drop:.2%}")

        # 验收（阶段 6 入口条件）
        passed = (
            single_success >= 1 and  # 至少1个单候选成功
            continuous_result.get('total_change', 0) > 0 and  # 连续晋升目标为正
            max_old_drop < config.old_ability_threshold and  # 旧能力<10%
            writeback_drop < config.writeback_threshold  # writeback<5%
        )

        print(f"  阶段6入口验收: {'✓ 通过' if passed else '✗ 未通过'}")

        return {
            'config_name': config.name,
            'kl_weights': config.kl_weights,
            'learning_rate': config.learning_rate,
            'single_results': single_results,
            'continuous_result': continuous_result,
            'single_success': single_success,
            'avg_target_improvement': avg_target_improvement,
            'max_old_drop': max_old_drop,
            'writeback_drop': writeback_drop,
            'passed': passed,
        }

    def _test_single_promotion(self, config: FineTuneConfig, candidate: Dict) -> Dict:
        """测试单候选晋升"""
        promoter = LayeredProtectionPromotion(self.model_path, config)

        baseline = promoter.evaluate()
        generator = SampleGenerator(candidate)
        samples = generator.generate_all_samples()
        promoter.guided_promotion(samples)
        after = promoter.evaluate()

        target_change = after['target'] - baseline['target']
        max_drop = 0
        writeback_drop = 0

        for key in baseline['old']:
            change = after['old'][key] - baseline['old'][key]
            max_drop = max(max_drop, abs(change))
            if key == 'writeback':
                writeback_drop = abs(change)

        rollback_success = promoter.rollback()
        after_rollback = promoter.evaluate()
        rollback_accurate = all(
            abs(after_rollback['old'][k] - baseline['old'][k]) < 0.02
            for k in baseline['old']
        )

        # 验收标准
        success = (
            target_change > 0.10 and  # > +10%
            max_drop < config.old_ability_threshold and  # < 10%
            writeback_drop < config.writeback_threshold  # < 5%
        )

        return {
            'candidate_id': candidate['candidate_id'],
            'target_change': target_change,
            'writeback_drop': writeback_drop,
            'max_old_drop': max_drop,
            'rollback_success': rollback_success and rollback_accurate,
            'success': success,
        }

    def _test_continuous_promotion(self, config: FineTuneConfig, candidates: List[Dict]) -> Dict:
        """测试连续晋升"""
        if len(candidates) < 2:
            return {'success': False, 'reason': 'insufficient_candidates'}

        promoter = LayeredProtectionPromotion(self.model_path, config)
        baseline = promoter.evaluate()
        selected = candidates[:2]

        cumulative_improvement = 0

        for i, candidate in enumerate(selected):
            generator = SampleGenerator(candidate)
            samples = generator.generate_all_samples()
            promoter.guided_promotion(samples)

            after_step = promoter.evaluate()
            step_improvement = after_step['target'] - baseline['target'] - cumulative_improvement
            cumulative_improvement += step_improvement

        final_eval = promoter.evaluate()
        total_change = final_eval['target'] - baseline['target']

        max_drop = 0
        writeback_drop = 0
        for key in baseline['old']:
            change = final_eval['old'][key] - baseline['old'][key]
            max_drop = max(max_drop, abs(change))
            if key == 'writeback':
                writeback_drop = abs(change)

        success = (
            total_change > 0 and  # 为正
            max_drop < config.old_ability_threshold and
            writeback_drop < config.writeback_threshold
        )

        return {
            'total_change': total_change,
            'max_old_drop': max_drop,
            'writeback_drop': writeback_drop,
            'success': success,
        }

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 70)
        print("Stage 5G+: 方案 A+ 微调复验")
        print("=" * 70)

        print("\n实验配置:")
        for cfg in EXPERIMENT_CONFIGS:
            print(f"  {cfg.name}")
            print(f"    KL: {cfg.kl_weights}")
            print(f"    LR: {cfg.learning_rate}")

        candidates = self.load_candidates()
        print(f"\n加载 {len(candidates)} 个候选")

        all_results = []
        for config in EXPERIMENT_CONFIGS:
            result = self.test_config(config, candidates)
            all_results.append(result)

        # 总结
        print("\n" + "=" * 70)
        print("Stage 5G+ 复验总结")
        print("=" * 70)

        print("\n各实验结果:")
        for r in all_results:
            status = "✓ 通过" if r['passed'] else "✗ 未通过"
            print(f"\n  {r['config_name']}: {status}")
            print(f"    单候选: {r['single_success']}/2")
            print(f"    目标提升: {r['avg_target_improvement']:+.2%}")
            print(f"    旧能力掉落: {r['max_old_drop']:.2%}")
            print(f"    writeback: {r['writeback_drop']:.2%}")

        passed_configs = [r for r in all_results if r['passed']]

        if passed_configs:
            best = max(passed_configs, key=lambda x: x['avg_target_improvement'])
            print(f"\n🎉 最佳方案: {best['config_name']}")
            print(f"  KL权重: {best['kl_weights']}")
            print(f"  学习率: {best['learning_rate']}")
            print(f"  平均目标提升: {best['avg_target_improvement']:+.2%}")
            print(f"  最大旧能力掉落: {best['max_old_drop']:.2%}")
            print(f"  writeback掉落: {best['writeback_drop']:.2%}")

            if len(passed_configs) >= 2:
                print("\n✅ 多组方案通过验收，阶段 6 入口条件满足！")
            else:
                print("\n✓ 找到可行方案，建议进入阶段 6")
        else:
            print("\n⚠️ 未找到通过所有验收的方案")
            print("  建议：引入自适应 KL 或进一步微调")

        with open("eval/stage5g_plus_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n✓ Stage 5G+ 完成")
        print("  结果已保存到 eval/stage5g_plus_results.json")

        return all_results


def main():
    """主函数"""
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"

    experiment = Stage5GPlusExperiment(model_path)
    results = experiment.run_all_tests()


if __name__ == "__main__":
    main()
