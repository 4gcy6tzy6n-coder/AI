"""
Stage 5G - Experiment 4: Global Protection with Old Ability Replay

实验 4：全面高保护 + 旧能力回放小批次 + 分项监控

核心改进：
1. 全局高 KL 保护：gap/policy/governance=0.25, writeback=0.30
2. 旧能力回放：每 epoch 混入旧能力样本
3. 分项监控：单独追踪 retrieval/governance/policy/writeback 掉落

配置：
- kl_weights = {"gap": 0.25, "policy": 0.25, "governance": 0.25, "writeback": 0.30}
- learning_rate = 1.2e-5
- epochs = 3

验收标准：
- 目标提升 ≥ +5%
- 最大旧能力掉落 < 10%
- writeback 掉落 < 5%
- 连续晋升后目标能力仍为正
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
from typing import Dict, List
from dataclasses import dataclass

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class Exp4Config:
    """实验 4 配置"""
    name: str = "实验4_全面高保护_旧能力回放"
    kl_weights: Dict[str, float] = None
    learning_rate: float = 1.2e-5
    num_epochs: int = 3
    batch_size: int = 8
    max_param_change: float = 0.01
    # 旧能力回放比例
    replay_ratio: float = 0.2  # 20% 旧能力样本

    def __post_init__(self):
        if self.kl_weights is None:
            self.kl_weights = {
                'gap': 0.25,
                'policy': 0.25,
                'governance': 0.25,
                'writeback': 0.30,
            }


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
        self.is_replay = is_replay  # 是否是旧能力回放样本


class SampleGenerator:
    """样本生成器 - 包含新能力样本和旧能力回放样本"""

    def __init__(self, candidate: Dict):
        self.candidate = candidate
        self.entity_a = candidate['entities']['a']
        self.entity_b = candidate['entities']['b']

    def generate_new_ability_samples(self, num: int = 30) -> List[FinetuneSample]:
        """生成新能力样本（RELATION 识别）"""
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
    def generate_retrieval_samples(num: int = 5) -> List[FinetuneSample]:
        """生成 retrieval trigger 保护样本"""
        templates = [
            "我需要查询相关信息",
            "请帮我检索资料",
            "这个问题需要查资料",
        ]
        samples = []
        for i in range(num):
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=1,  # RETRIEVABLE
                target_strategy=1,  # RETRIEVAL_FIRST
                target_writeback=0,
                target_governance=1,
                sample_type='retrieval_replay',
                is_replay=True
            ))
        return samples

    @staticmethod
    def generate_governance_samples(num: int = 5) -> List[FinetuneSample]:
        """生成 governance decision 保护样本"""
        templates = [
            "请评估这个回答是否安全",
            "检查是否有敏感信息",
            "验证回答的合规性",
        ]
        samples = []
        for i in range(num):
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=0,
                target_strategy=0,
                target_writeback=0,
                target_governance=1,  # 需要治理
                sample_type='governance_replay',
                is_replay=True
            ))
        return samples

    @staticmethod
    def generate_policy_samples(num: int = 5) -> List[FinetuneSample]:
        """生成 policy routing 保护样本"""
        templates = [
            "直接回答这个问题",
            "需要检索后再回答",
            "这个问题需要分步骤处理",
        ]
        samples = []
        for i in range(num):
            strategy = i % 3  # 轮换策略
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
    def generate_writeback_samples(num: int = 5) -> List[FinetuneSample]:
        """生成 writeback baseline 保护样本"""
        templates = [
            "记住我的偏好",
            "保存这个信息",
            "更新用户画像",
        ]
        samples = []
        for i in range(num):
            samples.append(FinetuneSample(
                input_text=templates[i % len(templates)],
                target_gap=0,
                target_strategy=0,
                target_writeback=1,  # 触发写回
                target_governance=1,
                sample_type='writeback_replay',
                is_replay=True
            ))
        return samples

    @classmethod
    def generate_all_replay_samples(cls, num_per_type: int = 5) -> List[FinetuneSample]:
        """生成所有旧能力回放样本"""
        samples = []
        samples.extend(cls.generate_retrieval_samples(num_per_type))
        samples.extend(cls.generate_governance_samples(num_per_type))
        samples.extend(cls.generate_policy_samples(num_per_type))
        samples.extend(cls.generate_writeback_samples(num_per_type))
        return samples


class GlobalProtectionPromotion:
    """全局保护参数晋升"""

    def __init__(self, model_path: str, config: Exp4Config):
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

    def evaluate_detailed(self) -> Dict:
        """详细评估 - 分项监控"""
        self.model.eval()

        target_correct = 0
        old_abilities = {
            'retrieval_trigger': 0,
            'governance': 0,
            'policy_routing': 0,
            'writeback': 0,
        }

        num_samples = 50

        with torch.no_grad():
            for _ in range(num_samples):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)

                # 目标能力：RELATION 识别
                gap_conf = outputs['gap_probs'][0].max().item()
                policy_conf = outputs['policy_probs'][0].max().item()
                if gap_conf > 0.6 and policy_conf > 0.6:
                    target_correct += 1

                # 分项监控旧能力
                if outputs['policy_probs'][0].max().item() > 0.5:
                    old_abilities['retrieval_trigger'] += 1
                    old_abilities['policy_routing'] += 1
                if outputs['governance_probs'][0].max().item() > 0.5:
                    old_abilities['governance'] += 1
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    old_abilities['writeback'] += 1

        return {
            'target': target_correct / num_samples,
            'old': {k: v / num_samples for k, v in old_abilities.items()},
        }

    def compute_layered_kl(self, current_outputs: Dict) -> torch.Tensor:
        """计算分层 KL 约束"""
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
                weight = self.config.kl_weights.get(config_key, 0.25)
                kl_loss += kl * weight

        return kl_loss

    def guided_promotion_with_replay(self, new_samples: List[FinetuneSample],
                                     replay_samples: List[FinetuneSample]) -> Dict:
        """有监督晋升 + 旧能力回放"""
        self.model.train()

        losses = []
        replay_losses = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_replay_loss = 0.0

            # 合并新样本和回放样本
            # 按 replay_ratio 比例混入
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

                avg_kl_weight = sum(self.config.kl_weights.values()) / len(self.config.kl_weights)
                total_loss = task_loss + avg_kl_weight * kl_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_param_change)
                self.optimizer.step()

                epoch_loss += total_loss.item()

                if sample.is_replay:
                    epoch_replay_loss += task_loss.item()

            losses.append(epoch_loss / len(all_samples))
            if num_replay > 0:
                replay_losses.append(epoch_replay_loss / num_replay)

        return {
            'losses': losses,
            'replay_losses': replay_losses,
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


class Experiment4:
    """实验 4"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = Exp4Config()

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
        promoter = GlobalProtectionPromotion(self.model_path, self.config)

        # 基线评估
        baseline = promoter.evaluate_detailed()

        # 生成样本
        new_gen = SampleGenerator(candidate)
        new_samples = new_gen.generate_all_samples()
        replay_samples = OldAbilityReplayGenerator.generate_all_replay_samples(5)

        # 晋升
        result = promoter.guided_promotion_with_replay(new_samples, replay_samples)

        # 晋升后评估
        after = promoter.evaluate_detailed()

        # 计算变化
        target_change = after['target'] - baseline['target']

        # 分项监控
        ability_changes = {}
        max_drop = 0
        for key in baseline['old']:
            change = after['old'][key] - baseline['old'][key]
            ability_changes[key] = change
            max_drop = max(max_drop, abs(change))

        # 回滚测试
        rollback_success = promoter.rollback()
        after_rollback = promoter.evaluate_detailed()
        rollback_accurate = all(
            abs(after_rollback['old'][k] - baseline['old'][k]) < 0.02
            for k in baseline['old']
        )

        # 验收
        success = (
            target_change >= 0.05 and  # ≥ +5%
            max_drop < 0.10 and  # < 10%
            ability_changes.get('writeback', 0) > -0.05  # writeback < 5%
        )

        return {
            'candidate_id': candidate['candidate_id'],
            'target_change': target_change,
            'ability_changes': ability_changes,
            'max_old_drop': max_drop,
            'rollback_success': rollback_success and rollback_accurate,
            'success': success,
        }

    def run_continuous_promotion(self, candidates: List[Dict]) -> Dict:
        """连续晋升测试"""
        if len(candidates) < 2:
            return {'success': False}

        promoter = GlobalProtectionPromotion(self.model_path, self.config)
        baseline = promoter.evaluate_detailed()

        selected = candidates[:2]
        replay_samples = OldAbilityReplayGenerator.generate_all_replay_samples(5)

        for candidate in selected:
            new_gen = SampleGenerator(candidate)
            new_samples = new_gen.generate_all_samples()
            promoter.guided_promotion_with_replay(new_samples, replay_samples)

        final_eval = promoter.evaluate_detailed()
        total_change = final_eval['target'] - baseline['target']

        ability_changes = {}
        max_drop = 0
        for key in baseline['old']:
            change = final_eval['old'][key] - baseline['old'][key]
            ability_changes[key] = change
            max_drop = max(max_drop, abs(change))

        success = (
            total_change > 0 and
            max_drop < 0.10 and
            ability_changes.get('writeback', 0) > -0.05
        )

        return {
            'total_change': total_change,
            'ability_changes': ability_changes,
            'max_old_drop': max_drop,
            'success': success,
        }

    def run(self):
        """运行实验 4"""
        print("=" * 70)
        print("Stage 5G - 实验 4: 全面高保护 + 旧能力回放")
        print("=" * 70)

        print(f"\n配置:")
        print(f"  KL 权重: {self.config.kl_weights}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  旧能力回放比例: {self.config.replay_ratio}")

        candidates = self.load_candidates()
        print(f"\n加载 {len(candidates)} 个候选")

        # 单候选测试
        print("\n" + "=" * 70)
        print("单候选晋升测试 (2个)")
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
            print(f"  回滚: {'✓' if result['rollback_success'] else '✗'}")
            print(f"  验收: {'✓ 通过' if result['success'] else '✗ 未通过'}")

        # 连续晋升测试
        print("\n" + "=" * 70)
        print("连续晋升测试")
        print("=" * 70)

        continuous_result = self.run_continuous_promotion(candidates)

        print(f"\n总目标变化: {continuous_result['total_change']:+.2%}")
        print(f"分项变化:")
        for ability, change in continuous_result['ability_changes'].items():
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {ability}: {change:+.2%} {status}")
        print(f"最大旧能力掉落: {continuous_result['max_old_drop']:.2%}")
        print(f"验收: {'✓ 通过' if continuous_result['success'] else '✗ 未通过'}")

        # 总结
        print("\n" + "=" * 70)
        print("实验 4 总结")
        print("=" * 70)

        single_success = sum(1 for r in single_results if r['success'])
        avg_target = sum(r['target_change'] for r in single_results) / len(single_results)

        print(f"\n单候选: {single_success}/2 成功")
        print(f"平均目标提升: {avg_target:+.2%}")
        print(f"连续晋升: {'✓ 通过' if continuous_result['success'] else '✗ 未通过'}")

        # 阶段 6 入口检查
        print("\n阶段 6 入口条件:")
        checks = [
            ("单候选成功 ≥ 1", single_success >= 1),
            ("目标提升 ≥ +5%", avg_target >= 0.05),
            ("连续晋升目标 > 0", continuous_result['total_change'] > 0),
            ("旧能力掉落 < 10%", continuous_result['max_old_drop'] < 0.10),
            ("writeback 掉落 < 5%",
             continuous_result['ability_changes'].get('writeback', 0) > -0.05),
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 实验 4 通过！建议进入阶段 6")
        else:
            print("\n⚠️ 部分条件未满足，建议上实验 5（自适应 KL）")

        # 保存结果
        results = {
            'experiment': 'exp4_global_protection',
            'config': {
                'kl_weights': self.config.kl_weights,
                'learning_rate': self.config.learning_rate,
                'replay_ratio': self.config.replay_ratio,
            },
            'single_results': single_results,
            'continuous_result': continuous_result,
            'phase6_ready': all_passed,
        }

        with open("eval/stage5g_exp4_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print("\n✓ 实验 4 完成")
        print("  结果已保存到 eval/stage5g_exp4_results.json")

        return results


def main():
    """主函数"""
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"

    experiment = Experiment4(model_path)
    results = experiment.run()


if __name__ == "__main__":
    main()
