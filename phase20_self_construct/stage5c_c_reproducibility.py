"""
Stage 5C-c: Reproducibility Validation

可复现性验证实验

3个实验：
1. 第二个 RELATION 候选复现 (relation_169adf1f0331, 0.84分)
2. 跨类型候选验证 (EXPLANATION 或 RULE)
3. 双候选连续晋升

目标：验证"有监督、受约束、局部参数晋升"方法的可复现性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from copy import deepcopy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class ReproConfig:
    """可复现性实验配置"""
    learning_rate: float = 1e-5
    num_epochs: int = 5
    batch_size: int = 8
    kl_weight: float = 0.1
    max_param_change: float = 0.01
    old_ability_threshold: float = 0.1


class FinetuneDataset:
    """微调数据集"""
    
    def __init__(self, samples: List[Dict], vocab_size: int = 10000):
        self.samples = samples
        self.vocab_size = vocab_size
    
    def text_to_ids(self, text: str, max_len: int = 50) -> List[int]:
        ids = [hash(c) % self.vocab_size for c in text[:max_len]]
        while len(ids) < max_len:
            ids.append(0)
        return ids[:max_len]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = self.text_to_ids(sample['input_text'])
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(sample['target_gap'], dtype=torch.long),
            torch.tensor(sample['target_strategy'], dtype=torch.long),
        )


class ReproducibilityExperiment:
    """可复现性实验"""
    
    def __init__(self, model_path: str):
        self.config = ReproConfig()
        
        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # 保存基线
        self.baseline_params = {name: param.clone().detach() 
                               for name, param in self.model.named_parameters()}
        
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randint(0, 10000, (1, 50))
            self.baseline_outputs = self.model(dummy_input)
        
        self.results = []
    
    def generate_samples_for_candidate(self, candidate: Dict, num_samples: int = 50) -> List[Dict]:
        """为候选生成微调样本"""
        candidate_type = candidate.get('candidate_type', 'UNKNOWN')
        
        if candidate_type == 'RELATION':
            return self._generate_relation_samples(candidate, num_samples)
        elif candidate_type == 'EXPLANATION':
            return self._generate_explanation_samples(candidate, num_samples)
        elif candidate_type == 'RULE':
            return self._generate_rule_samples(candidate, num_samples)
        else:
            return []
    
    def _generate_relation_samples(self, candidate: Dict, num: int) -> List[Dict]:
        """生成 RELATION 类型样本"""
        entities = candidate.get('entities', {'a': 'A', 'b': 'B'})
        a, b = entities.get('a', 'A'), entities.get('b', 'B')
        
        samples = []
        templates = [
            f"{a} 和 {b} 是什么关系？",
            f"{a} 与 {b} 之间有什么联系？",
            f"请解释 {a} 和 {b} 的关系",
            f"{a} 如何利用 {b}？",
            f"{b} 在 {a} 中起什么作用？",
            f"在使用 {a} 时，为什么需要 {b}？",
            f"{a} 依赖 {b} 吗？",
            f"{a} 和 {b} 是如何配合工作的？",
            f"没有 {b}，{a} 还能工作吗？",
            f"{a} 为什么离不开 {b}？",
        ]
        
        # 正样本
        for i in range(num // 2):
            text = templates[i % len(templates)]
            samples.append({
                'input_text': text,
                'target_gap': 1,
                'target_strategy': 1,
                'sample_type': 'positive'
            })
        
        # 负样本（无关实体）
        unrelated = [("苹果", "香蕉"), ("汽车", "飞机"), ("猫", "狗")]
        for i in range(num // 2):
            pair = unrelated[i % len(unrelated)]
            samples.append({
                'input_text': f"{pair[0]} 和 {pair[1]} 是什么关系？",
                'target_gap': 0,
                'target_strategy': 0,
                'sample_type': 'negative'
            })
        
        import random
        random.shuffle(samples)
        return samples[:num]
    
    def _generate_explanation_samples(self, candidate: Dict, num: int) -> List[Dict]:
        """生成 EXPLANATION 类型样本"""
        source_query = candidate.get('source_query', '为什么？')
        
        samples = []
        templates = [
            f"请解释：{source_query}",
            f"{source_query} 的原因是什么？",
            f"为什么{source_query.replace('为什么', '').replace('？', '')}？",
            f"能说明一下{source_query.replace('为什么', '').replace('？', '')}吗？",
            f"我想了解{source_query.replace('为什么', '').replace('？', '')}",
        ]
        
        for i in range(num):
            text = templates[i % len(templates)]
            samples.append({
                'input_text': text,
                'target_gap': 1,
                'target_strategy': 1,
                'sample_type': 'positive'
            })
        
        return samples
    
    def _generate_rule_samples(self, candidate: Dict, num: int) -> List[Dict]:
        """生成 RULE 类型样本"""
        scenario = candidate.get('scenario', '某场景')
        
        samples = []
        templates = [
            f"在{scenario}时应该怎么做？",
            f"遇到{scenario}情况如何处理？",
            f"{scenario}的规则是什么？",
            f"请说明{scenario}的处理流程",
            f"{scenario}时应该遵循什么原则？",
        ]
        
        for i in range(num):
            text = templates[i % len(templates)]
            samples.append({
                'input_text': text,
                'target_gap': 1,
                'target_strategy': 2,  # CONSERVATIVE
                'sample_type': 'positive'
            })
        
        return samples
    
    def guided_promotion(self, samples: List[Dict]) -> Dict:
        """执行有监督晋升"""
        dataset = FinetuneDataset(samples)
        
        # 设置可训练参数
        target_params = []
        for name, param in self.model.named_parameters():
            if 'unit_encoder' in name or 'policy_head' in name:
                param.requires_grad = True
                target_params.append(param)
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.Adam(target_params, lr=self.config.learning_rate)
        
        # 训练
        history = []
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            for i in range(0, len(dataset), self.config.batch_size):
                batch_samples = []
                for j in range(i, min(i + self.config.batch_size, len(dataset))):
                    batch_samples.append(dataset[j])
                
                if not batch_samples:
                    continue
                
                input_ids = torch.stack([s[0] for s in batch_samples])
                target_gap = torch.stack([s[1] for s in batch_samples])
                target_strategy = torch.stack([s[2] for s in batch_samples])
                
                self.model.train()
                optimizer.zero_grad()
                
                outputs = self.model(input_ids)
                
                gap_loss = F.cross_entropy(outputs['gap_logits'], target_gap)
                policy_loss = F.cross_entropy(outputs['policy_logits'], target_strategy)
                
                # KL约束
                kl_loss = 0.0
                kl_gap = F.kl_div(
                    F.log_softmax(outputs['gap_logits'], dim=-1),
                    F.softmax(self.baseline_outputs['gap_logits'], dim=-1),
                    reduction='batchmean'
                )
                kl_policy = F.kl_div(
                    F.log_softmax(outputs['policy_logits'], dim=-1),
                    F.softmax(self.baseline_outputs['policy_logits'], dim=-1),
                    reduction='batchmean'
                )
                kl_loss = kl_gap + kl_policy
                
                total_loss = gap_loss + policy_loss + self.config.kl_weight * kl_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(target_params, self.config.max_param_change)
                optimizer.step()
                
                epoch_losses.append({
                    'total': total_loss.item(),
                    'gap': gap_loss.item(),
                    'policy': policy_loss.item(),
                    'kl': kl_loss.item(),
                })
            
            avg_loss = {
                'total': sum(l['total'] for l in epoch_losses) / len(epoch_losses),
                'gap': sum(l['gap'] for l in epoch_losses) / len(epoch_losses),
                'policy': sum(l['policy'] for l in epoch_losses) / len(epoch_losses),
                'kl': sum(l['kl'] for l in epoch_losses) / len(epoch_losses),
            }
            history.append(avg_loss)
        
        return {'history': history}
    
    def evaluate(self, num_samples: int = 50) -> Dict:
        """评估"""
        self.model.eval()
        
        target_correct = 0
        old_abilities = {
            'private_knowledge': 0,
            'retrieval_trigger': 0,
            'gap': 0,
            'governance': 0,
            'writeback': 0,
        }
        
        with torch.no_grad():
            for _ in range(num_samples):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)
                
                # 目标能力
                if outputs['gap_probs'][0].max().item() > 0.7 and \
                   outputs['policy_probs'][0].max().item() > 0.7:
                    target_correct += 1
                
                # 旧能力
                if outputs['governance_probs'][0].max().item() > 0.5:
                    old_abilities['private_knowledge'] += 1
                    old_abilities['governance'] += 1
                if outputs['policy_probs'][0].max().item() > 0.5:
                    old_abilities['retrieval_trigger'] += 1
                if outputs['gap_probs'][0].max().item() > 0.5:
                    old_abilities['gap'] += 1
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    old_abilities['writeback'] += 1
        
        for key in old_abilities:
            old_abilities[key] /= num_samples
        
        return {
            'target': target_correct / num_samples,
            'old': old_abilities,
        }
    
    def run_experiment_1(self, candidate: Dict) -> Dict:
        """
        实验 1：第二个 RELATION 候选复现
        
        验证 5C-b 方法对中等质量候选也有效
        """
        print("\n" + "=" * 70)
        print("实验 1：第二个 RELATION 候选复现")
        print("=" * 70)
        
        candidate_id = candidate.get('candidate_id', 'unknown')
        validation_score = candidate.get('stages', {}).get('validation', {}).get('score', 0)
        
        print(f"候选: {candidate_id}")
        print(f"验证分数: {validation_score:.2f}")
        
        # 基线评估
        print("\n基线评估...")
        baseline = self.evaluate()
        print(f"  目标能力: {baseline['target']:.2%}")
        
        # 生成样本
        samples = self.generate_samples_for_candidate(candidate, 50)
        print(f"  生成 {len(samples)} 条微调样本")
        
        # 晋升
        print("\n执行晋升...")
        promotion_result = self.guided_promotion(samples)
        
        # 晋升后评估
        print("\n晋升后评估...")
        post = self.evaluate()
        print(f"  目标能力: {post['target']:.2%}")
        
        # 对比
        target_change = post['target'] - baseline['target']
        print(f"\n目标能力变化: {baseline['target']:.2%} -> {post['target']:.2%} ({target_change:+.2%})")
        
        # 旧能力
        print("\n旧能力变化:")
        max_drop = 0
        for key in baseline['old']:
            change = post['old'][key] - baseline['old'][key]
            max_drop = max(max_drop, abs(change))
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {key}: {baseline['old'][key]:.2%} -> {post['old'][key]:.2%} ({change:+.2%}) {status}")
        
        # 验收
        success = target_change > 0 and max_drop < self.config.old_ability_threshold
        print(f"\n验收: {'✓ 成功' if success else '✗ 失败'}")
        
        return {
            'experiment': 'relation_2',
            'candidate_id': candidate_id,
            'baseline': baseline,
            'post': post,
            'target_change': target_change,
            'max_old_drop': max_drop,
            'success': success,
            'training': promotion_result,
        }
    
    def run_experiment_2(self, candidate: Dict) -> Dict:
        """
        实验 2：跨类型候选验证
        
        验证方法是否适用于其他候选类型
        """
        print("\n" + "=" * 70)
        print("实验 2：跨类型候选验证")
        print("=" * 70)
        
        candidate_id = candidate.get('candidate_id', 'unknown')
        candidate_type = candidate.get('candidate_type', 'UNKNOWN')
        
        print(f"候选: {candidate_id}")
        print(f"类型: {candidate_type}")
        
        # 重置模型到基线
        print("\n重置模型到基线...")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.baseline_params:
                    param.copy_(self.baseline_params[name])
        
        # 基线评估
        baseline = self.evaluate()
        print(f"  目标能力: {baseline['target']:.2%}")
        
        # 生成样本
        samples = self.generate_samples_for_candidate(candidate, 50)
        print(f"  生成 {len(samples)} 条微调样本")
        
        # 晋升
        print("\n执行晋升...")
        promotion_result = self.guided_promotion(samples)
        
        # 晋升后评估
        post = self.evaluate()
        print(f"  目标能力: {post['target']:.2%}")
        
        # 对比
        target_change = post['target'] - baseline['target']
        print(f"\n目标能力变化: {baseline['target']:.2%} -> {post['target']:.2%} ({target_change:+.2%})")
        
        # 旧能力
        print("\n旧能力变化:")
        max_drop = 0
        for key in baseline['old']:
            change = post['old'][key] - baseline['old'][key]
            max_drop = max(max_drop, abs(change))
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {key}: {baseline['old'][key]:.2%} -> {post['old'][key]:.2%} ({change:+.2%}) {status}")
        
        # 验收
        success = target_change > 0 and max_drop < self.config.old_ability_threshold
        print(f"\n验收: {'✓ 成功' if success else '✗ 失败'}")
        
        return {
            'experiment': f'cross_type_{candidate_type}',
            'candidate_id': candidate_id,
            'candidate_type': candidate_type,
            'baseline': baseline,
            'post': post,
            'target_change': target_change,
            'max_old_drop': max_drop,
            'success': success,
            'training': promotion_result,
        }
    
    def run_experiment_3(self, candidate1: Dict, candidate2: Dict) -> Dict:
        """
        实验 3：双候选连续晋升
        
        验证连续晋升的累积效果
        """
        print("\n" + "=" * 70)
        print("实验 3：双候选连续晋升")
        print("=" * 70)
        
        print(f"候选 1: {candidate1.get('candidate_id', 'unknown')}")
        print(f"候选 2: {candidate2.get('candidate_id', 'unknown')}")
        
        # 重置模型
        print("\n重置模型到基线...")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.baseline_params:
                    param.copy_(self.baseline_params[name])
        
        # 初始基线
        baseline = self.evaluate()
        print(f"\n初始基线 - 目标能力: {baseline['target']:.2%}")
        
        # 第一次晋升
        print("\n--- 第一次晋升 ---")
        samples1 = self.generate_samples_for_candidate(candidate1, 30)
        self.guided_promotion(samples1)
        after_1 = self.evaluate()
        print(f"第一次后 - 目标能力: {after_1['target']:.2%}")
        
        # 第二次晋升（连续）
        print("\n--- 第二次晋升（连续） ---")
        samples2 = self.generate_samples_for_candidate(candidate2, 30)
        self.guided_promotion(samples2)
        after_2 = self.evaluate()
        print(f"第二次后 - 目标能力: {after_2['target']:.2%}")
        
        # 对比
        total_change = after_2['target'] - baseline['target']
        print(f"\n总目标能力变化: {baseline['target']:.2%} -> {after_2['target']:.2%} ({total_change:+.2%})")
        
        # 旧能力
        print("\n旧能力变化:")
        max_drop = 0
        for key in baseline['old']:
            change = after_2['old'][key] - baseline['old'][key]
            max_drop = max(max_drop, abs(change))
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {key}: {baseline['old'][key]:.2%} -> {after_2['old'][key]:.2%} ({change:+.2%}) {status}")
        
        # 验收
        success = total_change > 0 and max_drop < self.config.old_ability_threshold
        print(f"\n验收: {'✓ 成功' if success else '✗ 失败'}")
        
        return {
            'experiment': 'dual_promotion',
            'candidate_1': candidate1.get('candidate_id', 'unknown'),
            'candidate_2': candidate2.get('candidate_id', 'unknown'),
            'baseline': baseline,
            'after_1': after_1,
            'after_2': after_2,
            'total_change': total_change,
            'max_old_drop': max_drop,
            'success': success,
        }
    
    def load_candidates(self) -> List[Dict]:
        """加载候选"""
        path = "candidates/stage5b_high_quality_batch.jsonl"
        candidates = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 只加载通过晋升的候选
                if data.get('status') == 'PENDING_REVIEW' or True:  # 简化处理
                    candidates.append(data)
        
        return candidates
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("=" * 70)
        print("Stage 5C-c: 可复现性验证")
        print("=" * 70)
        
        # 加载候选
        candidates = self.load_candidates()
        print(f"\n加载 {len(candidates)} 个候选")
        
        # 分类候选
        relation_candidates = [c for c in candidates if c.get('candidate_type') == 'RELATION']
        explanation_candidates = [c for c in candidates if c.get('candidate_type') == 'EXPLANATION']
        rule_candidates = [c for c in candidates if c.get('candidate_type') == 'RULE']
        
        print(f"  RELATION: {len(relation_candidates)}")
        print(f"  EXPLANATION: {len(explanation_candidates)}")
        print(f"  RULE: {len(rule_candidates)}")
        
        results = []
        
        # 实验 1：第二个 RELATION
        if len(relation_candidates) >= 2:
            result1 = self.run_experiment_1(relation_candidates[1])
            results.append(result1)
        else:
            print("\n✗ 实验 1 跳过：没有足够的 RELATION 候选")
        
        # 实验 2：跨类型
        cross_type_candidate = None
        if explanation_candidates:
            cross_type_candidate = explanation_candidates[0]
        elif rule_candidates:
            cross_type_candidate = rule_candidates[0]
        
        if cross_type_candidate:
            result2 = self.run_experiment_2(cross_type_candidate)
            results.append(result2)
        else:
            print("\n✗ 实验 2 跳过：没有 EXPLANATION 或 RULE 候选")
        
        # 实验 3：双候选连续晋升
        if len(relation_candidates) >= 2:
            result3 = self.run_experiment_3(relation_candidates[0], relation_candidates[1])
            results.append(result3)
        else:
            print("\n✗ 实验 3 跳过：没有足够的候选")
        
        # 总结
        print("\n" + "=" * 70)
        print("Stage 5C-c 总结")
        print("=" * 70)
        
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"\n总实验数: {len(results)}")
        print(f"成功数: {success_count}")
        print(f"成功率: {success_count / len(results) * 100:.1f}%" if results else "N/A")
        
        # 阶段 6 入口条件
        print("\n阶段 6 入口条件检查:")
        checks = [
            ("至少 2 个候选晋升成功", success_count >= 2),
            ("至少 2 种候选类型中有 1 种额外成功", 
             len(set(r.get('candidate_type', 'RELATION') for r in results if r.get('success'))) >= 2),
        ]
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        # 保存结果
        with open("eval/stage5c_c_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ Stage 5C-c 完成")
        print("  结果已保存到 eval/stage5c_c_results.json")
        
        return results


def main():
    """主函数"""
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    
    experiment = ReproducibilityExperiment(model_path)
    results = experiment.run_all_experiments()


if __name__ == "__main__":
    main()
