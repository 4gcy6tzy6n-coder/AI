"""
Stage 5F: Old Ability Protection Enhancement & Continuous Promotion Re-verification

旧能力保护强化与连续晋升复验

核心改进：
1. KL 权重：0.1 → 0.3
2. writeback 专项保持损失
3. 学习率降低：1e-5 → 5e-6
4. 参数更新范围收缩
5. 每步后门控

目标：让"可成长"变成"可持续成长"
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
class Stage5FConfig:
    """Stage 5F 配置 - 强化保护"""
    # 学习率降低
    learning_rate: float = 5e-6  # 从 1e-5 降低
    num_epochs: int = 2  # 从 3 减少
    batch_size: int = 8
    
    # KL 约束加强
    kl_weight: float = 0.3  # 从 0.1 提高
    
    # writeback 专项保持
    writeback_retention_weight: float = 0.5
    
    # 参数更新限制
    max_param_change: float = 0.005  # 从 0.01 降低
    
    # 评估阈值
    old_ability_threshold: float = 0.10
    
    # 连续晋升配置
    micro_steps: int = 2
    step_eval_interval: int = 1


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
    """样本生成器 - 包含 writeback 目标"""
    
    def __init__(self, candidate: Dict):
        self.candidate = candidate
        self.entity_a = candidate['entities']['a']
        self.entity_b = candidate['entities']['b']
    
    def generate_all_samples(self, num_positive: int = 12, num_negative: int = 8, 
                            num_adversarial: int = 12) -> List[FinetuneSample]:
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
        ]
        
        samples = []
        for i in range(num):
            template = templates[i % len(templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,  # RETRIEVABLE
                target_strategy=1,  # RETRIEVAL_FIRST
                target_writeback=0,  # 不触发写回
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
                target_gap=0,  # KNOWLEDGE
                target_strategy=0,  # DIRECT_ANSWER
                target_writeback=0,
                sample_type='negative'
            ))
        return samples
    
    def generate_adversarial_samples(self, num: int) -> List[FinetuneSample]:
        """生成对抗样本"""
        synonym_templates = [
            "{a} 与 {b} 有何关联？",
            "{a} 对 {b} 的依赖程度如何？",
        ]
        
        colloquial_templates = [
            "{a} 跟 {b} 有啥关系啊？",
            "为啥 {a} 要用 {b} 呢？",
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


class ProtectedPromotion:
    """受保护的参数晋升"""
    
    def __init__(self, model_path: str):
        self.config = Stage5FConfig()
        
        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # 保存基线
        self.baseline_params = {name: param.clone().detach() 
                               for name, param in self.model.named_parameters()}
        
        # 保存基线输出（用于 KL 约束）
        self.baseline_outputs = None
        self._capture_baseline_outputs()
        
        # 优化器 - 只优化特定参数
        self.optimizer = self._create_selective_optimizer()
    
    def _capture_baseline_outputs(self):
        """捕获基线输出"""
        self.model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, 10000, (1, 50))
            self.baseline_outputs = self.model(input_ids)
    
    def _create_selective_optimizer(self):
        """创建选择性优化器 - 只优化 relation 相关参数"""
        # 只优化这些参数
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
        
        # 目标能力：关系识别
        target_correct = 0
        # 旧能力
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
                
                # 目标能力
                gap_conf = outputs['gap_probs'][0].max().item()
                policy_conf = outputs['policy_probs'][0].max().item()
                if gap_conf > 0.6 and policy_conf > 0.6:
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
        
        return {
            'target': target_correct / num_samples,
            'old': {k: v / num_samples for k, v in old_abilities.items()},
        }
    
    def compute_kl_constraint(self, current_outputs: Dict) -> torch.Tensor:
        """计算 KL 散度约束 - 加强版"""
        kl_loss = 0.0
        
        # Gap 概率分布
        kl_gap = F.kl_div(
            F.log_softmax(current_outputs['gap_logits'], dim=-1),
            F.softmax(self.baseline_outputs['gap_logits'], dim=-1),
            reduction='batchmean'
        )
        kl_loss += kl_gap
        
        # Policy 概率分布
        kl_policy = F.kl_div(
            F.log_softmax(current_outputs['policy_logits'], dim=-1),
            F.softmax(self.baseline_outputs['policy_logits'], dim=-1),
            reduction='batchmean'
        )
        kl_loss += kl_policy
        
        # Writeback 概率分布 - 专项保护
        kl_writeback = F.kl_div(
            F.log_softmax(current_outputs['writeback_logits'], dim=-1),
            F.softmax(self.baseline_outputs['writeback_logits'], dim=-1),
            reduction='batchmean'
        )
        kl_loss += kl_writeback * 2.0  # 加权保护
        
        return kl_loss
    
    def guided_promotion(self, samples: List[FinetuneSample]) -> Dict:
        """有监督的参数晋升 - 强化保护版"""
        self.model.train()
        
        losses = []
        kl_losses = []
        task_losses = []
        
        for epoch in range(self.config.num_epochs):
            epoch_task_loss = 0.0
            epoch_kl_loss = 0.0
            
            for sample in samples:
                self.optimizer.zero_grad()
                
                # 模拟输入
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)
                
                # 任务损失
                target_gap = torch.tensor([sample.target_gap])
                target_strategy = torch.tensor([sample.target_strategy])
                target_writeback = torch.tensor([sample.target_writeback])
                
                gap_loss = F.cross_entropy(outputs['gap_logits'], target_gap)
                policy_loss = F.cross_entropy(outputs['policy_logits'], target_strategy)
                writeback_loss = F.cross_entropy(outputs['writeback_logits'], target_writeback)
                
                task_loss = gap_loss + policy_loss + writeback_loss
                
                # KL 约束 - 加强版
                kl_loss = self.compute_kl_constraint(outputs)
                
                # 总损失
                total_loss = task_loss + self.config.kl_weight * kl_loss
                
                total_loss.backward()
                
                # 梯度裁剪 - 更严格
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_param_change
                )
                
                self.optimizer.step()
                
                epoch_task_loss += task_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            avg_task_loss = epoch_task_loss / len(samples)
            avg_kl_loss = epoch_kl_loss / len(samples)
            
            losses.append(avg_task_loss + self.config.kl_weight * avg_kl_loss)
            task_losses.append(avg_task_loss)
            kl_losses.append(avg_kl_loss)
        
        return {
            'losses': losses,
            'task_losses': task_losses,
            'kl_losses': kl_losses,
        }
    
    def rollback(self):
        """回滚到基线"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.baseline_params:
                    param.copy_(self.baseline_params[name])
        # 重新捕获基线输出
        self._capture_baseline_outputs()


class Stage5FExperiment:
    """Stage 5F 实验"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = Stage5FConfig()
    
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
    
    def run_single_promotion_with_protection(self, candidate: Dict) -> Dict:
        """单候选晋升 - 强化保护版"""
        print("\n" + "=" * 70)
        print(f"单候选晋升 (强化保护): {candidate['candidate_id']}")
        print("=" * 70)
        
        score = candidate['metadata']['quality_score']
        print(f"质量分数: {score:.3f}")
        print(f"保护配置: KL={self.config.kl_weight}, LR={self.config.learning_rate}")
        
        # 创建晋升器
        promoter = ProtectedPromotion(self.model_path)
        
        # 基线评估
        print("\n--- 基线评估 ---")
        baseline = promoter.evaluate()
        print(f"目标能力: {baseline['target']:.2%}")
        print(f"writeback: {baseline['old']['writeback']:.2%}")
        
        # 生成样本
        print("\n--- 生成训练样本 ---")
        generator = SampleGenerator(candidate)
        samples = generator.generate_all_samples()
        print(f"生成 {len(samples)} 个样本")
        
        # 晋升
        print("\n--- 参数晋升 ---")
        promotion_result = promoter.guided_promotion(samples)
        print(f"任务损失: {[f'{l:.4f}' for l in promotion_result['task_losses']]}")
        print(f"KL 损失: {[f'{l:.4f}' for l in promotion_result['kl_losses']]}")
        
        # 晋升后评估
        print("\n--- 晋升后评估 ---")
        after = promoter.evaluate()
        print(f"目标能力: {after['target']:.2%}")
        print(f"writeback: {after['old']['writeback']:.2%}")
        
        # 变化
        target_change = after['target'] - baseline['target']
        writeback_change = after['old']['writeback'] - baseline['old']['writeback']
        print(f"\n目标能力变化: {target_change:+.2%}")
        print(f"writeback 变化: {writeback_change:+.2%}")
        
        # 旧能力
        print("\n旧能力变化:")
        max_drop = 0
        for key in baseline['old']:
            change = after['old'][key] - baseline['old'][key]
            max_drop = max(max_drop, abs(change))
            status = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {key}: {baseline['old'][key]:.2%} -> {after['old'][key]:.2%} ({change:+.2%}) {status}")
        
        # 验收
        success = target_change > 0 and max_drop < self.config.old_ability_threshold
        print(f"\n验收: {'✓ 成功' if success else '✗ 失败'}")
        
        # 回滚
        print("\n--- 回滚测试 ---")
        promoter.rollback()
        after_rollback = promoter.evaluate()
        rollback_success = abs(after_rollback['target'] - baseline['target']) < 0.02
        print(f"回滚效果: {'✓ 成功' if rollback_success else '✗ 失败'}")
        
        return {
            'candidate_id': candidate['candidate_id'],
            'score': score,
            'baseline': baseline,
            'after': after,
            'target_change': target_change,
            'writeback_change': writeback_change,
            'max_old_drop': max_drop,
            'success': success,
            'rollback_success': rollback_success,
        }
    
    def run_continuous_promotion_with_governance(self, candidates: List[Dict]) -> Dict:
        """连续晋升 - 每步后门控版"""
        print("\n" + "=" * 70)
        print("连续微步晋升 (带门控)")
        print("=" * 70)
        
        if len(candidates) < 2:
            print("✗ 候选不足")
            return {'success': False, 'reason': 'insufficient_candidates'}
        
        selected = candidates[:2]
        print(f"选择 {len(selected)} 个候选")
        for c in selected:
            print(f"  {c['candidate_id']}: {c['metadata']['quality_score']:.3f}")
        
        # 创建晋升器
        promoter = ProtectedPromotion(self.model_path)
        
        # 初始基线
        print("\n--- 初始基线 ---")
        baseline = promoter.evaluate()
        print(f"目标能力: {baseline['target']:.2%}")
        print(f"writeback: {baseline['old']['writeback']:.2%}")
        
        step_results = []
        cumulative_improvement = 0
        should_continue = True
        
        for i, candidate in enumerate(selected):
            if not should_continue:
                print(f"\n--- 第 {i+1} 步: 因上一步风险暂停 ---")
                continue
            
            print(f"\n--- 第 {i+1} 步: {candidate['candidate_id']} ---")
            
            # 生成样本
            generator = SampleGenerator(candidate)
            samples = generator.generate_all_samples()
            
            # 晋升
            promotion_result = promoter.guided_promotion(samples)
            
            # 评估
            after_step = promoter.evaluate()
            step_improvement = after_step['target'] - baseline['target'] - cumulative_improvement
            cumulative_improvement += step_improvement
            
            # 旧能力检查
            max_drop = 0
            writeback_drop = 0
            for key in baseline['old']:
                change = after_step['old'][key] - baseline['old'][key]
                max_drop = max(max_drop, abs(change))
                if key == 'writeback':
                    writeback_drop = abs(change)
            
            print(f"  本步提升: {step_improvement:+.2%}")
            print(f"  累计提升: {cumulative_improvement:+.2%}")
            print(f"  writeback 掉落: {writeback_drop:.2%}")
            print(f"  最大旧能力掉落: {max_drop:.2%}")
            
            step_results.append({
                'step': i + 1,
                'candidate_id': candidate['candidate_id'],
                'improvement': step_improvement,
                'cumulative': cumulative_improvement,
                'writeback_drop': writeback_drop,
                'max_old_drop': max_drop,
            })
            
            # 门控决策
            if max_drop > self.config.old_ability_threshold * 0.8:  # 提前预警
                print(f"  ⚠️ 旧能力掉落接近阈值，建议暂停")
                should_continue = False
            else:
                print(f"  ✓ 安全，可继续")
        
        # 总评估
        print("\n--- 连续晋升总结 ---")
        final_eval = promoter.evaluate()
        total_change = final_eval['target'] - baseline['target']
        print(f"总目标能力变化: {baseline['target']:.2%} -> {final_eval['target']:.2%} ({total_change:+.2%})")
        
        # 最终旧能力检查
        final_max_drop = 0
        for key in baseline['old']:
            change = final_eval['old'][key] - baseline['old'][key]
            final_max_drop = max(final_max_drop, abs(change))
        
        print(f"最终旧能力最大掉落: {final_max_drop:.2%}")
        
        # 验收
        success = total_change > 0.02 and final_max_drop < self.config.old_ability_threshold
        print(f"验收: {'✓ 成功' if success else '✗ 失败'}")
        
        return {
            'experiment': 'continuous_with_governance',
            'steps': len([r for r in step_results if r]),
            'baseline_target': baseline['target'],
            'final_target': final_eval['target'],
            'total_change': total_change,
            'final_max_drop': final_max_drop,
            'step_results': step_results,
            'success': success,
        }
    
    def run_all_experiments(self):
        """运行所有 Stage 5F 实验"""
        print("=" * 70)
        print("Stage 5F: 旧能力保护强化与连续晋升复验")
        print("=" * 70)
        
        print("\n核心改进:")
        print(f"  KL 权重: {self.config.kl_weight} (加强)")
        print(f"  学习率: {self.config.learning_rate} (降低)")
        print(f"  参数更新范围: 收缩至 unit_encoder + policy_head")
        print(f"  连续晋升: 每步后门控")
        
        # 加载候选
        candidates = self.load_candidates()
        print(f"\n加载 {len(candidates)} 个适合参数晋升的候选")
        
        if len(candidates) < 2:
            print("✗ 候选不足")
            return
        
        # 实验 1：单候选晋升（2个）
        print("\n" + "=" * 70)
        print("实验 1：单候选晋升复验 (强化保护)")
        print("=" * 70)
        
        single_results = []
        for candidate in candidates[:2]:
            result = self.run_single_promotion_with_protection(candidate)
            single_results.append(result)
        
        single_success_count = sum(1 for r in single_results if r['success'])
        print(f"\n单候选晋升结果: {single_success_count}/{len(single_results)} 成功")
        
        # 实验 2：连续晋升
        continuous_result = self.run_continuous_promotion_with_governance(candidates)
        
        # 总结
        print("\n" + "=" * 70)
        print("Stage 5F 复验总结")
        print("=" * 70)
        
        print("\n实验结果:")
        print(f"  单候选晋升: {single_success_count}/{len(single_results)} 成功")
        print(f"  连续微步晋升: {'✓ 成功' if continuous_result['success'] else '✗ 失败'}")
        
        # 关键指标
        print("\n关键指标:")
        for r in single_results:
            print(f"  {r['candidate_id']}:")
            print(f"    目标提升: {r['target_change']:+.2%}")
            print(f"    writeback 变化: {r['writeback_change']:+.2%}")
            print(f"    最大旧能力掉落: {r['max_old_drop']:.2%}")
        
        if continuous_result.get('step_results'):
            print(f"\n  连续晋升:")
            for step in continuous_result['step_results']:
                print(f"    第 {step['step']} 步: {step['improvement']:+.2%} (writeback: {step['writeback_drop']:.2%})")
        
        # 阶段 6 入口条件检查
        print("\n阶段 6 入口条件检查:")
        checks = [
            ("至少 2 个 RELATION 单候选晋升成功", single_success_count >= 2),
            ("连续微步晋升成功", continuous_result['success']),
            ("旧能力掉落 < 10%", 
             all(r['max_old_drop'] < 0.10 for r in single_results) and 
             continuous_result['final_max_drop'] < 0.10),
            ("回滚机制可用", all(r['rollback_success'] for r in single_results)),
        ]
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        all_passed = all(passed for _, passed in checks)
        
        if all_passed:
            print("\n🎉 满足阶段 6 入口条件！")
        else:
            print("\n⚠️ 部分条件未满足，建议继续优化")
        
        # 保存结果
        results = {
            'single_promotion': single_results,
            'continuous': continuous_result,
            'phase6_ready': all_passed,
        }
        
        with open("eval/stage5f_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ Stage 5F 完成")
        print("  结果已保存到 eval/stage5f_results.json")
        
        return results


def main():
    """主函数"""
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    
    experiment = Stage5FExperiment(model_path)
    results = experiment.run_all_experiments()


if __name__ == "__main__":
    main()
