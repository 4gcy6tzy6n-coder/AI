"""
Dual Guard 参数搜索

目标: 找到 feature_guard_beta 和 head_lr_ratio 的最优组合

搜索策略:
1. 第一轮: 固定 head_lr_ratio=0.3, 搜索 beta [0.01, 0.02, 0.03, 0.04, 0.05]
2. 第二轮: 固定最优 beta, 搜索 head_lr_ratio [0.5, 1.0, 1.5, 2.0]

判优标准:
- Score Δ 越接近 0 越好 (但允许小幅正向)
- Output KL 越低越好
- Target > 10%
- Old Ability < 15%
- 优先选择 dual > head_only 的配置
"""

import torch
import torch.nn.functional as F
import copy
import hashlib
from typing import Dict, List, Tuple
from dataclasses import dataclass
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage7_dual_guard_orchestrator import DualGuardPromoter, DualGuardConfig


def compute_model_hash(model) -> str:
    """计算模型参数哈希"""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().tobytes())
    combined = b''.join(params)
    return hashlib.md5(combined).hexdigest()[:16]


class FixedSampleEvaluator:
    """固定样本评估器"""
    
    def __init__(self, num_samples: int = 30, seed: int = 42):
        torch.manual_seed(seed)
        self.fixed_samples = [
            torch.randint(0, 10000, (1, 50))
            for _ in range(num_samples)
        ]
    
    def evaluate(self, model) -> Dict:
        """评估writeback能力"""
        model.eval()
        scores = []
        all_logits = []
        
        with torch.no_grad():
            for input_ids in self.fixed_samples:
                outputs = model(input_ids)
                wb_logits = outputs['writeback_logits']
                wb_probs = F.softmax(wb_logits, dim=-1)
                
                score = wb_probs[0, 1].item() if wb_probs.shape[1] > 1 else wb_probs[0, 0].item()
                scores.append(score)
                all_logits.append(wb_logits.clone())
        
        return {
            'writeback_score': sum(scores) / len(scores),
            'all_logits': all_logits,
        }


@dataclass
class SearchResult:
    """搜索结果"""
    beta: float
    head_lr_ratio: float
    score_change: float
    output_kl: float
    target_gain: float
    old_ability_drop: float
    writeback_change: float
    
    def __str__(self):
        return (f"beta={self.beta:.2f}, lr_ratio={self.head_lr_ratio:.1f} | "
                f"Score Δ={self.score_change:+.4f}, KL={self.output_kl:.6f} | "
                f"Target={self.target_gain:+.2%}, Old={self.old_ability_drop:.2%}")
    
    def is_valid(self) -> bool:
        """检查是否满足约束"""
        return self.target_gain > 0.10 and self.old_ability_drop < 0.15
    
    def score(self) -> float:
        """
        综合评分 (越高越好)
        
        评分逻辑:
        - Score Δ 接近 0 得分高 (但允许小幅正向)
        - KL 低得分高
        - 必须满足约束
        """
        if not self.is_valid():
            return -1000
        
        # Score Δ: 越接近 0 越好, 小幅正向 (+0.05~0.10) 也可以接受
        if self.score_change < 0:
            score_delta_penalty = abs(self.score_change) * 2  # 负向惩罚
        elif self.score_change < 0.10:
            score_delta_penalty = 0  # 小幅正向不惩罚
        else:
            score_delta_penalty = (self.score_change - 0.10) * 5  # 大幅正向也惩罚
        
        # KL: 越低越好
        kl_score = -self.output_kl * 10
        
        return 100 - score_delta_penalty + kl_score


def run_single_config(
    beta: float,
    head_lr_ratio: float,
    base_state: Dict,
    base_system: Stage6SystemOrchestrator,
    num_steps: int = 50,
) -> SearchResult:
    """运行单组配置实验"""
    
    print(f"\n[Config] beta={beta:.2f}, head_lr_ratio={head_lr_ratio:.1f}")
    
    # 重置模型
    model = base_system.orchestrator.backbone.get_model()
    model.load_state_dict(base_state)
    for param in model.parameters():
        param.requires_grad = True
    
    # 评估器
    evaluator = FixedSampleEvaluator()
    
    # 训练前评估
    before_eval = evaluator.evaluate(model)
    
    # 创建 Dual Guard Promoter
    dual_config = DualGuardConfig(
        feature_guard_beta=beta,
        head_lr_ratio=head_lr_ratio,
        lambda_wb=0.1,
        num_epochs=3,
        learning_rate=1e-4,
    )
    
    promoter = DualGuardPromoter(
        model,
        dual_config,
        enable_feature_guard=True,
        enable_head_training=True,
    )
    
    # 训练
    for step in range(num_steps):
        model.train()
        
        for epoch in range(3):
            promoter.optimizer.zero_grad()
            
            torch.manual_seed(42 + step * 100 + epoch)
            input_ids = torch.randint(0, 10000, (1, 50))
            
            outputs = model(input_ids)
            
            # 主损失
            main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
            
            # Writeback损失
            writeback_probs = outputs['writeback_probs']
            gap_decision = outputs['gap_probs'][0, 1]
            policy_decision = outputs['policy_probs'][0].max()
            writeback_target = torch.tensor(
                1.0 if (gap_decision > 0.5 and policy_decision > 0.5) else 0.0
            )
            wb_loss = F.binary_cross_entropy(
                writeback_probs[0, 1:2],
                writeback_target.unsqueeze(0)
            )
            
            # Feature Guard损失
            guard_loss = promoter.feature_guard.compute_feature_guard_loss()
            
            # 总损失
            total_loss = main_loss + 0.1 * wb_loss + guard_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            promoter.optimizer.step()
    
    # 训练后评估
    after_eval = evaluator.evaluate(model)
    score_change = after_eval['writeback_score'] - before_eval['writeback_score']
    
    # 计算 KL
    total_kl = 0
    for before_logits, after_logits in zip(before_eval['all_logits'], after_eval['all_logits']):
        kl = F.kl_div(
            F.log_softmax(after_logits, dim=-1),
            F.softmax(before_logits, dim=-1),
            reduction='batchmean'
        ).item()
        total_kl += kl
    avg_kl = total_kl / len(before_eval['all_logits'])
    
    # 系统级评估
    eval_result = base_system.protocol.evaluate_with_protocol(
        f'search_beta{beta}_lr{head_lr_ratio}', num_samples=30
    )
    baseline = base_system.protocol.baseline.scores
    
    result = SearchResult(
        beta=beta,
        head_lr_ratio=head_lr_ratio,
        score_change=score_change,
        output_kl=avg_kl,
        target_gain=eval_result.scores.target_score - baseline.target_score,
        old_ability_drop=max(
            abs(eval_result.scores.retrieval_score - baseline.retrieval_score),
            abs(eval_result.scores.policy_score - baseline.policy_score),
        ),
        writeback_change=abs(eval_result.scores.writeback_score - baseline.writeback_score),
    )
    
    print(f"  Result: {result}")
    print(f"  Valid: {result.is_valid()}, Score: {result.score():.2f}")
    
    return result


def search_beta_values(base_state, base_system, num_steps=50) -> List[SearchResult]:
    """第一轮: 搜索 beta 值"""
    print("\n" + "="*70)
    print("第一轮: 搜索 feature_guard_beta")
    print("="*70)
    print("固定 head_lr_ratio=0.3, 搜索 beta [0.01, 0.02, 0.03, 0.04, 0.05]")
    
    beta_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    head_lr_ratio = 0.3
    
    results = []
    for beta in beta_values:
        result = run_single_config(beta, head_lr_ratio, base_state, base_system, num_steps)
        results.append(result)
    
    return results


def search_lr_ratio_values(best_beta: float, base_state, base_system, num_steps=50) -> List[SearchResult]:
    """第二轮: 搜索 head_lr_ratio 值"""
    print("\n" + "="*70)
    print("第二轮: 搜索 head_lr_ratio")
    print("="*70)
    print(f"固定 beta={best_beta:.2f}, 搜索 head_lr_ratio [0.5, 1.0, 1.5, 2.0]")
    
    lr_ratios = [0.5, 1.0, 1.5, 2.0]
    
    results = []
    for lr_ratio in lr_ratios:
        result = run_single_config(best_beta, lr_ratio, base_state, base_system, num_steps)
        results.append(result)
    
    return results


def print_comparison_table(results: List[SearchResult], title: str):
    """打印对比表"""
    print(f"\n{title}")
    print("-" * 100)
    print(f"{'Beta':<6} {'LR Ratio':<10} {'Score Δ':<10} {'Output KL':<12} {'Target':<10} {'Old Ability':<12} {'Valid':<8} {'Score':<8}")
    print("-" * 100)
    
    for r in results:
        valid_mark = "✓" if r.is_valid() else "✗"
        print(f"{r.beta:<6.2f} {r.head_lr_ratio:<10.1f} {r.score_change:>+8.4f}  {r.output_kl:>10.6f}  "
              f"{r.target_gain:>+8.2%}  {r.old_ability_drop:>10.2%}  {valid_mark:<8} {r.score():>7.2f}")


def main():
    """主搜索流程"""
    print("="*70)
    print("Dual Guard 参数搜索")
    print("="*70)
    
    # Step 1: 创建统一基线
    print("\n" + "="*70)
    print("Step 1: 创建统一基线模型")
    print("="*70)
    
    torch.manual_seed(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    base_system = Stage6SystemOrchestrator(
        experiment_id='dual_guard_param_search',
        custom_config=config,
    )
    base_system.establish_baseline()
    
    base_state = base_system.orchestrator.backbone.get_model().state_dict()
    print(f"基线模型已创建: {compute_model_hash(base_system.orchestrator.backbone.get_model())}")
    
    # 先跑对照组
    print("\n" + "="*70)
    print("对照组: head_comp_only (beta=0, lr_ratio=0.3)")
    print("="*70)
    head_only_result = run_single_config(0.0, 0.3, base_state, base_system, num_steps=50)
    
    # 第一轮: 搜索 beta
    beta_results = search_beta_values(base_state, base_system, num_steps=50)
    print_comparison_table(beta_results, "Beta 搜索结果")
    
    # 找到最优 beta
    valid_beta_results = [r for r in beta_results if r.is_valid()]
    if valid_beta_results:
        best_beta_result = max(valid_beta_results, key=lambda r: r.score())
        best_beta = best_beta_result.beta
        print(f"\n最优 beta: {best_beta:.2f} (Score: {best_beta_result.score():.2f})")
    else:
        print("\n警告: 没有满足约束的 beta 值, 使用默认 0.02")
        best_beta = 0.02
    
    # 第二轮: 搜索 head_lr_ratio
    lr_results = search_lr_ratio_values(best_beta, base_state, base_system, num_steps=50)
    print_comparison_table(lr_results, "LR Ratio 搜索结果")
    
    # 找到最优组合
    all_results = beta_results + lr_results + [head_only_result]
    valid_results = [r for r in all_results if r.is_valid()]
    
    if valid_results:
        best_result = max(valid_results, key=lambda r: r.score())
        
        print("\n" + "="*70)
        print("最优配置")
        print("="*70)
        print(f"Feature Guard Beta: {best_result.beta:.2f}")
        print(f"Head LR Ratio: {best_result.head_lr_ratio:.1f}")
        print(f"Score Change: {best_result.score_change:+.4f}")
        print(f"Output KL: {best_result.output_kl:.6f}")
        print(f"Target Gain: {best_result.target_gain:+.2%}")
        print(f"Old Ability Drop: {best_result.old_ability_drop:.2%}")
        print(f"综合评分: {best_result.score():.2f}")
        
        # 与 head_only 对比
        if best_result.score() > head_only_result.score():
            print(f"\n✓ Dual Guard 优于 head_only!")
            print(f"  提升: {best_result.score() - head_only_result.score():.2f} 分")
        else:
            print(f"\n⚠ Dual Guard 未超过 head_only")
            print(f"  建议使用 head_only 作为临时主候选")
            print(f"  或继续调整参数")
    else:
        print("\n警告: 没有找到满足约束的配置")
        print("建议使用 head_comp_only 作为临时主候选")
    
    # 最终建议
    print("\n" + "="*70)
    print("最终建议")
    print("="*70)
    print("1. 当前临时主候选: head_comp_only")
    print("   - Score Δ: +0.1052")
    print("   - 无 feature guard 约束, head 自由补偿")
    
    if valid_results and best_result.beta > 0:
        print(f"\n2. Dual Guard 候选配置:")
        print(f"   - beta={best_result.beta:.2f}, head_lr_ratio={best_result.head_lr_ratio:.1f}")
        print(f"   - Score Δ: {best_result.score_change:+.4f}")
        if best_result.score() > head_only_result.score():
            print(f"   - 已优于 head_only, 可作为正式方案")
        else:
            print(f"   - 需继续调优以超过 head_only")
    
    print(f"\n3. Feature Guard 安全对照:")
    print(f"   - beta=0.05 时 Score Δ=-0.0198")
    print(f"   - 可作为稳定性上限参考")


if __name__ == "__main__":
    main()
