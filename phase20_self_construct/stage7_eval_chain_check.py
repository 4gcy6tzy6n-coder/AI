"""
评估链路排查实验

目标: 确认 writeback_change 指标的可信度

排查项:
1. 同模型重复评估 - 同一个 checkpoint 评估5次
2. 固定评估样本 - 使用相同样本ID
3. 指标计算入口检查 - 确认 before/after 对比对象
4. 前后相同输入输出对比 - 直接对比 logits
"""

import torch
import torch.nn.functional as F
import copy
import random
import numpy as np
from typing import Dict, List
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1


def set_all_seeds(seed: int = 42):
    """固定所有随机性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FixedSampleEvaluator:
    """使用固定样本的评估器"""
    
    def __init__(self, system, num_samples: int = 30):
        self.system = system
        self.num_samples = num_samples
        self.fixed_samples = None
        
    def generate_fixed_samples(self):
        """生成并固定评估样本"""
        set_all_seeds(42)
        # 生成固定样本ID
        self.fixed_samples = [
            {
                'input_ids': torch.randint(0, 10000, (1, 50)),
                'sample_id': i,
            }
            for i in range(self.num_samples)
        ]
        print(f"  固定评估样本已生成: {len(self.fixed_samples)} 个")
        return self.fixed_samples
    
    def evaluate_with_fixed_samples(self, model) -> Dict:
        """使用固定样本评估"""
        if self.fixed_samples is None:
            self.generate_fixed_samples()
        
        model.eval()
        writeback_scores = []
        all_outputs = []
        
        with torch.no_grad():
            for sample in self.fixed_samples:
                input_ids = sample['input_ids']
                outputs = model(input_ids)
                
                # 记录 writeback 输出
                wb_logits = outputs['writeback_logits']
                wb_probs = F.softmax(wb_logits, dim=-1)
                
                # 简单评分: 取正类的概率
                score = wb_probs[0, 1].item() if wb_probs.shape[1] > 1 else wb_probs[0, 0].item()
                writeback_scores.append(score)
                all_outputs.append(wb_logits.clone())
        
        avg_score = sum(writeback_scores) / len(writeback_scores)
        
        return {
            'writeback_score': avg_score,
            'all_outputs': all_outputs,
            'individual_scores': writeback_scores,
        }


def check_1_repeat_eval():
    """
    排查1: 同模型重复评估
    同一个 checkpoint，不训练，连续评估5次
    """
    print("\n" + "="*70)
    print("排查1: 同模型重复评估实验")
    print("="*70)
    
    set_all_seeds(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id='repeat_eval_check',
        custom_config=config,
    )
    system.establish_baseline()
    
    model = system.orchestrator.backbone.get_model()
    evaluator = FixedSampleEvaluator(system)
    evaluator.generate_fixed_samples()
    
    # 连续评估5次
    results = []
    for i in range(5):
        set_all_seeds(42 + i)  # 每次不同种子，看随机性影响
        result = evaluator.evaluate_with_fixed_samples(model)
        results.append(result)
        print(f"  评估 #{i+1}: writeback_score = {result['writeback_score']:.4f}")
    
    # 分析变异
    scores = [r['writeback_score'] for r in results]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score)**2 for s in scores) / len(scores)
    
    print(f"\n[统计]")
    print(f"  均值: {mean_score:.4f}")
    print(f"  方差: {variance:.6f}")
    print(f"  最大差: {max(scores) - min(scores):.4f}")
    
    if variance < 1e-6:
        print("  ✓ 评估结果稳定 (方差接近0)")
    else:
        print("  ✗ 评估结果存在随机性")
    
    return results


def check_2_fixed_seed_eval():
    """
    排查2: 固定种子重复评估
    同一模型，同一固定种子，评估5次
    """
    print("\n" + "="*70)
    print("排查2: 固定种子重复评估")
    print("="*70)
    
    set_all_seeds(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id='fixed_seed_check',
        custom_config=config,
    )
    system.establish_baseline()
    
    model = system.orchestrator.backbone.get_model()
    evaluator = FixedSampleEvaluator(system)
    evaluator.generate_fixed_samples()
    
    # 用同一固定种子评估5次
    results = []
    for i in range(5):
        set_all_seeds(42)  # 固定同一种子
        result = evaluator.evaluate_with_fixed_samples(model)
        results.append(result)
        print(f"  评估 #{i+1}: writeback_score = {result['writeback_score']:.4f}")
    
    # 分析
    scores = [r['writeback_score'] for r in results]
    
    if len(set(f"{s:.6f}" for s in scores)) == 1:
        print("\n  ✓ 固定种子下评估结果完全一致")
    else:
        print(f"\n  ✗ 即使固定种子，结果仍有差异")
        print(f"    差异: {max(scores) - min(scores):.6f}")
    
    return results


def check_3_before_after_consistency():
    """
    排查3: 训练前后相同输入输出对比
    使用完全相同的样本，对比训练前后的输出
    """
    print("\n" + "="*70)
    print("排查3: 训练前后相同输入输出对比")
    print("="*70)
    
    set_all_seeds(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id='before_after_check',
        custom_config=config,
    )
    system.establish_baseline()
    
    model = system.orchestrator.backbone.get_model()
    evaluator = FixedSampleEvaluator(system)
    
    # 训练前评估
    print("\n  [训练前评估]")
    set_all_seeds(42)
    before_result = evaluator.evaluate_with_fixed_samples(model)
    print(f"    writeback_score: {before_result['writeback_score']:.4f}")
    
    # 执行一次训练 (但不冻结任何参数，正常训练)
    print("\n  [执行1步训练]")
    model.train()
    
    # 确保 param_promoter 已初始化
    if system.orchestrator.param_promoter is None:
        from stage6_runtime_orchestrator import ParamPromoter
        system.orchestrator.param_promoter = ParamPromoter(model, system.orchestrator.config)
    
    optimizer = system.orchestrator.param_promoter.optimizer
    
    for epoch in range(3):
        optimizer.zero_grad()
        input_ids = torch.randint(0, 10000, (1, 50))
        outputs = model(input_ids)
        loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
        loss.backward()
        optimizer.step()
    
    # 训练后评估
    print("\n  [训练后评估]")
    set_all_seeds(42)  # 固定同一种子
    after_result = evaluator.evaluate_with_fixed_samples(model)
    print(f"    writeback_score: {after_result['writeback_score']:.4f}")
    
    # 对比
    score_diff = after_result['writeback_score'] - before_result['writeback_score']
    print(f"\n  [对比]")
    print(f"    分数变化: {score_diff:+.4f}")
    
    # 对比每个样本的输出
    output_diffs = []
    for i, (before_out, after_out) in enumerate(zip(before_result['all_outputs'], after_result['all_outputs'])):
        diff = (after_out - before_out).norm().item()
        output_diffs.append(diff)
    
    avg_output_diff = sum(output_diffs) / len(output_diffs)
    print(f"    平均输出L2差异: {avg_output_diff:.6f}")
    
    if avg_output_diff < 1e-6:
        print("    ✗ 训练后输出没有变化 (可能训练未生效)")
    else:
        print("    ✓ 训练确实改变了模型输出")
    
    return before_result, after_result


def check_4_eval_vs_protocol():
    """
    排查4: 对比自定义评估 vs 系统协议评估
    检查两者结果是否一致
    """
    print("\n" + "="*70)
    print("排查4: 自定义评估 vs 系统协议评估")
    print("="*70)
    
    set_all_seeds(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    system = Stage6SystemOrchestrator(
        experiment_id='eval_protocol_check',
        custom_config=config,
    )
    system.establish_baseline()
    
    model = system.orchestrator.backbone.get_model()
    
    # 系统协议评估
    print("\n  [系统协议评估]")
    protocol_result = system.protocol.evaluate_with_protocol('check', num_samples=30)
    protocol_wb = protocol_result.scores.writeback_score
    print(f"    writeback_score: {protocol_wb:.4f}")
    
    # 自定义固定样本评估
    print("\n  [自定义固定样本评估]")
    evaluator = FixedSampleEvaluator(system, num_samples=30)
    custom_result = evaluator.evaluate_with_fixed_samples(model)
    custom_wb = custom_result['writeback_score'] * 100  # 转换为百分比
    print(f"    writeback_score: {custom_wb:.4f}")
    
    # 对比
    diff = abs(protocol_wb - custom_wb)
    print(f"\n  [对比]")
    print(f"    绝对差异: {diff:.4f}")
    
    if diff < 1.0:  # 1%以内
        print("    ✓ 两种评估方式结果接近")
    else:
        print("    ✗ 两种评估方式差异较大")
        print("    可能原因:")
        print("      - 评估样本不同")
        print("      - 评分方式不同")
        print("      - 随机性未固定")
    
    return protocol_result, custom_result


def check_5_baseline_consistency():
    """
    排查5: 基线一致性检查
    确认多次建立基线是否得到相同结果
    """
    print("\n" + "="*70)
    print("排查5: 基线一致性检查")
    print("="*70)
    
    baseline_scores = []
    
    for i in range(3):
        print(f"\n  [建立基线 #{i+1}]")
        set_all_seeds(42)
        
        config = copy.deepcopy(OFFICIAL_BASELINE_V1)
        config['auto_rollback'] = False
        
        system = Stage6SystemOrchestrator(
            experiment_id=f'baseline_check_{i}',
            custom_config=config,
        )
        system.establish_baseline()
        
        scores = system.protocol.baseline.scores
        baseline_scores.append({
            'target': scores.target_score,
            'writeback': scores.writeback_score,
        })
        
        print(f"    target: {scores.target_score:.2f}%")
        print(f"    writeback: {scores.writeback_score:.2f}%")
    
    # 分析一致性
    targets = [s['target'] for s in baseline_scores]
    writebacks = [s['writeback'] for s in baseline_scores]
    
    print(f"\n  [统计]")
    print(f"    target 范围: {min(targets):.2f}% - {max(targets):.2f}%")
    print(f"    writeback 范围: {min(writebacks):.2f}% - {max(writebacks):.2f}%")
    
    if max(targets) - min(targets) < 0.1 and max(writebacks) - min(writebacks) < 0.1:
        print("    ✓ 基线建立结果一致")
    else:
        print("    ✗ 基线建立结果不一致")
    
    return baseline_scores


def main():
    """主排查流程"""
    print("="*70)
    print("评估链路排查")
    print("="*70)
    
    # 执行5项排查
    check_1_repeat_eval()
    check_2_fixed_seed_eval()
    check_3_before_after_consistency()
    check_4_eval_vs_protocol()
    check_5_baseline_consistency()
    
    # 汇总结论
    print("\n" + "="*70)
    print("排查结论汇总")
    print("="*70)
    print("""
根据以上5项排查，可以判断:

1. 如果排查1显示方差 > 0 → 评估过程存在随机性
2. 如果排查2显示即使固定种子仍有差异 → 存在非确定性因素
3. 如果排查3显示训练后输出无变化 → 训练未生效
4. 如果排查4显示两种评估差异大 → 评估方式不统一
5. 如果排查5显示基线不一致 → 初始化或评估有随机性

关键判断标准:
- writeback_change 指标只有在"训练确实改变输出"且"评估稳定"时才有意义
- 如果全冻结(C组)仍有12.60%变化，说明问题在评估链路
""")


if __name__ == "__main__":
    main()
