"""
评估对齐检查

检查实验脚本和主线 orchestrator 的评估差异
"""

import torch
import torch.nn.functional as F
import copy
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage7_output_kl_guard import build_output_kl_guard


def check_evaluation_alignment():
    """检查评估对齐"""
    print("="*70)
    print("评估对齐检查")
    print("="*70)
    
    # 创建系统
    torch.manual_seed(42)
    system = Stage6SystemOrchestrator(
        experiment_id='alignment_check',
        custom_config={**OFFICIAL_BASELINE_V1, 'auto_rollback': False},
    )
    system.establish_baseline()
    
    model = system.orchestrator.backbone.get_model()
    baseline = system.protocol.baseline.scores
    
    print(f"\n[系统基线]")
    print(f"  Target: {baseline.target_score:.4f}")
    print(f"  Writeback: {baseline.writeback_score:.4f}")
    print(f"  Retrieval: {baseline.retrieval_score:.4f}")
    print(f"  Policy: {baseline.policy_score:.4f}")
    print(f"  Governance: {baseline.governance_score:.4f}")
    
    # 使用实验脚本的方式计算 WB Score
    torch.manual_seed(42)
    anchor_inputs = [torch.randint(0, 10000, (1, 50)) for _ in range(30)]
    
    model.eval()
    with torch.no_grad():
        wb_scores = []
        for input_ids in anchor_inputs:
            outputs = model(input_ids)
            probs = F.softmax(outputs['writeback_logits'], dim=-1)
            score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
            wb_scores.append(score)
        exp_wb_score = sum(wb_scores) / len(wb_scores)
    
    print(f"\n[实验脚本计算]")
    print(f"  WB Score (30 samples): {exp_wb_score:.4f}")
    print(f"  与系统基线差异: {exp_wb_score - baseline.writeback_score:+.4f}")
    
    # 检查 Output KL Guard 的参考输出
    guard = build_output_kl_guard(model, beta=0.2, num_samples=30, use_probs=True)
    
    print(f"\n[Output KL Guard]")
    print(f"  参考输出形状: {guard.ref_outputs.shape}")
    print(f"  参考输出均值: {guard.ref_outputs.mean().item():.4f}")
    print(f"  参考输出标准差: {guard.ref_outputs.std().item():.4f}")
    
    # 计算当前输出漂移
    drift = guard.compute_output_drift()
    print(f"  当前 Output Drift: {drift:.6f}")
    
    # 关键发现
    print(f"\n[关键发现]")
    if abs(exp_wb_score - baseline.writeback_score) > 0.1:
        print("  ⚠ 实验脚本与系统基线 WB Score 差异过大!")
        print("  可能原因:")
        print("    - 锚点样本生成方式不同")
        print("    - 评估时模型状态不同")
        print("    - 基线建立时机问题")
    else:
        print("  ✓ 实验脚本与系统基线 WB Score 一致")
    
    return {
        'system_baseline_wb': baseline.writeback_score,
        'exp_wb_score': exp_wb_score,
        'difference': exp_wb_score - baseline.writeback_score,
    }


if __name__ == "__main__":
    results = check_evaluation_alignment()
