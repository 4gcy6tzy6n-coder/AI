"""
正确的 Writeback 变化来源拆解实验

核心原则:
1. 同一初始模型 checkpoint
2. 同一份固定评估样本
3. 同一随机种子
4. 唯一变化因素只有冻结/训练策略

4组实验 (从同一基线分叉):
A: freeze head, train backbone
B: freeze backbone, train head
C: freeze both (控制组)
D: train both (对照组)
"""

import torch
import torch.nn.functional as F
import copy
import hashlib
from typing import Dict, List
from stage6_system_orchestrator import Stage6SystemOrchestrator, OFFICIAL_BASELINE_V1
from stage6_runtime_orchestrator import ParamPromoter


def compute_model_hash(model) -> str:
    """计算模型参数哈希"""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().tobytes())
    combined = b''.join(params)
    return hashlib.md5(combined).hexdigest()[:16]


def compute_writeback_head_hash(model) -> str:
    """计算 writeback head 参数哈希"""
    params = []
    for name, param in model.named_parameters():
        if 'writeback' in name.lower():
            params.append(param.data.cpu().numpy().tobytes())
    if not params:
        return "no_writeback_head"
    combined = b''.join(params)
    return hashlib.md5(combined).hexdigest()[:16]


class CorrectSourceAnalysis:
    """正确的来源分析器"""
    
    def __init__(self, base_system: Stage6SystemOrchestrator):
        self.base_system = base_system
        self.model = base_system.orchestrator.backbone.get_model()
        self.config = base_system.orchestrator.config
        self.baseline_scores = base_system.protocol.baseline.scores
        
        # 固定评估样本
        self.fixed_samples = self._generate_fixed_samples(30)
        
    def _generate_fixed_samples(self, num_samples: int) -> List[Dict]:
        """生成固定评估样本"""
        torch.manual_seed(42)
        samples = []
        for i in range(num_samples):
            samples.append({
                'input_ids': torch.randint(0, 10000, (1, 50)),
                'sample_id': i,
            })
        return samples
    
    def evaluate_fixed(self, model) -> Dict:
        """使用固定样本评估"""
        model.eval()
        writeback_scores = []
        all_logits = []
        
        with torch.no_grad():
            for sample in self.fixed_samples:
                outputs = model(sample['input_ids'])
                wb_logits = outputs['writeback_logits']
                wb_probs = F.softmax(wb_logits, dim=-1)
                
                score = wb_probs[0, 1].item() if wb_probs.shape[1] > 1 else wb_probs[0, 0].item()
                writeback_scores.append(score)
                all_logits.append(wb_logits.clone())
        
        return {
            'writeback_score': sum(writeback_scores) / len(writeback_scores),
            'all_logits': all_logits,
            'individual_scores': writeback_scores,
        }
    
    def run_experiment(
        self,
        exp_name: str,
        freeze_backbone: bool,
        freeze_head: bool,
        num_steps: int
    ) -> Dict:
        """
        运行单组实验
        
        关键: 从同一基线模型开始，不重新初始化
        """
        print(f"\n{'='*70}")
        print(f"实验: {exp_name}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Freeze head: {freeze_head}")
        print(f"  Steps: {num_steps}")
        print(f"{'='*70}")
        
        # 强制校验1: 初始哈希
        initial_model_hash = compute_model_hash(self.model)
        initial_wb_hash = compute_writeback_head_hash(self.model)
        print(f"\n[强制校验1: 初始模型哈希]")
        print(f"  Model hash: {initial_model_hash}")
        print(f"  Writeback head hash: {initial_wb_hash}")
        
        # 强制校验2: 评估输入
        print(f"\n[强制校验2: 评估输入]")
        print(f"  样本数量: {len(self.fixed_samples)}")
        print(f"  样本ID范围: 0-{len(self.fixed_samples)-1}")
        
        # 训练前评估
        print(f"\n[训练前评估]")
        before_eval = self.evaluate_fixed(self.model)
        print(f"  Writeback score: {before_eval['writeback_score']:.4f}")
        
        # 应用冻结策略
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'writeback' not in name.lower():
                    param.requires_grad = False
            print(f"\n[策略应用] Backbone 已冻结")
        
        if freeze_head:
            for name, param in self.model.named_parameters():
                if 'writeback' in name.lower():
                    param.requires_grad = False
            print(f"[策略应用] Writeback head 已冻结")
        
        # 创建优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if trainable_params:
            optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)
            print(f"[优化器] {len(trainable_params)} 个参数可训练")
        else:
            optimizer = None
            print(f"[优化器] 无参数可训练 (全冻结)")
        
        # 训练循环
        print(f"\n[训练 {num_steps} steps]")
        for step in range(num_steps):
            if optimizer is None:
                print(f"  Step {step+1}: 跳过 (无参数可训练)")
                continue
            
            self.model.train()
            for epoch in range(self.config.num_epochs):
                optimizer.zero_grad()
                
                # 使用固定种子生成训练数据
                torch.manual_seed(42 + step * 100 + epoch)
                input_ids = torch.randint(0, 10000, (1, 50))
                
                outputs = self.model(input_ids)
                
                # 计算损失
                main_loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
                
                # 如果head未冻结，加入writeback损失
                if not freeze_head:
                    writeback_probs = outputs['writeback_probs']
                    gap_decision = outputs['gap_probs'][0, 1]
                    policy_decision = outputs['policy_probs'][0].max()
                    writeback_target = torch.tensor(1.0 if (gap_decision > 0.5 and policy_decision > 0.5) else 0.0)
                    wb_loss = F.binary_cross_entropy(writeback_probs[0, 1:2], writeback_target.unsqueeze(0))
                    total_loss = main_loss + 0.1 * wb_loss
                else:
                    total_loss = main_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            if (step + 1) % 5 == 0 or step == 0:
                print(f"  Step {step+1} 完成")
        
        # 训练后评估
        print(f"\n[训练后评估]")
        after_eval = self.evaluate_fixed(self.model)
        print(f"  Writeback score: {after_eval['writeback_score']:.4f}")
        
        # 计算变化
        score_change = after_eval['writeback_score'] - before_eval['writeback_score']
        
        # 计算输出漂移 (KL散度)
        total_kl = 0
        for before_logits, after_logits in zip(before_eval['all_logits'], after_eval['all_logits']):
            kl = F.kl_div(
                F.log_softmax(after_logits, dim=-1),
                F.softmax(before_logits, dim=-1),
                reduction='batchmean'
            ).item()
            total_kl += kl
        avg_kl = total_kl / len(before_eval['all_logits'])
        
        # 计算参数变化
        param_delta = 0
        wb_param_delta = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # 只统计训练过的参数
                delta = param.norm().item()
                param_delta += delta
                if 'writeback' in name.lower():
                    wb_param_delta += delta
        
        result = {
            'exp_name': exp_name,
            'freeze_backbone': freeze_backbone,
            'freeze_head': freeze_head,
            'num_steps': num_steps,
            'initial_model_hash': initial_model_hash,
            'initial_wb_hash': initial_wb_hash,
            'before_score': before_eval['writeback_score'],
            'after_score': after_eval['writeback_score'],
            'score_change': score_change,
            'output_kl': avg_kl,
            'param_delta': param_delta,
            'wb_param_delta': wb_param_delta,
        }
        
        print(f"\n[结果汇总]")
        print(f"  Score change: {score_change:+.4f}")
        print(f"  Output KL: {avg_kl:.6f}")
        print(f"  Param delta: {param_delta:.6f}")
        print(f"  WB param delta: {wb_param_delta:.6f}")
        
        return result


def run_correct_analysis(num_steps: int = 10):
    """运行正确的来源分析"""
    print("="*70)
    print(f"正确的 Writeback 来源拆解 ({num_steps} steps)")
    print("="*70)
    
    # Step 1: 创建统一基线模型
    print("\n" + "="*70)
    print("Step 1: 创建统一基线模型")
    print("="*70)
    
    torch.manual_seed(42)
    config = copy.deepcopy(OFFICIAL_BASELINE_V1)
    config['auto_rollback'] = False
    
    base_system = Stage6SystemOrchestrator(
        experiment_id=f'correct_source_base_{num_steps}',
        custom_config=config,
    )
    base_system.establish_baseline()
    
    # 保存基线模型状态
    base_state = base_system.orchestrator.backbone.get_model().state_dict()
    print(f"\n基线模型已创建")
    print(f"  Model hash: {compute_model_hash(base_system.orchestrator.backbone.get_model())}")
    print(f"  WB hash: {compute_writeback_head_hash(base_system.orchestrator.backbone.get_model())}")
    
    # Step 2: 从同一基线分叉4组实验
    print("\n" + "="*70)
    print("Step 2: 从同一基线分叉4组实验")
    print("="*70)
    
    experiments = [
        ('A_freeze_head', False, True),
        ('B_freeze_backbone', True, False),
        ('C_freeze_both', True, True),
        ('D_train_both', False, False),
    ]
    
    results = []
    
    for exp_name, freeze_bb, freeze_head in experiments:
        # 重置模型到基线状态
        base_system.orchestrator.backbone.get_model().load_state_dict(base_state)
        
        # 重置所有参数的 requires_grad
        for param in base_system.orchestrator.backbone.get_model().parameters():
            param.requires_grad = True
        
        # 创建分析器并运行实验
        analyzer = CorrectSourceAnalysis(base_system)
        result = analyzer.run_experiment(exp_name, freeze_bb, freeze_head, num_steps)
        results.append(result)
    
    # 结果对比
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    
    print(f"\n{'实验':<20} {'Freeze BB':<12} {'Freeze Head':<12} {'Score Δ':<10} {'Output KL':<10} {'WB Param Δ':<10}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['exp_name']:<20} {str(r['freeze_backbone']):<12} {str(r['freeze_head']):<12} "
              f"{r['score_change']:>+8.4f}  {r['output_kl']:>8.6f}  {r['wb_param_delta']:>8.4f}")
    
    # 来源分析
    print("\n" + "="*70)
    print("来源分析")
    print("="*70)
    
    exp_a = next(r for r in results if r['exp_name'] == 'A_freeze_head')
    exp_b = next(r for r in results if r['exp_name'] == 'B_freeze_backbone')
    exp_c = next(r for r in results if r['exp_name'] == 'C_freeze_both')
    exp_d = next(r for r in results if r['exp_name'] == 'D_train_both')
    
    print("\n1. 控制组验证 (C: freeze both):")
    print(f"   Score change: {exp_c['score_change']:+.4f}")
    print(f"   Output KL: {exp_c['output_kl']:.6f}")
    if abs(exp_c['score_change']) < 0.01 and exp_c['output_kl'] < 1e-6:
        print("   ✓ 控制组无变化，实验设计正确")
    else:
        print("   ✗ 控制组有变化，存在问题")
    
    print("\n2. Head vs Backbone 影响对比:")
    print(f"   A (freeze head): score_change = {exp_a['score_change']:+.4f}")
    print(f"   B (freeze backbone): score_change = {exp_b['score_change']:+.4f}")
    
    if abs(exp_a['score_change']) > abs(exp_b['score_change']):
        print("   → Backbone 训练对 writeback 影响更大")
    else:
        print("   → Head 训练对 writeback 影响更大")
    
    print("\n3. 完整训练效果 (D: train both):")
    print(f"   Score change: {exp_d['score_change']:+.4f}")
    print(f"   与 A+B 对比: {exp_d['score_change']:.4f} vs {exp_a['score_change'] + exp_b['score_change']:.4f}")
    
    return results


def main():
    """主实验"""
    # 先做 10-step 验证设计正确性
    results_10 = run_correct_analysis(num_steps=10)
    
    # 再做 50-step 看长期趋势
    print("\n\n" + "="*70)
    print("="*70)
    print("现在执行 50-step 长期趋势分析")
    print("="*70)
    results_50 = run_correct_analysis(num_steps=50)
    
    # 最终结论
    print("\n\n" + "="*70)
    print("最终结论")
    print("="*70)
    print("""
基于正确的来源拆解实验，可以得出:

1. 如果 C组 (freeze both) 的 score_change ≈ 0:
   → 实验设计正确，测量可信

2. 如果 A组 (freeze head) 的 |score_change| > B组 (freeze backbone):
   → Writeback 漂移主要来自 backbone 特征变化
   → Guard 应该保护 backbone 特征

3. 如果 B组 |score_change| > A组:
   → Writeback 漂移主要来自 head 参数变化
   → Guard 应该保护 head 参数

4. 如果两者都有显著影响:
   → 需要 dual-level guard
""")


if __name__ == "__main__":
    main()
